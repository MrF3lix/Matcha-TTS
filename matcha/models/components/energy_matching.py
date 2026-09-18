"""Energy Matching decoder for Matcha-TTS.

Drop-in replacement for :class:`matcha.models.components.flow_matching.CFM` implementing
Energy Matching (Balcerak et al., 2025). Instead of regressing a time-dependent velocity
field v(x, t), we learn a *time-independent* scalar potential V(x | mu, spks) whose negative
gradient -grad V transports noise to data.

Training has two phases, selected by the trainer's global step:

  phase 1  ||grad V(x_t) - (x_0 - x_1)||^2 along linear interpolants x_t = (1 - t) x_0 + t x_1
           with t ~ U(0, tau_star). This is exactly the CFM objective with v = -grad V.
  phase 2  phase-1 loss + cd_weight * (V(data) - V(negatives)) / eps_max, where the negatives
           come from Langevin dynamics on V. This shapes V into a proper energy so that
           Langevin sampling targets the Boltzmann density p(x) ~ exp(-V(x) / eps_max).

Sampling runs gradient flow dx = -grad V dt for tau_star (Euler, `n_timesteps` steps, the
analogue of `CFM.solve_euler`), optionally followed by `sample_langevin_steps` Langevin steps
at temperature eps_max.
"""

import contextlib
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from matcha.models.components.decoder import Decoder
from matcha.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@contextlib.contextmanager
def _autograd_enabled():
    """Allow autograd (needed for grad V) inside `torch.inference_mode()` / `torch.no_grad()`.

    Matcha's `synthesise` is decorated with `torch.inference_mode()` and Lightning runs the
    validation loop under it as well; both would otherwise break the gradient computation.
    """
    with torch.inference_mode(False), torch.enable_grad():
        yield


def _regular(t):
    """Inference tensors cannot take part in autograd; clone them into regular tensors."""
    return t.clone() if isinstance(t, torch.Tensor) and t.is_inference() else t


def _slice(t, start, end):
    return None if t is None else t[start:end]


def _param(params, key, default):
    return default if params is None else getattr(params, key, default)


class EnergyMatching(torch.nn.Module):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__()
        self.n_feats = out_channel
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.sigma_min = _param(cfm_params, "sigma_min", 1e-4)

        # Potential / interpolation
        self.output_scale = _param(cfm_params, "output_scale", 1.0)
        self.tau_star = _param(cfm_params, "tau_star", 1.0)
        self.eps_max = _param(cfm_params, "eps_max", 0.05)
        self.ot_coupling = _param(cfm_params, "ot_coupling", False)

        # Phase 2 (contrastive term)
        self.phase1_steps = _param(cfm_params, "phase1_steps", 40_000)
        self.cd_weight = _param(cfm_params, "cd_weight", 1e-3)
        self.neg_langevin_steps = _param(cfm_params, "neg_langevin_steps", 30)
        self.neg_langevin_dt = _param(cfm_params, "neg_langevin_dt", 0.05)
        self.trim_alpha = _param(cfm_params, "trim_alpha", 0.1)
        self.clamp_beta = _param(cfm_params, "clamp_beta", 0.05)

        # Sampling
        self.sample_langevin_steps = _param(cfm_params, "sample_langevin_steps", 0)

        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        # Same U-Net as the CFM estimator, but with a single output channel. Summed over the
        # (masked) frames it gives the scalar potential V(x | mu, spks).
        self.estimator = Decoder(in_channels=in_channels, out_channels=1, **decoder_params)

        # Detached sub-losses of the last `compute_loss` call, handy for logging/debugging.
        self.loss_components = {}
        self._phase = None

    # ------------------------------------------------------------------ potential
    def energy(self, x, mask, mu, spks=None, cond=None):
        """Scalar potential V(x | mu, spks).

        Args:
            x (torch.Tensor): shape (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): shape (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): encoder output, shape (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): shape (batch_size, spk_emb_dim)

        Returns:
            torch.Tensor: shape (batch_size,). Sum over masked frames, so that grad_x V has a
                per-frame scale that does not depend on the sequence length.
        """
        t = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)  # V is time-independent
        out = self.estimator(x, mask, mu, t, spks, cond)  # (batch_size, 1, mel_timesteps), masked
        return out.sum(dim=(1, 2)) * self.output_scale

    def grad_energy(self, x, mask, mu, spks=None, cond=None, create_graph=False):
        """grad_x V(x | mu, spks) with the same shape as x.

        With `create_graph=True` the result is differentiable w.r.t. the parameters (needed
        for the phase-1 loss); this requires the math SDPA kernel since the fused attention
        kernels do not implement double backward.
        """
        kernel = sdpa_kernel(SDPBackend.MATH) if create_graph else contextlib.nullcontext()
        with _autograd_enabled(), kernel:
            x = x.detach().clone().requires_grad_(True)
            mask, mu, spks, cond = (_regular(v) for v in (mask, mu, spks, cond))
            energy = self.energy(x, mask, mu, spks, cond)
            (grad,) = torch.autograd.grad(energy.sum(), x, create_graph=create_graph)
        return grad

    # ------------------------------------------------------------------ dynamics
    def langevin(self, x, mask, mu, spks, cond, n_steps, dt, eps_fn):
        """Euler-Maruyama on dx = -grad V dt + sqrt(2 eps) dW, with eps = eps_fn(step).

        eps_fn returning 0 gives plain gradient flow (the deterministic transport, i.e. the
        analogue of the flow-matching ODE).
        """
        x = x.detach()
        for m in range(n_steps):
            grad = self.grad_energy(x, mask, mu, spks, cond, create_graph=False)
            x = x - dt * grad.detach()
            eps = eps_fn(m)
            if eps > 0:
                x = x + math.sqrt(2 * dt * eps) * torch.randn_like(x) * mask
            x = x.detach()
        return x

    def _eps_schedule(self, tau):
        """Zero temperature (pure transport) until tau_star, Langevin at eps_max afterwards."""
        return self.eps_max if tau >= self.tau_star else 0.0

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        """Generate a mel-spectrogram (same interface as `CFM.forward`).

        Args:
            mu (torch.Tensor): output of encoder, shape (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output mask, shape (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of Euler steps for the gradient flow over [0, tau_star]
            temperature (float, optional): scaling of the initial noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker embedding, shape (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram, shape (batch_size, n_feats, mel_timesteps)
        """
        x = torch.randn_like(mu) * temperature
        dt = self.tau_star / n_timesteps
        # Transport: gradient flow on V (deterministic), the counterpart of CFM.solve_euler.
        x = self.langevin(x, mask, mu, spks, cond, n_timesteps, dt, lambda m: 0.0)
        # Refinement: Langevin dynamics at temperature eps_max (only meaningful after phase 2).
        x = self.langevin(x, mask, mu, spks, cond, self.sample_langevin_steps, dt, lambda m: self.eps_max)
        return x * mask

    # ------------------------------------------------------------------ training
    def compute_loss(self, x1, mask, mu, spks=None, cond=None, step=None):
        """Computes the energy matching loss (same interface as `CFM.compute_loss`).

        Args:
            x1 (torch.Tensor): target, shape (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask, shape (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder, shape (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding, shape (batch_size, spk_emb_dim)
            step (int, optional): trainer global step, selects phase 1 / phase 2. None -> phase 1.

        Returns:
            loss: energy matching loss
            y: interpolant x_t, shape (batch_size, n_feats, mel_timesteps)
        """
        b = x1.shape[0]
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype) * self.tau_star
        z = torch.randn_like(x1)
        if self.ot_coupling:
            z = self._ot_couple(z, x1, mask)

        # Same interpolant and target as CFM; grad V has to match the *negative* velocity.
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        grad_v = self.grad_energy(y, mask, mu, spks, cond, create_graph=True)
        fm_loss = F.mse_loss(grad_v, -u * mask, reduction="sum") / (torch.sum(mask) * u.shape[1])

        phase = 2 if step is not None and step >= self.phase1_steps else 1
        if phase != self._phase:
            log.info(f"Energy matching: entering phase {phase} at step {step}")
            self._phase = phase

        loss = fm_loss
        cd_loss = torch.zeros((), device=loss.device)
        if phase == 2:
            cd_loss = self._contrastive_loss(x1, z, mask, mu, spks, cond)
            loss = loss + self.cd_weight * cd_loss

        self.loss_components = {"fm_loss": fm_loss.detach(), "cd_loss": cd_loss.detach(), "phase": phase}
        return loss, y

    def _contrastive_loss(self, x1, z, mask, mu, spks, cond):
        """(V(data) - V(negatives)) / eps_max with Langevin negatives, trimmed and clamped.

        Half of the negatives start at the data (explore the neighbourhood of the data
        manifold at temperature eps_max), the other half at noise (gradient flow until
        tau_star, then Langevin) to find spurious minima of V. Every negative keeps the
        conditioning (mu, mask, spks) of the sample it started from, so the energy gap is
        computed pairwise and per frame, which keeps variable-length batches comparable.
        """
        b = x1.shape[0]
        half = b // 2
        n, dt = self.neg_langevin_steps, self.neg_langevin_dt

        negatives = []
        if half > 0:
            negatives.append(
                self.langevin(
                    x1[:half],
                    mask[:half],
                    mu[:half],
                    _slice(spks, 0, half),
                    _slice(cond, 0, half),
                    n,
                    dt,
                    lambda m: self.eps_max,
                )
            )
        negatives.append(
            self.langevin(
                z[half:],
                mask[half:],
                mu[half:],
                _slice(spks, half, b),
                _slice(cond, half, b),
                n,
                dt,
                lambda m: self._eps_schedule(m * dt),
            )
        )
        x_neg = torch.cat(negatives, dim=0)

        v_data = self.energy(x1, mask, mu, spks, cond)
        v_neg = self.energy(x_neg, mask, mu, spks, cond)
        n_frames = mask.sum(dim=(1, 2)).clamp(min=1)
        gap = (v_data - v_neg) / n_frames

        # Drop the negatives with the highest energy (most negative gap): Langevin outliers.
        k = int(self.trim_alpha * gap.numel())
        gap = torch.sort(gap)[0][k:] if k > 0 else gap
        cd_loss = gap.mean() / self.eps_max
        return torch.clamp(cd_loss, min=-self.clamp_beta)

    @staticmethod
    def _ot_couple(z, x1, mask):
        """Re-pair noise and data samples within the minibatch by optimal transport.

        Cost is the masked squared distance ||mask_j * (z_i - x1_j)||^2, so the pairing is
        consistent with the loss normalisation. Note that for conditional generation pairs
        cross conditionings (z_i gets the mu of x1_j), which weakens the usual straightening
        benefit; off by default.
        """
        from scipy.optimize import (
            linear_sum_assignment,  # pylint: disable=import-outside-toplevel
        )

        x1 = x1 * mask
        m = mask.squeeze(1).to(z.dtype)  # (b, T)
        cost = (z**2).sum(dim=1) @ m.T  # ||mask_j * z_i||^2
        cost = cost - 2 * z.flatten(1) @ x1.flatten(1).T
        cost = cost + (x1**2).sum(dim=(1, 2))[None, :]
        row, col = linear_sum_assignment(cost.detach().float().cpu().numpy())
        z_new = torch.empty_like(z)
        z_new[torch.as_tensor(col, device=z.device)] = z[torch.as_tensor(row, device=z.device)]
        return z_new
