"""Vocode a few validation utterances each epoch and send them to the logger.

Kept out of the LightningModule on purpose: the vocoder is a separate model with its own
weights and device, and which vocoder is appropriate depends on which mel front-end the run
was trained for. As a callback it is opt-in per experiment and the model stays unaware of it.

Two vocoders are supported: VocBulwark for the `vocbulwark` mel front-end (96 mels, 24 kHz) and
HiFi-GAN for the standard `hifigan` front-end (80 mels, 22.05 kHz).
"""

from pathlib import Path

import torch
from lightning import Callback

from matcha.utils import pylogger
from matcha.utils.logging_utils import log_audio
from matcha.utils.model import denormalize
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir

log = pylogger.get_pylogger(__name__)

HIFIGAN_VOCODERS = ("hifigan_T2_v1", "hifigan_univ_v1")
HIFIGAN_SAMPLE_RATE = 22050


class LogAudioSamples(Callback):
    """Synthesise validation utterances and log them as audio.

    Args:
        vocoder: "vocbulwark", or one of the HiFi-GAN generators the CLI knows ("hifigan_T2_v1"
            for LJSpeech, "hifigan_univ_v1" for multi-speaker). HiFi-GAN weights are downloaded
            to the Matcha data dir on first use, like `matcha-tts` does.
        speaker_embedding: path to the .pt embedding the VocBulwark vocoder requires
            (see scripts/compute_speaker_embedding.py). For a multi-speaker run this is the
            [n_spks, 768] table, indexed by the batch's speaker ids. Ignored for HiFi-GAN.
        n_samples: how many utterances from the first validation batch to synthesise.
        every_n_epochs: log every N epochs. Vocoding is not free, so this is not 1 by default.
        n_timesteps: ODE steps for synthesis; 10 matches the mel plots logged alongside.
        vocoder_device: where to run the vocoder. Defaults to CPU so a ~0.5 GB model does not
            compete with training for GPU memory; set to "cuda" if you have headroom and want
            the logging step to be quicker.
        log_ground_truth: on the first logged epoch, also vocode the *ground-truth* mels. That
            is copy-synthesis, so it shows the vocoder's ceiling and gives you a fixed
            reference to compare every later epoch against.
        denoiser_strength: strength of the WaveGlow-style denoiser applied to HiFi-GAN output,
            same default as the CLI. Not used for VocBulwark.
    """

    def __init__(
        self,
        vocoder="vocbulwark",
        speaker_embedding=None,
        n_samples=2,
        every_n_epochs=1,
        n_timesteps=10,
        vocoder_device="cpu",
        log_ground_truth=True,
        denoiser_strength=0.00025,
    ):
        super().__init__()
        if vocoder != "vocbulwark" and vocoder not in HIFIGAN_VOCODERS:
            raise ValueError(f"unknown vocoder {vocoder!r}, expected 'vocbulwark' or one of {HIFIGAN_VOCODERS}")
        self.vocoder = vocoder
        self.speaker_embedding = None if speaker_embedding is None else Path(speaker_embedding)
        self.n_samples = n_samples
        self.every_n_epochs = every_n_epochs
        self.n_timesteps = n_timesteps
        self.vocoder_device = vocoder_device
        self.log_ground_truth = log_ground_truth
        self.denoiser_strength = denoiser_strength

        # Fail now rather than an epoch into a 24 h job.
        if self.vocoder == "vocbulwark" and (self.speaker_embedding is None or not self.speaker_embedding.is_file()):
            raise FileNotFoundError(
                f"speaker embedding not found: {self.speaker_embedding}\n"
                "Create one with: python scripts/compute_speaker_embedding.py "
                "--wav-dir data/LJSpeech-1.1/wavs --out data/ljspeech_speaker_embedding.pt"
            )

        self._vocoder = None
        self._denoiser = None
        self._sample_rate = None
        self._disabled = False
        self._logged_ground_truth = False

    def _load_vocoder(self):
        """Load lazily: the weights are fetched remotely, so defer until we truly need them."""
        if self._vocoder is None and not self._disabled:
            try:
                if self.vocoder == "vocbulwark":
                    from matcha.vocbulwark import SAMPLE_RATE, load_vocbulwark

                    self._vocoder = load_vocbulwark(self.speaker_embedding, self.vocoder_device)
                    self._sample_rate = SAMPLE_RATE
                else:
                    # Same weights, location and denoiser as `matcha-tts --vocoder <name>`.
                    from matcha.cli import VOCODER_URLS, load_hifigan
                    from matcha.hifigan.denoiser import Denoiser

                    checkpoint = get_user_data_dir() / self.vocoder
                    assert_model_downloaded(checkpoint, VOCODER_URLS[self.vocoder])
                    self._vocoder = load_hifigan(checkpoint, self.vocoder_device)
                    self._denoiser = Denoiser(self._vocoder, mode="zeros")
                    self._sample_rate = HIFIGAN_SAMPLE_RATE
                log.info("Audio sample logger: %s ready on %s", self.vocoder, self.vocoder_device)
            except Exception as exc:  # noqa: BLE001 - logging must never kill a training run
                self._disabled = True
                log.warning("Audio sample logger disabled, vocoder failed to load: %s", exc)
        return self._vocoder

    def _to_waveform(self, mel, spks=None):
        mel = mel.detach().float().to(self.vocoder_device)
        with torch.no_grad():
            if getattr(self._vocoder, "n_speakers", 1) > 1:
                if spks is None:
                    raise ValueError("multi-speaker VocBulwark table needs speaker ids, but the batch has none")
                audio = self._vocoder(mel, spks.to(self.vocoder_device)).clamp(-1, 1)
            else:
                audio = self._vocoder(mel).clamp(-1, 1)
            if self._denoiser is not None:
                audio = self._denoiser(audio.squeeze(), strength=self.denoiser_strength)
        return audio.squeeze().cpu().numpy()

    def on_validation_end(self, trainer, pl_module):
        if not trainer.is_global_zero or self._disabled or trainer.sanity_checking:
            return
        if self.every_n_epochs <= 0 or trainer.current_epoch % self.every_n_epochs != 0:
            return
        if self._load_vocoder() is None:
            return

        batch = next(iter(trainer.val_dataloaders))
        n = min(self.n_samples, batch["x"].shape[0])
        step = trainer.current_epoch

        try:
            # Copy-synthesis of the real mels: a fixed upper bound to compare against. The
            # batch holds normalised mels, so undo that first -- the vocoder wants the
            # feature space the mels were extracted in.
            if self.log_ground_truth and not self._logged_ground_truth:
                for i in range(n):
                    mel = batch["y"][i, :, : batch["y_lengths"][i]].unsqueeze(0)
                    mel = denormalize(mel, pl_module.mel_mean, pl_module.mel_std)
                    spks = batch["spks"][i].unsqueeze(0) if batch["spks"] is not None else None
                    for logger in trainer.loggers:
                        log_audio(logger, f"audio_original/{i}", self._to_waveform(mel, spks), step, self._sample_rate)
                self._logged_ground_truth = True

            for i in range(n):
                x = batch["x"][i].unsqueeze(0).to(pl_module.device)
                x_lengths = batch["x_lengths"][i].unsqueeze(0).to(pl_module.device)
                spks = batch["spks"][i].unsqueeze(0).to(pl_module.device) if batch["spks"] is not None else None

                # `synthesise` already denormalises its "mel" output.
                output = pl_module.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=self.n_timesteps, spks=spks)
                for logger in trainer.loggers:
                    log_audio(
                        logger, f"audio_generated/{i}", self._to_waveform(output["mel"], spks), step, self._sample_rate
                    )
        except Exception as exc:  # noqa: BLE001
            self._disabled = True
            log.warning("Audio sample logger disabled after an error: %s", exc)
