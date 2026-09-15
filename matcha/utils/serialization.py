"""Helpers for loading checkpoints with PyTorch >= 2.6.

Since PyTorch 2.6 ``torch.load`` defaults to ``weights_only=True``, which refuses to
unpickle anything that is not a tensor or a plain built-in. Matcha checkpoints are
written by Lightning and keep the OmegaConf configuration in ``hyper_parameters``,
so loading one fails with::

    UnpicklingError: Weights only load failed. [...]
    Unsupported global: GLOBAL omegaconf.dictconfig.DictConfig was not an allowed global by default.

Allow-listing the handful of classes that Matcha checkpoints actually contain keeps
the ``weights_only`` protection in place, so this is done for every load below.
If a checkpoint holds something else on top of that and you trust where it came
from, set ``MATCHA_UNSAFE_CHECKPOINT_LOAD=1`` to fall back to full unpickling.
"""

import collections
import functools
import os
import pickle
import typing
from contextlib import contextmanager

import torch
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode

UNSAFE_LOAD_ENV_VAR = "MATCHA_UNSAFE_CHECKPOINT_LOAD"

_SAFE_GLOBALS_REGISTERED = False


def register_safe_globals():
    """Allow-list the non-tensor classes stored inside Matcha/Lightning checkpoints.

    Call this before any ``torch.load`` / ``load_from_checkpoint`` call. It is
    idempotent and a no-op on PyTorch versions without the ``weights_only`` unpickler.
    """
    global _SAFE_GLOBALS_REGISTERED  # pylint: disable=global-statement
    if _SAFE_GLOBALS_REGISTERED or not hasattr(torch.serialization, "add_safe_globals"):
        return

    torch.serialization.add_safe_globals(
        [
            # the hydra config that Lightning stores in `hyper_parameters`
            DictConfig,
            ListConfig,
            ContainerMetadata,
            Metadata,
            AnyNode,
            # containers used by the OmegaConf internals above
            collections.defaultdict,
            typing.Any,
            dict,
            list,
            int,
            float,
            str,
            bool,
            # the training state that Lightning keeps in `callbacks` and `optimizer_states`
            functools.partial,
            torch.optim.Adam,
        ]
    )
    _SAFE_GLOBALS_REGISTERED = True


def unsafe_load_requested():
    """Whether the user opted into full (unrestricted) unpickling."""
    return os.environ.get(UNSAFE_LOAD_ENV_VAR, "0").lower() in ("1", "true", "yes")


@contextmanager
def checkpoint_loading(checkpoint_path=None):
    """Context manager that makes ``torch.load`` able to read Matcha checkpoints.

    It allow-lists the classes Matcha checkpoints contain, and - only when
    ``MATCHA_UNSAFE_CHECKPOINT_LOAD`` is set - disables the ``weights_only``
    restriction entirely. Both also cover loaders that call ``torch.load``
    internally, such as ``LightningModule.load_from_checkpoint``.
    """
    register_safe_globals()
    original_load = torch.load

    if unsafe_load_requested():

        def _full_unpickling_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        torch.load = _full_unpickling_load

    try:
        yield
    except pickle.UnpicklingError as exception:
        raise _explain_unpickling_error(exception, checkpoint_path) from exception
    finally:
        torch.load = original_load


def load_checkpoint(checkpoint_path, map_location=None):
    """``torch.load`` that can read Matcha and HiFi-GAN checkpoints on PyTorch >= 2.6."""
    with checkpoint_loading(checkpoint_path):
        return torch.load(checkpoint_path, map_location=map_location)


def _explain_unpickling_error(exception, checkpoint_path):
    """Append the Matcha specific escape hatch to PyTorch's ``weights_only`` error."""
    if unsafe_load_requested():
        return exception

    checkpoint = checkpoint_path if checkpoint_path is not None else "The checkpoint"
    return pickle.UnpicklingError(
        f"{exception}\n\n"
        f"[matcha] {checkpoint} contains an object that is not on the allow-list of "
        f"matcha.utils.serialization.register_safe_globals(). If you trust the source of this "
        f"checkpoint, re-run with {UNSAFE_LOAD_ENV_VAR}=1 to load it with weights_only=False."
    )
