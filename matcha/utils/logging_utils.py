from typing import Any, Dict

from lightning.pytorch.utilities import rank_zero_only
from omegaconf import OmegaConf

from matcha.utils import pylogger

log = pylogger.get_pylogger(__name__)


def log_image(logger, key, image, step):
    """Log a single HWC image, whichever logger is configured.

    `logger.experiment` is logger specific: TensorBoard exposes a `SummaryWriter` (`add_image`),
    while wandb exposes a `Run` (no `add_image`, hence `AttributeError` when the TensorBoard API is
    used directly). Loggers that support images through Lightning implement `log_image`.
    """
    if hasattr(logger, "log_image"):  # wandb, comet, neptune, mlflow
        logger.log_image(key=key, images=[image], step=step)
    elif hasattr(logger.experiment, "add_image"):  # tensorboard
        logger.experiment.add_image(key, image, step, dataformats="HWC")
    else:
        log.debug("Logger %s cannot log images, skipping %s", type(logger).__name__, key)


@rank_zero_only
def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"])
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
