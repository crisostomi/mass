import logging
from pathlib import Path
from pydoc import locate
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.callbacks import NNTemplateCore
from nn_core.model_logging import NNLogger
from omegaconf import DictConfig

from mass.modules.heads import get_classification_head
import torch

from nn_core.serialization import load_model

from mass.modules.encoder import ClassificationHead, ImageEncoder

pylogger = logging.getLogger(__name__)


def get_classification_heads(cfg: DictConfig):
    classification_heads = []

    for dataset_name in cfg.eval_datasets:

        classification_head = get_classification_head(
            cfg.nn.module.encoder.model_name,
            dataset_name,
            cfg.nn.data.data_path,
            cfg.misc.ckpt_path,
            cache_dir=cfg.misc.cache_dir,
            openclip_cachedir=cfg.misc.openclip_cachedir,
        )

        classification_heads.append(classification_head)

    return classification_heads


def boilerplate(cfg):
    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    num_tasks = len(cfg.eval_datasets)
    cfg.core.tags.append(f"n{num_tasks}")
    cfg.core.tags.append(f"{cfg.nn.module.encoder.model_name}")

    template_core = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )

    logger.upload_source()

    return logger, template_core


def load_model_from_artifact(run, artifact_path):
    pylogger.info(f"Loading model from artifact {artifact_path}")

    artifact = run.use_artifact(artifact_path)
    artifact.download()

    ckpt_path = Path(artifact.file())

    model_class = locate(artifact.metadata["model_class"])

    if model_class == ImageEncoder:
        model = model_class(**artifact.metadata)
    elif model_class == ClassificationHead:
        model = model_class(normalize=True, **artifact.metadata)

    model.load_state_dict(torch.load(ckpt_path))

    return model


def load_model_from_disk(model_path, model_name=None) -> ImageEncoder:

    loaded = torch.load(model_path)

    # if it's a statedict, we need to create the model first
    if not isinstance(loaded, ImageEncoder):

        state_dict = loaded

        model = ImageEncoder(model_name)
        model.load_state_dict(state_dict)
        return model

    return loaded


def get_class(model):
    return model.__class__.__module__ + "." + model.__class__.__qualname__
