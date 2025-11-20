import logging
import os
from pathlib import Path
from pydoc import locate
import tempfile

from nn_core.callbacks import NNTemplateCore
from nn_core.common.utils import enforce_tags
from nn_core.model_logging import NNLogger
from huggingface_hub import hf_hub_download

from omegaconf import DictConfig
import torch

from mass.modules.encoder import ClassificationHead, ImageEncoder
from mass.modules.heads import get_classification_head
from huggingface_hub import HfApi, create_repo, upload_folder

from mass import PROJECT_ROOT

pylogger = logging.getLogger(__name__)


def get_class(model):
    return model.__class__.__module__ + "." + model.__class__.__qualname__


def get_classification_heads(datasets, encoder_name, ckpt_path, cache_dir):
    classification_heads = []

    for dataset_name in datasets:
        classification_head = get_classification_head(
            encoder_name,
            dataset_name,
            ckpt_path,
            openclip_cachedir=cache_dir,
        )

        classification_heads.append(classification_head)

    return classification_heads


def boilerplate(cfg):
    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    template_core = NNTemplateCore(
        restore_cfg=None,  # Disable checkpoint restoration for evaluation
    )
    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )

    # logger.upload_source()

    return logger, template_core


def load_model_from_disk(model_path, model_name=None) -> ImageEncoder:

    pylogger.info(f"Loading model from disk {model_path}")

    loaded = torch.load(model_path)

    # if it's a statedict, we need to create the model first
    if not isinstance(loaded, ImageEncoder):

        state_dict = loaded

        model = ImageEncoder(model_name)
        model.load_state_dict(state_dict)
        return model

    return loaded


def load_model_from_hf(model_name, dataset_name="base") -> ImageEncoder:

    model_path = f"crisostomi/{model_name}-{dataset_name}"

    ckpt_path = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
    state_dict = torch.load(ckpt_path, map_location="cpu")

    model = ImageEncoder(model_name)
    model.load_state_dict(state_dict)
    return model


def upload_model_to_hf(model, model_name, dataset_name):
    """ """

    with open(f"{PROJECT_ROOT}/secrets.txt", "r") as f:
        hf_token = f.readline().strip()

    repo_id = f"crisostomi/{model_name}-{dataset_name}"
    create_repo(
        repo_id, repo_type="model", private=False, exist_ok=True, token=hf_token
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        weights_path = os.path.join(tmpdir, "pytorch_model.bin")
        torch.save(model.state_dict(), weights_path)

        upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Initial upload",
            token=hf_token,
        )

    print(f"✅ Uploaded to https://huggingface.co/{repo_id}")
