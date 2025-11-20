import logging
import os

from typing import Dict, List, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningModule
from tqdm import tqdm

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

from mass.modules.encoder import ImageEncoder
from mass.modules.heads import get_classification_head
from mass.pl_module.image_classifier import ImageClassifier
from mass.utils.io_utils import (
    load_model_from_hf,
    upload_model_to_hf,
)
from hydra.utils import instantiate

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig):
    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )

    classification_head = get_classification_head(
        cfg.nn.encoder.model_name,
        cfg.dataset.name,
        ckpt_path=cfg.misc.ckpt_path,
        openclip_cachedir=cfg.misc.openclip_cachedir,
        device=cfg.device,
    )

    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )

    model: ImageClassifier = hydra.utils.instantiate(
        cfg.nn.module,
        encoder=zeroshot_encoder,
        classifier=classification_head,
        _recursive_=False,
    )

    model.task_name = cfg.dataset.name

    dataset = instantiate(
        cfg.dataset,
        preprocess_fn=zeroshot_encoder.val_preprocess,
        batch_size=cfg.train.batch_size,
    )

    model.freeze_head()

    storage_dir: str = cfg.core.storage_dir

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        logger=logger,
        enable_checkpointing=False,  # Completely disable checkpointing
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(
        model=model,
        train_dataloaders=dataset.train_loader,
    )

    pylogger.info("Starting testing!")
    trainer.test(model=model, dataloaders=dataset.test_loader)

    upload_model_to_hf(model.encoder, cfg.nn.encoder.model_name, cfg.dataset.name)

    logger.log_configuration(model, cfg)

    if logger is not None:
        logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
