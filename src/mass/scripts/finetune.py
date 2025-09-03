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

from mass.data.datasets.registry import get_dataset
from mass.modules.encoder import ImageEncoder
from mass.modules.heads import get_classification_head
from mass.pl_module.image_classifier import ImageClassifier
from mass.utils.io_utils import get_class, load_model_from_artifact, upload_model_to_hf
from mass.utils.utils import build_callbacks

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
        cfg.nn.module.model.model_name,
        cfg.nn.data.train_dataset,
        cfg.nn.data.data_path,
        cfg.misc.ckpt_path,
        cache_dir=cfg.misc.cache_dir,
        openclip_cachedir=cfg.misc.openclip_cachedir,
    )

    model_class = get_class(classification_head)


    model: ImageClassifier = hydra.utils.instantiate(
        cfg.nn.module,
        encoder=image_encoder,
        classifier=classification_head,
        _recursive_=False,
    )

    dataset = 

    model.freeze_head()

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    storage_dir: str = cfg.core.storage_dir

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        max_epochs=cfg.nn.data.dataset.ft_epochs,
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(
        model=model,
        train_dataloaders=dataset.train_loader,
        ckpt_path=template_core.trainer_ckpt_path,
    )

    pylogger.info("Starting testing!")
    trainer.test(model=model, dataloaders=dataset.test_loader)

    upload_model_to_hf(model.encoder, cfg.nn.module.model.model_name, dataset)

    if logger is not None:
        logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
