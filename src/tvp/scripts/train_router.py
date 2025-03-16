## Imports
import copy
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import open_clip

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import tvp  # noqa
from tvp.data.datasets.registry import get_task_evaluation_dataset
from tvp.modules.encoder import ImageEncoder
from tvp.pl_module.router_task_classifier import RouterTaskClassifier
from tvp.modules.router import AbstractRouter
from tvp.utils.io_utils import load_model_from_disk
from tvp.utils.utils import (
    compute_task_dict, 
    apply_dict_to_model,
    build_callbacks,
    get_finetuning_accuracies,
    add_normalized_accuracy,
    compute_avg_accuracy,
    print_memory,
    get_routing_weights,
)
from tvp.task_vectors.task_singular_vectors import *
import json
import os

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def boilerplate(cfg):
    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    template_core = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )

    return logger, template_core


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """

    seed_index_everything(cfg)

    logger, template_core = boilerplate(cfg)

    ntasks = len(cfg.eval_datasets)

    # Temporarily disable struct mode to allow dynamic update
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.ntasks = ntasks  # Now we can safely update it
    omegaconf.OmegaConf.set_struct(cfg, True)  # Re-enable struct mode

    pylogger.info(f"Number of tasks (ntasks): {cfg.ntasks}")

    # only has vision encoder, no text transformer
    zeroshot_encoder_statedict = load_model_from_disk(cfg.misc.pretrained_checkpoint)

    finetuned_name = (
        lambda name: cfg.misc.ckpt_path
        + "/"
        + name
        + "Val"
        + "/"
        + "nonlinear_finetuned.pt"
    )
    finetuned_models = {
        dataset: load_model_from_disk(finetuned_name(dataset))
        for dataset in cfg.task_vectors.to_apply
    }

    pylogger.info(f"Checkpoint saving to: {cfg.misc.checkpoint_dir}")
    # pylogger.info(f"Number of finetuned models: {len(finetuned_models)}")
    pylogger.info(f"Finetuned models: {list(finetuned_models.keys())}")

    task_dicts = {}
    for dataset in cfg.task_vectors.to_apply:
        task_dicts[dataset] = compute_task_dict(
            zeroshot_encoder_statedict, finetuned_models[dataset]
        )
        del finetuned_models[dataset]  # Delete one model at a time
        torch.cuda.empty_cache()

    print_memory("after computing task dicts")

    svd_dict = get_svd_dict(task_dicts, cfg.eval_datasets, cfg.misc.svd_path)
    print_memory("after computing svd dict")

    multi_task_vector = sum_svd(
        ref_state_dict=copy.deepcopy(zeroshot_encoder_statedict),
        svd_dicts=svd_dict,
    )
    print_memory("after computing merged vector")

    seed_index_everything(cfg)
    pylogger.info(f"Training {cfg.nn.module.router.name}-MASS")

    assert (
        cfg.nn.module.router.name == "linear"
    ), "The only trainable router is the linear one!"

    zeroshot_encoder: ImageEncoder = instantiate(
        cfg.nn.module.encoder
    )  # the second pass backbone
    zeroshot_encoder.load_state_dict(zeroshot_encoder_statedict, strict=False)

    merged_encoder: ImageEncoder = copy.deepcopy(
        zeroshot_encoder
    )  # the first pass backbone
    merged_encoder = apply_dict_to_model(multi_task_vector, merged_encoder)

    router: AbstractRouter = instantiate(
        cfg.nn.module.router,
        encoder=merged_encoder,
        svd_dict=svd_dict,
        cfg=cfg,
        _recursive_=False,
    )

    model: RouterTaskClassifier = instantiate(
        cfg.nn.module,
        router=router,
        _recursive_=False,
    )

    pylogger.info(f"Training linear router on {len(cfg.eval_datasets)} tasks")
    model.set_metrics(len(cfg.eval_datasets))

    print_memory("before train")

    dataset = get_task_evaluation_dataset(
        cfg.eval_datasets,
        preprocess_fn=zeroshot_encoder.train_preprocess,
        location=cfg.nn.data.data_path,
        batch_size=cfg.nn.data.batch_size.train,
    )

    callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )
    trainer = pl.Trainer(
        default_root_dir=cfg.core.storage_dir,
        plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
        logger=logger,
        callbacks=callbacks,
        limit_train_batches= cfg.number_of_train_batches, 
        limit_val_batches= cfg.number_of_val_batches,
        **cfg.train.trainer,
    )

    pylogger.info(f"Training for task inference!")


    trainer.fit(
        model=model,
        train_dataloaders=dataset.train_loader,
        val_dataloaders=dataset.test_loader,
    )
    for key, value in trainer.callback_metrics.items():
        pylogger.info(f"{key}: {value}")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="train_router.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
