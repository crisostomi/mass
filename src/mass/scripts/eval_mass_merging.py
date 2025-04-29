import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import open_clip
import wandb
import torch.nn as nn

import hydra
import omegaconf
import pytorch_lightning as pl
import tqdm
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO
from mass.pl_module.image_classifier import ImageClassifier

# Force the execution of __init__.py if this file is executed directly.
import mass  # noqa
from mass.data.datasets.registry import get_dataset
from mass.modules.encoder import ClassificationHead, ImageEncoder
from mass.modules.projection_router import ProjectionRouter
from mass.modules.nn_router import NNRouter
from mass.modules.heads import get_classification_head
from mass.modules.router import AbstractRouter
from mass.pl_module.encoder import EncoderWrapper
from mass.utils.io_utils import load_model_from_disk
from mass.utils.plots import plot_interactive_radar_chart
from mass.utils.utils import (
    compute_task_dict, 
    apply_dict_to_model,
    build_callbacks,
    get_finetuning_accuracies,
    add_normalized_accuracy,
    compute_avg_accuracy,
    print_memory,
    get_routing_weights,
    svd_key_from_layer
)
from mass.task_vectors.task_singular_vectors import *
import json
import os

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def boilerplate(cfg):
    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))

    num_tasks = len(cfg.eval_datasets)
    cfg.core.tags.append(f"n{num_tasks}")
    cfg.core.tags.append(f'{cfg.nn.module.encoder.model_name}')
    cfg.core.tags.append(f'optim_notebook')

    template_core = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )
    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )

    logger.upload_source()

    return logger, template_core

def run(cfg: DictConfig) -> str:

    seed_index_everything(cfg)

    logger, template_core = boilerplate(cfg)


    # upperbound accuracies, used for logging the normalized accuracy
    finetuned_accuracies = get_finetuning_accuracies(cfg.misc.finetuned_accuracy_path)

    # only has vision encoder, no text transformer
    zeroshot_encoder_statedict = load_model_from_disk(cfg.misc.pretrained_checkpoint)

    zeroshot_encoder: ImageEncoder = instantiate(
        cfg.nn.module.encoder
    )  # the second pass backbone

    zeroshot_encoder.load_state_dict(zeroshot_encoder_statedict, strict=False)

    finetuned_name = (
    lambda name: Path(cfg.misc.ckpt_path) / f"{name}Val" / "nonlinear_finetuned.pt"
    )
    finetuned_models = {
        dataset: load_model_from_disk(finetuned_name(dataset))
        for dataset in cfg.task_vectors.to_apply
    }

    num_tasks = len(cfg.eval_datasets)

    pylogger.info(f"Number of tasks: {len(cfg.eval_datasets)}")
    pylogger.info(f"Finetuned models: {list(finetuned_models.keys())}")

    datasets = {dataset_name: get_dataset(
            dataset_name,
            preprocess_fn=zeroshot_encoder.val_preprocess,
            location=cfg.nn.data.data_path,
            batch_size=cfg.nn.data.batch_size.train,
        ) for dataset_name in cfg.task_vectors.to_apply}
    
    embed_dt = EmbeddingsDataset(finetuned_models, datasets, cfg.number_of_train_batches, cfg)

    data = embed_dt.generate_layer_datasets()

    pylogger.info(data.keys())

    task_dicts = {}
    for dataset in cfg.task_vectors.to_apply:
        task_dicts[dataset] = compute_task_dict(
            zeroshot_encoder_statedict, finetuned_models[dataset]
        )

    svd_dicts = get_svd_dict(
        task_dicts, cfg.eval_datasets, cfg.misc.svd_path, cfg.svd_compress_factor
    )

    pylogger.info(svd_dicts.keys())
    pylogger.info(svd_dicts['FER2013'].keys())

    problem = LayerOptimProblem(cfg, svd_dicts, data, 'cuda')
    problem.fit()

    merged_vector = merge_parameters(zeroshot_encoder, problem)

    merged_encoder: ImageEncoder = instantiate(
        cfg.nn.module.encoder
    )  # the second pass backbone

    merged_encoder.load_state_dict(merged_vector, strict=False)

    for dataset in cfg.task_vectors.to_apply:

        model = ImageClassifier(
                encoder=merged_encoder,
                x_key='x',
                y_key='y',
                classifier=get_classification_head(
                    cfg.nn.module.encoder.model_name,
                    dataset,
                    cfg.nn.data.data_path,
                    cfg.misc.ckpt_path,
                    cache_dir=cfg.misc.cache_dir,
                    openclip_cachedir=cfg.misc.openclip_cachedir,
                ),
            )
        
        evaluate(model, dataset, zeroshot_encoder.val_preprocess)
    
    wandb.finish()




@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="optimisation.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()