## Imports
import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import open_clip
import wandb

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from lightning.pytorch import Callback
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import tvp  # noqa
from tvp.data.datasets.registry import get_dataset
from tvp.modules.encoder import ClassificationHead, ImageEncoder
from tvp.modules.projection_router import ProjectionRouter
from tvp.modules.nn_router import NNRouter
from tvp.modules.heads import get_classification_head
from tvp.modules.router import AbstractRouter
from tvp.utils.io_utils import load_model_from_disk
from tvp.utils.plots import plot_interactive_radar_chart
from tvp.utils.utils import (
    compute_task_dict,
    apply_dict_to_model,
    build_callbacks,
    get_finetuning_accuracies,
    add_normalized_accuracy,
    compute_avg_accuracy,
    print_memory,
    get_routing_weights,
    svd_key_from_layer,
)
from tvp.task_vectors.task_singular_vectors import *
import json
import os

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


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


def get_merged_base(
    cfg,
    merging_method,
    zeroshot_encoder: ImageEncoder,
    svd_dicts: Dict[str, Any],
):

    coefficient = 1

    if merging_method == "isotropic":

        multi_task_vector = isotropic_sum(
            ref_state_dict=copy.deepcopy(zeroshot_encoder.state_dict()),
            svd_dict=svd_dicts,
        )

        model_name = cfg.nn.module.encoder.model_name

        if (
            model_name in cfg.optimal_alphas
            and len(cfg.eval_datasets) in cfg.optimal_alphas[model_name]
        ):
            coefficient = cfg.optimal_alphas[model_name][len(cfg.eval_datasets)]

    elif merging_method == "tsvm":

        multi_task_vector = sum_svd_no_redundant_tasks_simple(
            ref_state_dict=copy.deepcopy(zeroshot_encoder.state_dict()),
            svd_dict=svd_dicts,
            similarity_threshold=cfg.similarity_threshold,
        )
    elif merging_method == "zeroshot":
        return zeroshot_encoder
    else:
        raise NotImplementedError

    merged_encoder: ImageEncoder = copy.deepcopy(zeroshot_encoder)

    merged_encoder = apply_dict_to_model(
        multi_task_vector,
        merged_encoder,
        coefficient=coefficient,
    )

    return merged_encoder  # , svd_dicts


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


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """

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

    task_dicts = {}
    for dataset in cfg.task_vectors.to_apply:
        task_dicts[dataset] = compute_task_dict(
            zeroshot_encoder_statedict, finetuned_models[dataset]
        )
        del finetuned_models[dataset]  # Delete one model at a time
        torch.cuda.empty_cache()

    print_memory("after computing task dicts")

    svd_dict = get_svd_dict(
        task_dicts, cfg.eval_datasets, cfg.misc.svd_path, cfg.svd_compress_factor
    )

    if (
        cfg.nn.module.router.name == "proj"
        and cfg.nn.module.router.use_constant_compressed_routing_weights
    ):
        pylogger.info("Using constant compression for routing weights")
        un_compressed_routing_weights = get_uncompressed_weights(
            task_dicts,
            cfg.nn.module.router.constant_compressed_ratio,
            svd_key_from_layer(
                cfg.nn.module.encoder.layer_to_hook,
                cfg.nn.module.encoder.layer_num_to_hook,
            ),
        )
    else:
        un_compressed_routing_weights = None

    print_memory("after computing svd dict")

    merged_encoder = get_merged_base(
        cfg, cfg.nn.module.base_merging_method, zeroshot_encoder, svd_dict
    )

    router: AbstractRouter = instantiate(
        cfg.nn.module.router,
        encoder=merged_encoder,
        svd_dict=svd_dict,
        routing_weights=un_compressed_routing_weights,
        cfg=cfg,
        _recursive_=False,
    )

    classification_heads: List[ClassificationHead] = get_classification_heads(cfg)

    model = instantiate(
        cfg.nn.module,
        encoder=merged_encoder,
        zeroshot_model=zeroshot_encoder,
        router=router,
        svd_dicts=svd_dict,
        classification_heads=classification_heads,
        _recursive_=False,
    )

    logger.log_configuration(model, cfg)

    results = {}
    print_memory("before eval")
    for dataset_name in cfg.eval_datasets:

        dataset = get_dataset(
            dataset_name,
            preprocess_fn=zeroshot_encoder.val_preprocess,
            location=cfg.nn.data.data_path,
            batch_size=cfg.nn.data.batch_size.train,
            number_of_train_batches=cfg.number_of_train_batches,
        )

        model.set_metrics(len(dataset.classnames))
        model.set_task(dataset_name)
        model.set_finetuning_accuracy(
            finetuned_accuracies[
                dataset_name + "Val" if cfg.eval_on_train else dataset_name
            ]
        )

        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        trainer = pl.Trainer(
            default_root_dir=cfg.core.storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=logger,
            callbacks=callbacks,
            **cfg.train.trainer,
        )

        if cfg.eval_on_train:
            pylogger.error("For now evaluation supported only on val-set")
            pylogger.info(f"Evaluating on {dataset_name} the training set")
            test_results = trainer.test(model=model, dataloaders=dataset.train_loader)

        else:
            pylogger.info(f"Evaluating on the {dataset_name} test set!")
            test_results = trainer.test(model=model, dataloaders=dataset.test_loader)

        results[dataset_name] = test_results

    avg = compute_avg_accuracy(results)
    results["avg"] = [
        avg
    ]  # as a list for consistency due to lightning logging stuff this way

    logger.experiment.log(avg)

    pylogger.info(results)

    results_path = Path(cfg.misc.results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"{len(cfg.eval_datasets)}.json", "w+") as f:
        json.dump(results, f, indent=4)

    radarchart = plot_interactive_radar_chart(results, title="Radar Chart")
    logger.experiment.log({"radar": wandb.Plotly(radarchart)})

    pylogger.info(f"Results saved to {cfg.misc.results_path}")

    logger.experiment.log_artifact(
        wandb.Artifact(
            f"results_{cfg.nn.module.encoder.model_name}_{num_tasks}",
            type="results",
            metadata={"results": results_path},
        )
    )

    if logger is not None:
        logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
