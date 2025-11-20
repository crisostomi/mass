## Imports
import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mass.data.dataset import HFImageClassification
from mass.pl_module.image_classifier import ImageClassifier
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
import mass  # noqa
from mass.modules.encoder import ClassificationHead, ImageEncoder
from mass.modules.heads import (
    get_classification_head,
)
from mass.utils.io_utils import (
    boilerplate,
    load_model_from_hf,
)
from mass.utils.plots import plot_interactive_radar_chart
from mass.utils.utils import (
    build_callbacks,
    get_finetuning_accuracies,
    compute_avg_accuracy,
    print_memory,
)
from mass.utils.task_vectors import *
import json
import os

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def load_config(
    config_path: str,
    config_name: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """
    Load a Hydra config without launching a full Hydra app.

    Args:
        config_path (str): Path to the folder containing your configs (relative to project root).
        config_name (str): Name of the YAML config file (without `.yaml`).
        overrides (list[str], optional): List of override strings, e.g. ["trainer.max_epochs=20"].

    Returns:
        DictConfig: The loaded configuration.
    """
    overrides = overrides or []
    abs_config_path = str(Path(config_path).absolute())

    with hydra.initialize(config_path=abs_config_path, version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides)

    return cfg


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """

    seed_index_everything(cfg)

    num_tasks = len(cfg.benchmark.datasets)
    cfg.core.tags.append(f"n{num_tasks}")
    cfg.core.tags.append(f"{cfg.nn.encoder.model_name}")

    logger, template_core = boilerplate(cfg)

    num_tasks = len(cfg.benchmark.datasets)

    # Temporarily disable struct mode to allow dynamic update
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.num_tasks = num_tasks  # Now we can safely update it
    omegaconf.OmegaConf.set_struct(cfg, True)  # Re-enable struct mode

    # upperbound accuracies, used for logging the normalized accuracy
    finetuned_accuracies: Dict[str, float] = get_finetuning_accuracies(
        cfg.misc.finetuned_accuracy_path
    )[cfg.nn.encoder.model_name]

    # only has vision encoder, no text transformer
    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )

    finetuned_models = {
        dataset: load_model_from_hf(
            model_name=cfg.nn.encoder.model_name, dataset_name=dataset
        ).state_dict()  # TODO: uniform to use both without state_dict()
        for dataset in cfg.benchmark.datasets
    }

    pylogger.info(f"Number of tasks: {cfg.num_tasks}")
    pylogger.info(f"Finetuned models: {list(finetuned_models.keys())}")

    merger = instantiate(cfg.merger)

    merged_encoder = merger.merge(zeroshot_encoder, finetuned_models)

    logger.log_configuration(merged_encoder, cfg)

    results = {}
    print_memory("before eval")
    for dataset_name in cfg.benchmark.datasets:

        dataset_cfg = OmegaConf.load(
            PROJECT_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
        )

        dataset = instantiate(
            dataset_cfg, preprocess_fn=zeroshot_encoder.val_preprocess
        )

        classification_head = get_classification_head(
            cfg.nn.encoder.model_name,
            dataset_name,
            ckpt_path=cfg.misc.ckpt_path,
            openclip_cachedir=cfg.misc.openclip_cachedir,
            device=cfg.device,
        )

        model = ImageClassifier(
            encoder=merged_encoder,
            classifier=classification_head,
            x_key=cfg.conventions.x_key,
            y_key=cfg.conventions.y_key,
        )

        model.set_metrics(len(dataset.classnames))
        model.set_task(dataset_name)
        model.set_finetuning_accuracy(finetuned_accuracies[dataset_name])

        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        trainer = pl.Trainer(
            default_root_dir=cfg.core.storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=logger,
            callbacks=callbacks,
            limit_test_batches=(
                cfg.number_of_train_batches if cfg.eval_on_train else None
            ),
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
    with open(results_path / f"{len(cfg.benchmark.datasets)}.json", "w+") as f:
        json.dump(results, f, indent=4)

    radarchart = plot_interactive_radar_chart(results, title="Radar Chart")
    logger.experiment.log({"radar": wandb.Plotly(radarchart)})

    pylogger.info(f"Results saved to {cfg.misc.results_path}")

    logger.experiment.log_artifact(
        wandb.Artifact(
            f"results_{cfg.nn.encoder.model_name}_{num_tasks}",
            type="results",
            metadata={"results": results_path},
        )
    )

    if logger is not None:
        logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="static_merging.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
