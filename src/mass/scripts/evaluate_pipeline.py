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
from hydra.utils import instantiate
from lightning.pytorch import Callback

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.serialization import NNCheckpointIO

# Force the execution of __init__.py if this file is executed directly.
import mass  # noqa
from mass.modules.encoder import ClassificationHead, ImageEncoder
from mass.modules.router import AbstractRouter
from mass.utils.io_utils import (
    boilerplate,
    get_classification_heads,
    load_model_from_hf,
)
from mass.utils.plots import plot_interactive_radar_chart
from mass.utils.utils import (
    compute_task_dict,
    apply_dict_to_model,
    build_callbacks,
    get_finetuning_accuracies,
    compute_avg_accuracy,
)
from mass.task_vectors.task_singular_vectors import *
import json
import os

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def get_merged_base(
    cfg,
    zeroshot_encoder: ImageEncoder,
    svd_dicts: Dict[str, Any],
):
        
    multi_task_vector = (
        sum_svd_no_redundant_tasks_simple( 
            ref_state_dict=copy.deepcopy(zeroshot_encoder.state_dict()),
            svd_dict=svd_dicts,
            similarity_threshold=cfg.similarity_threshold,
        )
    )

    merged_encoder: ImageEncoder = copy.deepcopy(zeroshot_encoder)

    merged_encoder = apply_dict_to_model(
        multi_task_vector,
        merged_encoder,
    )

    return merged_encoder  # , svd_dicts


@torch.no_grad()
def run(cfg: omegaconf.DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """
    cfg.core.tags.append("mass")
    if cfg.base == "zeroshot":
        cfg.core.tags.append("zeroshot_base")
        
    pylogger.info(f"Starting MASS eval")
    seed_index_everything(cfg)

    logger, template_core = boilerplate(cfg)

    num_tasks = len(cfg.eval_datasets)

    # Temporarily disable struct mode to allow dynamic update
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.num_tasks = num_tasks  # Now we can safely update it
    omegaconf.OmegaConf.set_struct(cfg, True)  # Re-enable struct mode

    # upperbound accuracies, used for logging the normalized accuracy
    finetuned_accuracies: Dict[str, float] = get_finetuning_accuracies(
        cfg.misc.finetuned_accuracy_path
    )[cfg.nn.encoder.model_name]

    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )

    finetuned_models = {
        dataset: load_model_from_hf(
            model_name=cfg.nn.encoder.model_name, dataset_name=dataset
        )
        for dataset in cfg.benchmark.datasets
    }

    pylogger.info(f"Number of tasks: {cfg.num_tasks}")
    pylogger.info(f"Finetuned models: {list(finetuned_models.keys())}")

    task_dicts = {}
    for dataset in cfg.benchmark.datasets:
        task_dicts[dataset] = compute_task_dict(
            zeroshot_encoder.state_dict(), finetuned_models[dataset].state_dict()
        )
        del finetuned_models[dataset]  # Delete one model at a time
        torch.cuda.empty_cache()


    svd_dict = get_svd_dict(
        task_dicts, cfg.benchmark.datasets, cfg.misc.svd_path, cfg.svd_compress_factor
    )

    del task_dicts

    pylogger.info(f"Retrieving merged model {cfg.base}")
    
    if cfg.base == "tsvm":
        merged_encoder = get_merged_base(
            cfg, zeroshot_encoder, svd_dict
        )
    elif cfg.base == "zeroshot":
        merged_encoder = copy.deepcopy(zeroshot_encoder)   
    else:
        raise ValueError(f"Unknown base {cfg.base}")

    pylogger.info(f"Instantiating router")
    router: AbstractRouter = instantiate(
        cfg.nn.module.router,
        encoder=merged_encoder,
        svd_dict=svd_dict,
        cfg=cfg,
        _recursive_=False,
    )

    classification_heads: List[ClassificationHead] = get_classification_heads(cfg)

    pylogger.info(f"Instantiating final model")
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
    torch.cuda.empty_cache()
    pylogger.info(f"Starting evaluation")
    for dataset_name in cfg.benchmark.datasets:

        dataset_cfg = omegaconf.OmegaConf.load(
            PROJECT_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
        )
        
        dataset = instantiate(
            dataset_cfg, preprocess_fn=zeroshot_encoder.val_preprocess, batch_size=cfg.data_batch_size
        )

        model.set_metrics(len(dataset.classnames))
        model.set_task(dataset.name)
        model.set_finetuning_accuracy(
            finetuned_accuracies[
                dataset.name + "Val" if cfg.eval_on_train else dataset.name
            ]
        )

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
            pylogger.info(f"Evaluating on {dataset.name} the training set")
            test_results = trainer.test(model=model, dataloaders=dataset.train_loader)

        else:
            pylogger.info(f"Evaluating on the {dataset.name} test set!")
            test_results = trainer.test(model=model, dataloaders=dataset.test_loader)

        results[dataset.name] = test_results

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


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
