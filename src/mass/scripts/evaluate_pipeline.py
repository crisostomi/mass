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
from nn_core.common.utils import enforce_tags, seed_index_everything
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
    print_memory,
    svd_key_from_layer,
)
from mass.task_vectors.task_singular_vectors import *
import json
import os

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


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

        model_name = cfg.nn.encoder.model_name

        if (
            model_name in cfg.optimal_alphas
            and len(cfg.eval_datasets) in cfg.optimal_alphas[model_name]
        ):
            coefficient = cfg.optimal_alphas[model_name][len(cfg.eval_datasets)]

    elif merging_method == "tsvm":

        multi_task_vector = (
            sum_svd_no_redundant_tasks(  # TODO: restore no redundancy for proj
                ref_state_dict=copy.deepcopy(zeroshot_encoder.state_dict()),
                svd_dict=svd_dicts,
                similarity_threshold=cfg.similarity_threshold,
            )
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


@torch.no_grad()
def run(cfg: omegaconf.DictConfig) -> str:
    """Generic train loop.

    Args:
        cfg: run configuration, defined by Hydra in /conf

    Returns:
        the run directory inside the storage_dir used by the current experiment
    """

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
    )

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

    print_memory("after computing task dicts")

    svd_dict = get_svd_dict(
        task_dicts, cfg.benchmark.datasets, cfg.misc.svd_path, cfg.svd_compress_factor
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
                cfg.nn.encoder.layer_to_hook,
                cfg.nn.encoder.layer_num_to_hook,
            ),
        )
    else:
        un_compressed_routing_weights = None

    del task_dicts

    print_memory("after computing svd dict")

    merged_encoder = get_merged_base(
        cfg, cfg.nn.module.base_merging_method, zeroshot_encoder, svd_dict
    )
    print_memory("before router")
    router: AbstractRouter = instantiate(
        cfg.nn.module.router,
        encoder=merged_encoder,
        svd_dict=svd_dict,
        routing_weights=un_compressed_routing_weights,
        cfg=cfg,
        _recursive_=False,
    )

    print_memory("after router")

    if cfg.nn.module.router.name == "linear":
        linear_path = os.path.join(
            os.path.join(cfg.misc.checkpoint_dir, cfg.nn.module.router.filename),
            "checkpoint.ckpt",
        )
        state_dict = torch.load(linear_path)["state_dict"]["router"]
        router.load_state_dict(state_dict, True)

    print_memory("before heads")
    classification_heads: List[ClassificationHead] = get_classification_heads(cfg)

    print_memory("after heads")

    model = instantiate(
        cfg.nn.module,
        encoder=merged_encoder,
        zeroshot_model=zeroshot_encoder,
        router=router,
        svd_dicts=svd_dict,
        classification_heads=classification_heads,
        _recursive_=False,
    )

    print_memory("after MASS")

    logger.log_configuration(model, cfg)

    results = {}
    torch.cuda.empty_cache()
    print_memory("before eval")
    for dataset_name in cfg.benchmark.datasets:

        dataset_cfg = omegaconf.OmegaConf.load(
            PROJECT_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
        )

        dataset = instantiate(
            dataset_cfg, preprocess_fn=zeroshot_encoder.val_preprocess
        )

        print_memory("after data load")

        model.set_metrics(len(dataset.classnames))
        model.set_task(dataset_name)
        model.set_finetuning_accuracy(
            finetuned_accuracies[
                dataset_name + "Val" if cfg.eval_on_train else dataset_name
            ]
        )

        print_memory("after matrics")

        callbacks: List[Callback] = build_callbacks(cfg.train.callbacks, template_core)

        print_memory("after callbacks")

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

        print_memory("after trainer")

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


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="task_vectors.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
