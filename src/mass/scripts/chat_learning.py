import json
import hydra
import logging
import omegaconf
import wandb
import os
import time
import numpy as np
import torch
import einops
import tqdm
import open_clip
import copy
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Any, Dict, List, Optional
from lightning.pytorch import Callback
from nn_core.serialization import NNCheckpointIO
from hydra.utils import instantiate
from hydra import initialize, compose

# Import boilerplate dependencies from your training framework
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

# Import the functions from your modified files.
from mass.clip_interpret.compute_complete_text_set import run_completeness
from mass.clip_interpret.compute_text_set_projection import run_text_features
from mass.utils.io_utils import load_model_from_disk
from mass.utils.utils import compute_task_dict
from mass.task_vectors.task_singular_vectors import get_svd_dict
from mass.task_vectors.task_singular_vectors import compress_tv
from mass.pl_module.image_multihead_classifier import MultiHeadImageClassifier
from mass.data.datasets.registry import get_dataset
from mass.modules.encoder import ClassificationHead, ImageEncoder
from mass.modules.heads import get_classification_head
from mass.utils.utils import (
    apply_dict_to_model,
    build_callbacks,
    compute_avg_accuracy,
    get_finetuning_accuracies,
)
from mass.task_vectors.task_singular_vectors import *

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def boilerplate(cfg: DictConfig):
    """
    Full boilerplate initialization.
    This sets up tags, restores any checkpoints, and initializes the advanced logger.
    """
    # Enforce and update tags
    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))
    num_tasks = len(cfg.eval_datasets)
    cfg.core.tags.append(f"n{num_tasks}")
    cfg.core.tags.append(f"{cfg.nn.module.encoder.model_name}")
    cfg.core.tags.append(f"chat_learning")

    # Initialize template core for resuming experiments
    template_core = NNTemplateCore(restore_cfg=cfg.train.get("restore", None))
    # Initialize the NNLogger with advanced logging configuration
    logger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )
    logger.upload_source()
    return logger, template_core

def get_merged_method(
    cfg,
    merging_method: str,
    zeroshot_encoder: ImageEncoder,
    svd_dicts: Dict[str, Any],
) -> ImageEncoder:
    """
    Merge the encoder based on the specified method.
    """
    coefficient = 1.0
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
        multi_task_vector = sum_svd(
            ref_state_dict=copy.deepcopy(zeroshot_encoder.state_dict()),
            svd_dicts=svd_dicts,
            # similarity_threshold=cfg.similarity_threshold,
        )

    elif merging_method == "zeroshot":
        return zeroshot_encoder
    
    elif merging_method == "tsvm_modified":
        multi_task_vector = sum_svd_modified(
            ref_state_dict=copy.deepcopy(zeroshot_encoder.state_dict()),
            svd_dicts=svd_dicts,
            layer_to_modify=cfg.layer,
        )
    else:
        raise NotImplementedError

    merged_encoder: ImageEncoder = copy.deepcopy(zeroshot_encoder)
    merged_encoder = apply_dict_to_model(
        multi_task_vector,
        merged_encoder,
        coefficient=coefficient,
    )

    return merged_encoder  # , svd_dicts


def get_classification_heads(cfg: DictConfig) -> List[ClassificationHead]:
    """
    Instantiate classification heads for each evaluation dataset.
    """

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

class ModelWrapper(MultiHeadImageClassifier):
    """
    Wraps a multi-head classifier to allow dynamically setting the active head.
    """
    def __init__(self, model, head_idx: int = 0, **kwargs):
        super().__init__(
            encoder=model.encoder,
            classification_heads=model.classification_heads,
            **kwargs,
        )
        self.model = model
        self.head_idx = head_idx

    def forward(self, x, head_idx=None):
        return super().forward(x, head_idx = head_idx if head_idx is not None else self.head_idx)

    def __call__(self, inputs, head_idx=None):
        return super().__call__(inputs, self.head_idx)

    def set_head(self, head_idx: int):
        self.head_idx = head_idx

def evaluate_model(cfg, zeroshot_encoder, model, finetuned_accuracies: Dict[str, Any], logger, template_core) -> Dict[str, Any]:
    """
    Evaluate the given model on each evaluation dataset.
    """
    results = {}
    for i, dataset_name in enumerate(cfg.eval_datasets):
        dataset = get_dataset(
            dataset_name,
            preprocess_fn=zeroshot_encoder.val_preprocess,
            location=cfg.nn.data.data_path,
            batch_size=cfg.nn.data.batch_size.train,
            num_workers=cfg.nn.data.num_workers.test,
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
            limit_test_batches=(cfg.number_of_train_batches if cfg.eval_on_train else None),
            **cfg.train.trainer,
        )

        if cfg.eval_on_train:
            pylogger.error("For now evaluation supported only on val-set")
            pylogger.info(f"Evaluating on {dataset_name} the training set")
            test_results = trainer.test(model=model, dataloaders=dataset.train_loader)

        else:
            model.set_head(i)
            pylogger.info(f"Evaluating on the {dataset_name} test set!")
            test_results = trainer.test(
                model=model, dataloaders=dataset.test_loader
            )

        results[dataset_name] = test_results
    return results



def run(cfg: DictConfig) -> str:
    seed_index_everything(cfg)

    logger, template_core = boilerplate(cfg)

    ntasks = len(cfg.eval_datasets)

    # Temporarily disable struct mode to allow dynamic update
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.ntasks = ntasks  # Now we can safely update it
    omegaconf.OmegaConf.set_struct(cfg, True)  # Re-enable struct mode

    # name = cfg.text_descriptions.replace(".txt", "")
    # output_path = os.path.join(cfg.misc.output_dir, f"{name}_{cfg.model}.npy")

    # if os.path.exists(output_path):
    #     pylogger.info(f"Output file already exists: {output_path}. Skipping computation.")
    # else:
    #     pylogger.info("Running text feature extraction...")
    #     run_text_features(cfg)
    # pylogger.info("Translating into text descriptions...")

    compression_factor_b32 = cfg.svd_compress_factor or len(cfg.eval_datasets)
    svd_path_b32 = (
        Path(cfg.misc.svd_path).with_suffix("").as_posix()
        + f"_compress_{compression_factor_b32}.pt"
    )
    svd_dict_b32 = get_or_compute_svd_dict(compression_factor_b32, svd_path_b32, cfg)
    pylogger.info(f"Loaded svd dict: {svd_path_b32}")

    # compression_factor_b16 = cfg.svd_compress_factor or len(cfg.eval_datasets)
    # svd_path_b16 = (
    #     Path("./svd_dict_ViT-B-16.pt").with_suffix("").as_posix()
    #     + f"_compress_{compression_factor_b16}.pt"
    # )
    # svd_dict_b16 = get_or_compute_svd_dict(compression_factor_b16, svd_path_b16, cfg)
    # pylogger.info(f"Loaded svd dict: {svd_path_b16}")

    # model_b32, _, preprocess_b32 = open_clip.create_model_and_transforms(
    # cfg.model, pretrained="openai", cache_dir=cfg.misc.openclip_cachedir
    # )
    # model_b32.to(cfg.device)
    # all_labels_b32 = set()

    # model_b16, _, preprocess_b16 = open_clip.create_model_and_transforms(
    #     "ViT-B-16", pretrained="openai", cache_dir=cfg.misc.openclip_cachedir
    # )
    # model_b16.to(cfg.device)
    # all_labels_b16 = set()

    # Load text features from file
    # name = cfg.text_descriptions.replace(".txt", "")
    # text_features_path_b32 = os.path.join(cfg.misc.output_dir, f"{name}_{cfg.model}.npy")
    # text_features_path_b16 = os.path.join(
    #     cfg.misc.output_dir.replace("ViT-B-32", "ViT-B-16"), f"{name}_ViT-B-16.npy"
    # )

    # with open(text_features_path_b32, "rb") as f:
    #     text_features_b32 = np.load(f)
    # pylogger.info(f"Loaded text features from {text_features_path_b32}")

    # with open(text_features_path_b16, "rb") as f:
    #     text_features_b16 = np.load(f)
    # pylogger.info(f"Loaded text features from {text_features_path_b16}")

    # # Load text descriptions (each line is one text)
    # text_file = os.path.join(cfg.misc.description_dir, f"{cfg.text_descriptions}")
    # with open(text_file, "r") as f:
    #     lines = [line.strip() for line in f.readlines()]
    # pylogger.info(f"Loaded text descriptions from {text_file}")


    # upperbound accuracies, used for logging the normalized accuracy
    finetuned_accuracies = get_finetuning_accuracies(cfg.misc.finetuned_accuracy_path)

    # only has vision encoder, no text transformer
    zeroshot_encoder_statedict = load_model_from_disk(cfg.misc.pretrained_checkpoint)
    zeroshot_encoder: ImageEncoder = instantiate(cfg.nn.module.encoder)  # the second pass backbone
    zeroshot_encoder.load_state_dict(zeroshot_encoder_statedict, strict=False)

    # create the merged encoder using the (zeroshot|tsvm|isotropic) merging method
    merged_encoder = get_merged_method(
        cfg, cfg.nn.module.base_merging_method, zeroshot_encoder, svd_dict_b32
    )

    # Instantiate classification heads and the main model
    classification_heads: List[ClassificationHead] = get_classification_heads(cfg)
    model = instantiate(
        cfg.nn.module,
        encoder=merged_encoder,
        zeroshot_model=zeroshot_encoder,
        svd_dicts=svd_dict_b32,
        classification_heads=classification_heads,
        x_key=cfg.nn.module.x_key,
        y_key=cfg.nn.module.y_key,
        _recursive_=False,
    )
    logger.log_configuration(model, cfg)

    # Wrap the model for multi-head usage.
    wrapped_model = ModelWrapper(model, head_idx=0)
    wrapped_model.hparams.x_key = cfg.nn.module.x_key
    wrapped_model.hparams.y_key = cfg.nn.module.y_key
    wrapped_model = wrapped_model.to(cfg.device)

    # Evaluate on all datasets.
    results = evaluate_model(cfg, zeroshot_encoder, wrapped_model, finetuned_accuracies, logger, template_core)
    avg = compute_avg_accuracy(results)
    results["avg"] = [avg]
    pylogger.info(f"Results: {results}")
    
    logger.experiment.log(avg)

    results_path = Path(cfg.misc.results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    with open(results_path / f"{cfg.layer}_{len(cfg.eval_datasets)}.json", "w+") as f:
        json.dump(results, f, indent=4)

    pylogger.info(f"Results saved to {cfg.misc.results_path}")

    logger.experiment.log_artifact(
        wandb.Artifact(
            f"results_{cfg.nn.module.encoder.model_name}_{ntasks}",
            type="results",
            metadata={"results": results_path},
        )
    )

    if logger is not None:
        logger.experiment.finish()



@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "conf"), config_name="clip_interpret.yaml")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
