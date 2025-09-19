## Imports

import logging
from typing import List


import open_clip
import wandb

import hydra
import omegaconf
import torch

from hydra.utils import instantiate
from nn_core.serialization import NNCheckpointIO

import pytorch_lightning as pl

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

# Force the execution of __init__.py if this file is executed directly.
import mass  # noqa

import os

from mass.utils.io_utils import boilerplate
from mass.utils.utils import print_memory, build_callbacks

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


@torch.no_grad()
def run(cfg: omegaconf.DictConfig) -> str:

    seed_index_everything(cfg)
    
    logger, template_core = boilerplate(cfg)

    num_tasks = len(cfg.eval_datasets)

    # Temporarily disable struct mode to allow dynamic update
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.num_tasks = num_tasks  # Now we can safely update it
    omegaconf.OmegaConf.set_struct(cfg, True)  # Re-enable struct mode
    
    zeroshot_encoder = instantiate(cfg.nn.encoder.model)
    
    finetuned_models = {
        dataset: instantiate(cfg.nn.encoder.model, pretrained_model_name_or_path=cfg.nn.encoder.model.pretrained_model_name_or_path.replace("google/", f"tanganke/") + f"_glue-{dataset}",
        )
        for dataset in cfg.benchmark.datasets
    }
    
    pylogger.info(f"Finetuned models: {finetuned_models.keys()}")
    
    pylogger.info(f"{cfg.eval_datasets}")
    
    moerging = instantiate(
        cfg.nn.module,
        zeroshot_model=zeroshot_encoder,
        finetuned_models=finetuned_models,
    )

    tokenizer = instantiate(cfg.nn.tokenizer)
    
    task_model = instantiate(cfg.nn.task, moe_model=moerging.model.cuda(), tokenizer=tokenizer)
    
    # TODO: add task specific layer
    
    pylogger.info(f"Model instantiated: {moerging}")
    
    
    for dataset_name in cfg.benchmark.datasets:

        dataset_cfg = omegaconf.OmegaConf.load(
            PROJECT_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
        )
        
        dataset = instantiate(
            dataset_cfg, tokenizer=tokenizer) #cache_dir="~/.cache/huggingface/datasets/glue")
        
        pylogger.info(f"Dataset {dataset_name} loaded: {dataset}")
        pylogger.info(f"{type(dataset)}")
        
        callbacks = build_callbacks(cfg.train.callbacks, template_core)

        # TODO: check if this would work
        task_model.set_metrics()
        task_model.set_task(dataset_name)

        trainer = pl.Trainer(
            default_root_dir=cfg.core.storage_dir,
            plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
            logger=logger,
            callbacks=callbacks,
            limit_test_batches=None,
            **cfg.train.trainer,
        )

        test_results = trainer.test(model=task_model, dataloaders=dataset.val_loader)
        
        pylogger.info(f"Test results on {dataset_name}: {test_results}")    
    

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="eval_language.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
