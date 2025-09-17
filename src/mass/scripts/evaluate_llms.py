## Imports

import logging


import open_clip
import wandb

import hydra
import omegaconf
import torch

from hydra.utils import instantiate


from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

# Force the execution of __init__.py if this file is executed directly.
import mass  # noqa

import os

from mass.utils.io_utils import boilerplate

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
    
    zeroshot_encoder = instantiate(cfg.nn.encoder)
    
    finetuned_models = {
        dataset: instantiate(cfg.nn.encoder, pretrained_model_name_or_path=cfg.nn.encoder.pretrained_model_name_or_path.replace("google/", f"tanganke/") + f"_glue-{dataset}",
        )
        for dataset in cfg.benchmark.datasets
    }
    
    finetuned_models_list = list(finetuned_models.values())
    
    pylogger.info(f"Finetuned models: {finetuned_models.keys()}")
    
    moerging = instantiate(
        cfg.nn.module,
        zeroshot_model=zeroshot_encoder,
        finetuned_models=finetuned_models_list,
    )
    
    # TODO: add task specific layer
    
    pylogger.info(f"Model instantiated: {moerging}")
    
    
    for dataset_name in cfg.benchmark.datasets:

        dataset_cfg = omegaconf.OmegaConf.load(
            PROJECT_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
        )
        
        dataset = instantiate(
            dataset_cfg, preprocess_fn=zeroshot_encoder.val_preprocess, batch_size=cfg.data_batch_size
        )
    

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="eval_language.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
