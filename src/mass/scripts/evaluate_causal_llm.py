## Imports

import logging
from pathlib import Path
from typing import Dict

import wandb

import hydra
import omegaconf
import torch

from hydra.utils import instantiate
from nn_core.serialization import NNCheckpointIO

import pytorch_lightning as pl

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from lm_eval.__main__ import check_argument_types, cli_evaluate, setup_parser

# Force the execution of __init__.py if this file is executed directly.
import mass  # noqa

from mass.utils.io_utils import boilerplate
from mass.utils.plots import plot_interactive_radar_chart
from mass.utils.utils import compute_avg_accuracy, get_finetuning_accuracies, build_callbacks
from mass.pl_module.language_classifier import get_task_config_name

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

EXPERTS = {"1":"meta-math/MetaMath-Mistral-7B","2":"cognitivecomputations/dolphin-2.1-mistral-7b","3":"uukuguy/speechless-code-mistral-7b-v1.0"}


@torch.no_grad()
def run(cfg: omegaconf.DictConfig) -> str:

    seed_index_everything(cfg)
    
    cfg.core.tags.append(f"{cfg.nn.encoder.model_name}")
    
    logger, template_core = boilerplate(cfg)



    # Temporarily disable struct mode to allow dynamic update
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.num_tasks = 0  # Now we can safely update it
    omegaconf.OmegaConf.set_struct(cfg, True)  # Re-enable struct mode
    
    zeroshot_encoder = instantiate(cfg.nn.encoder.model)
    for name, layer in zeroshot_encoder.named_modules():
        pylogger.info(f"Layer name: {name} | Layer type: {type(layer)}")
    
    finetuned_models = {
        dataset: instantiate(cfg.nn.encoder.model, pretrained_model_name_or_path=EXPERTS[dataset],
        )
        for dataset in cfg.benchmark.datasets
    }
    
    pylogger.info(f"Finetuned models: {finetuned_models.keys()}")
    
    
    moerging = instantiate(
        cfg.nn.module,
        zeroshot_model=zeroshot_encoder,
        finetuned_models=finetuned_models,
    )

    tokenizer = instantiate(cfg.nn.tokenizer)
    
    pylogger.info(f"Model instantiated: {moerging}")
    
    # results = {}
    # for dataset_name in cfg.benchmark.datasets:

    #     dataset_cfg = omegaconf.OmegaConf.load(
    #         PROJECT_ROOT / "conf" / "dataset" / f"{dataset_name}.yaml"
    #     )
        
    #     dataset = instantiate(
    #         dataset_cfg, tokenizer=tokenizer) #cache_dir="~/.cache/huggingface/datasets/glue")
        
    #     pylogger.info(f"Dataset {dataset_name} loaded: {dataset}")
        
    #     # Get appropriate task configuration and instantiate
    #     task_config_name = get_task_config_name(dataset_name)
    #     task_cfg = omegaconf.OmegaConf.load(
    #         PROJECT_ROOT / "conf" / "nn" / "task" / f"{task_config_name}.yaml"
    #     )
        
    #     task_model = instantiate(
    #         task_cfg, 
    #         moe_model=moerging.model.cuda(), 
    #         tokenizer=tokenizer,
    #         custom_logger=logger
    #     )
        
    #     task_model.set_task_name(dataset_name)
        
    #     pylogger.info(f"Using {task_config_name} for {dataset_name}")
        
    #     callbacks = build_callbacks(cfg.train.callbacks, template_core)

    #     task_model.set_metrics()
    #     task_model.set_task(dataset_name)
    #     task_model.set_finetuning_accuracy(
    #         finetuned_accuracies[
    #             dataset_name
    #         ]
    #     )

    #     trainer = pl.Trainer(
    #         default_root_dir=cfg.core.storage_dir,
    #         plugins=[NNCheckpointIO(jailing_dir=logger.run_dir)],
    #         logger=logger,
    #         callbacks=callbacks,
    #         limit_test_batches=None,
    #         **cfg.train.trainer,
    #     )
        
    #     results[dataset_name] = trainer.test(model=task_model, dataloaders=dataset.data_loader)
        
    # pylogger.info(f"{results}")
        
    # avg = compute_avg_accuracy(results)
    # results["avg"] = [
    #     avg
    # ]  # as a list for consistency due to lightning logging stuff this way

    # logger.experiment.log(avg)
    
    # pylogger.info(results)
    
    # results_path = Path(cfg.misc.results_path)
    
    # radarchart = plot_interactive_radar_chart(results, title="Radar Chart")
    # logger.experiment.log({"radar": wandb.Plotly(radarchart)})
    
    # logger.experiment.log_artifact(
    #     wandb.Artifact(
    #         f"results_{cfg.nn.encoder.model_name}_{num_tasks}",
    #         type="results",
    #         metadata={"results": results_path},
    #     )
    # )

    if logger is not None:
        logger.experiment.finish()
          
    

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="eval_causal_language.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()