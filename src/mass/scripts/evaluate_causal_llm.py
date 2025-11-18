## Imports

import json
import logging
from pathlib import Path
from typing import Dict

import lm_eval
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
from lm_eval.evaluator import simple_evaluate 

# Force the execution of __init__.py if this file is executed directly.
import mass  # noqa

from mass.utils.io_utils import boilerplate
from mass.utils.plots import plot_interactive_radar_chart
from mass.utils.utils import compute_avg_accuracy, get_finetuning_accuracies, build_callbacks, print_memory
from mass.pl_module.language_classifier import get_task_config_name

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

EXPERTS = {"1":"meta-math/MetaMath-Mistral-7B","2":"cognitivecomputations/dolphin-2.1-mistral-7b","3":"uukuguy/speechless-code-mistral-7b-v1.0"}

def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if hasattr(obj, "item"):  # numpy / torch scalar
        return obj.item()
    if isinstance(obj, (Path,)):
        return str(obj)
    return obj


@torch.no_grad()
def run(cfg: omegaconf.DictConfig):
    seed_index_everything(cfg)
    cfg.core.tags.append(f"{cfg.nn.encoder.model_name}")
    logger, template_core = boilerplate(cfg)
    
    num_tasks = len(cfg.eval_datasets)

    # Temporarily disable struct mode to allow dynamic update
    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.num_tasks = num_tasks  # Now we can safely update it
    omegaconf.OmegaConf.set_struct(cfg, True)  # Re-enable struct mode
    
    print_memory("after boilerplate")
    zeroshot_encoder = instantiate(cfg.nn.encoder.model, torch_dtype=torch.bfloat16)
    print_memory("after loading zeroshot")

    finetuned_models = {
        dataset: instantiate(cfg.nn.encoder.model, pretrained_model_name_or_path=EXPERTS[dataset], torch_dtype=torch.bfloat16)
        for dataset in cfg.benchmark.datasets
    }
    moerging = instantiate(
        cfg.nn.module,
        zeroshot_model=zeroshot_encoder,
        finetuned_models=finetuned_models,
    )

    print_memory("before starting eval")
    
    eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=moerging.model,
        use_fast_tokenizer=False
    )

    results = simple_evaluate(
        model=eval_model,
        tasks=[cfg.benchmark.name], 
        batch_size=8,
        limit=20, 
    )

    safe_results = json_safe(results)
    pylogger.info(f"Evaluation results:\n{json.dumps(safe_results, indent=2)}")

    if logger is not None and 'results' in results:
        flat_results = {}
        for task_name, metrics in results['results'].items():
            for metric_name, value in metrics.items():
                if not metric_name.endswith("_stderr"):
                    flat_results[f"eval/{task_name}/{metric_name}"] = json_safe(value)
        
        pylogger.info(f"Logging the following metrics to W&B: {flat_results}")
        logger.experiment.log(flat_results)

    if logger is not None:
        logger.experiment.finish()
          
    

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="eval_causal_language.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()