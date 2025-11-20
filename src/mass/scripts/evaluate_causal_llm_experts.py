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
from mass.utils.utils import (
    compute_avg_accuracy,
    get_finetuning_accuracies,
    build_callbacks,
    print_memory,
)
from mass.pl_module.language_classifier import get_task_config_name

pylogger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")

EXPERTS = {
    "1": "meta-math/MetaMath-Mistral-7B",
    "2": "cognitivecomputations/dolphin-2.1-mistral-7b",
    "3": "uukuguy/speechless-code-mistral-7b-v1.0",
}


@torch.no_grad()
def run(cfg: omegaconf.DictConfig):
    seed_index_everything(cfg)
    cfg.core.tags.append(f"{cfg.nn.encoder.model_name}")

    logger, template_core = boilerplate(cfg)

    num_tasks = len(cfg.eval_datasets)

    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.num_tasks = num_tasks
    omegaconf.OmegaConf.set_struct(cfg, True)

    eval_model = lm_eval.models.huggingface.HFLM(
        pretrained=instantiate(
            cfg.nn.encoder.model,
            pretrained_model_name_or_path=EXPERTS[str(cfg.expert)],
            torch_dtype=torch.bfloat16,
        ).cuda(),
        use_fast_tokenizer=False,
    )

    command_line_args = []

    command_line_args.extend(["--model", "hf"])

    tasks = cfg.benchmark.name
    if isinstance(tasks, (list, omegaconf.ListConfig)):
        command_line_args.extend(["--tasks", ",".join(tasks)])
    else:
        command_line_args.extend(["--tasks", str(tasks)])

    command_line_args.extend(["--batch_size", "8"])
    command_line_args.extend(["--verbosity", "INFO"])

    wandb_params = [
        f"project={cfg.core.project_name}",
        f"entity={cfg.core.entity}",
        f"name={cfg.core.tags[-1]}_eval",
    ]
    command_line_args.extend(["--wandb_args", ",".join(wandb_params)])

    parser = setup_parser()
    check_argument_types(parser)
    args = parser.parse_args(args=command_line_args)

    args.model = eval_model

    cli_evaluate(args)

    pylogger.info("Evaluation completed.")

    if logger is not None:
        logger.experiment.finish()


@hydra.main(
    config_path=str(PROJECT_ROOT / "conf"), config_name="eval_causal_language.yaml"
)
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
