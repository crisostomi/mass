import logging

import hydra
import omegaconf

import torch
from omegaconf import DictConfig

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger

from mass.data.datasets.registry import get_dataset
from mass.utils.io_utils import load_model_from_artifact

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


def run(cfg: DictConfig):
    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )

    zeroshot_identifier = f"{cfg.nn.module.model.model_name}_pt"

    image_encoder = load_model_from_artifact(
        artifact_path=f"{zeroshot_identifier}:latest", run=logger.experiment
    )

    dataset = get_dataset(
        cfg.nn.data.train_dataset,
        preprocess_fn=None,
        location=cfg.nn.data.data_path,
        batch_size=cfg.nn.data.batch_size.train,
    )

    pylogger.info(f"Dataset {cfg.nn.data.train_dataset} works fine;")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
