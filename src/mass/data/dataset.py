import functools
import logging
import hydra
import omegaconf
from torch.utils.data import Dataset

from nn_core.common import PROJECT_ROOT
import torch
from pathlib import Path


pylogger = logging.getLogger(__name__)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    _: Dataset = hydra.utils.instantiate(cfg.nn.data.dataset, _recursive_=False)


if __name__ == "__main__":
    main()
