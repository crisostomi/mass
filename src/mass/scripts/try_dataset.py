#!/usr/bin/env python3
import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger
import omegaconf


def run(cfg):

    dataset_names = []
    for entry in cfg.defaults:

        # entries are plain names like 'cifar10', 'fer2013', ...
        if isinstance(entry, str):
            dataset_names.append(entry)

        elif isinstance(entry, DictConfig):
            # allow hydra-style dict defaults like {'dataset': 'cifar10'}
            # take any value of the dict as the dataset file name
            dataset_names.extend([v for v in entry.values()])

        else:
            raise ValueError(f"Unsupported defaults entry: {entry}")

    print(f"Found {len(dataset_names)} datasets")

    loaded = {}

    for name in dataset_names:
        ds_cfg = load_dataset_cfg(args.conf_root, name)

        if "_target_" not in ds_cfg:
            raise ValueError(
                f"{name}.yaml must define a _target_ (got keys: {list(ds_cfg.keys())})"
            )

        # Hydra instantiate expects the dict with _target_ at the root
        # Ensure it's a DictConfig to keep OmegaConf behavior
        target_cfg = DictConfig(ds_cfg)

        print(f"\n→ Loading '{name}' via {target_cfg._target_} ...")
        try:
            obj = instantiate(target_cfg)
            loaded[name] = obj
            print(f"   Loaded: {short_summarize(obj)}")

            if args.print_example:
                sample = None
                # Try common access patterns
                try:
                    if hasattr(obj, "select") and hasattr(obj, "__getitem__"):
                        # 🤗 datasets.Dataset
                        sample = obj[0]
                    elif hasattr(obj, "keys") and "train" in obj:
                        # DatasetDict
                        sample = obj["train"][0]
                    elif hasattr(obj, "__getitem__"):
                        sample = obj[0]
                except Exception:
                    sample = None

                if sample is not None:
                    # Avoid dumping giant tensors; convert to a light preview
                    try:

                        def sanitize(x):
                            try:
                                import torch
                                import numpy as np

                                if isinstance(x, (torch.Tensor,)):
                                    return f"Tensor(shape={tuple(x.shape)}, dtype={x.dtype})"
                                if isinstance(x, (np.ndarray,)):
                                    return f"ndarray(shape={x.shape}, dtype={x.dtype})"
                            except Exception:
                                pass
                            if hasattr(x, "size") and hasattr(x, "mode"):  # PIL Image
                                return f"PIL.Image(size={x.size}, mode={x.mode})"
                            return x

                        if isinstance(sample, dict):
                            preview = {k: sanitize(v) for k, v in sample.items()}
                        else:
                            preview = sanitize(sample)
                        print(
                            "   Example[0] preview:",
                            json.dumps(preview, default=str)[:500],
                        )
                    except Exception as e:
                        print(f"   (Could not preview example: {e})")

        except Exception as e:
            print(f"   ERROR loading '{name}': {e}")

    print(f"\nDone. Successfully loaded: {list(loaded.keys())}")

    failed = [name for name in dataset_names if name not in loaded]
    if failed:
        print(f"Failed to load: {failed}")


def load_dataset_cfg(cfg_dir: Path, name: str) -> DictConfig:
    """
    Each dataset file is expected to look like:
      <name>.yaml:
        <name>:
          _target_: package.func
          key: value
    """
    cfg = OmegaConf.load(cfg_dir / f"{name}.yaml")

    if name not in cfg:

        # allow files where the single top-level key is the dataset name
        if len(cfg.keys()) == 1:
            only = next(iter(cfg.keys()))
            return cfg[only]

        raise KeyError(
            f"Top-level key '{name}' not found in {name}.yaml (has: {list(cfg.keys())})"
        )

    return cfg[name]


def short_summarize(obj: Any) -> str:
    try:
        # Try to be informative for common dataset types
        # 🤗 datasets
        if obj.__class__.__module__.startswith("datasets"):
            # obj might be Dataset or DatasetDict
            try:
                import datasets as hfds  # noqa: F401
            except Exception:
                pass
            if hasattr(obj, "num_rows"):
                return f"HuggingFace Dataset(num_rows={obj.num_rows})"
            if hasattr(obj, "keys"):
                # DatasetDict
                d = {k: getattr(v, "num_rows", "?") for k, v in obj.items()}
                return f"HuggingFace DatasetDict({d})"
        # torchvision-style or others
        if hasattr(obj, "__len__"):
            return f"{obj.__class__.__name__}(len={len(obj)})"
        return obj.__class__.__name__
    except Exception:
        return obj.__class__.__name__


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
