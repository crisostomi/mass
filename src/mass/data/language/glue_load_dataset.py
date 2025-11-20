import logging
import os
from typing import Mapping, Optional

import numpy as np
import torch

from datasets import load_dataset, load_from_disk
from omegaconf import DictConfig

from hydra.utils import instantiate


from .glue_preprocessors import glue_processors
from .glue_prompt_templates import glue_prompt_templates

log = logging.getLogger(__name__)


def torch_default_data_collator(features):
    import torch

    if not isinstance(features[0], Mapping):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.from_numpy(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def _load_glue_dataset(name, tokenizer):
    if isinstance(tokenizer, (DictConfig, dict)):
        tokenizer = instantiate(tokenizer)

    dataset = load_dataset("glue", name)
    preprocessor = glue_processors[name](
        template=glue_prompt_templates[name],
        tokenizer=tokenizer,
        tokenizer_kwargs={
            "padding": "max_length",
            "truncation": True,
            "return_tensors": "pt",
        },
    )
    dataset = dataset.map(
        preprocessor,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=1,
    )
    return dataset


def load_glue_dataset(
    name,
    tokenizer,
    batch_size: int = 16,
    cache_dir: Optional[str] = "../.cache",
    split: Optional[str] = None,
):

    if cache_dir is not None:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        cache_path = os.path.join(cache_dir, "flan-t5", f"_load_{name}_dataset_cached")
        if os.path.exists(cache_path):
            dataset = load_from_disk(cache_path)
        else:
            dataset = _load_glue_dataset(name, tokenizer)
            log.info(f"Saving {name} dataset to {cache_path}")
            dataset.save_to_disk(cache_path)
    else:
        dataset = _load_glue_dataset(name, tokenizer)

    return GlueDataset(dataset, batch_size=batch_size, split=split)


class GlueDataset:
    def __init__(
        self,
        dataset,
        batch_size: int = 16,
        split: Optional[str] = "validation",
    ):
        self.data_loader = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=batch_size,
            collate_fn=torch_default_data_collator,
            num_workers=16,
        )
