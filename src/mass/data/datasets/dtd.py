import os
from typing import Callable, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image


class _HFDatasetAdapter(Dataset):
    """
    Wraps a 🤗 `Dataset` so that it behaves like a torchvision
    ImageFolder: returns (PIL.Image, int) and exposes .classes
    and .class_to_idx for downstream code compatibility.
    """

    def __init__(
        self,
        split,  # a HF Dataset split
        transform: Callable,
        class_names: List[str],
        class_to_idx: Dict[str, int],
    ):
        self._ds = split
        self.transform = transform
        self.classes = class_names
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        record = self._ds[idx]
        img: Image.Image = record["image"]  # already decoded
        label: int = int(record["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class DTD:
    """
    Same constructor signature you had before; `location` is kept
    only so that existing calls don't break, but it is ignored.
    """

    def __init__(
        self,
        preprocess: Callable,
        location: str = os.path.expanduser("~/data"),
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        ds = load_dataset("tanganke/dtd")

        # 2. Build label ↔ index mapping
        class_names: List[str] = ds["train"].features["label"].names
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(class_names)
        }

        # 3. Wrap each split so it looks like ImageFolder
        self.train_dataset = _HFDatasetAdapter(
            ds["train"], preprocess, class_names, self.class_to_idx
        )
        self.test_dataset = _HFDatasetAdapter(
            ds["test"], preprocess, class_names, self.class_to_idx
        )

        # 4. Standard DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )

        # 5. Convenience list of readable class names
        self.classnames = [c.replace("_", " ") for c in class_names]
