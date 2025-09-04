import numpy as np
from typing import Optional, Sequence, Dict, Union, Callable
from torch.utils.data import Dataset, DataLoader
from datasets import (
    Dataset as HFDataset,
    DatasetDict,
    Features,
    ClassLabel,
    load_dataset as load_hf_dataset,
)
from hydra.utils import instantiate
import torchvision


def convert(x):
    import torchvision.transforms.functional as F

    if isinstance(x, np.ndarray):
        return F.to_pil_image(x)
    return x


def _prepend_convert(transform):
    try:
        if hasattr(transform, "transforms") and isinstance(transform.transforms, list):
            if not transform.transforms or transform.transforms[0] is not convert:
                transform.transforms.insert(0, convert)
    except Exception:
        pass
    return transform


class _HFImageTorchDataset(Dataset):
    """Torch Dataset over an HF split, with optional label remap and torchvision-style transform."""

    def __init__(
        self,
        hf_split: HFDataset,
        transform=None,
        label_map: Optional[
            Union[Dict[int, int], Sequence[int], np.ndarray, Callable[[int], int]]
        ] = None,
    ):
        self.hf_split = hf_split
        self.transform = _prepend_convert(transform)
        self.label_map = label_map

    def __len__(self):
        return self.hf_split.num_rows

    def _map_label(self, y: int) -> int:
        # Convert boolean labels to integers
        if isinstance(y, bool):
            y = int(y)

        if self.label_map is None:
            return y
        if callable(self.label_map):
            return int(self.label_map(y))
        return int(self.label_map[y])

    def __getitem__(self, idx):
        ex = self.hf_split[idx]
        img = ex["image"]
        y = self._map_label(ex["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, y


class HFImageClassification:
    """
    Strict adapter around an ALREADY-LOADED HF DatasetDict.
    Exposes: train_dataset/test_dataset, train_loader/test_loader, classnames.
    - Requires actual 'train' and 'test' splits (or explicit split_map pointing to existing keys).
    - No auto-splitting, no validation remap, no heuristics.
    """

    def __init__(
        self,
        hf_ds: DatasetDict,
        preprocess,
        ft_epochs: int,
        split_map: Optional[
            Dict[str, str]
        ] = None,  # e.g., {"train":"train","test":"test"}
        batch_size: int = 128,
        num_workers: int = 6,
        label_map: Optional[
            Union[Dict[int, int], Sequence[int], np.ndarray, Callable[[int], int]]
        ] = None,
        classnames_override: Optional[Sequence[str]] = None,
        pin_memory: bool = True,
    ):

        if split_map is None:
            assert (
                "train" in hf_ds and "test" in hf_ds
            ), "Expected 'train' and 'test' splits in the provided DatasetDict."
            train_key, test_key = "train", "test"
        else:
            assert (
                "train" in split_map and "test" in split_map
            ), "split_map must contain 'train' and 'test' keys."
            train_key, test_key = split_map["train"], split_map["test"]
            assert (
                train_key in hf_ds and test_key in hf_ds
            ), f"split_map points to missing splits: got {list(hf_ds.keys())}"

        self.train_dataset = _HFImageTorchDataset(
            hf_ds[train_key], transform=preprocess, label_map=label_map
        )
        self.test_dataset = _HFImageTorchDataset(
            hf_ds[test_key], transform=preprocess, label_map=label_map
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        if classnames_override is not None:
            self.classnames = list(classnames_override)
        else:
            self.classnames = self._extract_classnames_strict(hf_ds)

        # mirror torchvision attr some libs expect
        self.train_dataset.classes = self.classnames
        self.test_dataset.classes = self.classnames
        self.ft_epochs = ft_epochs

    @staticmethod
    def _extract_classnames_strict(ds: DatasetDict) -> Sequence[str]:
        # Find a split with ClassLabel and use its names; else fail loudly.
        for split in ds.values():
            feats: Features = split.features
            if "label" in feats and isinstance(feats["label"], ClassLabel):
                return list(feats["label"].names)
        raise AssertionError(
            "No ClassLabel found for 'label'. "
            "Provide classnames_override or ensure the dataset uses ClassLabel for 'label'."
        )


def load_fer2013():
    dataset = load_hf_dataset("clip-benchmark/wds_fer2013")
    dataset = dataset.remove_columns(["__key__", "__url__"])
    dataset = dataset.rename_columns({"jpg": "image", "cls": "label"})

    return dataset


def emnist_preprocess_fn(preprocess_fn):
    new_preprocess_fn = torchvision.transforms.Compose(
        [
            preprocess_fn,
            lambda img: torchvision.transforms.functional.rotate(img, -90),
            lambda img: torchvision.transforms.functional.hflip(img),
        ]
    )
    return new_preprocess_fn


def _norm(name: str) -> str:
    # Optional normalization to match styles like "forest" vs "Forest", underscores vs spaces
    return name.strip().lower().replace("_", " ")


def compute_label_map_from_names(
    current_names: Sequence[
        str
    ],  # names in the dataset's *current* order (index = old label)
    desired_order: Sequence[str],  # your target ordering (index = new label)
    normalize: Callable[[str], str] = _norm,
) -> np.ndarray:
    cur_norm = [_norm(n) for n in current_names]
    des_norm = [_norm(n) for n in desired_order]

    # Strict checks
    if len(set(cur_norm)) != len(cur_norm):
        raise ValueError("Duplicate names found in current_names after normalization.")
    if len(set(des_norm)) != len(des_norm):
        raise ValueError("Duplicate names found in desired_order after normalization.")

    if set(cur_norm) != set(des_norm):
        missing_in_desired = set(cur_norm) - set(des_norm)
        missing_in_current = set(des_norm) - set(cur_norm)
        raise ValueError(
            f"Name mismatch.\n"
            f"• Present only in current: {sorted(missing_in_desired)}\n"
            f"• Present only in desired: {sorted(missing_in_current)}"
        )

    name_to_new = {name: i for i, name in enumerate(des_norm)}
    # old label idx -> new label idx
    label_map = np.array([name_to_new[name] for name in cur_norm], dtype=int)
    return label_map


def load_dataset(
    name,
    hf_dataset,
    preprocess_fn,
    ft_epochs,
    split_map=None,
    batch_size=128,
    label_map=None,
    classnames_override=None,
):
    if "EMNIST" in name:
        preprocess_fn = emnist_preprocess_fn(preprocess_fn)

    if not split_map:
        split_map = {"train": "train", "test": "test"}

    hf_dataset = instantiate(hf_dataset)

    dataset = HFImageClassification(
        hf_ds=hf_dataset,
        preprocess=preprocess_fn,
        split_map=split_map,
        batch_size=batch_size,
        ft_epochs=ft_epochs,
        label_map=label_map,
        classnames_override=classnames_override,
    )

    return dataset


def maybe_dictionarize(batch, x_key, y_key):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {x_key: batch[0], y_key: batch[1]}
    elif len(batch) == 3:
        batch = {x_key: batch[0], y_key: batch[1], "metadata": batch[2]}
    else:
        raise ValueError(f"Unexpected number of elements: {len(batch)}")

    return batch
