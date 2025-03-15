import abc
import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader


# modified from: https://github.com/microsoft/torchgeo
class VisionDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets lacking geospatial information.
    This base class is designed for datasets with pre-defined image chips.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and labels at that index
        Raises:
            IndexError: if index is out of range of the dataset
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.
        Returns:
            length of the dataset
        """

    def __str__(self) -> str:
        """Return the informal string representation of the object.
        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: VisionDataset
    size: {len(self)}"""


class VisionClassificationDataset(VisionDataset, ImageFolder):
    """Abstract base class for classification datasets lacking geospatial information.
    This base class is designed for datasets with pre-defined image chips which
    are separated into separate folders per class.
    """

    def __init__(
        self,
        root: str,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        loader: Optional[Callable[[str], Any]] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Initialize a new VisionClassificationDataset instance.
        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            loader: a callable function which takes as input a path to an image and
                returns a PIL Image or numpy array
            is_valid_file: A function that takes the path of an Image file and checks if
                the file is a valid file
        """
        # When transform & target_transform are None, ImageFolder.__getitem__(index)
        # returns a PIL.Image and int for image and label, respectively
        super().__init__(
            root=root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        # Must be set after calling super().__init__()
        self.transforms = transforms

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.
        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)

        if self.transforms is not None:
            return self.transforms(image), label

        return image, label

    def __len__(self) -> int:
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.imgs)

    def _load_image(self, index: int) -> Tuple[Tensor, Tensor]:
        """Load a single image and it's class label.
        Args:
            index: index to return
        Returns:
            the image
            the image class label
        """
        img, label = ImageFolder.__getitem__(self, index)
        label = torch.tensor(label)
        return img, label


class RESISC45Dataset(VisionClassificationDataset):
    """RESISC45 dataset with updated label mapping.
    Note: This version updates the datapoint labels according to the new class ordering.
    """
    directory = "resisc45/NWPU-RESISC45"

    splits = ["train", "val", "test"]
    split_urls = {
        "train": "https://storage.googleapis.com/remote_sensing_representations/resisc45-train.txt",
        "val": "https://storage.googleapis.com/remote_sensing_representations/resisc45-val.txt",
        "test": "https://storage.googleapis.com/remote_sensing_representations/resisc45-test.txt",
    }
    split_md5s = {
        "train": "b5a4c05a37de15e4ca886696a85c403e",
        "val": "a0770cee4c5ca20b8c32bbd61e114805",
        "test": "3dda9e4988b47eb1de9f07993653eb08",
    }
    classes = [
        "airplane",         # 0
        "forest",           # 1, moved here for consistency with EuroSAT
        "airport",          # 2
        "baseball_diamond", # 3
        "industrial_area",  # 4, moved here for consistency with EuroSAT
        "basketball_court", # 5
        "beach",            # 6
        "bridge",           # 7
        "river",            # 8, moved here for consistency with EuroSAT
        "chaparral",
        "church",
        "circular_farmland",
        "cloud",
        "commercial_area",
        "dense_residential",  # one of this under residential
        "desert",
        "freeway",
        "golf_course",
        "ground_track_field",
        "harbor",
        "intersection",
        "island",
        "lake",             # this
        "meadow",
        "medium_residential",  # one of this under residential
        "mobile_home_park",
        "mountain",
        "overpass",
        "palace",
        "parking_lot",
        "railway",
        "railway_station",
        "rectangular_farmland",
        "roundabout",
        "runway",
        "sea_ice",
        "ship",
        "snowberg",
        "sparse_residential",  # one of this under residential
        "stadium",
        "storage_tank",
        "tennis_court",
        "terrace",
        "thermal_power_station",
        "wetland",
    ]

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
    ) -> None:
        assert split in self.splits
        self.root = root

        valid_fns = set()
        with open(os.path.join(self.root, "resisc45", f"resisc45-{split}.txt")) as f:
            for fn in f:
                valid_fns.add(fn.strip())
        is_in_split: Callable[[str], bool] = lambda x: os.path.basename(x) in valid_fns

        super().__init__(
            root=os.path.join(root, self.directory),
            transforms=transforms,
            is_valid_file=is_in_split,
        )

        # Override classes and mapping to reflect the new ordering
        self.classes = self.__class__.classes
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def _load_image(self, index: int) -> Tuple[Tensor, Tensor]:
        # Load the image using the default loader (PIL)
        path, _ = self.imgs[index]
        img = pil_loader(path)
        # Extract the class name from the parent folder name
        class_name = os.path.basename(os.path.dirname(path))
        # Remap the label to the new index
        new_label = self.class_to_idx[class_name]
        return img, torch.tensor(new_label)


class RESISC45:
    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=32, num_workers=8, train_batches=-1):
        self.train_dataset = RESISC45Dataset(root=location, split="train", transforms=preprocess)

        if train_batches > 0:
            num_samples = train_batches * batch_size
            indices = list(range(num_samples))
            self.train_dataset = Subset(self.train_dataset, indices)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = RESISC45Dataset(root=location, split="test", transforms=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers
        )

        # class names have _ so split on this for better zero-shot head
        self.classnames = [" ".join(c.split("_")) for c in RESISC45Dataset.classes]
