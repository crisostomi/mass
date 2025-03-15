import os

import torch
from torch.utils.data import Subset
import torchvision.datasets as datasets

canonized_classnames = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

canonized_classnames = sorted(canonized_classnames)

import os
import torch
import torchvision.datasets as datasets
from torchvision import transforms


class STL10Canonized(datasets.STL10):
    """
    A wrapper around the standard STL10 dataset that remaps
    the integer labels so they match CIFAR-10's class index order.
    """

    def __init__(
        self, root, split="train", transform=None, download=False, label_map=None
    ):
        super().__init__(root=root, split=split, transform=transform, download=download)
        self.label_map = label_map  # A dict that maps original_label -> new_label

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        mapped_label = self.label_map[label]
        return image, mapped_label


class STL10:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=8,
        train_batches=-1,
    ):

        # STL10         ->  CANONICAL INDEX:
        # 0: airplane   ->  0: airplane
        # 2: car        ->  1: automobile/car
        # 1: bird       ->  2: bird
        # 3: cat        ->  3: cat
        # 4: deer       ->  4: deer
        # 5: dog        ->  5: dog
        # None          ->  6: frog
        # 6: horse      ->  7: horse
        # 7: monkey      -> 8: monkey
        # 8: ship       ->  9: ship
        # 9: truck      ->  10: truck

        stl_to_canon_index = {
            0: 0,  # airplane
            2: 1,  # car
            1: 2,  # bird
            3: 3,  # cat
            4: 4,  # deer
            5: 5,  # dog
            6: 7,  # horse
            7: 8,  # monkey
            8: 9,  # ship
            9: 10,  # truck
        }

        location = os.path.join(location, "stl10")
        self.train_dataset = STL10Canonized(
            root=location,
            download=True,
            split="train",
            transform=preprocess,
            label_map=stl_to_canon_index,
        )

        self.train_dataset.classes = canonized_classnames

        if train_batches > 0:
            num_samples = train_batches * batch_size
            indices = list(range(num_samples))
            self.train_dataset = Subset(self.train_dataset, indices)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = STL10Canonized(
            root=location,
            download=True,
            split="test",
            transform=preprocess,
            label_map=stl_to_canon_index,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.test_dataset.classes = canonized_classnames
        self.classnames = canonized_classnames
