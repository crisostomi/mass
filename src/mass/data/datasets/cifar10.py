import os

import datasets
import numpy as np
import PIL
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10 as PyTorchCIFAR10
from torchvision.datasets import VisionDataset
from torch.utils.data import Subset

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


class CIFAR10Canonized(PyTorchCIFAR10):
    """
    A wrapper around the standard Cifar10 dataset that remaps the integer labels so they match a standardized class index order.
    """

    def __init__(self, root, label_map=None, **kwargs):
        super().__init__(root=root, **kwargs)
        self.label_map = label_map

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)

        mapped_label = self.label_map[label]
        return image, mapped_label


class CIFAR10:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=8,
    ):

        # CIFAR10       ->  CANONICAL INDEX:
        # 0: airplane   ->  0: airplane
        # 1: automobile ->  1: automobile/car
        # 2: bird       ->  2: bird
        # 3: cat        ->  3: cat
        # 4: deer       ->  4: deer
        # 5: dog        ->  5: dog
        # 6: frog       ->  6: frog
        # 7: horse      ->  7: horse
        # None          ->  8: monkey
        # 8: ship       ->  9: ship
        # 9: truck      ->  10: truck

        cifar10_to_canon_index = {
            0: 0,  # airplane
            1: 1,  # automobile
            2: 2,  # bird
            3: 3,  # cat
            4: 4,  # deer
            5: 5,  # dog
            6: 6,  # frog
            7: 7,  # horse
            8: 9,  # ship
            9: 10,  # truck
        }

        self.train_dataset = CIFAR10Canonized(
            root=location,
            download=True,
            train=True,
            transform=preprocess,
            label_map=cifar10_to_canon_index,
        )

        self.train_dataset.classes = canonized_classnames

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = CIFAR10Canonized(
            root=location,
            download=True,
            train=False,
            transform=preprocess,
            label_map=cifar10_to_canon_index,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.test_dataset.classes = canonized_classnames
        self.classnames = canonized_classnames


def convert(x):
    if isinstance(x, np.ndarray):
        return torchvision.transforms.functional.to_pil_image(x)
    return x


class BasicVisionDataset(VisionDataset):
    def __init__(self, images, targets, transform=None, target_transform=None):
        if transform is not None:
            transform.transforms.insert(0, convert)
        super(BasicVisionDataset, self).__init__(
            root=None, transform=transform, target_transform=target_transform
        )
        assert len(images) == len(targets)

        self.images = images
        self.targets = targets

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.targets[index]

    def __len__(self):
        return len(self.targets)
