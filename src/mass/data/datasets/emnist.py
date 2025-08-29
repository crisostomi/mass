import os

import torch
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import Subset


class EMNIST:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=6,
    ):

        preprocess = torchvision.transforms.Compose(
            [
                preprocess,
                lambda img: torchvision.transforms.functional.rotate(img, -90),
                lambda img: torchvision.transforms.functional.hflip(img),
            ]
        )

        # location = os.path.join(location, "EMNIST")
        self.train_dataset = datasets.EMNIST(
            root=location,
            download=True,
            split="digits",
            transform=preprocess,
            train=True,
        )

        self.classnames = self.train_dataset.classes

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.EMNIST(
            root=location,
            download=True,
            split="digits",
            transform=preprocess,
            train=False,
        )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
