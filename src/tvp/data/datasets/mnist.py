import os

import torch
import torchvision.datasets as datasets
from torch.utils.data import Subset


class MNIST:
    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=128, num_workers=8, train_batches=-1):
        self.train_dataset = datasets.MNIST(root=location, download=True, train=True, transform=preprocess)

        if train_batches > 0:
            num_samples = train_batches * batch_size
            indices = list(range(num_samples))
            self.train_dataset = Subset(self.train_dataset, indices)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        self.test_dataset = datasets.MNIST(root=location, download=True, train=False, transform=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        self.classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
