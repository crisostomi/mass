import os

import torch
import torchvision.datasets as datasets
from torch.utils.data import Subset

class DTD:
    def __init__(self, preprocess, location=os.path.expanduser("~/data"), batch_size=32, num_workers=8, train_batches=-1):
        # Data loading code
        traindir = os.path.join(location, "dtd", "train")
        valdir = os.path.join(location, "dtd", "val")

        self.train_dataset = datasets.ImageFolder(traindir, transform=preprocess)
        self.class_to_idx = self.train_dataset.class_to_idx

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

        self.test_dataset = datasets.ImageFolder(valdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        idx_to_class = dict((v, k) for k, v in self.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]
