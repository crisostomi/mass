import os
import re

import torch
import torchvision.datasets as datasets
from torch.utils.data import Subset


def pretify_classname(classname):
    tmp = re.findall(r"[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))", classname)
    tmp = [i.lower() for i in tmp]
    out = " ".join(tmp)
    if out.endswith("al"):
        return out + " area"
    return out


class EuroSATBase:
    def __init__(self, preprocess, test_split, location="~/datasets", batch_size=32, num_workers=8, train_batches=-1):
        # Data loading code
        traindir = os.path.join(location, "EuroSAT_splits", "train")
        testdir = os.path.join(location, "EuroSAT_splits", test_split)

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

        self.test_dataset = datasets.ImageFolder(testdir, transform=preprocess)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=batch_size, num_workers=num_workers
        )
        idx_to_class = dict((v, k) for k, v in self.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace("_", " ") for i in range(len(idx_to_class))]
        self.classnames = [pretify_classname(c) for c in self.classnames]
        ours_to_open_ai = {
            "annual crop": "annual crop land", # 0
            "forest": "forest", # 1
            "herbaceous vegetation": "brushland or shrubland", # 2
            "highway": "highway or road", # 3
            "industrial area": "industrial buildings or commercial buildings", # 4
            "pasture": "pasture land", # 5
            "permanent crop": "permanent crop land", # 6
            "residential area": "residential buildings or homes or apartments", # 7
            "river": "river", # 8
            "sea lake": "lake or sea", # 9
        }
        for i in range(len(self.classnames)):
            self.classnames[i] = ours_to_open_ai[self.classnames[i]]


class EuroSAT(EuroSATBase):
    def __init__(self, preprocess, location="~/datasets", batch_size=32, num_workers=16, train_batches=-1):
        super().__init__(preprocess, "test", location, batch_size, num_workers, train_batches=train_batches)


class EuroSATVal(EuroSATBase):
    def __init__(self, preprocess, location="~/datasets", batch_size=32, num_workers=16):
        super().__init__(preprocess, "val", location, batch_size, num_workers)
