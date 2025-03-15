import copy
import inspect
import random
import sys
from tqdm import tqdm
import logging

import torch
from torch.utils.data import DataLoader, Dataset, Sampler, ConcatDataset, Subset
from torch.utils.data.dataset import random_split

from tvp.data.datasets.common import TaskDataset
from tvp.data.datasets.templates import get_dataset_to_label, get_dataset_label

from tvp.data.datasets.cars import Cars
from tvp.data.datasets.cifar10 import CIFAR10
from tvp.data.datasets.cifar100 import CIFAR100
from tvp.data.datasets.dtd import DTD
from tvp.data.datasets.emnist import EMNIST
from tvp.data.datasets.eurosat import EuroSAT, EuroSATVal
from tvp.data.datasets.fashionmnist import FashionMNIST

from tvp.data.datasets.fer2013 import FER2013
from tvp.data.datasets.flowers102 import Flowers102
from tvp.data.datasets.food101 import Food101
from tvp.data.datasets.gtsrb import GTSRB
from tvp.data.datasets.imagenet import ImageNet
from tvp.data.datasets.kmnist import KMNIST
from tvp.data.datasets.mnist import MNIST

from tvp.data.datasets.oxfordpets import OxfordIIITPet
from tvp.data.datasets.pcam import PCAM
from tvp.data.datasets.resisc45 import RESISC45
from tvp.data.datasets.sst2 import RenderedSST2
from tvp.data.datasets.stl10 import STL10
from tvp.data.datasets.sun397 import SUN397
from tvp.data.datasets.svhn import SVHN

registry = {name: obj for name, obj in inspect.getmembers(sys.modules[__name__], inspect.isclass)}

pylogger = logging.getLogger(__name__)


class GenericDataset(object):
    def __init__(self):
        self.train_dataset = None
        self.train_loader = None
        self.test_dataset = None
        self.test_loader = None
        self.classnames = None


def split_train_into_train_val(
    dataset, new_dataset_class_name, batch_size, num_workers, val_fraction, max_val_samples=None, seed=0
):
    assert val_fraction > 0.0 and val_fraction < 1.0
    total_size = len(dataset.train_dataset)
    val_size = int(total_size * val_fraction)
    if max_val_samples is not None:
        val_size = min(val_size, max_val_samples)
    train_size = total_size - val_size

    assert val_size > 0
    assert train_size > 0

    lengths = [train_size, val_size]

    trainset, valset = random_split(dataset.train_dataset, lengths, generator=torch.Generator().manual_seed(seed))
    if new_dataset_class_name == "MNISTVal":
        assert trainset.indices[0] == 36044

    new_dataset = None

    new_dataset_class = type(new_dataset_class_name, (GenericDataset,), {})
    new_dataset = new_dataset_class()

    new_dataset.train_dataset = trainset
    new_dataset.train_loader = torch.utils.data.DataLoader(
        new_dataset.train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    new_dataset.test_dataset = valset
    new_dataset.test_loader = torch.utils.data.DataLoader(
        new_dataset.test_dataset, batch_size=batch_size, num_workers=num_workers
    )

    new_dataset.classnames = copy.copy(dataset.classnames)

    return new_dataset


def get_dataset(
    dataset_name, preprocess_fn, location, batch_size=128, num_workers=1, val_fraction=0.1, max_val_samples=5000, number_of_train_batches=-1
):
    if dataset_name.endswith("Val"):
        # Handle val splits
        if dataset_name in registry:
            dataset_class = registry[dataset_name]
        else:
            base_dataset_name = dataset_name.split("Val")[0]
            base_dataset = get_dataset(base_dataset_name, preprocess_fn, location, batch_size, num_workers)
            dataset = split_train_into_train_val(
                base_dataset, dataset_name, batch_size, num_workers, val_fraction, max_val_samples
            )
            return dataset
    else:
        assert (
            dataset_name in registry
        ), f"Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}"
        dataset_class = registry[dataset_name]
        dataset = dataset_class(preprocess_fn, location=location, batch_size=batch_size, num_workers=num_workers, train_batches=number_of_train_batches)
    return dataset

def get_task_evaluation_dataset(
    dataset_names,
    preprocess_fn,
    location,
    batch_size=128,
    num_workers=8,
    train_samples=8e3,
    test_samples=5e2,
    seed=42
):
    train_datasets = []
    test_datasets = []

    for dataset_name in tqdm(dataset_names, desc="Loading Datasets"):
        dataset = get_dataset(dataset_name, preprocess_fn, location, batch_size, num_workers)
        if dataset is None:
            continue 

        full_train_dataset = dataset.train_dataset
        full_test_dataset = dataset.test_dataset

        num_train = len(full_train_dataset)
        num_test = len(full_test_dataset)

        train_indices = list(range(min(int(train_samples), int(num_train))))
        test_indices = list(range(min(int(test_samples), int(num_test))))

        if int(num_train) < int(train_samples):
            pylogger.warning(f"Insufficient train samples for {dataset_name}")
        if int(num_test) < int(test_samples):
            pylogger.warning(f"Insufficient test samples for {dataset_name}")

        train_subset = Subset(full_train_dataset, train_indices)
        test_subset = Subset(full_test_dataset, test_indices)

        task_index = get_dataset_label(dataset_name)

        train_datasets.append(TaskDataset(train_subset, task_index))
        test_datasets.append(TaskDataset(test_subset, task_index))

    unified_train_dataset = ConcatDataset(train_datasets)
    unified_test_dataset = ConcatDataset(test_datasets)

    unified_train_loader = DataLoader(
        unified_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    unified_test_loader = DataLoader(
        unified_test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    unified_dataset = GenericDataset()
    unified_dataset.train_dataset = unified_train_dataset
    unified_dataset.test_dataset = unified_test_dataset
    unified_dataset.train_loader = unified_train_loader
    unified_dataset.test_loader = unified_test_loader

    unified_dataset.classnames = None

    return unified_dataset