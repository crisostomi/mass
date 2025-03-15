import io
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from datasets import load_from_disk
from torch.utils.data import Subset


class CustomFER2013Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = Image.open(io.BytesIO(sample["img_bytes"])).convert("L")  # Convert to grayscale
        label = sample["labels"]

        if self.transform:
            image = self.transform(image)

        return image, label


class FER2013:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=6,
        train_batches=-1
    ):
        train_dataset_path = os.path.join(location, "fer-2013/train")
        test_dataset_path = os.path.join(location, "fer-2013/test")

        # Ensure the training dataset is loaded from disk
        if os.path.exists(train_dataset_path) and os.path.exists(test_dataset_path):
            print(f"Loading training dataset from {train_dataset_path}...")
            train_data = load_from_disk(train_dataset_path)

            print(f"Loading test dataset from {test_dataset_path}...")
            test_data = load_from_disk(test_dataset_path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {train_dataset_path} or {test_dataset_path}. Ensure they are correctly stored."
            )

        # Instantiate the custom PyTorch training dataset
        self.train_dataset = CustomFER2013Dataset(train_data, transform=preprocess)

        if train_batches > 0:
            num_samples = train_batches * batch_size
            indices = list(range(num_samples))
            self.train_dataset = Subset(self.train_dataset, indices)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        # Instantiate the custom PyTorch test dataset
        self.test_dataset = CustomFER2013Dataset(test_data, transform=preprocess)

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = [
            ["angry"],
            ["disgusted"],
            ["fearful"],
            ["happy", "smiling"],
            ["sad", "depressed"],
            ["surprised", "shocked", "spooked"],
            ["neutral", "bored"],
        ]
