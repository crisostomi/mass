import os
import pathlib
import h5py
import torch
from typing import Any, Callable, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset

class PCAMDataset(Dataset):
    """Custom PCAM Dataset Loader (based on torchvision.datasets.PCAM)

    This implementation removes MD5 checksum verification and manual downloading
    from Google Drive but follows the same structure as the original torchvision class.

    Args:
        location (str): Root directory where the PCAM dataset is stored.
        split (str, optional): Dataset split - "train", "test", or "val".
        transform (callable, optional): Image transform function.
    """

    _FILES = {
        "train": {
            "images": "camelyonpatch_level_2_split_train_x.h5",
            "targets": "camelyonpatch_level_2_split_train_y.h5",
        },
        "test": {
            "images": "camelyonpatch_level_2_split_test_x.h5",
            "targets": "camelyonpatch_level_2_split_test_y.h5",
        },
        "val": {
            "images": "camelyonpatch_level_2_split_valid_x.h5",
            "targets": "camelyonpatch_level_2_split_valid_y.h5",
        },
    }

    def __init__(
        self,
        location: str,
        split: str,
        transform: Optional[Callable] = None,
    ):
        self.root = pathlib.Path(location) / "PCAM"
        self.split = split
        self.transform = transform

        # Validate split argument
        if split not in self._FILES:
            raise ValueError(f"Invalid split '{split}'. Choose from 'train', 'test', 'val'.")

        # Define file paths
        self.images_path = self.root / self._FILES[split]["images"]
        self.targets_path = self.root / self._FILES[split]["targets"]

        # Load the dataset
        self._load_h5_files()

    def _load_h5_files(self):
        """Loads images and labels from HDF5 files."""
        if not self.images_path.exists() or not self.targets_path.exists():
            raise RuntimeError(f"Dataset files not found in {self.root}. Please check the dataset location.")

        with h5py.File(self.images_path, "r") as images_data, h5py.File(self.targets_path, "r") as targets_data:
            self.data = images_data["x"][:]  # Image data
            self.labels = targets_data["y"][:]  # Labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image = self.data[idx]
        label = int(self.labels[idx, 0, 0, 0])  # Extract label properly

        # Convert to PIL Image (to match torchvision behavior)
        image = Image.fromarray(image).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class PCAM:
    """PCAM DataLoader wrapper class

    Args:
        preprocess (callable, optional): Image transformation function.
        location (str, optional): Path where the PCAM dataset is stored.
        batch_size (int, optional): Number of samples per batch.
        num_workers (int, optional): Number of worker threads for DataLoader.
    """

    def __init__(
        self,
        preprocess: Optional[Callable] = None,
        location: str = os.path.expanduser("~/data"),
        batch_size: int = 128,
        num_workers: int = 6
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create datasets manually
        self.train_dataset = PCAMDataset(location, split="train", transform=preprocess)

        self.test_dataset = PCAMDataset(location, split="test", transform=preprocess)

        # Data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

        self.classnames = [
            "lymph node",
            "lymph node containing metastatic tumor tissue",
        ]
