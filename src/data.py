from typing import Tuple

import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def get_mnist_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, Dataset]:
    """
    Download MNIST and return train & test DataLoaders, plus the raw test Dataset.

    Args:
        batch_size: Batch size for loaders.
        num_workers: Number of worker processes for data loading.

    Returns:
        train_loader, test_loader, test_dataset
    """
    transform = transforms.ToTensor()

    train_ds = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_ds = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, test_ds


def shift_tensor(img: torch.Tensor, max_shift: int) -> torch.Tensor:
    """
    Shift a [1, 28, 28] tensor by up to +/- max_shift pixels in x and y with zero padding.

    Args:
        img: Tensor of shape [1, H, W] (MNIST: [1, 28, 28]).
        max_shift: Maximum absolute shift in pixels along each axis.

    Returns:
        Shifted tensor of the same shape.
    """
    _, h, w = img.shape
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)

    shifted = torch.zeros_like(img)

    x_start_src = max(0, -dx)
    y_start_src = max(0, -dy)
    x_end_src = min(w, w - dx)
    y_end_src = min(h, h - dy)

    x_start_dst = max(0, dx)
    y_start_dst = max(0, dy)
    x_end_dst = x_start_dst + (x_end_src - x_start_src)
    y_end_dst = y_start_dst + (y_end_src - y_start_src)

    shifted[:, y_start_dst:y_end_dst, x_start_dst:x_end_dst] = img[
        :, y_start_src:y_end_src, x_start_src:x_end_src
    ]
    return shifted


class ShiftedMNIST(Dataset):
    """
    Wrap the MNIST test dataset and apply random spatial shifts on-the-fly.

    Useful for measuring accuracy under translations.
    """

    def __init__(self, base_dataset: Dataset, max_shift: int) -> None:
        self.base = base_dataset
        self.max_shift = max_shift

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, label = self.base[idx]
        img = shift_tensor(img, self.max_shift)
        return img, label
