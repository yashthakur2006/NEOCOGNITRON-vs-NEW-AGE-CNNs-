from typing import Dict, Any

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import shift_tensor


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """
    Train the model for a single epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    acc = total_correct / total

    print(f"[Train] Epoch {epoch:03d}: loss={avg_loss:.4f}, acc={acc*100:.2f}%")
    return {"loss": avg_loss, "acc": acc}


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "test",
) -> float:
    """
    Evaluate classification accuracy of the model on a given DataLoader.
    """
    model.eval()
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)

    acc = total_correct / total
    print(f"[Eval] {desc} accuracy: {acc*100:.2f}%")
    return acc


@torch.no_grad()
def invariance_score(
    model: nn.Module,
    base_dataset: Dataset,
    device: torch.device,
    max_shift: int,
    num_transforms: int = 5,
    num_samples: int = 2000,
) -> float:
    """
    Compute translation invariance score.

    For each sampled test image x:
      - compute base prediction on x
      - generate num_transforms random shifts of x within [-max_shift, max_shift]
      - compute the fraction of those shifted samples where the prediction stays the same
    Then average over all sampled images.

    Returns:
        Invariance score in [0, 1].
    """
    model.eval()
    indices = list(range(len(base_dataset)))
    random.shuffle(indices)
    indices = indices[:num_samples]

    consistent_counts = 0
    total_counts = 0

    for idx in indices:
        img, _ = base_dataset[idx]
        img = img.to(device).unsqueeze(0)  # [1, 1, 28, 28]
        base_pred = model(img).argmax(dim=1).item()

        for _ in range(num_transforms):
            shifted = shift_tensor(img.squeeze(0).cpu(), max_shift)
            shifted = shifted.to(device).unsqueeze(0)
            pred_shifted = model(shifted).argmax(dim=1).item()
            if pred_shifted == base_pred:
                consistent_counts += 1
            total_counts += 1

    score = consistent_counts / total_counts if total_counts > 0 else 0.0
    print(
        f"[Invariance] max_shift={max_shift}: "
        f"score={score:.4f} (consistent {consistent_counts}/{total_counts})"
    )
    return score
