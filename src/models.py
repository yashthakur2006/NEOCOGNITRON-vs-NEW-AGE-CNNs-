from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeoCognitronLike(nn.Module):
    """
    Neocognitron-inspired S/C architecture, trained with backprop.

    Structure:
    - S1: Conv(1 -> 16, k=5), ReLU
    - C1: MaxPool 2x2
    - S2: Conv(16 -> 32, k=5), ReLU
    - C2: MaxPool 2x2
    - S3: Conv(32 -> 64, k=3), ReLU
    - Global average pooling
    - Linear classifier (64 -> 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # S1-C1
        self.s1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.c1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # S2-C2
        self.s2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.c2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # S3
        self.s3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Classifier: GAP -> FC
        self.classifier = nn.Linear(64, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.s1(x))
        x = self.c1(x)

        x = F.relu(self.s2(x))
        x = self.c2(x)

        x = F.relu(self.s3(x))

        # Global average pooling to [B, 64]
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        logits = self.classifier(x)
        return logits


class TinyCNN(nn.Module):
    """
    Small modern-ish CNN baseline.

    Structure:
    - Conv(1 -> 32, k=3), ReLU
    - Conv(32 -> 32, k=3), ReLU
    - MaxPool 2x2
    - Conv(32 -> 64, k=3), ReLU
    - MaxPool 2x2
    - Flatten
    - FC(64*7*7 -> 128), ReLU
    - FC(128 -> 10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)

        x = F.relu(self.conv3(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
