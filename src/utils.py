import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Optional, but makes CUDA more deterministic (can slow things down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: Optional[str] = None) -> torch.device:
    """
    Return a torch.device, preferring CUDA if available (unless overridden).

    Args:
        preferred: "cpu" or "cuda" or None. If None, choose automatically.

    Returns:
        torch.device
    """
    if preferred == "cpu":
        return torch.device("cpu")
    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("Requested CUDA but it's not available; falling back to CPU.")
            return torch.device("cpu")

    # Auto mode
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
