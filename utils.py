# utils.py
"""
General-purpose utility functions for Few-Shot Learning (FSL) experiments.

This module provides helper functions for:
    - Reproducibility setup
    - Metric tracking
    - Filesystem utilities
    - Model parameter counting
    - Device handling

These utilities support consistent experimental pipelines
for both training and evaluation phases.
"""

import os
import random
import numpy as np
import torch


# ============================================================
# Reproducibility Utilities
# ============================================================

def set_seed(seed: int = 42):
    """
    Sets global random seeds to ensure reproducible experiments.

    Applies deterministic behavior to:
        - Python's built-in random module
        - NumPy random generator
        - PyTorch (CPU and CUDA backends)

    Args:
        seed (int): Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Metric Tracking
# ============================================================

class AverageMeter:
    """
    Tracks and updates a running average of a scalar metric.

    Useful for monitoring loss, accuracy, or other statistics
    during training and evaluation loops.

    Example:
        >>> meter = AverageMeter()
        >>> meter.update(loss.item())
        >>> print(meter.avg)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.value = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        """
        Updates the running average with a new value.

        Args:
            val (float): New metric value.
            n (int): Weight or batch size (default: 1).
        """
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


# ============================================================
# Filesystem Utilities
# ============================================================

def ensure_dir(path: str):
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        path (str): Directory path to create.
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ============================================================
# Model Inspection
# ============================================================

def count_params(model):
    """
    Counts the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): Model instance.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Device Management
# ============================================================

def to_device(batch, device):
    """
    Moves a tensor or a sequence of tensors to a specified device.

    Recursively traverses lists or tuples to transfer each element.

    Args:
        batch (Tensor | list | tuple): Tensor(s) to move.
        device (torch.device or str): Target device ('cuda', 'mps', or 'cpu').

    Returns:
        Tensor | list | tuple: Tensor(s) moved to the target device.

    Example:
        >>> support, support_y, query, query_y = to_device(
        ...     (support, support_y, query, query_y), device
        ... )
    """
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device) for x in batch]
    elif torch.is_tensor(batch):
        return batch.to(device)
    else:
        return batch
