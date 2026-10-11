"""Reproducible random seed helpers."""

import random


def set_global_seed(seed=42):
    """Seed random, NumPy, and PyTorch when they are installed."""
    random.seed(seed)

    try:
        import numpy
    except ImportError:
        pass
    else:
        numpy.random.seed(seed)

    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
