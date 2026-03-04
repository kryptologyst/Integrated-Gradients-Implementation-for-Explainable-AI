"""
Seeding utilities for reproducible experiments.
"""

import logging
import random
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_deterministic_seed(seed: int) -> None:
    """
    Set deterministic seed for all random number generators.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Set deterministic seed: {seed}")


def get_random_seed() -> int:
    """
    Get a random seed for experiments.
    
    Returns:
        Random seed value
    """
    return random.randint(0, 2**32 - 1)
