"""
Device management utilities for PyTorch models.
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch computations.
    
    Priority: CUDA -> MPS (Apple Silicon) -> CPU
    
    Returns:
        torch.device: The best available device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    return device


def move_to_device(
    obj: any, 
    device: Optional[torch.device] = None
) -> any:
    """
    Move a PyTorch object to the specified device.
    
    Args:
        obj: PyTorch object (tensor, model, etc.)
        device: Target device (auto-detected if None)
        
    Returns:
        Object moved to device
    """
    if device is None:
        device = get_device()
    
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    else:
        logger.warning(f"Cannot move object of type {type(obj)} to device")
        return obj
