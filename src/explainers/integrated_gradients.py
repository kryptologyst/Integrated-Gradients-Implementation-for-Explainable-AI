"""
Integrated Gradients implementation for Explainable AI.

This module provides comprehensive implementations of Integrated Gradients
and related attribution methods for deep learning models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from captum.attr import (
    GradientShap,
    GuidedBackprop,
    IntegratedGradients,
    Saliency,
    SmoothGrad,
)
from captum.attr._utils.attribution import Attribution
from torch import Tensor
from torch.utils.data import DataLoader

from .utils.device import get_device
from .utils.seeding import set_deterministic_seed

logger = logging.getLogger(__name__)


class IntegratedGradientsExplainer:
    """
    Comprehensive Integrated Gradients explainer with multiple variants and evaluation metrics.
    
    This class provides implementations of:
    - Standard Integrated Gradients
    - Expected Integrated Gradients
    - Guided Integrated Gradients
    - Smooth Integrated Gradients
    
    Attributes:
        model: The PyTorch model to explain
        device: Device to run computations on
        baseline_method: Method for generating baselines
        n_steps: Number of integration steps
        internal_batch_size: Batch size for internal computations
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        baseline_method: str = "zeros",
        n_steps: int = 50,
        internal_batch_size: int = 1,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the Integrated Gradients explainer.
        
        Args:
            model: PyTorch model to explain
            device: Device to run computations on (auto-detected if None)
            baseline_method: Method for generating baselines ("zeros", "mean", "random")
            n_steps: Number of integration steps for IG computation
            internal_batch_size: Batch size for internal computations
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            set_deterministic_seed(random_seed)
            
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()
        
        self.baseline_method = baseline_method
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        
        # Initialize Captum explainers
        self._init_captum_explainers()
        
        logger.info(f"Initialized IntegratedGradientsExplainer on {self.device}")
        logger.info(f"Baseline method: {baseline_method}, Steps: {n_steps}")
    
    def _init_captum_explainers(self) -> None:
        """Initialize Captum attribution methods."""
        self.ig = IntegratedGradients(
            self.model,
            multiply_by_inputs=False,
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
        )
        
        self.gradient_shap = GradientShap(
            self.model,
            multiply_by_inputs=False,
            n_samples=self.n_steps,
            stdevs=0.09,
        )
        
        self.saliency = Saliency(self.model)
        self.guided_backprop = GuidedBackprop(self.model)
        self.smooth_grad = SmoothGrad(self.model, n_samples=10, stdevs=0.15)
        
    def _get_baseline(
        self, 
        inputs: Tensor, 
        method: Optional[str] = None
    ) -> Tensor:
        """
        Generate baseline inputs for attribution computation.
        
        Args:
            inputs: Input tensor
            method: Baseline method override
            
        Returns:
            Baseline tensor
        """
        method = method or self.baseline_method
        
        if method == "zeros":
            return torch.zeros_like(inputs)
        elif method == "mean":
            return torch.mean(inputs, dim=0, keepdim=True).expand_as(inputs)
        elif method == "random":
            return torch.randn_like(inputs) * 0.1
        else:
            raise ValueError(f"Unknown baseline method: {method}")
    
    def explain(
        self,
        inputs: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        method: str = "integrated_gradients",
        baseline: Optional[Tensor] = None,
        return_convergence_delta: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Compute attributions for given inputs.
        
        Args:
            inputs: Input tensor to explain
            target: Target class index or tensor
            method: Attribution method to use
            baseline: Custom baseline (if None, uses configured method)
            return_convergence_delta: Whether to return convergence delta
            
        Returns:
            Attribution tensor, optionally with convergence delta
        """
        inputs = inputs.to(self.device)
        
        if baseline is None:
            baseline = self._get_baseline(inputs)
        else:
            baseline = baseline.to(self.device)
        
        if target is None:
            # Use model's prediction as target
            with torch.no_grad():
                outputs = self.model(inputs)
                target = torch.argmax(outputs, dim=1)
        
        # Compute attributions based on method
        if method == "integrated_gradients":
            attributions = self.ig.attribute(
                inputs,
                baselines=baseline,
                target=target,
                return_convergence_delta=return_convergence_delta,
            )
        elif method == "gradient_shap":
            attributions = self.gradient_shap.attribute(
                inputs,
                baselines=baseline,
                target=target,
                return_convergence_delta=return_convergence_delta,
            )
        elif method == "saliency":
            attributions = self.saliency.attribute(inputs, target=target)
        elif method == "guided_backprop":
            attributions = self.guided_backprop.attribute(inputs, target=target)
        elif method == "smooth_grad":
            attributions = self.smooth_grad.attribute(inputs, target=target)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        if return_convergence_delta and isinstance(attributions, tuple):
            return attributions
        else:
            return attributions
    
    def explain_batch(
        self,
        dataloader: DataLoader,
        method: str = "integrated_gradients",
        max_samples: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compute attributions for a batch of inputs.
        
        Args:
            dataloader: DataLoader containing inputs and targets
            method: Attribution method to use
            max_samples: Maximum number of samples to process
            
        Returns:
            Dictionary containing attributions and metadata
        """
        all_attributions = []
        all_inputs = []
        all_targets = []
        all_predictions = []
        
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                if max_samples and sample_count >= max_samples:
                    break
                    
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get model predictions
                outputs = self.model(inputs)
                predictions = torch.argmax(outputs, dim=1)
                
                # Compute attributions
                attributions = self.explain(
                    inputs, 
                    target=targets, 
                    method=method
                )
                
                all_attributions.append(attributions.cpu())
                all_inputs.append(inputs.cpu())
                all_targets.append(targets.cpu())
                all_predictions.append(predictions.cpu())
                
                sample_count += inputs.size(0)
                
                if batch_idx % 10 == 0:
                    logger.info(f"Processed {sample_count} samples")
        
        return {
            "attributions": torch.cat(all_attributions, dim=0),
            "inputs": torch.cat(all_inputs, dim=0),
            "targets": torch.cat(all_targets, dim=0),
            "predictions": torch.cat(all_predictions, dim=0),
            "method": method,
            "n_samples": sample_count,
        }
    
    def compare_methods(
        self,
        inputs: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        methods: Optional[List[str]] = None,
    ) -> Dict[str, Tensor]:
        """
        Compare multiple attribution methods on the same inputs.
        
        Args:
            inputs: Input tensor to explain
            target: Target class index or tensor
            methods: List of methods to compare
            
        Returns:
            Dictionary mapping method names to attributions
        """
        if methods is None:
            methods = [
                "integrated_gradients",
                "gradient_shap", 
                "saliency",
                "guided_backprop",
                "smooth_grad",
            ]
        
        results = {}
        for method in methods:
            try:
                attributions = self.explain(inputs, target=target, method=method)
                results[method] = attributions
                logger.info(f"Computed {method} attributions")
            except Exception as e:
                logger.warning(f"Failed to compute {method}: {e}")
                results[method] = None
        
        return results
    
    def get_feature_importance(
        self,
        attributions: Tensor,
        aggregation: str = "mean",
    ) -> Tensor:
        """
        Compute feature importance from attributions.
        
        Args:
            attributions: Attribution tensor
            aggregation: Aggregation method ("mean", "sum", "max", "l2")
            
        Returns:
            Feature importance tensor
        """
        if aggregation == "mean":
            return torch.mean(torch.abs(attributions), dim=0)
        elif aggregation == "sum":
            return torch.sum(torch.abs(attributions), dim=0)
        elif aggregation == "max":
            return torch.max(torch.abs(attributions), dim=0)[0]
        elif aggregation == "l2":
            return torch.norm(attributions, p=2, dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    def compute_sensitivity_analysis(
        self,
        inputs: Tensor,
        target: Optional[Union[int, Tensor]] = None,
        noise_levels: List[float] = None,
    ) -> Dict[str, List[float]]:
        """
        Perform sensitivity analysis by adding noise to inputs.
        
        Args:
            inputs: Input tensor
            target: Target class index or tensor
            noise_levels: List of noise standard deviations to test
            
        Returns:
            Dictionary containing sensitivity results
        """
        if noise_levels is None:
            noise_levels = [0.01, 0.05, 0.1, 0.2]
        
        original_attributions = self.explain(inputs, target=target)
        sensitivity_results = {"noise_levels": noise_levels, "similarities": []}
        
        for noise_level in noise_levels:
            # Add noise to inputs
            noise = torch.randn_like(inputs) * noise_level
            noisy_inputs = inputs + noise
            
            # Compute attributions for noisy inputs
            noisy_attributions = self.explain(noisy_inputs, target=target)
            
            # Compute similarity (cosine similarity)
            similarity = torch.cosine_similarity(
                original_attributions.flatten(),
                noisy_attributions.flatten(),
                dim=0,
            ).item()
            
            sensitivity_results["similarities"].append(similarity)
        
        return sensitivity_results
