"""
Evaluation metrics for attribution methods.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats
from torch import Tensor

logger = logging.getLogger(__name__)


class AttributionMetrics:
    """
    Comprehensive evaluation metrics for attribution methods.
    """
    
    @staticmethod
    def faithfulness_deletion(
        model: torch.nn.Module,
        inputs: Tensor,
        attributions: Tensor,
        targets: Tensor,
        deletion_ratio: float = 0.1,
    ) -> float:
        """
        Compute faithfulness using deletion test.
        
        Remove top-k% most important features and measure performance drop.
        
        Args:
            model: PyTorch model
            inputs: Input tensor
            attributions: Attribution tensor
            targets: Target tensor
            deletion_ratio: Fraction of features to delete
            
        Returns:
            Faithfulness score (higher is better)
        """
        model.eval()
        
        # Get original predictions
        with torch.no_grad():
            original_outputs = model(inputs)
            original_probs = torch.softmax(original_outputs, dim=1)
            original_conf = original_probs.gather(1, targets.unsqueeze(1)).squeeze()
        
        # Get top-k most important features
        k = int(deletion_ratio * attributions.numel())
        _, top_indices = torch.topk(torch.abs(attributions).flatten(), k)
        
        # Create masked inputs
        masked_inputs = inputs.clone()
        flat_inputs = masked_inputs.flatten()
        flat_inputs[top_indices] = 0
        masked_inputs = flat_inputs.reshape_as(inputs)
        
        # Get predictions on masked inputs
        with torch.no_grad():
            masked_outputs = model(masked_inputs)
            masked_probs = torch.softmax(masked_outputs, dim=1)
            masked_conf = masked_probs.gather(1, targets.unsqueeze(1)).squeeze()
        
        # Compute faithfulness as confidence drop
        faithfulness = (original_conf - masked_conf).mean().item()
        
        return faithfulness
    
    @staticmethod
    def faithfulness_insertion(
        model: torch.nn.Module,
        inputs: Tensor,
        attributions: Tensor,
        targets: Tensor,
        insertion_ratio: float = 0.1,
    ) -> float:
        """
        Compute faithfulness using insertion test.
        
        Add top-k% most important features to baseline and measure performance gain.
        
        Args:
            model: PyTorch model
            inputs: Input tensor
            attributions: Attribution tensor
            targets: Target tensor
            insertion_ratio: Fraction of features to insert
            
        Returns:
            Faithfulness score (higher is better)
        """
        model.eval()
        
        # Create baseline (zeros)
        baseline = torch.zeros_like(inputs)
        
        # Get baseline predictions
        with torch.no_grad():
            baseline_outputs = model(baseline)
            baseline_probs = torch.softmax(baseline_outputs, dim=1)
            baseline_conf = baseline_probs.gather(1, targets.unsqueeze(1)).squeeze()
        
        # Get top-k most important features
        k = int(insertion_ratio * attributions.numel())
        _, top_indices = torch.topk(torch.abs(attributions).flatten(), k)
        
        # Create insertion inputs
        insertion_inputs = baseline.clone()
        flat_inputs = insertion_inputs.flatten()
        flat_original = inputs.flatten()
        flat_inputs[top_indices] = flat_original[top_indices]
        insertion_inputs = flat_inputs.reshape_as(inputs)
        
        # Get predictions on insertion inputs
        with torch.no_grad():
            insertion_outputs = model(insertion_inputs)
            insertion_probs = torch.softmax(insertion_outputs, dim=1)
            insertion_conf = insertion_probs.gather(1, targets.unsqueeze(1)).squeeze()
        
        # Compute faithfulness as confidence gain
        faithfulness = (insertion_conf - baseline_conf).mean().item()
        
        return faithfulness
    
    @staticmethod
    def stability_spearman(
        attributions1: Tensor,
        attributions2: Tensor,
    ) -> float:
        """
        Compute Spearman correlation between two attribution maps.
        
        Args:
            attributions1: First attribution tensor
            attributions2: Second attribution tensor
            
        Returns:
            Spearman correlation coefficient
        """
        flat1 = attributions1.flatten().detach().cpu().numpy()
        flat2 = attributions2.flatten().detach().cpu().numpy()
        
        correlation, _ = stats.spearmanr(flat1, flat2)
        return correlation
    
    @staticmethod
    def stability_kendall(
        attributions1: Tensor,
        attributions2: Tensor,
    ) -> float:
        """
        Compute Kendall tau correlation between two attribution maps.
        
        Args:
            attributions1: First attribution tensor
            attributions2: Second attribution tensor
            
        Returns:
            Kendall tau correlation coefficient
        """
        flat1 = attributions1.flatten().detach().cpu().numpy()
        flat2 = attributions2.flatten().detach().cpu().numpy()
        
        correlation, _ = stats.kendalltau(flat1, flat2)
        return correlation
    
    @staticmethod
    def stability_iou(
        attributions1: Tensor,
        attributions2: Tensor,
        threshold: float = 0.1,
    ) -> float:
        """
        Compute Intersection over Union (IoU) for top-k attribution regions.
        
        Args:
            attributions1: First attribution tensor
            attributions2: Second attribution tensor
            threshold: Threshold for binary attribution maps
            
        Returns:
            IoU score
        """
        # Create binary masks
        mask1 = torch.abs(attributions1) > threshold
        mask2 = torch.abs(attributions2) > threshold
        
        # Compute IoU
        intersection = (mask1 & mask2).sum().float()
        union = (mask1 | mask2).sum().float()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (intersection / union).item()
    
    @staticmethod
    def sanity_check_randomization(
        model: torch.nn.Module,
        inputs: Tensor,
        targets: Tensor,
        random_seed: int = 42,
    ) -> Dict[str, float]:
        """
        Perform sanity check by randomizing model weights.
        
        Args:
            model: PyTorch model
            inputs: Input tensor
            targets: Target tensor
            random_seed: Random seed for weight randomization
            
        Returns:
            Dictionary containing sanity check results
        """
        # Save original weights
        original_state_dict = model.state_dict().copy()
        
        # Randomize weights
        torch.manual_seed(random_seed)
        for param in model.parameters():
            param.data = torch.randn_like(param.data) * 0.1
        
        # Get attributions with randomized weights
        from ..explainers.integrated_gradients import IntegratedGradientsExplainer
        
        explainer = IntegratedGradientsExplainer(model)
        random_attributions = explainer.explain(inputs, target=targets)
        
        # Restore original weights
        model.load_state_dict(original_state_dict)
        
        # Get attributions with original weights
        explainer = IntegratedGradientsExplainer(model)
        original_attributions = explainer.explain(inputs, target=targets)
        
        # Compute similarity metrics
        spearman_corr = AttributionMetrics.stability_spearman(
            original_attributions, random_attributions
        )
        kendall_corr = AttributionMetrics.stability_kendall(
            original_attributions, random_attributions
        )
        iou_score = AttributionMetrics.stability_iou(
            original_attributions, random_attributions
        )
        
        return {
            "spearman_correlation": spearman_corr,
            "kendall_correlation": kendall_corr,
            "iou_score": iou_score,
            "is_sane": spearman_corr < 0.1,  # Threshold for sanity
        }
    
    @staticmethod
    def compute_comprehensive_metrics(
        model: torch.nn.Module,
        inputs: Tensor,
        attributions: Tensor,
        targets: Tensor,
        method_name: str = "unknown",
    ) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics for attributions.
        
        Args:
            model: PyTorch model
            inputs: Input tensor
            attributions: Attribution tensor
            targets: Target tensor
            method_name: Name of the attribution method
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {
            "method": method_name,
            "faithfulness_deletion": AttributionMetrics.faithfulness_deletion(
                model, inputs, attributions, targets
            ),
            "faithfulness_insertion": AttributionMetrics.faithfulness_insertion(
                model, inputs, attributions, targets
            ),
        }
        
        # Add sanity check
        sanity_results = AttributionMetrics.sanity_check_randomization(
            model, inputs, targets
        )
        metrics.update(sanity_results)
        
        return metrics
