"""
Visualization utilities for attribution methods.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch import Tensor

logger = logging.getLogger(__name__)


class AttributionVisualizer:
    """
    Comprehensive visualization tools for attribution methods.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("husl")
    
    def visualize_attributions(
        self,
        inputs: Tensor,
        attributions: Tensor,
        targets: Optional[Tensor] = None,
        predictions: Optional[Tensor] = None,
        method_name: str = "Integrated Gradients",
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Visualize attributions for a batch of inputs.
        
        Args:
            inputs: Input images
            attributions: Attribution maps
            targets: Ground truth targets
            predictions: Model predictions
            method_name: Name of attribution method
            class_names: List of class names
            save_path: Path to save figure
        """
        batch_size = inputs.size(0)
        n_cols = min(4, batch_size)
        n_rows = (batch_size + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(n_cols * 6, n_rows * 3))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            row = i // n_cols
            col = i % n_cols
            
            # Original image
            ax_orig = axes[row, col * 2]
            self._plot_image(
                ax_orig, 
                inputs[i], 
                title=f"Original {i+1}",
                target=targets[i] if targets is not None else None,
                prediction=predictions[i] if predictions is not None else None,
                class_names=class_names,
            )
            
            # Attribution map
            ax_attr = axes[row, col * 2 + 1]
            self._plot_attribution(
                ax_attr,
                inputs[i],
                attributions[i],
                title=f"{method_name} {i+1}",
            )
        
        # Hide unused subplots
        for i in range(batch_size, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col * 2].set_visible(False)
            axes[row, col * 2 + 1].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def _plot_image(
        self,
        ax: plt.Axes,
        image: Tensor,
        title: str,
        target: Optional[int] = None,
        prediction: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """Plot a single image."""
        # Convert tensor to numpy
        if image.dim() == 3:
            img_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            img_np = image.cpu().numpy()
        
        # Denormalize if needed
        if img_np.min() < 0:
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        ax.imshow(img_np)
        ax.set_title(title)
        ax.axis('off')
        
        # Add target/prediction info
        if target is not None or prediction is not None:
            info_text = ""
            if target is not None:
                target_name = class_names[target] if class_names else f"Class {target}"
                info_text += f"Target: {target_name}"
            if prediction is not None:
                pred_name = class_names[prediction] if class_names else f"Class {prediction}"
                info_text += f"\nPred: {pred_name}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _plot_attribution(
        self,
        ax: plt.Axes,
        image: Tensor,
        attribution: Tensor,
        title: str,
    ) -> None:
        """Plot attribution map."""
        # Convert to numpy
        if attribution.dim() == 3:
            attr_np = attribution.permute(1, 2, 0).cpu().numpy()
        else:
            attr_np = attribution.cpu().numpy()
        
        # Handle multi-channel attributions
        if attr_np.shape[-1] > 1:
            # Sum across channels for visualization
            attr_np = np.sum(np.abs(attr_np), axis=-1)
        
        # Normalize
        attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
        
        # Create heatmap
        im = ax.imshow(attr_np, cmap='jet', alpha=0.7)
        ax.set_title(title)
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def compare_methods(
        self,
        inputs: Tensor,
        attribution_dict: dict,
        targets: Optional[Tensor] = None,
        predictions: Optional[Tensor] = None,
        class_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Compare multiple attribution methods side by side.
        
        Args:
            inputs: Input images
            attribution_dict: Dictionary mapping method names to attributions
            targets: Ground truth targets
            predictions: Model predictions
            class_names: List of class names
            save_path: Path to save figure
        """
        batch_size = inputs.size(0)
        n_methods = len(attribution_dict)
        
        fig, axes = plt.subplots(
            batch_size, n_methods + 1, 
            figsize=((n_methods + 1) * 3, batch_size * 3)
        )
        
        if batch_size == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(batch_size):
            # Original image
            ax_orig = axes[i, 0]
            self._plot_image(
                ax_orig,
                inputs[i],
                f"Original {i+1}",
                target=targets[i] if targets is not None else None,
                prediction=predictions[i] if predictions is not None else None,
                class_names=class_names,
            )
            
            # Attribution maps
            for j, (method_name, attributions) in enumerate(attribution_dict.items()):
                if attributions is not None:
                    ax_attr = axes[i, j + 1]
                    self._plot_attribution(
                        ax_attr,
                        inputs[i],
                        attributions[i],
                        f"{method_name} {i+1}",
                    )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved comparison to {save_path}")
        
        plt.show()
    
    def plot_attribution_statistics(
        self,
        attribution_dict: dict,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot statistics comparing different attribution methods.
        
        Args:
            attribution_dict: Dictionary mapping method names to attributions
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        methods = list(attribution_dict.keys())
        valid_methods = [m for m in methods if attribution_dict[m] is not None]
        
        # Attribution magnitudes
        ax1 = axes[0, 0]
        magnitudes = []
        for method in valid_methods:
            attr = attribution_dict[method]
            mag = torch.mean(torch.abs(attr)).item()
            magnitudes.append(mag)
        
        ax1.bar(valid_methods, magnitudes)
        ax1.set_title("Average Attribution Magnitude")
        ax1.set_ylabel("Magnitude")
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Attribution sparsity
        ax2 = axes[0, 1]
        sparsities = []
        for method in valid_methods:
            attr = attribution_dict[method]
            # Count non-zero elements
            non_zero = torch.count_nonzero(attr).item()
            total = attr.numel()
            sparsity = 1 - (non_zero / total)
            sparsities.append(sparsity)
        
        ax2.bar(valid_methods, sparsities)
        ax2.set_title("Attribution Sparsity")
        ax2.set_ylabel("Sparsity (1 - density)")
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Attribution distribution
        ax3 = axes[1, 0]
        for method in valid_methods:
            attr = attribution_dict[method]
            flat_attr = attr.flatten().cpu().numpy()
            ax3.hist(flat_attr, alpha=0.5, label=method, bins=50)
        
        ax3.set_title("Attribution Distribution")
        ax3.set_xlabel("Attribution Value")
        ax3.set_ylabel("Frequency")
        ax3.legend()
        
        # Method comparison heatmap
        ax4 = axes[1, 1]
        if len(valid_methods) > 1:
            similarities = np.zeros((len(valid_methods), len(valid_methods)))
            for i, method1 in enumerate(valid_methods):
                for j, method2 in enumerate(valid_methods):
                    if i == j:
                        similarities[i, j] = 1.0
                    else:
                        attr1 = attribution_dict[method1].flatten()
                        attr2 = attribution_dict[method2].flatten()
                        similarity = torch.cosine_similarity(attr1, attr2, dim=0).item()
                        similarities[i, j] = similarity
            
            im = ax4.imshow(similarities, cmap='coolwarm', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(valid_methods)))
            ax4.set_yticks(range(len(valid_methods)))
            ax4.set_xticklabels(valid_methods, rotation=45, ha='right')
            ax4.set_yticklabels(valid_methods)
            ax4.set_title("Method Similarity")
            
            # Add text annotations
            for i in range(len(valid_methods)):
                for j in range(len(valid_methods)):
                    text = ax4.text(j, i, f'{similarities[i, j]:.2f}',
                                   ha="center", va="center", color="black")
            
            plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved statistics to {save_path}")
        
        plt.show()
    
    def plot_evaluation_metrics(
        self,
        metrics_dict: dict,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot evaluation metrics for different methods.
        
        Args:
            metrics_dict: Dictionary mapping method names to metrics
            save_path: Path to save figure
        """
        methods = list(metrics_dict.keys())
        metrics = list(metrics_dict[methods[0]].keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):  # Plot first 4 metrics
            ax = axes[i]
            values = [metrics_dict[method][metric] for method in methods]
            
            bars = ax.bar(methods, values)
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_ylabel("Score")
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics to {save_path}")
        
        plt.show()
