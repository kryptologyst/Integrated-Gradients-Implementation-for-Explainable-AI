#!/usr/bin/env python3
"""
Simple example script demonstrating Integrated Gradients usage.

This script shows how to:
1. Load a pre-trained model
2. Create sample data
3. Compute Integrated Gradients attributions
4. Visualize results
5. Compare different attribution methods
"""

import logging
import matplotlib.pyplot as plt
import torch

from src.data.datasets import create_sample_batch
from src.explainers.integrated_gradients import IntegratedGradientsExplainer
from src.models.model_utils import load_pretrained_model
from src.utils.device import get_device
from src.utils.seeding import set_deterministic_seed
from src.viz.visualization import AttributionVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main example function."""
    logger.info("Starting Integrated Gradients example...")
    
    # Set random seed for reproducibility
    set_deterministic_seed(42)
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading pre-trained model...")
    model = load_pretrained_model("resnet18", num_classes=10, pretrained=True)
    model.to(device)
    model.eval()
    
    # Create sample data
    logger.info("Creating sample data...")
    inputs, targets = create_sample_batch("cifar10", batch_size=4)
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    logger.info(f"Sample shape: {inputs.shape}")
    logger.info(f"Targets: {targets.tolist()}")
    
    # Initialize explainer
    logger.info("Initializing Integrated Gradients explainer...")
    explainer = IntegratedGradientsExplainer(
        model=model,
        device=device,
        baseline_method="zeros",
        n_steps=50,
    )
    
    # Compute Integrated Gradients attributions
    logger.info("Computing Integrated Gradients attributions...")
    ig_attributions = explainer.explain(
        inputs=inputs,
        target=targets,
        method="integrated_gradients",
    )
    
    logger.info(f"Attribution shape: {ig_attributions.shape}")
    logger.info(f"Attribution range: [{ig_attributions.min():.4f}, {ig_attributions.max():.4f}]")
    
    # Compare multiple methods
    logger.info("Comparing multiple attribution methods...")
    methods = ["integrated_gradients", "gradient_shap", "saliency", "guided_backprop"]
    comparison_results = explainer.compare_methods(
        inputs=inputs,
        target=targets,
        methods=methods,
    )
    
    # Print comparison results
    logger.info("Method comparison results:")
    for method, attributions in comparison_results.items():
        if attributions is not None:
            logger.info(f"  {method}: shape={attributions.shape}, range=[{attributions.min():.4f}, {attributions.max():.4f}]")
        else:
            logger.info(f"  {method}: Failed to compute")
    
    # Create visualizations
    logger.info("Creating visualizations...")
    visualizer = AttributionVisualizer()
    
    # Visualize Integrated Gradients
    visualizer.visualize_attributions(
        inputs=inputs.cpu(),
        attributions=ig_attributions.cpu(),
        targets=targets.cpu(),
        method_name="Integrated Gradients",
        save_path="assets/ig_example.png",
    )
    
    # Compare methods
    valid_comparison = {k: v for k, v in comparison_results.items() if v is not None}
    if valid_comparison:
        visualizer.compare_methods(
            inputs=inputs.cpu(),
            attribution_dict=valid_comparison,
            targets=targets.cpu(),
            save_path="assets/method_comparison_example.png",
        )
    
    # Compute feature importance
    logger.info("Computing feature importance...")
    importance = explainer.get_feature_importance(ig_attributions, aggregation="mean")
    logger.info(f"Feature importance shape: {importance.shape}")
    logger.info(f"Top 5 most important features: {torch.topk(importance.flatten(), 5).indices.tolist()}")
    
    # Sensitivity analysis
    logger.info("Performing sensitivity analysis...")
    sensitivity = explainer.compute_sensitivity_analysis(
        inputs=inputs,
        target=targets,
        noise_levels=[0.01, 0.05, 0.1],
    )
    
    logger.info("Sensitivity analysis results:")
    for noise_level, similarity in zip(sensitivity["noise_levels"], sensitivity["similarities"]):
        logger.info(f"  Noise level {noise_level}: similarity = {similarity:.4f}")
    
    logger.info("Example completed successfully!")
    logger.info("Check the 'assets/' directory for saved visualizations.")


if __name__ == "__main__":
    main()
