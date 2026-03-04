"""
Main evaluation script for Integrated Gradients.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from omegaconf import OmegaConf

from src.data.datasets import get_data_loaders
from src.explainers.integrated_gradients import IntegratedGradientsExplainer
from src.metrics.attribution_metrics import AttributionMetrics
from src.models.model_utils import load_pretrained_model, evaluate_model
from src.utils.device import get_device
from src.utils.seeding import set_deterministic_seed
from src.viz.visualization import AttributionVisualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_integrated_gradients(
    config: Dict,
    save_results: bool = True,
) -> Dict:
    """
    Comprehensive evaluation of Integrated Gradients and related methods.
    
    Args:
        config: Configuration dictionary
        save_results: Whether to save results
        
    Returns:
        Dictionary containing evaluation results
    """
    # Set random seed
    if config.get('random_seed'):
        set_deterministic_seed(config['random_seed'])
    
    # Get device
    device = get_device()
    
    # Load data
    logger.info("Loading dataset...")
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name=config['dataset']['name'],
        data_dir=config['dataset']['data_dir'],
        batch_size=config['dataset']['batch_size'],
        num_workers=config['dataset']['num_workers'],
    )
    
    # Load model
    logger.info("Loading model...")
    model = load_pretrained_model(
        model_name=config['model']['name'],
        num_classes=config['dataset']['num_classes'],
        pretrained=config['model']['pretrained'],
    )
    model.to(device)
    
    # Evaluate model performance
    logger.info("Evaluating model performance...")
    test_metrics = evaluate_model(model, test_loader, device)
    logger.info(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    
    # Initialize explainer
    logger.info("Initializing explainer...")
    explainer = IntegratedGradientsExplainer(
        model=model,
        device=device,
        baseline_method=config['explainer']['baseline_method'],
        n_steps=config['explainer']['n_steps'],
        internal_batch_size=config['explainer']['internal_batch_size'],
        random_seed=config.get('random_seed'),
    )
    
    # Get sample batch for evaluation
    sample_inputs, sample_targets = next(iter(test_loader))
    sample_inputs = sample_inputs[:config['evaluation']['sample_size']]
    sample_targets = sample_targets[:config['evaluation']['sample_size']]
    
    # Compute attributions for different methods
    logger.info("Computing attributions...")
    methods_to_evaluate = config['evaluation']['methods']
    attribution_results = {}
    evaluation_metrics = {}
    
    for method in methods_to_evaluate:
        logger.info(f"Evaluating {method}...")
        
        try:
            # Compute attributions
            attributions = explainer.explain(
                sample_inputs,
                target=sample_targets,
                method=method,
            )
            attribution_results[method] = attributions
            
            # Compute evaluation metrics
            metrics = AttributionMetrics.compute_comprehensive_metrics(
                model=model,
                inputs=sample_inputs,
                attributions=attributions,
                targets=sample_targets,
                method_name=method,
            )
            evaluation_metrics[method] = metrics
            
            logger.info(f"{method} - Faithfulness (deletion): {metrics['faithfulness_deletion']:.4f}")
            logger.info(f"{method} - Faithfulness (insertion): {metrics['faithfulness_insertion']:.4f}")
            logger.info(f"{method} - Spearman correlation: {metrics['spearman_correlation']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {method}: {e}")
            attribution_results[method] = None
            evaluation_metrics[method] = None
    
    # Create visualizations
    if config['evaluation']['create_visualizations']:
        logger.info("Creating visualizations...")
        visualizer = AttributionVisualizer()
        
        # Visualize individual methods
        for method, attributions in attribution_results.items():
            if attributions is not None:
                visualizer.visualize_attributions(
                    inputs=sample_inputs,
                    attributions=attributions,
                    targets=sample_targets,
                    method_name=method,
                    save_path=f"assets/{method}_attributions.png",
                )
        
        # Compare methods
        visualizer.compare_methods(
            inputs=sample_inputs,
            attribution_dict=attribution_results,
            targets=sample_targets,
            save_path="assets/method_comparison.png",
        )
        
        # Plot statistics
        visualizer.plot_attribution_statistics(
            attribution_dict=attribution_results,
            save_path="assets/attribution_statistics.png",
        )
        
        # Plot evaluation metrics
        valid_metrics = {k: v for k, v in evaluation_metrics.items() if v is not None}
        if valid_metrics:
            visualizer.plot_evaluation_metrics(
                metrics_dict=valid_metrics,
                save_path="assets/evaluation_metrics.png",
            )
    
    # Compile results
    results = {
        'model_performance': test_metrics,
        'attribution_results': {k: v is not None for k, v in attribution_results.items()},
        'evaluation_metrics': evaluation_metrics,
        'config': config,
    }
    
    # Save results
    if save_results:
        results_path = Path("assets/evaluation_results.yaml")
        results_path.parent.mkdir(exist_ok=True)
        
        # Convert tensors to lists for YAML serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            yaml.dump(serializable_results, f, default_flow_style=False)
        
        logger.info(f"Results saved to {results_path}")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate Integrated Gradients")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        default=True,
        help="Save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run evaluation
    results = evaluate_integrated_gradients(
        config=config,
        save_results=args.save_results,
    )
    
    # Print summary
    logger.info("Evaluation Summary:")
    logger.info(f"Model accuracy: {results['model_performance']['accuracy']:.4f}")
    
    valid_methods = [k for k, v in results['attribution_results'].items() if v]
    logger.info(f"Successfully evaluated methods: {valid_methods}")
    
    for method in valid_methods:
        metrics = results['evaluation_metrics'][method]
        logger.info(f"{method}:")
        logger.info(f"  Faithfulness (deletion): {metrics['faithfulness_deletion']:.4f}")
        logger.info(f"  Faithfulness (insertion): {metrics['faithfulness_insertion']:.4f}")
        logger.info(f"  Spearman correlation: {metrics['spearman_correlation']:.4f}")


if __name__ == "__main__":
    main()
