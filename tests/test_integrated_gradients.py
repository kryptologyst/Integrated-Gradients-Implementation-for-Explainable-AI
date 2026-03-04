"""
Tests for Integrated Gradients implementation.
"""

import pytest
import torch
import torch.nn as nn

from src.explainers.integrated_gradients import IntegratedGradientsExplainer
from src.models.model_utils import SimpleCNN
from src.utils.device import get_device
from src.utils.seeding import set_deterministic_seed


class TestIntegratedGradients:
    """Test cases for Integrated Gradients explainer."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        model = SimpleCNN(num_classes=3, input_size=(32, 32))
        return model
    
    @pytest.fixture
    def sample_inputs(self):
        """Create sample inputs for testing."""
        return torch.randn(2, 3, 32, 32)
    
    @pytest.fixture
    def sample_targets(self):
        """Create sample targets for testing."""
        return torch.tensor([0, 1])
    
    def test_explainer_initialization(self, simple_model):
        """Test explainer initialization."""
        explainer = IntegratedGradientsExplainer(
            model=simple_model,
            baseline_method="zeros",
            n_steps=10,
        )
        
        assert explainer.model == simple_model
        assert explainer.baseline_method == "zeros"
        assert explainer.n_steps == 10
        assert explainer.device is not None
    
    def test_baseline_generation(self, simple_model, sample_inputs):
        """Test baseline generation methods."""
        explainer = IntegratedGradientsExplainer(simple_model)
        
        # Test zeros baseline
        baseline = explainer._get_baseline(sample_inputs, method="zeros")
        assert torch.allclose(baseline, torch.zeros_like(sample_inputs))
        
        # Test mean baseline
        baseline = explainer._get_baseline(sample_inputs, method="mean")
        expected_mean = torch.mean(sample_inputs, dim=0, keepdim=True).expand_as(sample_inputs)
        assert torch.allclose(baseline, expected_mean)
        
        # Test random baseline
        baseline = explainer._get_baseline(sample_inputs, method="random")
        assert baseline.shape == sample_inputs.shape
    
    def test_integrated_gradients_attribution(self, simple_model, sample_inputs, sample_targets):
        """Test Integrated Gradients attribution computation."""
        explainer = IntegratedGradientsExplainer(
            simple_model,
            n_steps=10,
        )
        
        attributions = explainer.explain(
            sample_inputs,
            target=sample_targets,
            method="integrated_gradients",
        )
        
        assert attributions.shape == sample_inputs.shape
        assert not torch.isnan(attributions).any()
        assert not torch.isinf(attributions).any()
    
    def test_multiple_methods(self, simple_model, sample_inputs, sample_targets):
        """Test multiple attribution methods."""
        explainer = IntegratedGradientsExplainer(simple_model, n_steps=10)
        
        methods = ["integrated_gradients", "saliency"]
        
        for method in methods:
            attributions = explainer.explain(
                sample_inputs,
                target=sample_targets,
                method=method,
            )
            
            assert attributions.shape == sample_inputs.shape
            assert not torch.isnan(attributions).any()
    
    def test_method_comparison(self, simple_model, sample_inputs, sample_targets):
        """Test method comparison functionality."""
        explainer = IntegratedGradientsExplainer(simple_model, n_steps=10)
        
        comparison = explainer.compare_methods(
            sample_inputs,
            target=sample_targets,
            methods=["integrated_gradients", "saliency"],
        )
        
        assert len(comparison) == 2
        assert "integrated_gradients" in comparison
        assert "saliency" in comparison
        
        for method, attributions in comparison.items():
            if attributions is not None:
                assert attributions.shape == sample_inputs.shape
    
    def test_feature_importance(self, simple_model, sample_inputs, sample_targets):
        """Test feature importance computation."""
        explainer = IntegratedGradientsExplainer(simple_model, n_steps=10)
        
        attributions = explainer.explain(
            sample_inputs,
            target=sample_targets,
            method="integrated_gradients",
        )
        
        importance = explainer.get_feature_importance(attributions, aggregation="mean")
        
        assert importance.shape == sample_inputs.shape[1:]  # Remove batch dimension
        assert torch.all(importance >= 0)  # Importance should be non-negative
    
    def test_sensitivity_analysis(self, simple_model, sample_inputs, sample_targets):
        """Test sensitivity analysis."""
        explainer = IntegratedGradientsExplainer(simple_model, n_steps=10)
        
        sensitivity = explainer.compute_sensitivity_analysis(
            sample_inputs,
            target=sample_targets,
            noise_levels=[0.01, 0.05],
        )
        
        assert "noise_levels" in sensitivity
        assert "similarities" in sensitivity
        assert len(sensitivity["similarities"]) == 2
        
        # Similarities should decrease with higher noise
        similarities = sensitivity["similarities"]
        assert similarities[0] >= similarities[1]  # Lower noise should have higher similarity
    
    def test_deterministic_seeding(self, simple_model, sample_inputs, sample_targets):
        """Test deterministic behavior with seeding."""
        set_deterministic_seed(42)
        
        explainer1 = IntegratedGradientsExplainer(simple_model, n_steps=10, random_seed=42)
        attributions1 = explainer1.explain(
            sample_inputs,
            target=sample_targets,
            method="integrated_gradients",
        )
        
        set_deterministic_seed(42)
        
        explainer2 = IntegratedGradientsExplainer(simple_model, n_steps=10, random_seed=42)
        attributions2 = explainer2.explain(
            sample_inputs,
            target=sample_targets,
            method="integrated_gradients",
        )
        
        # Results should be identical with same seed
        assert torch.allclose(attributions1, attributions2, atol=1e-6)
    
    def test_device_handling(self, simple_model, sample_inputs, sample_targets):
        """Test device handling."""
        device = get_device()
        
        explainer = IntegratedGradientsExplainer(simple_model, device=device)
        
        # Move inputs to device
        sample_inputs = sample_inputs.to(device)
        sample_targets = sample_targets.to(device)
        
        attributions = explainer.explain(
            sample_inputs,
            target=sample_targets,
            method="integrated_gradients",
        )
        
        assert attributions.device == device
        assert attributions.shape == sample_inputs.shape


if __name__ == "__main__":
    pytest.main([__file__])
