# Integrated Gradients Implementation for Explainable AI

Comprehensive implementation of Integrated Gradients and related attribution methods for deep learning models, with extensive evaluation metrics and interactive visualization tools.

## ⚠️ IMPORTANT DISCLAIMER

**This project is for research and educational purposes only.** XAI outputs may be unstable or misleading and should not be used as a substitute for human judgment in regulated decisions. Always verify explanations with domain experts and consider multiple explanation methods.

## Features

- **Comprehensive Attribution Methods**: Integrated Gradients, Gradient SHAP, Saliency, Guided Backprop, SmoothGrad
- **Multiple Baselines**: Zero, mean, and random baseline support
- **Extensive Evaluation**: Faithfulness metrics, stability analysis, sanity checks
- **Interactive Demo**: Streamlit-based visualization tool
- **Modern Stack**: PyTorch 2.x, Captum, comprehensive type hints
- **Device Support**: CUDA → MPS (Apple Silicon) → CPU fallback
- **Reproducible**: Deterministic seeding and configuration management

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Integrated-Gradients-Implementation-for-Explainable-AI.git
cd Integrated-Gradients-Implementation-for-Explainable-AI

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Basic Usage

```python
import torch
from src.explainers.integrated_gradients import IntegratedGradientsExplainer
from src.models.model_utils import load_pretrained_model
from src.data.datasets import create_sample_batch

# Load model and data
model = load_pretrained_model("resnet18", num_classes=10)
inputs, targets = create_sample_batch("cifar10", batch_size=4)

# Initialize explainer
explainer = IntegratedGradientsExplainer(
    model=model,
    baseline_method="zeros",
    n_steps=50
)

# Compute attributions
attributions = explainer.explain(
    inputs=inputs,
    target=targets,
    method="integrated_gradients"
)

# Visualize results
from src.viz.visualization import AttributionVisualizer
visualizer = AttributionVisualizer()
visualizer.visualize_attributions(inputs, attributions, targets)
```

### Run Evaluation

```bash
# Run comprehensive evaluation
python scripts/evaluate.py --config configs/default.yaml

# Run with custom configuration
python scripts/evaluate.py --config configs/custom.yaml --save-results
```

### Launch Interactive Demo

```bash
# Start Streamlit demo
streamlit run demo/streamlit_app.py
```

## Project Structure

```
├── src/                          # Source code
│   ├── explainers/              # Attribution methods
│   │   └── integrated_gradients.py
│   ├── metrics/                 # Evaluation metrics
│   │   └── attribution_metrics.py
│   ├── models/                  # Model utilities
│   │   └── model_utils.py
│   ├── data/                    # Data loading
│   │   └── datasets.py
│   ├── viz/                     # Visualization
│   │   └── visualization.py
│   └── utils/                   # Utilities
│       ├── device.py
│       └── seeding.py
├── configs/                     # Configuration files
│   └── default.yaml
├── scripts/                     # Evaluation scripts
│   └── evaluate.py
├── demo/                        # Interactive demo
│   └── streamlit_app.py
├── tests/                       # Unit tests
├── assets/                      # Generated outputs
├── data/                        # Dataset storage
└── notebooks/                   # Jupyter notebooks
```

## Attribution Methods

### Integrated Gradients
- **Description**: Integrates gradients along the path from baseline to input
- **Axioms**: Satisfies sensitivity and implementation invariance
- **Parameters**: `n_steps` (integration steps), `baseline_method`

### Gradient SHAP
- **Description**: Gradient-based SHAP values using multiple baselines
- **Advantages**: More stable than standard SHAP for deep models
- **Parameters**: `n_samples` (number of baseline samples)

### Saliency
- **Description**: Raw gradients as feature importance
- **Use Case**: Quick baseline for gradient-based methods
- **Limitations**: May be noisy and less stable

### Guided Backprop
- **Description**: Guided gradients for better visualization
- **Advantages**: Cleaner visualizations than raw gradients
- **Use Case**: Image classification tasks

### SmoothGrad
- **Description**: Averages gradients over noisy inputs
- **Advantages**: Reduces noise in gradient-based attributions
- **Parameters**: `n_samples`, `stdevs` (noise standard deviation)

## Evaluation Metrics

### Faithfulness Metrics
- **Deletion Test**: Performance drop when removing important features
- **Insertion Test**: Performance gain when adding important features
- **Higher is better**: Indicates more faithful explanations

### Stability Metrics
- **Spearman Correlation**: Rank correlation between attribution methods
- **Kendall Tau**: Alternative rank correlation measure
- **IoU Score**: Intersection over Union for top-k attribution regions

### Sanity Checks
- **Randomization Test**: Compare attributions with randomized model weights
- **Baseline Sensitivity**: Test sensitivity to different baseline choices
- **Convergence Analysis**: Check convergence of integrated gradients

## Configuration

The project uses YAML configuration files for easy customization:

```yaml
# configs/default.yaml
random_seed: 42

dataset:
  name: "cifar10"
  batch_size: 32
  num_classes: 10

model:
  name: "resnet18"
  pretrained: true

explainer:
  baseline_method: "zeros"
  n_steps: 50

evaluation:
  methods: ["integrated_gradients", "gradient_shap", "saliency"]
  sample_size: 8
```

## Datasets

### CIFAR-10
- **Description**: 10-class image classification dataset
- **Size**: 50,000 training, 10,000 test images
- **Classes**: Airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Preprocessing**: Normalized with ImageNet statistics

### Synthetic Dataset
- **Description**: Generated synthetic images with class-specific patterns
- **Use Case**: Testing and demonstration
- **Patterns**: Horizontal lines, vertical lines, diagonals, circles, squares
- **Configurable**: Number of samples, classes, image size

## Models

### Pre-trained Models
- **ResNet18/50**: Residual networks with ImageNet pre-training
- **VGG16**: Deep convolutional network
- **AlexNet**: Classic CNN architecture

### Custom Models
- **SimpleCNN**: Lightweight CNN for demonstration
- **Configurable**: Number of classes, input size, architecture

## Interactive Demo

The Streamlit demo provides:

### Local Explanations Tab
- Upload custom images or select from dataset
- Choose attribution method and parameters
- View attribution heatmaps and evaluation metrics
- Real-time parameter adjustment

### Method Comparison Tab
- Side-by-side comparison of all methods
- Visual comparison of attribution maps
- Batch processing for multiple samples

### Evaluation Metrics Tab
- Comprehensive evaluation of all methods
- Performance comparison charts
- Statistical analysis of attribution quality

### About Tab
- Detailed explanation of methods
- Technical documentation
- Usage examples and best practices

## Advanced Usage

### Custom Baselines

```python
# Use custom baseline
custom_baseline = torch.randn_like(inputs) * 0.1
attributions = explainer.explain(
    inputs=inputs,
    baseline=custom_baseline,
    method="integrated_gradients"
)
```

### Batch Processing

```python
# Process entire dataset
results = explainer.explain_batch(
    dataloader=test_loader,
    method="integrated_gradients",
    max_samples=100
)
```

### Method Comparison

```python
# Compare multiple methods
comparison = explainer.compare_methods(
    inputs=inputs,
    target=targets,
    methods=["integrated_gradients", "gradient_shap", "saliency"]
)
```

### Sensitivity Analysis

```python
# Analyze sensitivity to noise
sensitivity = explainer.compute_sensitivity_analysis(
    inputs=inputs,
    target=targets,
    noise_levels=[0.01, 0.05, 0.1, 0.2]
)
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_integrated_gradients.py
```

### Code Formatting

```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff check src/ scripts/ demo/

# Run pre-commit hooks
pre-commit run --all-files
```

### Adding New Methods

1. Implement attribution method in `src/explainers/`
2. Add evaluation metrics in `src/metrics/`
3. Update configuration files
4. Add tests in `tests/`
5. Update documentation

## Limitations and Considerations

### Method Limitations
- **Baseline Sensitivity**: Results depend on baseline choice
- **Computational Cost**: Higher `n_steps` increases computation time
- **Feature Interactions**: May not capture complex interactions
- **Model Dependencies**: Performance varies across model architectures

### Evaluation Limitations
- **Ground Truth**: No ground truth for feature importance
- **Metric Interpretation**: Metrics may not reflect human understanding
- **Dataset Bias**: Results depend on training data characteristics

### Best Practices
- **Multiple Methods**: Always compare multiple attribution methods
- **Baseline Comparison**: Test different baseline choices
- **Domain Expertise**: Validate explanations with domain experts
- **Error Analysis**: Analyze failure cases and edge cases

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@software{integrated_gradients_xai,
  title={Integrated Gradients Implementation for Explainable AI},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Integrated-Gradients-Implementation-for-Explainable-AI}
}
```

## References

- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. ICML.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. NeurIPS.
- Simonyan, K., Vedaldi, A., & Zisserman, A. (2013). Deep inside convolutional networks: Visualising image classification models and saliency maps. ICLR Workshop.
- Springenberg, J. T., Dosovitskiy, A., Brox, T., & Riedmiller, M. (2014). Striving for simplicity: The all convolutional net. ICLR Workshop.
- Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). Smoothgrad: removing noise by adding noise. ICML Workshop.
# Integrated-Gradients-Implementation-for-Explainable-AI
