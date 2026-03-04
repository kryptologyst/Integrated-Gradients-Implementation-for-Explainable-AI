"""
Streamlit demo for Integrated Gradients visualization.
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor

from src.data.datasets import get_data_loaders, create_sample_batch
from src.explainers.integrated_gradients import IntegratedGradientsExplainer
from src.metrics.attribution_metrics import AttributionMetrics
from src.models.model_utils import load_pretrained_model, evaluate_model
from src.utils.device import get_device
from src.utils.seeding import set_deterministic_seed
from src.viz.visualization import AttributionVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Integrated Gradients Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
DISCLAIMER = """
⚠️ **IMPORTANT DISCLAIMER**

This demo is for research and educational purposes only. XAI outputs may be unstable or misleading and should not be used as a substitute for human judgment in regulated decisions. Always verify explanations with domain experts and consider multiple explanation methods.
"""


@st.cache_resource
def load_model_and_data(dataset_name: str, model_name: str, random_seed: int):
    """Load model and data with caching."""
    set_deterministic_seed(random_seed)
    
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders(
        dataset_name=dataset_name,
        batch_size=32,
        num_workers=0,  # Streamlit doesn't support multiprocessing
    )
    
    # Load model
    model = load_pretrained_model(
        model_name=model_name,
        num_classes=10,
        pretrained=True,
    )
    
    device = get_device()
    model.to(device)
    
    return model, test_loader, device


def preprocess_uploaded_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Tensor:
    """Preprocess uploaded image for model input."""
    # Resize image
    image = image.resize(target_size)
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    tensor = transform(image)
    return tensor.unsqueeze(0)  # Add batch dimension


def main():
    """Main Streamlit app."""
    # Header
    st.markdown('<h1 class="main-header">🧠 Integrated Gradients Demo</h1>', unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown(f'<div class="warning-box">{DISCLAIMER}</div>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_name = st.sidebar.selectbox(
        "Dataset",
        ["cifar10", "synthetic"],
        index=0,
        help="Choose the dataset to use for demonstration"
    )
    
    # Model selection
    model_name = st.sidebar.selectbox(
        "Model",
        ["resnet18", "resnet50", "vgg16", "simple_cnn"],
        index=0,
        help="Choose the pre-trained model"
    )
    
    # Explainer parameters
    st.sidebar.subheader("Explainer Parameters")
    baseline_method = st.sidebar.selectbox(
        "Baseline Method",
        ["zeros", "mean", "random"],
        index=0,
        help="Method for generating baseline inputs"
    )
    
    n_steps = st.sidebar.slider(
        "Integration Steps",
        min_value=10,
        max_value=100,
        value=50,
        help="Number of steps for Integrated Gradients computation"
    )
    
    # Load model and data
    with st.spinner("Loading model and data..."):
        model, test_loader, device = load_model_and_data(dataset_name, model_name, 42)
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Local Explanations", "📊 Method Comparison", "📈 Evaluation Metrics", "ℹ️ About"])
    
    with tab1:
        st.header("Local Explanations")
        
        # Input selection
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Selection")
            input_method = st.radio(
                "Choose input method:",
                ["Sample from dataset", "Upload image"],
                index=0
            )
            
            if input_method == "Sample from dataset":
                sample_idx = st.slider("Sample index", 0, 99, 0)
                
                # Get sample
                sample_inputs, sample_targets = create_sample_batch(
                    dataset_name=dataset_name,
                    batch_size=100,
                )
                
                input_image = sample_inputs[sample_idx:sample_idx+1]
                target = sample_targets[sample_idx].item()
                
                # Display original image
                st.image(
                    input_image.squeeze().permute(1, 2, 0).numpy(),
                    caption=f"Sample {sample_idx}",
                    use_column_width=True
                )
                
            else:  # Upload image
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['png', 'jpg', 'jpeg'],
                    help="Upload an image to analyze"
                )
                
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    input_image = preprocess_uploaded_image(image)
                    target = None  # Will use model prediction
                    
                    st.image(image, caption="Uploaded image", use_column_width=True)
                else:
                    st.info("Please upload an image to get started")
                    return
        
        with col2:
            st.subheader("Attribution Method")
            method = st.selectbox(
                "Select attribution method:",
                ["integrated_gradients", "gradient_shap", "saliency", "guided_backprop", "smooth_grad"],
                index=0
            )
            
            # Compute attribution
            if st.button("Compute Attribution", type="primary"):
                with st.spinner("Computing attribution..."):
                    # Initialize explainer
                    explainer = IntegratedGradientsExplainer(
                        model=model,
                        device=device,
                        baseline_method=baseline_method,
                        n_steps=n_steps,
                    )
                    
                    # Get model prediction
                    with torch.no_grad():
                        outputs = model(input_image.to(device))
                        prediction = torch.argmax(outputs, dim=1).item()
                        confidence = torch.softmax(outputs, dim=1)[0, prediction].item()
                    
                    # Compute attribution
                    attribution = explainer.explain(
                        input_image.to(device),
                        target=target,
                        method=method,
                    )
                    
                    # Display results
                    st.success(f"Prediction: Class {prediction} (confidence: {confidence:.3f})")
                    
                    # Visualization
                    visualizer = AttributionVisualizer()
                    
                    # Create attribution heatmap
                    attr_np = attribution.squeeze().cpu().numpy()
                    if attr_np.ndim == 3:
                        attr_np = np.sum(np.abs(attr_np), axis=0)
                    
                    # Normalize
                    attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
                    
                    # Display attribution
                    st.image(
                        attr_np,
                        caption=f"{method.replace('_', ' ').title()} Attribution",
                        use_column_width=True
                    )
                    
                    # Compute metrics
                    if target is not None:
                        metrics = AttributionMetrics.compute_comprehensive_metrics(
                            model=model,
                            inputs=input_image,
                            attributions=attribution,
                            targets=torch.tensor([target]),
                            method_name=method,
                        )
                        
                        st.subheader("Evaluation Metrics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Faithfulness (Deletion)",
                                f"{metrics['faithfulness_deletion']:.3f}",
                                help="Performance drop when removing important features"
                            )
                        
                        with col2:
                            st.metric(
                                "Faithfulness (Insertion)",
                                f"{metrics['faithfulness_insertion']:.3f}",
                                help="Performance gain when adding important features"
                            )
                        
                        with col3:
                            st.metric(
                                "Spearman Correlation",
                                f"{metrics['spearman_correlation']:.3f}",
                                help="Correlation with randomized model (sanity check)"
                            )
    
    with tab2:
        st.header("Method Comparison")
        
        if st.button("Compare All Methods", type="primary"):
            with st.spinner("Computing attributions for all methods..."):
                # Get sample
                sample_inputs, sample_targets = create_sample_batch(
                    dataset_name=dataset_name,
                    batch_size=4,
                )
                
                # Initialize explainer
                explainer = IntegratedGradientsExplainer(
                    model=model,
                    device=device,
                    baseline_method=baseline_method,
                    n_steps=n_steps,
                )
                
                # Compute attributions for all methods
                methods = ["integrated_gradients", "gradient_shap", "saliency", "guided_backprop", "smooth_grad"]
                attribution_results = {}
                
                for method in methods:
                    try:
                        attributions = explainer.explain(
                            sample_inputs.to(device),
                            target=sample_targets.to(device),
                            method=method,
                        )
                        attribution_results[method] = attributions
                    except Exception as e:
                        st.warning(f"Failed to compute {method}: {e}")
                        attribution_results[method] = None
                
                # Display comparison
                st.subheader("Attribution Comparison")
                
                for i in range(sample_inputs.size(0)):
                    st.write(f"**Sample {i+1}**")
                    
                    # Original image
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    with col1:
                        st.image(
                            sample_inputs[i].permute(1, 2, 0).numpy(),
                            caption="Original",
                            use_column_width=True
                        )
                    
                    # Attribution maps
                    for j, (method, attributions) in enumerate(attribution_results.items()):
                        if attributions is not None:
                            attr_np = attributions[i].squeeze().cpu().numpy()
                            if attr_np.ndim == 3:
                                attr_np = np.sum(np.abs(attr_np), axis=0)
                            
                            attr_np = (attr_np - attr_np.min()) / (attr_np.max() - attr_np.min() + 1e-8)
                            
                            with st.columns(6)[j+1]:
                                st.image(
                                    attr_np,
                                    caption=method.replace('_', ' ').title(),
                                    use_column_width=True
                                )
                    
                    st.divider()
    
    with tab3:
        st.header("Evaluation Metrics")
        
        if st.button("Run Comprehensive Evaluation", type="primary"):
            with st.spinner("Running evaluation..."):
                # Get sample batch
                sample_inputs, sample_targets = create_sample_batch(
                    dataset_name=dataset_name,
                    batch_size=8,
                )
                
                # Initialize explainer
                explainer = IntegratedGradientsExplainer(
                    model=model,
                    device=device,
                    baseline_method=baseline_method,
                    n_steps=n_steps,
                )
                
                # Evaluate all methods
                methods = ["integrated_gradients", "gradient_shap", "saliency", "guided_backprop", "smooth_grad"]
                all_metrics = {}
                
                for method in methods:
                    try:
                        attributions = explainer.explain(
                            sample_inputs.to(device),
                            target=sample_targets.to(device),
                            method=method,
                        )
                        
                        metrics = AttributionMetrics.compute_comprehensive_metrics(
                            model=model,
                            inputs=sample_inputs,
                            attributions=attributions,
                            targets=sample_targets,
                            method_name=method,
                        )
                        
                        all_metrics[method] = metrics
                        
                    except Exception as e:
                        st.warning(f"Failed to evaluate {method}: {e}")
                
                # Display metrics
                if all_metrics:
                    st.subheader("Method Performance Comparison")
                    
                    # Create metrics dataframe
                    import pandas as pd
                    
                    metrics_df = pd.DataFrame(all_metrics).T
                    metrics_df = metrics_df.drop(columns=['method'])  # Remove redundant column
                    
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Create bar charts
                    st.subheader("Metrics Visualization")
                    
                    for metric in ['faithfulness_deletion', 'faithfulness_insertion', 'spearman_correlation']:
                        if metric in metrics_df.columns:
                            st.bar_chart(metrics_df[metric])
    
    with tab4:
        st.header("About Integrated Gradients")
        
        st.markdown("""
        ## What is Integrated Gradients?
        
        Integrated Gradients is a method for explaining the predictions of machine learning models by attributing the prediction to individual features. It works by calculating the integral of the gradients of the model's output with respect to the input features along a straight line from a baseline (e.g., a black image) to the actual input.
        
        ## Key Features
        
        - **Axiomatic Properties**: Satisfies sensitivity and implementation invariance
        - **Baseline Flexibility**: Can use different baseline inputs (zeros, mean, random)
        - **Gradient Integration**: Averages gradients along the path from baseline to input
        - **Feature Attribution**: Assigns importance scores to individual features
        
        ## Method Comparison
        
        This demo compares several attribution methods:
        
        - **Integrated Gradients**: Integrates gradients along the path from baseline to input
        - **Gradient SHAP**: Uses gradient-based SHAP values
        - **Saliency**: Uses raw gradients as attributions
        - **Guided Backprop**: Uses guided gradients for better visualization
        - **SmoothGrad**: Averages gradients over noisy inputs
        
        ## Evaluation Metrics
        
        - **Faithfulness (Deletion)**: Performance drop when removing important features
        - **Faithfulness (Insertion)**: Performance gain when adding important features
        - **Spearman Correlation**: Correlation with randomized model (sanity check)
        - **Kendall Tau**: Rank correlation between attribution methods
        - **IoU Score**: Intersection over Union for top-k attribution regions
        
        ## Limitations
        
        - Explanations may be unstable across different baselines
        - Computational cost increases with number of integration steps
        - May not capture complex feature interactions
        - Results depend on choice of baseline and integration path
        """)
        
        st.subheader("Technical Details")
        
        st.code("""
# Example usage
explainer = IntegratedGradientsExplainer(
    model=model,
    baseline_method="zeros",
    n_steps=50
)

attributions = explainer.explain(
    inputs=input_tensor,
    target=target_class,
    method="integrated_gradients"
)
        """, language="python")


if __name__ == "__main__":
    main()
