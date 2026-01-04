"""
image_utility.py
Enhanced utility functions for Image Captioning with interpretability features
Supports multiple dataset images, uploads, and LVLM-interpret integration
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients, GuidedGradCam, LayerGradCam
import gradio as gr
import io

# Try to import LVLM-interpret for advanced features
try:
    from lvlm_interpret import LVLMInterpreter
    LVLM_AVAILABLE = True
    print("LVLM-interpret library loaded successfully")
except ImportError:
    LVLM_AVAILABLE = False
    print("LVLM-interpret not available, using standard Captum methods")

# -------------------------
# GLOBAL MODEL SETUP
# -------------------------
# MODIFIED: Replace single model_id with tracking variable
# OLD: model_id = "microsoft/git-large-coco"
# NEW:
_current_model_name = "microsoft/git-large-coco"

# ADDED: Model registry
IMAGE_MODELS = {
    "microsoft/git-large-coco": {
        "model_id": "microsoft/git-large-coco",
        "type": "git",
        "description": "Microsoft's Generative Image-to-text Transformer"
    },
    "Salesforce/blip-image-captioning-base": {
        "model_id": "Salesforce/blip-image-captioning-base",
        "type": "blip",
        "description": "Salesforce's BLIP base model"
    },
    "Salesforce/blip-image-captioning-large": {
        "model_id": "Salesforce/blip-image-captioning-large",
        "type": "blip",
        "description": "Salesforce's BLIP large model"
    },
    "nlpconnect/vit-gpt2-image-captioning": {
        "model_id": "nlpconnect/vit-gpt2-image-captioning",
        "type": "vit-gpt2",
        "description": "Vision Transformer + GPT-2 decoder"
    }
}


processor = None
vision_model = None
sample_images = []
test_img = None

def initialize_model(model_name=None):
    """Initialize the vision model and processor"""
    global processor, vision_model, _current_model_name
    
    if model_name is not None:
        _current_model_name = model_name
    
    if processor is None or model_name is not None:
        print(f"Loading image captioning model: {_current_model_name}...")
        model_id = IMAGE_MODELS[_current_model_name]["model_id"]
        
        try:
            processor = AutoProcessor.from_pretrained(model_id)
            vision_model = AutoModelForCausalLM.from_pretrained(model_id).eval()
            print(f"Model loaded successfully: {_current_model_name}")
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            # Fall back to default
            if _current_model_name != "microsoft/git-large-coco":
                print("Falling back to default model...")
                _current_model_name = "microsoft/git-large-coco"
                processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
                vision_model = AutoModelForCausalLM.from_pretrained("microsoft/git-large-coco").eval()

# ADDED: Model management functions
def get_image_model_choices():
    """Return list of available image captioning models"""
    return list(IMAGE_MODELS.keys())

def get_current_model_name():
    """Return the currently selected model name"""
    return _current_model_name

def switch_image_model(model_name):
    """Switch to a different image captioning model"""
    global processor, vision_model, _current_model_name
    
    if model_name not in IMAGE_MODELS:
        return f"❌ Unknown model: {model_name}", f"**Model:** {_current_model_name}"
    
    # Reset and load new model
    processor = None
    vision_model = None
    
    try:
        initialize_model(model_name)
        model_info = IMAGE_MODELS[model_name]
        status_msg = f"✅ Loaded: {model_name}\n\n*{model_info['description']}*"
        model_display = f"**Model:** {model_name}"
        return status_msg, model_display
    except Exception as e:
        return f"❌ Error loading model: {str(e)}", f"**Model:** {_current_model_name}"

def load_sample_images(num_images=5):
    """Load sample images from the dataset"""
    global sample_images
    
    if len(sample_images) == 0:
        print(f"Loading {num_images} sample images from dataset...")
        
        # Stream dataset to get sample images
        ds_stream = load_dataset(
            "seaurkin/facial_exrpressions",
            split="train",
            streaming=True
        )
        
        sample_images = []
        for i, example in enumerate(ds_stream):
            if i >= num_images:
                break
            img = example["image"]
            # Resize for consistency
            img = img.resize((224, 224))
            sample_images.append(img)
        
        print(f"Loaded {len(sample_images)} sample images")
    
    return sample_images

# -------------------------
# Initialize and load first test image
# -------------------------
initialize_model()
sample_images = load_sample_images(num_images=5)
test_img = sample_images[0] if sample_images else None

# -------------------------
# PREDICTION FUNCTIONS
# -------------------------
def predict_caption(pil_imgs):
    """Generate captions for a list of PIL images"""
    initialize_model()
    
    if not isinstance(pil_imgs, list):
        pil_imgs = [pil_imgs]
    
    inputs = processor(images=pil_imgs, return_tensors="pt")
    with torch.no_grad():
        out = vision_model.generate(**inputs, max_length=40)
    captions = processor.batch_decode(out, skip_special_tokens=True)
    return captions

def generate_caption_only(image, image_source):
    """Just generate a caption without explanation"""
    initialize_model()
    
    # Handle image source selection
    if image_source == "Upload":
        if image is None:
            return "Please upload an image"
        pil_img = Image.fromarray(image.astype(np.uint8)).resize((224, 224)) if isinstance(image, np.ndarray) else image
    else:
        # Extract index from "Sample 1", "Sample 2", etc.
        try:
            idx = int(image_source.split()[1]) - 1
            pil_img = sample_images[idx]
        except:
            pil_img = sample_images[0]
    
    caption = predict_caption([pil_img])[0]
    return caption

# -------------------------
# INTEGRATED GRADIENTS (Standard)
# -------------------------
def run_integrated_gradients(image, image_source, num_tokens=3):
    """
    Run Integrated Gradients analysis on the image
    Supports both dataset samples and uploaded images
    """
    initialize_model()
    
    # Determine which image to use
    if image_source == "Upload":
        if image is None:
            return "Please upload an image", None
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            test_img_pil = Image.fromarray(image.astype(np.uint8))
        else:
            test_img_pil = image
    else:
        # Use sample from dataset
        try:
            idx = int(image_source.split()[1]) - 1
            test_img_pil = sample_images[idx]
        except:
            test_img_pil = sample_images[0]
    
    print("Generating caption...")
    caption = predict_caption([test_img_pil])[0]
    print(f"Caption: {caption}")
    
    # Prepare for IG
    test_img_resized = test_img_pil.resize((224, 224))
    inputs = processor(images=test_img_resized, return_tensors="pt")
    
    # Generate caption to get token IDs
    with torch.no_grad():
        generated_ids = vision_model.generate(**inputs, max_length=40)
    
    pixel_values = inputs['pixel_values'].requires_grad_(True)
    baseline = torch.zeros_like(pixel_values)
    
    # Get original image for visualization
    original_img = pixel_values.squeeze().cpu().detach().numpy()
    original_img = np.transpose(original_img, (1, 2, 0))
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
    
    # Limit number of tokens to visualize
    num_tokens_to_show = min(num_tokens, len(generated_ids[0]) - 1)
    
    fig, axes = plt.subplots(2, num_tokens_to_show, figsize=(5*num_tokens_to_show, 10))
    if num_tokens_to_show == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(num_tokens_to_show):
        token_position = idx + 1
        target_token_id = generated_ids[0, token_position].item()
        token_text = processor.decode([target_token_id])
        
        # Define forward function for this token
        def forward_func(pixel_values, pos=token_position, tok_id=target_token_id):
            outputs = vision_model(
                pixel_values=pixel_values,
                input_ids=generated_ids[:, :pos],
            )
            logits = outputs.logits[:, -1, :]
            return logits[:, tok_id]
        
        # Compute attributions
        ig = IntegratedGradients(forward_func)
        attributions = ig.attribute(
            pixel_values,
            baselines=baseline,
            n_steps=50,
            internal_batch_size=1
        )
        
        attr_np = attributions.squeeze().cpu().detach().numpy()
        attr_np = np.transpose(attr_np, (1, 2, 0))
        attr_sum = np.sum(np.abs(attr_np), axis=2)
        
        # Plot original with token label
        axes[0, idx].imshow(original_img)
        axes[0, idx].set_title(f'Token {idx+1}: "{token_text}"', fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Plot attribution heatmap
        axes[1, idx].imshow(original_img)
        axes[1, idx].imshow(attr_sum, cmap='hot', alpha=0.6)
        axes[1, idx].set_title(f'Attribution Heatmap', fontsize=10)
        axes[1, idx].axis('off')
    
    plt.suptitle(f'Caption: "{caption}"', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    result_img = Image.open(buf)
    plt.close()
    
    return caption, result_img

# -------------------------
# GRADCAM ANALYSIS
# -------------------------
def run_gradcam_analysis(image, image_source):
    """
    Run GradCAM analysis for spatial attribution
    Shows which regions of the image are most important for the caption
    """
    initialize_model()
    
    # Determine which image to use
    if image_source == "Upload":
        if image is None:
            return "Please upload an image", None
        if isinstance(image, np.ndarray):
            test_img_pil = Image.fromarray(image.astype(np.uint8))
        else:
            test_img_pil = image
    else:
        try:
            idx = int(image_source.split()[1]) - 1
            test_img_pil = sample_images[idx]
        except:
            test_img_pil = sample_images[0]
    
    print("Generating caption with GradCAM...")
    caption = predict_caption([test_img_pil])[0]
    
    # Prepare inputs
    test_img_resized = test_img_pil.resize((224, 224))
    inputs = processor(images=test_img_resized, return_tensors="pt")
    
    with torch.no_grad():
        generated_ids = vision_model.generate(**inputs, max_length=40)
    
    pixel_values = inputs['pixel_values'].requires_grad_(True)
    
    # Get original image
    original_img = pixel_values.squeeze().cpu().detach().numpy()
    original_img = np.transpose(original_img, (1, 2, 0))
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
    
    # Create visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    try:
        # Try to use GuidedGradCam if available
        # For now, use a simplified approach showing overall importance
        
        # Forward pass to get activations
        def forward_func(pixel_values):
            outputs = vision_model(
                pixel_values=pixel_values,
                input_ids=generated_ids[:, :-1],
            )
            # Return the mean of output logits as proxy for importance
            return outputs.logits.mean()
        
        # Compute gradients
        ig = IntegratedGradients(forward_func)
        attributions = ig.attribute(pixel_values, n_steps=30)
        
        attr_np = attributions.squeeze().cpu().detach().numpy()
        attr_np = np.transpose(attr_np, (1, 2, 0))
        attr_sum = np.sum(np.abs(attr_np), axis=2)
        
        # Overlay heatmap on original
        ax.imshow(original_img)
        im = ax.imshow(attr_sum, cmap='jet', alpha=0.5)
        ax.set_title(f'GradCAM: Overall Importance\nCaption: "{caption}"', 
                     fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    except Exception as e:
        print(f"GradCAM error: {e}")
        ax.imshow(original_img)
        ax.set_title(f'Image\nCaption: "{caption}"', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    result_img = Image.open(buf)
    plt.close()
    
    return caption, result_img

# -------------------------
# LVLM-INTERPRET INTEGRATION (if available)
# -------------------------
def run_lvlm_interpret(image, image_source, manipulation_type="mask"):
    """
    Run LVLM-interpret analysis if available
    Supports direct manipulation features like masking and perturbation
    """
    if not LVLM_AVAILABLE:
        return "LVLM-interpret library not available. Please install: pip install lvlm-interpret", None
    
    initialize_model()
    
    # Determine which image to use
    if image_source == "Upload":
        if image is None:
            return "Please upload an image", None
        if isinstance(image, np.ndarray):
            test_img_pil = Image.fromarray(image.astype(np.uint8))
        else:
            test_img_pil = image
    else:
        try:
            idx = int(image_source.split()[1]) - 1
            test_img_pil = sample_images[idx]
        except:
            test_img_pil = sample_images[0]
    
    try:
        # Initialize LVLM interpreter
        interpreter = LVLMInterpreter(vision_model, processor)
        
        # Generate base caption
        caption = predict_caption([test_img_pil])[0]
        
        # Prepare image
        test_img_resized = test_img_pil.resize((224, 224))
        
        # Run interpretation based on manipulation type
        if manipulation_type == "mask":
            # Region masking analysis
            result = interpreter.mask_regions(test_img_resized, caption)
        elif manipulation_type == "perturbation":
            # Perturbation analysis
            result = interpreter.perturb_image(test_img_resized, caption)
        else:
            # Default: attention visualization
            result = interpreter.visualize_attention(test_img_resized, caption)
        
        return caption, result
        
    except Exception as e:
        print(f"LVLM-interpret error: {e}")
        return f"Error running LVLM-interpret: {str(e)}", None

# -------------------------
# COMPARISON ANALYSIS
# -------------------------
def compare_multiple_images(num_images=3):
    """
    Compare captions and attributions across multiple sample images
    """
    initialize_model()
    
    images_to_compare = sample_images[:min(num_images, len(sample_images))]
    
    fig, axes = plt.subplots(2, len(images_to_compare), figsize=(6*len(images_to_compare), 12))
    if len(images_to_compare) == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, img in enumerate(images_to_compare):
        # Generate caption
        caption = predict_caption([img])[0]
        
        # Prepare for attribution
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            generated_ids = vision_model.generate(**inputs, max_length=40)
        
        pixel_values = inputs['pixel_values'].requires_grad_(True)
        baseline = torch.zeros_like(pixel_values)
        
        # Forward function for overall importance
        def forward_func(pixel_values):
            outputs = vision_model(
                pixel_values=pixel_values,
                input_ids=generated_ids[:, :-1],
            )
            return outputs.logits.mean()
        
        # Compute attributions
        ig = IntegratedGradients(forward_func)
        attributions = ig.attribute(pixel_values, baselines=baseline, n_steps=30)
        
        # Process for visualization
        original_img = pixel_values.squeeze().cpu().detach().numpy()
        original_img = np.transpose(original_img, (1, 2, 0))
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        
        attr_np = attributions.squeeze().cpu().detach().numpy()
        attr_np = np.transpose(attr_np, (1, 2, 0))
        attr_sum = np.sum(np.abs(attr_np), axis=2)
        
        # Plot original
        axes[0, idx].imshow(original_img)
        axes[0, idx].set_title(f'Image {idx+1}\n"{caption}"', fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')
        
        # Plot attribution
        axes[1, idx].imshow(original_img)
        axes[1, idx].imshow(attr_sum, cmap='hot', alpha=0.6)
        axes[1, idx].set_title('Attribution Map', fontsize=10)
        axes[1, idx].axis('off')
    
    plt.suptitle('Multi-Image Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    result_img = Image.open(buf)
    plt.close()
    
    return "Comparison complete", result_img

# -------------------------
# EXPORT FUNCTIONS
# -------------------------
def get_sample_image_choices():
    """Return list of sample image options for dropdown"""
    num_samples = len(sample_images)
    return [f"Sample {i+1}" for i in range(num_samples)] + ["Upload"]

def get_sample_image_by_index(choice):
    """Get a specific sample image by choice string"""
    if choice == "Upload":
        return None
    
    try:
        idx = int(choice.split()[1]) - 1
        return sample_images[idx]
    except:
        return sample_images[0]