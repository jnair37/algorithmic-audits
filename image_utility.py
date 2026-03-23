"""
image_utility.py
Enhanced utility functions for Image Captioning with interpretability features
Supports multiple dataset images, uploads, and LVLM-interpret integration
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BlipProcessor, BlipForConditionalGeneration
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients, GuidedGradCam, LayerGradCam
import gradio as gr
import io
import traceback
import tempfile
import csv

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
            vision_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            ).eval()
            print(f"Model loaded successfully: {_current_model_name}")
        except Exception as e:
            print(f"Error loading model {model_id}: {e}")
            # Fall back to default
            if _current_model_name != "microsoft/git-large-coco":
                print("Falling back to default model...")
                _current_model_name = "microsoft/git-large-coco"
                processor = AutoProcessor.from_pretrained("microsoft/git-large-coco")
                vision_model = AutoModelForCausalLM.from_pretrained(
                    "microsoft/git-large-coco",
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                ).eval()

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


# Accepted pipeline / model types for vision-language auditing
_VL_PIPELINE_TAGS = {"image-to-text", "visual-question-answering", "image-captioning"}
_VL_MODEL_TYPES = {"blip", "git", "vit-gpt2", "vit", "clip", "idefics", "llava", "paligemma"}


def validate_and_load_image_model(hf_model_id: str):
    """
    Validate and load a HuggingFace vision-language model by URL or ID.

    Returns:
        (status_markdown, model_display_markdown)
    """
    global processor, vision_model, _current_model_name

    if not hf_model_id or not hf_model_id.strip():
        return "⚠️ Please enter a HuggingFace model ID.", f"**Model:** {_current_model_name}"

    model_id = hf_model_id.strip()
    if model_id.startswith("https://huggingface.co/"):
        model_id = model_id[len("https://huggingface.co/"):]
    model_id = model_id.rstrip("/")

    try:
        from huggingface_hub import model_info as hf_model_info
        from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

        try:
            info = hf_model_info(model_id)
        except RepositoryNotFoundError:
            return (
                f"❌ **Repository not found:** `{model_id}`\n\nPlease check the model ID or URL.",
                f"**Model:** {_current_model_name}",
            )
        except GatedRepoError:
            return (
                f"❌ **Gated repository:** `{model_id}`\n\nAccept the model license on HuggingFace first.",
                f"**Model:** {_current_model_name}",
            )

        pipeline_tag = (info.pipeline_tag or "").lower()
        model_type = ""
        if info.config and isinstance(info.config, dict):
            model_type = info.config.get("model_type", "").lower()

        accepted = pipeline_tag in _VL_PIPELINE_TAGS or model_type in _VL_MODEL_TYPES

        if not accepted and pipeline_tag and pipeline_tag not in ("", "null"):
            return (
                f"❌ **Unsupported model type:** `{model_id}`\n\n"
                f"Detected pipeline: `{pipeline_tag}` / architecture: `{model_type or 'unknown'}`\n\n"
                "Only image-to-text / visual-question-answering models are supported.",
                f"**Model:** {_current_model_name}",
            )

        # Attempt load via AutoProcessor + AutoModelForCausalLM
        processor = None
        vision_model = None
        _current_model_name = model_id

        print(f"Loading VL model: {model_id} …")
        processor = AutoProcessor.from_pretrained(model_id)
        vision_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).eval()

        status_msg = f"✅ **Loaded:** `{model_id}`"
        model_display = f"**Model:** {model_id}"
        return status_msg, model_display

    except Exception as e:
        return (
            f"❌ **Error loading `{model_id}`:** {str(e)}",
            f"**Model:** {_current_model_name}",
        )

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
# MODIFIED: Defer dataset loading until the image captioning tab is actually used
# The tab is currently disabled, and the dataset URL is broken
# try:
#     sample_images = load_sample_images(num_images=5)
#     test_img = sample_images[0] if sample_images else None
# except Exception as e:
#     test_img = None
test_img = None  # Will be loaded lazily when needed

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
        # Lazy-load dataset if needed
        if len(sample_images) == 0:
            load_sample_images(num_images=5)
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
        # Lazy-load dataset if needed
        if len(sample_images) == 0:
            load_sample_images(num_images=5)
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
        # Lazy-load dataset if needed
        if len(sample_images) == 0:
            load_sample_images(num_images=5)
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
        # Lazy-load dataset if needed
        if len(sample_images) == 0:
            load_sample_images(num_images=5)
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
    
    # Lazy-load dataset if needed
    if len(sample_images) == 0:
        load_sample_images(num_images=5)
    
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
# BLIP VL MODEL FUNCTIONS (LVLM-Interpret integration)
# -------------------------
# These are self-contained and use a separate global model from the GIT/ViT-GPT2 above.

_blip_processor = None
_blip_model = None
_blip_device = "cuda" if torch.cuda.is_available() else "cpu"

def load_blip_model():
    """Lazy-load the BLIP model for token-level IG analysis."""
    global _blip_processor, _blip_model
    if _blip_model is None:
        print("DEBUG: [load_blip_model] Starting to load BLIP model...")
        print("DEBUG: [load_blip_model] Loading processor...")
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        print("DEBUG: [load_blip_model] Processor loaded. Now loading model weights...")
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("DEBUG: [load_blip_model] Model weights loaded. Setting to eval mode...")
        _blip_model.eval()
        print(f"DEBUG: [load_blip_model] BLIP model loaded on {_blip_device} successfully.")
    return _blip_processor, _blip_model


def blip_compute_integrated_gradients(image_pil, caption, target_token_idx, n_steps=50):
    """
    Compute Integrated Gradients for a specific token in the BLIP-generated caption.
    Returns a normalised (H x W) attribution map as a numpy array.
    """
    try:
        proc, mdl = load_blip_model()

        inputs = proc(image_pil, return_tensors="pt").to(_blip_device)
        pixel_values = inputs.pixel_values

        text_inputs = proc(
            text=caption,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(_blip_device)
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        def forward_func(pv):
            outputs = mdl(
                pixel_values=pv,
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.logits
            return logits[:, target_token_idx, input_ids[:, target_token_idx]]

        ig = IntegratedGradients(forward_func)
        baseline = torch.zeros_like(pixel_values)
        attributions = ig.attribute(
            pixel_values,
            baselines=baseline,
            n_steps=n_steps,
            internal_batch_size=32,
        )

        attr = attributions.squeeze(0).cpu().detach().numpy()
        attr = np.abs(attr).sum(axis=0)  # collapse channels
        attr_min, attr_max = attr.min(), attr.max()
        if attr_max > attr_min:
            attr = (attr - attr_min) / (attr_max - attr_min)
        return attr

    except Exception:
        traceback.print_exc()
        raise


def blip_create_attribution_overlay(image_pil, attribution_map, opacity=0.6):
    """
    Blend a smooth jet-colourmap heatmap over the original image.
    Applies Gaussian smoothing + cubic interpolation to remove the
    blocky patch-grid artefact that comes from ViT patch-level gradients.
    """
    try:
        from scipy.ndimage import zoom, gaussian_filter

        # Unwrap Gradio ImageEditor dict if needed
        if isinstance(image_pil, dict):
            raw = image_pil.get("composite", image_pil.get("image", None))
        else:
            raw = image_pil

        img_array = np.array(raw)
        if img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        h, w = img_array.shape[:2]

        # ── 1. Smooth the raw attribution map at its native resolution ─────
        # Gaussian sigma of ~1 at patch level (~14 patches) blends adjacent squares
        attr_smooth = gaussian_filter(attribution_map, sigma=1.0)

        # ── 2. Upsample with cubic interpolation (order=3) ────────────────
        zoom_factors = (h / attr_smooth.shape[0], w / attr_smooth.shape[1])
        attr_resized = zoom(attr_smooth, zoom_factors, order=3)

        # ── 3. Light second-pass smoothing in pixel space ──────────────────
        attr_resized = gaussian_filter(attr_resized, sigma=max(h, w) * 0.01)

        # ── 4. Percentile-clip normalization (removes outlier spikes) ──────
        lo, hi = np.percentile(attr_resized, 2), np.percentile(attr_resized, 98)
        if hi > lo:
            attr_norm = np.clip((attr_resized - lo) / (hi - lo), 0, 1)
        else:
            attr_norm = attr_resized

        # ── 5. Colourise and blend ─────────────────────────────────────────
        cmap = plt.get_cmap("jet")
        colored = (cmap(attr_norm)[:, :, :3] * 255).astype(np.uint8)

        alpha = attr_norm[:, :, np.newaxis] * opacity
        blended = (img_array * (1 - alpha) + colored * alpha).astype(np.uint8)
        return Image.fromarray(blended)

    except Exception:
        traceback.print_exc()
        raise


def blip_generate_caption_only(image_pil):
    """
    Generate a caption + token list for an image using BLIP.
    Accepts a plain PIL Image or the Gradio ImageEditor dict.

    Returns:
        caption (str), tokens_str (str), None,
        original_pil (PIL.Image), caption (str), tokens (list)
    """
    if image_pil is None:
        return "Please upload an image.", "", None, None, "", []

    # Unwrap Gradio ImageEditor dict
    if isinstance(image_pil, dict):
        image_pil = image_pil.get("composite", image_pil.get("image", None))
    if image_pil is None:
        return "No image found in editor.", "", None, None, "", []

    try:
        print("DEBUG: [blip_generate_caption_only] Calling load_blip_model...")
        proc, mdl = load_blip_model()
        print("DEBUG: [blip_generate_caption_only] Successfully grabbed model instance.")

        print("DEBUG: [blip_generate_caption_only] Prepping inputs...")
        inputs = proc(image_pil, return_tensors="pt").to(_blip_device)
        print("DEBUG: [blip_generate_caption_only] Generating forward pass...")
        with torch.no_grad():
            out = mdl.generate(**inputs, max_length=50)
        print("DEBUG: [blip_generate_caption_only] Generation complete. Decoding...")
        caption = proc.decode(out[0], skip_special_tokens=True)
        print(f"DEBUG: [blip_generate_caption_only] Caption: {caption}")

        # Tokenise so the user can pick a token index
        print("DEBUG: [blip_generate_caption_only] Tokenising caption...")
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false" # Prevent deadlock in fast tokenizers
        text_inputs = proc(
            text=caption,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(_blip_device)
        print("DEBUG: [blip_generate_caption_only] Tokenisation completed. Extracting individual tokens...")
        tokens = []
        for tid in text_inputs["input_ids"][0]:
            tok = proc.decode([tid], skip_special_tokens=True)
            if tok.strip():
                tokens.append(tok)

        print("DEBUG: [blip_generate_caption_only] Tokens extracted.")
        tokens_str = "\n".join([f"{i}: {t}" for i, t in enumerate(tokens)])
        return caption, tokens_str, None, image_pil, caption, tokens

    except Exception as exc:
        print(f"DEBUG: [blip_generate_caption_only] Exception caught: {exc}")
        traceback.print_exc()
        return f"Error: {exc}", "", None, None, "", []


def blip_analyze_image(image_pil, opacity, n_steps, target_token_idx, caption, tokens):
    """
    Compute token-level Integrated Gradients for the selected token and
    return an attribution heatmap overlaid on the (possibly edited) image.

    Returns:
        caption (str), tokens_str (str),
        attribution_overlay (PIL.Image), original_pil (PIL.Image),
        caption (str), tokens (list)
    """
    if image_pil is None:
        return "Please upload an image.", "", None, None, "", []
    if not caption or not tokens:
        return "Please generate a caption first.", "", None, image_pil, "", []

    # Unwrap dict for display; keep original dict for IG (uses composite)
    raw_pil = image_pil
    if isinstance(image_pil, dict):
        raw_pil = image_pil.get("composite", image_pil.get("image", None))

    tokens_str = "\n".join([f"{i}: {t}" for i, t in enumerate(tokens)])

    if target_token_idx is None or not (0 <= int(target_token_idx) < len(tokens)):
        return caption, tokens_str, None, raw_pil, caption, tokens

    try:
        attribution_map = blip_compute_integrated_gradients(
            raw_pil, caption, int(target_token_idx), int(n_steps)
        )
        overlay = blip_create_attribution_overlay(raw_pil, attribution_map, opacity)
        return caption, tokens_str, overlay, raw_pil, caption, tokens

    except Exception as exc:
        traceback.print_exc()
        return f"Error: {exc}", tokens_str, None, raw_pil, caption, tokens


def blip_occlude_then_analyze(image_editor_val, opacity, n_steps, target_token_idx):
    """
    For the batch occlusion workflow:
      1. Extract the edited/occluded composite from the ImageEditor dict.
      2. Re-generate a NEW caption from that (potentially occluded) image.
      3. Compute token-level Integrated Gradients for the selected token.

    This ensures that painting/erasing regions actually changes the caption
    and the attribution, rather than recycling the stale pre-occlusion state.

    Returns:
        caption (str), tokens_str (str),
        attribution_overlay (PIL.Image), edited_pil (PIL.Image),
        caption (str), tokens (list)
    """
    if image_editor_val is None:
        return "Please load an image into the occlusion editor.", "", None, None, "", []

    # Extract the composite (edited) image
    if isinstance(image_editor_val, dict):
        raw_pil = image_editor_val.get("composite", image_editor_val.get("image", None))
    else:
        raw_pil = image_editor_val

    if raw_pil is None:
        return "No image found in occlusion editor.", "", None, None, "", []

    try:
        # Step 1: generate a fresh caption from the (possibly occluded) image
        cap, toks_str, _, _, cap_str, tokens = blip_generate_caption_only(raw_pil)

        if not tokens:
            return cap, "", None, raw_pil, cap, []

        # Clamp token index
        idx = max(0, min(int(target_token_idx), len(tokens) - 1))

        # Step 2: compute IG attribution with the new caption
        attribution_map = blip_compute_integrated_gradients(
            raw_pil, cap, idx, int(n_steps)
        )

        # Step 3: overlay heatmap
        overlay = blip_create_attribution_overlay(raw_pil, attribution_map, opacity)

        return cap, toks_str, overlay, raw_pil, cap, tokens

    except Exception as exc:
        traceback.print_exc()
        return f"Error: {exc}", "", None, raw_pil, "", []


def blip_batch_caption_images(file_list):
    """
    Caption a batch of uploaded image files.

    Args:
        file_list: list of file dicts from gr.File (with a 'name' path key),
                   or list of PIL Images directly.

    Returns:
        results: list of dicts {image (PIL), caption (str), tokens (list)}
        gallery_items: list of (PIL Image, caption) tuples for gr.Gallery
    """
    if not file_list:
        return [], []

    results = []
    gallery_items = []

    print(f"DEBUG: [blip_batch_caption_images] Starting batch captioning for {len(file_list)} items...")
    for idx, item in enumerate(file_list):
        print(f"DEBUG: [blip_batch_caption_images] Processing item {idx+1}/{len(file_list)}...")
        try:
            # Handle item being a PIL Image already
            if isinstance(item, Image.Image):
                pil_img = item.convert("RGB")
            # gr.File returns a dict with 'name' (temp path) or a NamedString
            elif isinstance(item, dict):
                path = item.get("name", item.get("path", ""))
                pil_img = Image.open(path).convert("RGB")
            elif hasattr(item, "name"):
                path = item.name
                pil_img = Image.open(path).convert("RGB")
            else:
                path = str(item)
                pil_img = Image.open(path).convert("RGB")
        except Exception as exc:
            print(f"Could not open image {item}: {exc}")
            continue

        caption, tokens_str, _, _, cap_str, tokens = blip_generate_caption_only(pil_img)
        results.append({
            "image": pil_img,
            "caption": caption,
            "tokens": tokens,
        })
        gallery_items.append((pil_img, caption))

    return results, gallery_items


def generate_batch_images(prompt, num_images):
    """
    Generate a set of images using a lightweight text-to-image pipeline.
    
    Args:
        prompt: text prompt for generation.
        num_images: number of images to generate (max 5).
        
    Returns:
        List of PIL Images.
    """
    try:
        from diffusers import AutoPipelineForText2Image
        import torch
    except ImportError:
        print("diffusers library not installed.")
        return []

    num_images = min(int(num_images), 5)
    if num_images <= 0 or not prompt.strip():
        return []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = AutoPipelineForText2Image.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        safety_checker=None
    )
    pipeline.to(device)
    
    images = pipeline(prompt=prompt, num_images_per_prompt=num_images, num_inference_steps=25).images
    return images

def generate_word_freq_chart(results_g1, results_g2):
    """
    Generate a grouped bar chart comparing the top N most common words in both groups.
    """
    from collections import Counter
    import string
    import matplotlib.pyplot as plt
    import io
    from PIL import Image

    stop_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'with', 'and', 'or', 'of', 'to', 'for', 'it', 'this', 'that'}
    
    def get_word_counts(results):
        words = []
        for res in results:
            caption = res.get('caption', '').lower()
            # Remove punctuation
            caption = caption.translate(str.maketrans('', '', string.punctuation))
            for word in caption.split():
                if word not in stop_words and len(word) > 1:
                    words.append(word)
        return Counter(words)
        
    counts_g1 = get_word_counts(results_g1)
    counts_g2 = get_word_counts(results_g2)
    
    # Get top 5 words from both groups combined
    all_words = set(list(counts_g1.keys()) + list(counts_g2.keys()))
    combined_counts = {w: counts_g1.get(w, 0) + counts_g2.get(w, 0) for w in all_words}
    
    top_words = sorted(combined_counts.keys(), key=lambda w: combined_counts[w], reverse=True)[:8]
    
    if not top_words:
        # Return empty image if no words
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No words found", ha='center', va='center')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img
        
    g1_freqs = [counts_g1.get(w, 0) for w in top_words]
    g2_freqs = [counts_g2.get(w, 0) for w in top_words]
    
    import numpy as np
    x = np.arange(len(top_words))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, g1_freqs, width, label='Group 1', color='#3498db')
    ax.bar(x + width/2, g2_freqs, width, label='Group 2', color='#e74c3c')
    
    ax.set_ylabel('Frequency')
    ax.set_title('Most Common Words in Captions (excluding stop words)')
    ax.set_xticks(x)
    ax.set_xticklabels(top_words, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


# -------------------------
# IMAGE VERSION SAVE / COMPARE (mirrors resume_utility pattern)
# -------------------------

_image_versions = {}   # label -> {caption, attr_b64, original_b64, timestamp}


def _pil_to_b64(pil_img):
    """Encode a PIL image as a base64 PNG data-URI."""
    import base64, io as _io
    if pil_img is None:
        return ""
    buf = _io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def save_image_version(caption, attribution_pil, original_pil, batch_mode=False, batch_graph=None, results_g1=None, results_g2=None, auto_label=None):
    """
    Save a captioning session for later comparison.
    Supports single image or batch mode results.

    Returns:
        status_msg (str), dropdown_update (gr.update)
    """
    from datetime import datetime as _dt
    timestamp = _dt.now().strftime("%Y-%m-%d %H:%M")
    
    desc = caption[:30] + "..." if len(caption) > 30 else caption
    if not desc:
        desc = "No caption" if not batch_mode else "Batch Analysis"
        
    label = auto_label or f"{desc} | {timestamp}"

    # Prepare batch data if needed
    stored_g1 = []
    stored_g2 = []
    if batch_mode:
        if results_g1:
            for item in results_g1:
                stored_g1.append({
                    "caption": item.get("caption", ""),
                    "img_b64": _pil_to_b64(item.get("image")),
                })
        if results_g2:
            for item in results_g2:
                stored_g2.append({
                    "caption": item.get("caption", ""),
                    "img_b64": _pil_to_b64(item.get("image")),
                })

    _image_versions[label] = {
        "caption": caption or "(no caption)",
        "attr_b64": _pil_to_b64(attribution_pil),
        "original_b64": _pil_to_b64(original_pil),
        "batch_mode": batch_mode,
        "graph_b64": _pil_to_b64(batch_graph) if batch_mode else "",
        "results_g1": stored_g1,
        "results_g2": stored_g2,
        "timestamp": timestamp,
    }
    choices = list(_image_versions.keys())
    return (
        f"Saved: *{label}*",
        gr.update(choices=choices, value=label),
    )


def load_image_version(label):
    """Return an HTML comparison block for the selected saved version."""
    if not label or label not in _image_versions:
        return "<p>Select a saved version from the dropdown.</p>"
    v = _image_versions[label]
    
    html = f"""
    <div style="font-family:sans-serif;padding:12px;background:#fff;color:#000;border-radius:8px;border:1px solid #ddd;">
      <p><strong>Saved:</strong> {label}</p>
      <p><strong>Timestamp:</strong> {v.get('timestamp', 'N/A')}</p>
    """

    if v.get("batch_mode"):
        # Batch Display
        if v.get("graph_b64"):
            html += f'<div style="margin-bottom:20px;"><p style="font-weight:600;">Word Frequency Comparison</p><img src="{v["graph_b64"]}" style="max-width:100%;border-radius:6px;border:1px solid #eee;"/></div>'
        
        for g_idx, group_key in enumerate(["results_g1", "results_g2"]):
            res_list = v.get(group_key, [])
            if res_list:
                html += f'<div><p style="font-weight:600;">Group {g_idx+1} Images</p>'
                html += '<div style="display:flex;gap:10px;overflow-x:auto;padding-bottom:10px;">'
                for item in res_list:
                    html += f"""
                    <div style="flex:0 0 200px;border:1px solid #eee;padding:8px;border-radius:4px;">
                        <img src="{item["img_b64"]}" style="width:100%;height:150px;object-fit:cover;border-radius:2px;"/>
                        <p style="font-size:12px;margin:8px 0 0;line-height:1.2;">{item["caption"]}</p>
                    </div>
                    """
                html += '</div></div>'
    else:
        # Single Image Display
        attr_html = (
            f'<img src="{v["attr_b64"]}" style="max-width:100%;border-radius:6px;"/>'
            if v["attr_b64"]
            else "<p><em>No attribution heatmap saved.</em></p>"
        )
        orig_html = (
            f'<img src="{v["original_b64"]}" style="max-width:100%;border-radius:6px;"/>'
            if v["original_b64"]
            else ""
        )
        html += f"""
          <p><strong>Caption:</strong> {v['caption']}</p>
          <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;">
            <div><p style="font-weight:600;margin:0 0 4px">Original</p>{orig_html}</div>
            <div><p style="font-weight:600;margin:0 0 4px">Attribution Heatmap</p>{attr_html}</div>
          </div>
        """
    
    html += "</div>"
    return html


def clear_image_comparison():
    """Clear all saved versions."""
    _image_versions.clear()
    return (
        "<p>Comparison cleared.</p>",
        gr.update(choices=[], value=None),
    )


def get_image_version_choices():
    """Return saved version labels for the dropdown."""
    return list(_image_versions.keys())


def export_all_html():
    global _image_versions
    if not _image_versions:
        return None
    
    html = "<html><body style='font-family:sans-serif;padding:20px;'><h1>All Saved Image Variations</h1><hr>"
    for label, v in _image_versions.items():
        html += f"<h2>{label}</h2>"
        if v.get("batch_mode"):
            if v.get("graph_b64"):
                html += f'<h3>Word Frequency Comparison</h3><img src="{v["graph_b64"]}" style="max-width:800px;border-radius:6px;"/><br>'
            for g_idx, group_key in enumerate(["results_g1", "results_g2"]):
                res_list = v.get(group_key, [])
                if res_list:
                    html += f'<h4>Group {g_idx+1} Images</h4><div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(200px, 1fr));gap:20px;">'
                    for item in res_list:
                        html += f"""
                        <div style="border:1px solid #ddd;padding:10px;border-radius:6px;">
                            <img src="{item["img_b64"]}" style="width:100%;border-radius:4px;"/>
                            <p style="font-size:14px;margin:10px 0 0;">{item["caption"]}</p>
                        </div>
                        """
                    html += "</div>"
        else:
            attr_html = (f'<img src="{v["attr_b64"]}" style="max-width:400px;border-radius:6px;"/>' if v["attr_b64"] else "")
            orig_html = (f'<img src="{v["original_b64"]}" style="max-width:400px;border-radius:6px;"/>' if v["original_b64"] else "")
            html += f"<p><strong>Caption:</strong> {v['caption']}</p><div style='display:flex;gap:20px;'>{orig_html}{attr_html}</div>"
        html += "<hr>"
    html += "</body></html>"
    
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "all_image_sessions.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path

def export_selected_html(label):
    global _image_versions
    if not label or label not in _image_versions:
        return None
        
    v = _image_versions[label]
    html = f"<html><body style='font-family:sans-serif;padding:20px;'><h1>{label}</h1><hr>"
    
    if v.get("batch_mode"):
        if v.get("graph_b64"):
            html += f'<h3>Word Frequency Comparison</h3><img src="{v["graph_b64"]}" style="max-width:800px;border-radius:6px;"/><br>'
        for g_idx, group_key in enumerate(["results_g1", "results_g2"]):
            res_list = v.get(group_key, [])
            if res_list:
                html += f'<h4>Group {g_idx+1} Images</h4><div style="display:grid;grid-template-columns:repeat(auto-fill, minmax(250px, 1fr));gap:20px;">'
                for item in res_list:
                    html += f"""
                    <div style="border:1px solid #ddd;padding:15px;border-radius:8px;">
                        <img src="{item["img_b64"]}" style="width:100%;border-radius:6px;"/>
                        <p style="font-size:16px;margin:12px 0 0;">{item["caption"]}</p>
                    </div>
                    """
                html += "</div>"
    else:
        attr_html = (f'<img src="{v["attr_b64"]}" style="max-width:400px;border-radius:6px;"/>' if v["attr_b64"] else "")
        orig_html = (f'<img src="{v["original_b64"]}" style="max-width:400px;border-radius:6px;"/>' if v["original_b64"] else "")
        html += f"<p><strong>Caption:</strong> {v['caption']}</p><div style='display:flex;gap:20px;'>{orig_html}{attr_html}</div>"
        
    html += "</body></html>"
    
    temp_dir = tempfile.mkdtemp()
    # Sanitize label for filename
    safe_label = "".join([c if c.isalnum() else "_" for c in label])
    path = os.path.join(temp_dir, f"image_session_{safe_label}.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path

def export_batch_csv(results_g1, results_g2):
    if not results_g1 and not results_g2:
        return None
        
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "image_batch_results.csv")
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Group', 'Caption', 'Tokens'])
        
        for res in (results_g1 or []):
            writer.writerow(['Group 1', res.get('caption', ''), ", ".join(res.get('tokens', []))])
            
        for res in (results_g2 or []):
            writer.writerow(['Group 2', res.get('caption', ''), ", ".join(res.get('tokens', []))])
            
    return path

# -------------------------
# EXPORT FUNCTIONS
# -------------------------
def get_sample_image_choices():
    """Return list of sample image options for dropdown"""
    # Lazy-load dataset if not already loaded
    if len(sample_images) == 0:
        try:
            load_sample_images(num_images=5)
        except Exception as e:
            print(f"Warning: Could not load sample images: {e}")
            return ["Upload"]  # Only allow uploads if dataset fails
    
    num_samples = len(sample_images)
    return [f"Sample {i+1}" for i in range(num_samples)] + ["Upload"]

def get_sample_image_by_index(choice):
    """Get a specific sample image by choice string"""
    if choice == "Upload":
        return None
    
    # Lazy-load dataset if needed
    if len(sample_images) == 0:
        try:
            load_sample_images(num_images=5)
        except Exception as e:
            print(f"Warning: Could not load sample images: {e}")
            return None
    
    try:
        idx = int(choice.split()[1]) - 1
        return sample_images[idx]
    except:
        return sample_images[0] if sample_images else None