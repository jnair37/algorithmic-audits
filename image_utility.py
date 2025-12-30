import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients
import gradio as gr
import io

# -------------------------
# 1. STREAMING DATASET LOAD
# -------------------------
# streaming=True avoids downloading the entire dataset
ds_stream = load_dataset(
    "seaurkin/facial_exrpressions",
    split="train",
    streaming=True
)

# -------------------------
# Utility: get first N images from a streaming dataset
# -------------------------
def take_images_from_stream(stream, n, resize_to=(224,224)):
    imgs = []
    for example in stream:
        img = example["image"]
        if resize_to is not None:
            img = img.resize(resize_to)
        imgs.append(img)
        if len(imgs) == n:
            break
    return imgs

# -------------------------
# 2. LOAD CAPTIONING MODEL
# -------------------------
model_id = "microsoft/git-large-coco"
processor = AutoProcessor.from_pretrained(model_id)
vision_model = AutoModelForCausalLM.from_pretrained(model_id).eval()

# -------------------------
# Prediction function
# -------------------------
def predict_caption(pil_imgs):
    inputs = processor(images=pil_imgs, return_tensors="pt")
    with torch.no_grad():
        out = vision_model.generate(**inputs, max_length=40)
    captions = processor.batch_decode(out, skip_special_tokens=True)
    return captions

# -------------------------
# 3. Extract oen test image from stream
# -------------------------
# Create a fresh iterator (streaming datasets are single-pass)
ds_stream_2 = load_dataset(
    "seaurkin/facial_exrpressions",
    split="train",
    streaming=True
)

# quick sanity check
test_img = None
for example in ds_stream_2:
    test_img = example["image"]
    break

# -------------------------
# 4. Run caption + Integrated Gradients
# -------------------------
def run_integrated_gradients(test_img):
    """Run IG analysis on the provided test image"""

    # Convert numpy array to PIL Image if needed
    if isinstance(test_img, np.ndarray):
        test_img = Image.fromarray(test_img.astype(np.uint8))

    print("Generated caption:")
    caption = predict_caption([test_img])[0]
    print(caption)

    # Prepare for IG - ensure PIL Image for resize
    if isinstance(test_img, np.ndarray):
        test_img_resized = Image.fromarray(test_img.astype(np.uint8)).resize((224, 224))
    else:
        test_img_resized = test_img.resize((224, 224))

    print("resized")

    inputs = processor(images=test_img_resized, return_tensors="pt")

    print("generating")
    # Generate caption to get token IDs
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_length=40)

    pixel_values = inputs['pixel_values'].requires_grad_(True)
    baseline = torch.zeros_like(pixel_values)

    print("visualizing")

    # Get original image for visualization
    original_img = pixel_values.squeeze().cpu().detach().numpy()
    original_img = np.transpose(original_img, (1, 2, 0))
    original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())

    # Compute attributions for 1 token at a time
    num_tokens = 1 # min(3, len(generated_ids[0]) - 1)

    fig, axes = plt.subplots(2, num_tokens, figsize=(15, 10))
    if num_tokens == 1:
        axes = axes.reshape(-1, 1)

    for idx in range(num_tokens):
        token_position = idx + 1
        target_token_id = generated_ids[0, token_position].item()
        token_text = processor.decode([target_token_id])

        # Define forward function for this token
        def forward_func(pixel_values, pos=token_position, tok_id=target_token_id):
            outputs = model(
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

        # Plot original
        axes[0, idx].imshow(original_img)
        axes[0, idx].set_title(f'Token {idx+1}: "{token_text}"', fontsize=12, fontweight='bold')
        axes[0, idx].axis('off')

        # Plot attribution
        axes[1, idx].imshow(original_img)
        axes[1, idx].imshow(attr_sum, cmap='hot', alpha=0.6)
        axes[1, idx].set_title(f'Attribution Heatmap', fontsize=10)
        axes[1, idx].axis('off')

    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    result_img = Image.open(buf)
    plt.close()

    return caption, result_img