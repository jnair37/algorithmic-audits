# backend.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
import base64
import io
import numpy as np
from captum.attr import IntegratedGradients
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)
CORS(app)

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)
model.eval()

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    img_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(io.BytesIO(img_data)).convert('RGB')
    
    # Generate caption
    inputs = processor(image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Compute integrated gradients
    def forward_func(pixel_values):
        outputs = model(pixel_values=pixel_values)
        return outputs.logits[:, 0, :]  # First token logits
    
    ig = IntegratedGradients(forward_func)
    baseline = torch.zeros_like(inputs.pixel_values)
    attributions = ig.attribute(
        inputs.pixel_values,
        baseline,
        target=0,
        n_steps=50
    )
    
    # Process attributions
    attr = attributions.squeeze().cpu().numpy()
    attr = np.abs(attr).sum(axis=0)  # Sum over channels
    attr = (attr - attr.min()) / (attr.max() - attr.min())
    
    return jsonify({
        "caption": caption,
        "attributions": attr.flatten().tolist(),
        "width": attr.shape[1],
        "height": attr.shape[0]
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)

# Install: pip install flask flask-cors torch transformers captum pillow
# Run: python backend.py