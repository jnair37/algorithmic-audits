# Imports

import torch
import clip
import shap
import json
import urllib.request
import numpy as np
from PIL import Image
import plotly.graph_objects as go
from skimage.transform import resize
from matplotlib import pyplot as plt
import cv2

from lime import lime_image
from skimage.segmentation import mark_boundaries

def debug_print(x):
    with open("debuglog.txt", "a") as f:
      print("debug:", x, file=f)

def precompute():
  # Storing imagenet/clip class name text features to make the interface run faster
  # (largest bottleneck is on encoding the text)

  # get model to encode
  debug_print("Load clip function??")
  # load model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  debug_print("Stuck here?")
  model, preprocess = clip.load('ViT-B/32', device=device)
  debug_print("Or stuck here?")
  model.eval()

  debug_print('loaded cuda')
  # Get datasets
  X, y = shap.datasets.imagenet50()
  debug_print("xy retrieved")


  # Get class names
  url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
  with open(shap.datasets.cache(url)) as file:
      class_names = [v[1] for v in json.load(file).values()]

  debug_print("got dset")
  debug_print("next thing")
  text_prompts = [f"a photo of a {name.replace('_',' ')}" for name in class_names]
  debug_print("where is this going")
  text_tokens = clip.tokenize(text_prompts).to(device)
  debug_print("Something")

  with torch.no_grad():
      text_features = model.encode_text(text_tokens)
      debug_print("wheee")
      text_features /= text_features.norm(dim=-1, keepdim=True)  # normalize

  debug_print("just trying to print some text")
  torch.cuda.synchronize()
  debug_print("Synchronized")

  # Move to CPU and save
  text_features = text_features.cpu()
  debug_print("TYPE OF TEXT FEATURES??")
  debug_print(type(text_features))

  np.save('text_ftrs.npy', text_features.numpy())   # ✅ FIXED
  np.save('class_names.npy', np.array(class_names))

  # Run this ONCE before starting the Dash app to pre-compute everything

  model_tuple = load_clip('ViT-B/32')
  model, f, class_names, X = model_tuple

  print(type(model))
  print(type(f))
  print(type(class_names))
  print(type(X))

  # Pre-compute SHAP values (this takes time, but only happens once)
  masker = shap.maskers.Image("inpaint_telea", X[0].shape)
  explainer = shap.Explainer(f, masker, output_names=class_names)

  print("Computing SHAP values... this may take a few minutes...")
  shap_values = explainer(X[:5], max_evals=500, batch_size=50, outputs=shap.Explanation.argsort.flip[:1])

  # Cache it
  np.save('shap_values_data.npy', shap_values.data)
  np.save('shap_values_values.npy', shap_values.values)
  np.save('shap_values_output_names.npy', shap_values.output_names)

  print("✓ SHAP values cached! Now the Dash app will load instantly.")

  # Precompute LIME values ?


def load_clip(substring):
  debug_print("Load clip function??")
  # load model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  debug_print("Stuck here?")
  model, preprocess = clip.load(substring, device=device)
  debug_print("Or stuck here?")
  model.eval()

  debug_print('loaded cuda')
  # Get datasets
  X, y = shap.datasets.imagenet50()
  debug_print("xy retrieved")

  # Get class names
  # url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
  # with open(shap.datasets.cache(url)) as file:
  #     class_names = [v[1] for v in json.load(file).values()]
  class_names = list(np.load('class_names.npy'))

  debug_print("got dset")
  # # Specialized: convert class names to CLIP text prompts + encode
  # text_prompts = [f"a photo of a {name.replace('_',' ')}" for name in class_names]
  # text_tokens = clip.tokenize(text_prompts).to(device)

  # with torch.no_grad():
  #     text_features = model.encode_text(text_tokens)
  #     text_features /= text_features.norm(dim=-1, keepdim=True)  # normalize
  text_features = np.load('text_ftrs.npy')

  debug_print("encoded text")
  # Define f and encode
  def f(X):
      debug_print("running f")
      tmp = []
      for i in range(len(X)):
          img = Image.fromarray(X[i].astype("uint8"))
          tmp.append(preprocess(img))

      debug_print("running tensor f")
      X_tensor = torch.stack(tmp).to(device)

      debug_print("encoding f")
      with torch.no_grad():
          # Encode images
          image_features = model.encode_image(X_tensor)
          image_features /= image_features.norm(dim=-1, keepdim=True)
          debug_print("mid encoding")
          # Similarity to all class text embeddings
          sim = image_features.cpu() @ text_features.T  # shape [batch, num_classes]

      debug_print("sim")
      sim = sim.cpu().numpy()
      debug_print('this si working')
      debug_print(type(sim))
      return sim

  debug_print("got to teh end of load func")
  return model, f, class_names, X

def plot_out_map(model, map_type, block_size):

  model, f, class_names, X = model
  images = np.load('shap_values_data.npy')
  i = 1
  img = images[i]
  classification = np.load('shap_values_output_names.npy')
  out = classification[i] 

  if (map_type == 'shap'):
    debug_print("Shap function")
    values = np.load('shap_values_values.npy')
    sv_raw = values[i]
    shap_min = np.min(sv_raw)
    shap_max = np.max(sv_raw)
    sv = (2 * (sv_raw - shap_min) / (shap_max - shap_min)) - 1
    custom_text = [[f"Scaled Value={sv[i][j][0][0]:.2f}. This region is interpreted to have a ({'larger' if abs(sv[i][j][0][0]) >= 1 else 'smaller'}) ({'positive' if sv[i][j][0][0]*100000 >= 0 else 'negative'}) impact on this classification result."
                for j in range(sv[:,:,0,0].shape[1])] for i in range(sv[:,:,0,0].shape[0])]
    # Parameters
    H, W, _, _ = sv.shape
    h_blocks, w_blocks = H // block_size, W // block_size  # should be 14x14

    # Compute block averages
    coarse = sv[:h_blocks*block_size, :w_blocks*block_size, 0, 0].reshape(
        h_blocks, block_size, w_blocks, block_size
    ).mean(axis=(1, 3))  # shape (14, 14)

    coarse_resized = resize(coarse, (H, W), order=0, preserve_range=True, anti_aliasing=False)

  elif (map_type == 'lime'):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img, f, top_labels=5, hide_color=0, num_samples=1000)
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=False,  # allow both positive & negative
        num_features=10,
        hide_rest=False
    )
    debug_print("Lime function")
    # Parameters
    block_size = 16
    H, W = mask.shape
    h_blocks, w_blocks = H // block_size, W // block_size  # should be 14x14

    # Compute block averages
    coarse_lime = mask[:h_blocks*block_size, :w_blocks*block_size].reshape(
        h_blocks, block_size, w_blocks, block_size
    ).mean(axis=(1, 3))  # shape (14, 14)

    print(coarse_lime)

    coarse_lime_resized = resize(coarse_lime, (H, W), order=0, preserve_range=True, anti_aliasing=False)
    print(np.shape(coarse_lime_resized))

    custom_text = [[f"Scaled Value={coarse_lime_resized[i][j]:.2f}. This region is interpreted to have a ({'larger' if abs(coarse_lime_resized[i][j]) >= 0.75 else 'smaller'}) ({'positive' if coarse_lime_resized[i][j] >= 0 else 'negative'}) impact on this classification result."
                    for j in range(coarse_lime_resized.shape[1])] for i in range(coarse_lime_resized.shape[0])]

    coarse_resized = coarse_lime_resized

  else:
    return("boo")

  

  debug_print("Got here starting plot")

  # plot basic image
  fig = go.Figure()

  fig.add_trace(go.Image(z=img, hoverinfo="none"))

  
  debug_print("resize calculated")

  fig.add_trace(go.Heatmap(
      z=coarse_resized,
      colorscale="RdBu",
      showscale=True,
      opacity=0.6,
      text=custom_text,
      hoverinfo="text"
  ))

  # Layout tweaks
  fig.update_layout(
      title="Explanation Map: " + map_type + ". The model classifies this as " + out[0],
      xaxis=dict(showticklabels=False),
      yaxis=dict(showticklabels=False)
  )


  fig.update_yaxes(autorange="reversed")

  #fig.show()
  fig.write_image("imgs/test.png")

  debug_print("Got through plotting")

  return coarse_resized, fig

def graph_interactive_explanation(model, map_type):
  debug_print('graphing')
  # Todo: dont hardcode these
  if (map_type == 'shap'):
    _, fig = plot_out_map(model, 'shap', 16)
  elif (map_type == 'lime'):
    _, fig = plot_out_map(model, 'lime', 16)
  return fig

def create_and_graph(map_type):
  print('creating and graphing')
  m = load_clip('ViT-B/32')
  fig = graph_interactive_explanation(m, map_type)
  return m, fig

def explain_output(map_type):
  if (map_type == 'shap' or map_type == 'lime'):
    return create_and_graph(map_type)
  else:
    return ("Oops!", -1)