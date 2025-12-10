import torch
from PIL import Image
import numpy as np
from output_maps import load_clip
import time
from dash import Dash, html, dcc, Input, Output, State, callback_context
from matplotlib import pyplot as plt
import clip
import cv2
import plotly.graph_objects as go

def debug_print(x):
    with open("debuglog.txt", "a") as f:
      print("debug:", x, file=f)

def preprocess_imagenet50_for_clip(X, preprocess):
    tensors = []
    print(len(X))
    for i in range(len(X)):
        img = Image.fromarray(X[i].astype(np.uint8))
        tensors.append(preprocess(img))
    return torch.stack(tensors)

def show_activation_interface(model, preproc, device):
  model, f, class_names, X = model
  X_clip = preprocess_imagenet50_for_clip(X, preproc).to(device)

  image = X_clip[1].unsqueeze(0).to(device)

  #image = preprocess(img).unsqueeze(0).to(device)

  # ---- Hook to capture all attention maps ----
  attn_maps = {}

  def get_attention_hook(layer_idx):
      def hook(module, input, output):
          attn_maps[layer_idx] = output[0].detach().cpu()
      return hook

  handles = []
  for i, block in enumerate(model.visual.transformer.resblocks):
      h = block.attn.register_forward_hook(get_attention_hook(i))
      handles.append(h)

  # ---- Forward pass ----
  with torch.no_grad():
      _ = model.encode_image(image)

  # ---- Remove hooks ----
  for h in handles:
      h.remove()

  # ---- Process attention maps ----
  def get_cls_attention_map(attn):
      # attn: (tokens, tokens, heads)
      attn_mean = attn.mean(dim=2) # avg over heads
      cls_attn = attn_mean[1:]  # CLS token attention to image patches
      grid_size = int(cls_attn.shape[0] ** 0.5)
      cls_attn = cls_attn.reshape(grid_size, grid_size)
      cls_attn -= cls_attn.min()
      cls_attn /= cls_attn.max()
      return cls_attn.numpy()

  cls_maps = [get_cls_attention_map(attn_maps[i]) for i in sorted(attn_maps.keys())]
  mpl_figs = []

  # ---- Overlay function ----
  def overlay_attention(img, attn_map, alpha=0.5):
      attn_resized = cv2.resize(attn_map.astype('float32'), (224, 224))
      mpl_figs.append(attn_resized)
      heatmap = plt.cm.jet(attn_resized)[..., :3]
      overlay = (1 - alpha) * np.array(img) / 255.0 + alpha * heatmap
      overlay /= overlay.max()
      return overlay

  # ---- Display progression ----
  #fig, ax = plt.subplots(figsize=(4, 4))
  for i, attn_map in enumerate(cls_maps):
      overlay = overlay_attention(X[1], attn_map)
      plt.figure()
      fig, ax = plt.subplots()
      ax.imshow(overlay)
      ax.set_title(f"Layer {i+1}")
      ax.axis('off')
      #plt.pause(0.5)

  plt.show()

  return mpl_figs, X[1], image

def show_output_maps(input):
  debug_print("got here")
  m, fig = create_and_graph_shap()
  fig.write_image("test.png")
  print("wrote image??")
  return fig

def make_dash_app(flask_server, appdir):

  dash_app = Dash(__name__, server=flask_server, url_base_pathname=appdir)

  # --- Simulated processing functions ---
  def process_func1(user_input):
      s = show_output_maps(user_input)
      debug_print("test")
      s.write_image("imgs/test3.png")
      return s

  def process_func2(user_input):
      time.sleep(2)
      return f"Function 2 processed: {user_input.lower()} (with length {len(user_input)})"

  # --- Layout ---
  dash_app.layout = html.Div([
      html.H2("Multi-Tab Function Dashboard"),

      # Tabs container
      dcc.Tabs(
          id="tabs",
          value="home",
          children=[
              dcc.Tab(label="Home", value="home"),
              dcc.Tab(label="Function 1", value="func1", disabled=True),
              dcc.Tab(label="Function 2", value="func2", disabled=True),
          ]
      ),

      # --- All tab contents exist at once ---
      html.Div(id="home-tab", children=[
          html.H4("Enter input:"),
          dcc.Input(id="user-input", type="text", placeholder="Type something...", debounce=True),
          html.Button("Submit", id="submit-btn", n_clicks=0),
          html.Div(id="home-message", style={"marginTop": "15px", "fontWeight": "bold"})
      ]),

      dcc.Loading(html.Div(id="func1-tab", children=[
          html.H4("Function 1 Result:"),
          dcc.Graph(id='func1-graph')
      ])),

      dcc.Loading(html.Div(id="func2-tab", children=[
        html.H4("Function 2 Result:"),
        html.Div(id="func2-output"),
        html.Div(id="attention-container")  # NEW: attention visualization placeholder
    ])),
  ])

  # --- Show/hide tabs dynamically ---
  @dash_app.callback(
      Output("home-tab", "style"),
      Output("func1-tab", "style"),
      Output("func2-tab", "style"),
      Input("tabs", "value")
  )
  def display_tab(tab):
      style_hide = {"display": "none"}
      style_show = {"display": "block", "marginTop": "20px"}
      if tab == "home":
          return style_show, style_hide, style_hide
      elif tab == "func1":
          return style_hide, style_show, style_hide
      elif tab == "func2":
          return style_hide, style_hide, style_show
      return style_show, style_hide, style_hide

  # --- Unlock other tabs once input is submitted ---
  @dash_app.callback(
      Output("tabs", "children"),
      Output("home-message", "children"),
      Input("submit-btn", "n_clicks"),
      State("user-input", "value"),
      prevent_initial_call=True
  )
  def unlock_tabs(n_clicks, user_input):
      if not user_input:
          msg = "Please enter something first!"
          tabs = [
              dcc.Tab(label="Home", value="home"),
              dcc.Tab(label="Function 1", value="func1", disabled=True),
              dcc.Tab(label="Function 2", value="func2", disabled=True),
          ]
      else:
          msg = f"Input '{user_input}' stored! You can now access the other tabs."
          tabs = [
              dcc.Tab(label="Home", value="home"),
              dcc.Tab(label="Function 1", value="func1"),
              dcc.Tab(label="Function 2", value="func2"),
          ]
      return tabs, msg

  # --- Function 1 callback ---
  @dash_app.callback(
      Output("func1-graph", "figure"),
      Input("tabs", "value"),
      State("user-input", "value")
  )
  def update_func1(tab, user_input):
      if tab == "func1" and user_input:
          result = process_func1(user_input)
          result.write_image("imgs/test4.png")
          return result
      return {}

  # --- Function 2 callback ---
  @dash_app.callback(
      Output("func2-output", "children"),
      Output("attention-container", "children"),
      Input("tabs", "value"),
      State("user-input", "value")
  )
  def update_func2(tab, user_input):
      if tab == "func2" and user_input:
          result = process_func2(user_input)
          return result, get_attn_layout()
      return "", None

  return dash_app

def get_attn_layout():
    return html.Div([
        html.H4("Layer-by-Layer Attention Maps"),
        dcc.Graph(id='attention-main'),
        dcc.Graph(id='attention-hover', config={'displayModeBar': False})
    ])

def register_attn_callbacks(app, image_arrays, raw_img):
    num_layers = len(image_arrays)
    point_labels = [f"Layer {i+1}" for i in range(num_layers)]
    main_x = list(range(num_layers))

    @app.callback(
        Output('attention-main', 'figure'),
        Output('attention-hover', 'figure'),
        Input('attention-main', 'hoverData'),
        prevent_initial_call='initial_duplicate'
    )
    def update_hover(hover_data):
        idx = hover_data['points'][0]['x'] if hover_data else 0

        # Main line
        main_fig = go.Figure(go.Scatter(
            x=main_x, y=[5]*num_layers,
            mode='markers+lines',
            marker=dict(size=20, color='rgb(55,128,191)'),
            hovertemplate='<b>Layer %{x}</b><extra></extra>'
        ))
        main_fig.update_layout(
            title='Hover over a layer',
            yaxis=dict(showticklabels=False, range=[0, 10]),
            height=250
        )

        # Overlay figure
        overlay = go.Figure()
        overlay.add_trace(go.Image(z=raw_img))
        overlay.add_trace(go.Heatmap(
            z=image_arrays[idx],
            colorscale="RdBu",
            opacity=0.6
        ))
        overlay.update_layout(
            title=f"Layer {idx}: {point_labels[idx]}",
            height=600,
            margin=dict(t=50, b=20, l=20, r=20)
        )

        return main_fig, overlay

def explain_output(dash_app):

  debug_print("Load clip function??")
  # load model
  device = "cuda" if torch.cuda.is_available() else "cpu"
  debug_print("Stuck here?")
  model, preprocess = clip.load('ViT-B/32', device=device)

  m = load_clip('ViT-B/32')
  twelve_blocks, raw_img, img = show_activation_interface(m, preprocess, device)
  #dash_app = make_dash_app(a, appdir)
  register_attn_callbacks(dash_app, twelve_blocks, raw_img)
  
