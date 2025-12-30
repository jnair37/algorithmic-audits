#!pip install captum

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients
import gradio as gr
import io

# ============================================================================
# SHAP-Compatible Output Format
# ============================================================================

@dataclass
class UnifiedExplanation:
    """
    Unified format that mimics SHAP output structure.
    Compatible with existing analyze_text() function.

    For causal LMs: Explains contribution of each input token to next-token prediction
    """
    values: np.ndarray      # Shape: (1, n_tokens, vocab_size) - attribution scores
    data: List[List[str]]   # Shape: (1, n_tokens) - token strings
    base_values: np.ndarray # Shape: (1, vocab_size) - base prediction values
    output_names: List[str] # Top predicted tokens or all vocab

    def __getitem__(self, key):
        """Allow array-like indexing for compatibility"""
        if key == 0:
            return self
        raise IndexError(f"Index {key} out of range")


# ============================================================================
# Helper Functions for Causal LM
# ============================================================================

def _get_next_token_logits(model, input_ids, position=-1):
    """
    Get logits for next token prediction at specified position.

    Args:
        position: Position to predict next token for (-1 = last position)
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        # Get logits for next token at specified position
        logits = outputs.logits[0, position, :]  # (vocab_size,)
    return logits


def _get_top_k_tokens(logits, tokenizer, k=10):
    """Get top-k predicted tokens and their probabilities."""
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=k)

    top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

    return top_tokens, top_probs.cpu().numpy(), top_indices.cpu().numpy()


def _generate_continuation(text, model, tokenizer, max_new_tokens=10, temperature=1.0):
    """
    Generate a continuation of the text.

    Args:
        text: Input text
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (1.0 = neutral, lower = more deterministic)

    Returns:
        continuation: Generated text (without the input)
        full_text: Input + generated text
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_length = inputs['input_ids'].shape[1]

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0 else False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )

    # Decode full output
    full_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

    # Extract just the generated part
    continuation = tokenizer.decode(
        outputs.sequences[0][input_length:],
        skip_special_tokens=True
    )

    return continuation, full_text


# ============================================================================
# Method Implementations for Causal LM
# ============================================================================

def _get_integrated_gradients_causal(text, model, tokenizer, target_token_id=None,
                                     n_steps=50, position=-1):
    """
    Compute Integrated Gradients for causal LM.

    Args:
        text: Input text (may include partial generation)
        target_token_id: Specific token ID to explain
        n_steps: Number of integration steps
        position: Position to predict next token for (-1 = last position)
    """
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        raise ImportError("Captum required. Install: pip install captum")

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs['input_ids']

    # Get next token prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, position, :]

        if target_token_id is None:
            target_token_id = next_token_logits.argmax().item()

    # Get embedding layer based on model architecture
    # compatible with multiple types of models for when adding future selection feature
    if hasattr(model, 'transformer'):
        # GPT-2 style
        embed_layer = model.transformer.wte
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        # LLaMA style
        embed_layer = model.model.embed_tokens
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        # OPT style
        embed_layer = model.model.decoder.embed_tokens
    else:
        raise ValueError("Cannot find embedding layer for this model architecture")

    # Wrapper class to make model return just the target logit (scalar)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, position, target_token_id):
            super().__init__()
            self.model = model
            self.position = position
            self.target_token_id = target_token_id

        def forward(self, input_ids):
            outputs = self.model(input_ids)
            # Return just the scalar logit for the target token
            return outputs.logits[:, self.position, self.target_token_id]

    wrapped_model = ModelWrapper(model, position, target_token_id)

    # Create Layer IG explainer with the wrapped model
    lig = LayerIntegratedGradients(wrapped_model, embed_layer)

    # Compute attributions
    attributions = lig.attribute(
        input_ids,
        n_steps=n_steps,
        internal_batch_size=32
    )

    # Sum across embedding dimension to get per-token attribution
    attr_scores = attributions.squeeze().sum(dim=-1).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Get base value (prediction with zero embeddings)
    with torch.no_grad():
        zero_embeds = torch.zeros((1, input_ids.shape[1], embed_layer.embedding_dim)).to(input_ids.device)
        attention_mask = torch.ones_like(input_ids)
        base_outputs = model(inputs_embeds=zero_embeds, attention_mask=attention_mask)
        base_logits = base_outputs.logits[0, position, :]
        base_values = torch.softmax(base_logits, dim=-1).cpu().numpy()

    # For output names, just return the target token being explained
    target_token_str = tokenizer.decode([target_token_id])
    clean_target = target_token_str.replace('Ġ', ' ').replace('Â', '').strip()
    if not clean_target:
        clean_target = target_token_str
    output_names = [f"Target: {clean_target}"]

    return attr_scores, tokens, base_values, output_names, next_token_logits.cpu().numpy()


#### ======================================================================================================
#### Templates for additional methods to add; not currently working in the interface so they are not called
#### ======================================================================================================

def _get_attention_weights_causal(text, model, tokenizer, layer=-1, position=-1, target_token_id=None):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # Get attention from specified layer
    # Shape: (batch, num_heads, seq_len, seq_len)
    attentions = outputs.attentions[layer][0]

    # For causal models, look at attention TO the last token (or specified position)
    # This shows which tokens were important for predicting next token
    if position == -1:
        position = attentions.shape[-1] - 1

    # Average across heads, get attention pattern for the position
    token_importance = attentions[:, position, :].mean(dim=0).cpu().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Get next token logits
    next_token_logits = outputs.logits[0, position, :]
    probs = torch.softmax(next_token_logits, dim=-1).cpu().numpy()

    # For output names, show what token we're explaining (if provided)
    if target_token_id is not None:
        target_token_str = tokenizer.decode([target_token_id])
        clean_target = target_token_str.replace('Ġ', ' ').replace('Â', '').strip()
        if not clean_target:
            clean_target = target_token_str
        output_names = [f"Target: {clean_target}"]
    else:
        # Fallback: show top predicted token
        predicted_token_id = next_token_logits.argmax().item()
        target_token_str = tokenizer.decode([predicted_token_id])
        clean_target = target_token_str.replace('Ġ', ' ').replace('Â', '').strip()
        output_names = [f"Predicted: {clean_target}"]

    return token_importance, tokens, probs, output_names, next_token_logits.cpu().numpy()


def _get_gradient_x_input_causal(text, model, tokenizer, target_token_id=None, position=-1):

    try:
        from captum.attr import LayerGradientXActivation
    except ImportError:
        raise ImportError("Captum required. Install: pip install captum")

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs['input_ids']

    # Get next token prediction
    model.eval()
    outputs = model(input_ids)
    next_token_logits = outputs.logits[0, position, :]

    if target_token_id is None:
        target_token_id = next_token_logits.argmax().item()

    # Get embedding layer
    if hasattr(model, 'transformer'):
        embed_layer = model.transformer.wte
    elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_layer = model.model.embed_tokens
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        embed_layer = model.model.decoder.embed_tokens
    else:
        raise ValueError("Cannot find embedding layer")

    # Wrapper class to return just the target logit
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, position, target_token_id):
            super().__init__()
            self.model = model
            self.position = position
            self.target_token_id = target_token_id

        def forward(self, input_ids):
            outputs = self.model(input_ids)
            return outputs.logits[:, self.position, self.target_token_id]

    wrapped_model = ModelWrapper(model, position, target_token_id)

    # Compute Layer Gradient × Activation with wrapped model
    lgxa = LayerGradientXActivation(wrapped_model, embed_layer)
    attributions = lgxa.attribute(input_ids)

    # Sum across embedding dimension
    attr_scores = attributions.squeeze().sum(dim=-1).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Get base values
    with torch.no_grad():
        zero_embeds = torch.zeros((1, input_ids.shape[1], embed_layer.embedding_dim)).to(input_ids.device)
        attention_mask = torch.ones_like(input_ids)
        base_outputs = model(inputs_embeds=zero_embeds, attention_mask=attention_mask)
        base_logits = base_outputs.logits[0, position, :]
        base_values = torch.softmax(base_logits, dim=-1).cpu().numpy()

    # Output names
    target_token_str = tokenizer.decode([target_token_id])
    clean_target = target_token_str.replace('Ġ', ' ').replace('Â', '').strip()
    if not clean_target:
        clean_target = target_token_str
    output_names = [f"Target: {clean_target}"]

    return attr_scores, tokens, base_values, output_names, next_token_logits.cpu().numpy()


def _get_layer_integrated_gradients_causal(text, model, tokenizer, target_token_id=None,
                                           n_steps=20, position=-1):
    """
    Compute Layer Integrated Gradients for causal LM.
    """
    try:
        from captum.attr import LayerIntegratedGradients
    except ImportError:
        raise ImportError("Captum required. Install: pip install captum")

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs['input_ids']

    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, position, :]

        if target_token_id is None:
            target_token_id = next_token_logits.argmax().item()

    # Define forward function
    def forward_func(input_ids):
        outputs = model(input_ids)
        return outputs.logits[:, position, target_token_id]

    # Get embedding layer
    if hasattr(model, 'transformer'):
        # GPT-2 style
        embed_layer = model.transformer.wte
    elif hasattr(model, 'model'):
        # LLaMA style
        embed_layer = model.model.embed_tokens
    else:
        raise ValueError("Cannot find embedding layer for this model architecture")

    # Use embedding layer for attribution
    lig = LayerIntegratedGradients(forward_func, embed_layer)

    attributions = lig.attribute(
        input_ids,
        n_steps=n_steps
    )

    # Sum across embedding dimension
    attr_scores = attributions.squeeze().sum(dim=-1).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Get base values
    with torch.no_grad():
        base_outputs = model(torch.zeros_like(input_ids))
        base_logits = base_outputs.logits[0, position, :]
        base_values = torch.softmax(base_logits, dim=-1).cpu().numpy()

    # Get output names
    top_tokens, _, top_indices = _get_top_k_tokens(next_token_logits, tokenizer, k=10)
    output_names = []
    for token in top_tokens:
        clean_token = token.replace('Ġ', ' ').replace('Â', '').strip()
        if not clean_token:
            clean_token = token
        output_names.append(clean_token)

    return attr_scores, tokens, base_values, output_names, next_token_logits.cpu().numpy()


def _get_shap_values_causal(text, model, tokenizer):
    """
    Original SHAP implementation for causal LM.

    Note: SHAP should already handle causal LMs correctly.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP required. Install: pip install shap")

    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer([text])

    return shap_values


def _select_target_token_from_continuation(continuation, tokenizer, position=-1):
    """
    Select the most interesting token from a continuation to explain.

    Strategy:
    1. Check for yes/no tokens (important for Q&A)
    2. Otherwise, select the longest token (usually most meaningful)

    Args:
        continuation: Generated text
        tokenizer: Tokenizer
        position: Which token position in continuation to explain (-1 = first token)

    Returns:
        target_token_id: Token ID to explain
        target_token_str: String representation of the token
        token_position: Position in continuation
    """
    # Tokenize the continuation
    tokens = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)
    token_ids = tokens['input_ids'][0].tolist()

    if len(token_ids) == 0:
        return None, None, None

    # Convert token IDs to strings
    token_strings = [tokenizer.decode([tid]) for tid in token_ids]

    # Strategy 1: Check for yes/no (case insensitive)
    for i, token_str in enumerate(token_strings):
        clean = token_str.replace('Ġ', '').strip().lower()
        if clean in ['yes', 'no']:
            return token_ids[i], token_str, i

    # Strategy 2: Select the longest token (usually most meaningful)
    # Ignore very short tokens (articles, punctuation, etc.)
    meaningful_tokens = [
        (i, tid, tstr) for i, (tid, tstr) in enumerate(zip(token_ids, token_strings))
        if len(tstr.strip()) > 1  # At least 2 characters
    ]

    if meaningful_tokens:
        # Get longest token
        longest = max(meaningful_tokens, key=lambda x: len(x[2]))
        return longest[1], longest[2], longest[0]

    # Fallback: just use the first token
    return token_ids[0], token_strings[0], 0


def explain_generation(
    text: str,
    model,
    tokenizer,
    method: str = "integrated_gradients",
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    **kwargs
) -> Tuple[UnifiedExplanation, str, str, int]:
    """
    Generate continuation first, then explain why the model generated it.

    Workflow:
    1. Generate: "The cat sat on the" → " mat and slept peacefully"
    2. Select: "peacefully" (most interesting token)
    3. Extend input: "The cat sat on the mat and slept" (original + generated up to target)
    4. Explain: Why does "The cat sat on the mat and slept" predict "peacefully"?

    Args:
        text: Input text
        model: AutoModelForCausalLM
        tokenizer: Tokenizer
        method: Attribution method to use
        max_new_tokens: How many tokens to generate
        temperature: Generation temperature
        **kwargs: Additional arguments for attribution method (e.g., n_steps)

    Returns:
        explanation: Attribution explaining the selected token
        continuation: Full generated continuation
        target_token: The specific token being explained
        extended_input: The input text used for explanation (original + partial generation)
    """
    print(f"\n{'='*70}")
    print("STEP 1: GENERATING CONTINUATION...")
    print(f"{'='*70}")

    # Step 1: Generate continuation
    continuation, full_text = _generate_continuation(
        text, model, tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

    print(f"Input: '{text}'")
    print(f"Generated: '{continuation}'")
    print(f"Full: '{full_text}'")

    # Step 2: Select target token from continuation
    print(f"\n{'='*70}")
    print("STEP 2: SELECTING TARGET TOKEN TO EXPLAIN...")
    print(f"{'='*70}")

    target_token_id, target_token_str, token_position = _select_target_token_from_continuation(
        continuation, tokenizer
    )

    if target_token_id is None:
        raise ValueError("Could not select a target token from continuation")

    clean_target = target_token_str.replace('Ġ', ' ').replace('Â', '').strip()
    if not clean_target:
        clean_target = target_token_str

    print(f"Selected token: '{clean_target}' (position {token_position} in continuation)")

    # Step 3: Create extended input
    # If target is at position 0, use original input
    # If target is at position n, use original + first n tokens of continuation
    print(f"\n{'='*70}")
    print("STEP 3: CREATING EXTENDED INPUT FOR EXPLANATION...")
    print(f"{'='*70}")

    if token_position == 0:
        # Target is first token of continuation - use original input
        extended_input = text
        print(f"Target is first generated token - using original input")
    else:
        # Target is later in continuation - extend input with generated tokens up to target
        continuation_tokens = tokenizer(continuation, return_tensors="pt", add_special_tokens=False)
        tokens_before_target = continuation_tokens['input_ids'][0][:token_position]

        # Decode the tokens before target
        partial_continuation = tokenizer.decode(tokens_before_target, skip_special_tokens=True)
        extended_input = text + partial_continuation
        print(f"Target is at position {token_position} - extending input with partial generation")
        print(f"Partial continuation added: '{partial_continuation}'")

    print(f"Extended input: '{extended_input}'")

    # Step 4: Get explanation for why this specific token was generated
    print(f"\n{'='*70}")
    print(f"STEP 4: COMPUTING ATTRIBUTIONS FOR '{clean_target}'...")
    print(f"{'='*70}")

    explanation = get_explanation(
        extended_input,  # Use extended input!
        model,
        tokenizer,
        method=method,
        target_token_id=target_token_id,
        position=-1,  # Predict at last position of extended input
        **kwargs
    )

    print(f"\nExplanation complete!")
    print(f"Question answered: Which tokens in '{extended_input}'")
    print(f"                   contributed to generating '{clean_target}'?")
    print(f"{'='*70}\n")

    return explanation, continuation, clean_target, extended_input


def analyze_generation(
    text: str,
    model,
    tokenizer,
    method: str = "integrated_gradients",
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    **kwargs
):
    """
    Complete pipeline: Generate → Select → Explain → Visualize

    Args:
        text: Input text
        model: AutoModelForCausalLM
        tokenizer: Tokenizer
        method: Attribution method
        max_new_tokens: Tokens to generate
        temperature: Generation temperature
        **kwargs: Additional attribution arguments

    Returns:
        highlights: List of (start, end, label, color) tuples for visualization
        continuation: Generated text
        target_token: Token that was explained
        extended_input: The input text that was actually explained

    """
    # Step 1: Generate and explain
    explanation, continuation, target_token, extended_input = explain_generation(
        text, model, tokenizer, method, max_new_tokens, temperature, **kwargs
    )

    # Step 2: Convert to highlights for visualization
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    all_values = explanation.values
    data_values = explanation.data

    # Get attributions for the target token
    token_with_values = None
    for i in range(all_values.shape[2]):
        if np.any(all_values[0, :, i] != 0):
            token_with_values = i
            break

    if token_with_values is None:
        all_values = all_values[0, :, -1]
    else:
        all_values = all_values[0, :, token_with_values]

    # Normalize values
    min_val = np.min(all_values)
    max_val = np.max(all_values)

    if max_val - min_val == 0:
        normalized = np.ones_like(all_values) * 0.5
    else:
        normalized = (all_values - min_val) / (max_val - min_val)

    # Choose colormap
    cmap = plt.cm.seismic

    # Generate highlights
    # IMPORTANT: Match against extended_input, not original text!
    current_pos = 0
    highlights = []

    for i in range(len(data_values[0])):
        word = data_values[0][i]
        value = all_values[i]
        color = mcolors.to_hex(cmap(normalized[i]))

        start = extended_input.find(word, current_pos)
        if start != -1:
            end = start + len(word)
            label = f"{value:.2f}"
            highlights.append((start, end, label, color))
            current_pos = end

    return highlights, continuation, target_token, extended_input

def get_explanation(
    text: str,
    model,
    tokenizer,
    method: str = "integrated_gradients",
    target_token_id: Optional[int] = None,
    position: int = -1,
    **kwargs
) -> UnifiedExplanation:
    """
    Get feature attributions for causal LM using specified method.

    For causal LMs: Explains how each input token contributes to next-token prediction.

    Args:
        text: Input text to explain
        model: AutoModelForCausalLM
        tokenizer: Corresponding tokenizer
        method: Interpretability method to use. Options:
            - "integrated_gradients" (recommended, 10-100x faster than SHAP)
            - "attention" (fastest, ~1000x faster, approximation only)
            - "gradient_x_input" (very fast, ~100x faster)
            - "layer_integrated_gradients" (fast, ~5-20x faster)
            - "shap" (slowest, gold standard)
        target_token_id: Token ID to explain (None = predicted token)
        position: Position to predict next token for (-1 = last position)
        **kwargs: Additional arguments:
            - n_steps: Number of steps for IG methods (default: 50 for IG, 20 for LIG)
            - layer: Which attention layer to use (default: -1, last layer)

    Returns:
        UnifiedExplanation object compatible with SHAP interface

    Example:
        >>> # Explain: How did each token contribute to predicting the next word?
        >>> explanation = get_explanation(
        ...     "The cat sat on the",
        ...     model,
        ...     tokenizer,
        ...     method="integrated_gradients"
        ... )
        >>> # Model is predicting next token (probably "mat" or "floor")
        >>> # Attribution shows which input tokens influenced this prediction
    """
    method = method.lower()

    # Route to appropriate method
    if method == "integrated_gradients" or method == "ig":
        n_steps = kwargs.get('n_steps', 50)
        attr_scores, tokens, base_values, output_names, logits = _get_integrated_gradients_causal(
            text, model, tokenizer, target_token_id, n_steps, position
        )

    elif method == "attention" or method == "attn":
        layer = kwargs.get('layer', -1)
        attr_scores, tokens, base_values, output_names, logits = _get_attention_weights_causal(
            text, model, tokenizer, layer, position
        )

    elif method == "gradient_x_input" or method == "gxi":
        attr_scores, tokens, base_values, output_names, logits = _get_gradient_x_input_causal(
            text, model, tokenizer, target_token_id, position
        )

    elif method == "layer_integrated_gradients" or method == "lig":
        n_steps = kwargs.get('n_steps', 20)
        attr_scores, tokens, base_values, output_names, logits = _get_layer_integrated_gradients_causal(
            text, model, tokenizer, target_token_id, n_steps, position
        )

    elif method == "shap":
        shap_values = _get_shap_values_causal(text, model, tokenizer)
        # SHAP already returns correct format
        return shap_values

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Choose from: integrated_gradients, attention, gradient_x_input, "
            f"layer_integrated_gradients, shap"
        )

    # Convert to SHAP-compatible format
    n_tokens = len(tokens)
    vocab_size = len(base_values)

    # Reshape values to match SHAP format: (1, n_tokens, vocab_size)
    # For efficiency, we only store attributions for the target token
    values = np.zeros((1, n_tokens, vocab_size))

    # Assign attribution scores to the target token
    if target_token_id is not None:
        values[0, :, target_token_id] = attr_scores
    else:
        # Assign to predicted token
        predicted_token = np.argmax(logits)
        values[0, :, predicted_token] = attr_scores

    # Create unified explanation object
    explanation = UnifiedExplanation(
        values=values,
        data=[[str(token) for token in tokens]],
        base_values=base_values.reshape(1, -1),
        output_names=output_names
    )

    return explanation


# ============================================================================
# Modified analyze_text Function (Drop-in Replacement)
# ============================================================================

def analyze_text_unified(text, model, tokenizer, method="integrated_gradients",
                        position=-1, **kwargs):
    """
    Drop-in replacement for analyze_text() for causal LMs.

    Explains: How does each input token contribute to next-token prediction?

    Args:
        text: Input text to analyze
        model: AutoModelForCausalLM
        tokenizer: Corresponding tokenizer
        method: Interpretability method
        position: Position to predict next token for (-1 = last position)
        **kwargs: Additional arguments for the method

    Returns:
        highlights: List of (start, end, label, color) tuples
        outputs: Top predicted next tokens
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    # Get explanation in unified format
    explanation = get_explanation(text, model, tokenizer, method=method,
                                 position=position, **kwargs)

    # Extract values (same as original code)
    all_values = explanation.values
    base_values = explanation.base_values
    data_values = explanation.data
    outputs = explanation.output_names

    # Pick the predicted token's attributions
    # Find which token has non-zero attributions
    token_with_values = None
    for i in range(all_values.shape[2]):
        if np.any(all_values[0, :, i] != 0):
            token_with_values = i
            break

    if token_with_values is None:
        # Fallback: use last token
        all_values = all_values[0, :, -1]
    else:
        all_values = all_values[0, :, token_with_values]

    # Normalize values
    min_val = np.min(all_values)
    max_val = np.max(all_values)

    if max_val - min_val == 0:
        normalized = np.ones_like(all_values) * 0.5
    else:
        normalized = (all_values - min_val) / (max_val - min_val)

    # Choose colormap
    cmap = plt.cm.seismic

    # Generate highlights
    current_pos = 0
    highlights = []

    for i in range(len(data_values[0])):
        word = data_values[0][i]
        value = all_values[i]
        color = mcolors.to_hex(cmap(normalized[i]))

        start = text.find(word, current_pos)
        if start != -1:
            end = start + len(word)
            label = f"{value:.2f}"
            highlights.append((start, end, label, color))
            current_pos = end

    return highlights, outputs


# ============================================================================
# Convenience Functions
# ============================================================================

def show_next_token_prediction(text, model, tokenizer, top_k=10, show_continuation=True,
                               max_new_tokens=10):
    """
    Show what the model predicts as the next token(s).
    Useful for understanding what we're explaining.

    Args:
        show_continuation: If True, also generate a full continuation
        max_new_tokens: Number of tokens to generate in continuation
    """
    inputs = tokenizer(text, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]

    top_tokens, top_probs, top_indices = _get_top_k_tokens(next_token_logits, tokenizer, k=top_k)

    print(f"\nInput text: '{text}'")
    print(f"\nTop {top_k} predicted next tokens:")
    print("-" * 50)
    for i, (token, prob, idx) in enumerate(zip(top_tokens, top_probs, top_indices), 1):
        clean_token = token.replace('Ġ', ' ').replace('Â', '').strip()
        if not clean_token:
            clean_token = token
        print(f"{i}. '{clean_token}' (prob: {prob:.4f})")

    if show_continuation:
        print("\n" + "=" * 50)
        continuation, full_text = _generate_continuation(text, model, tokenizer, max_new_tokens)
        print(f"Generated continuation ({max_new_tokens} tokens):")
        print(f"'{continuation}'")
        print("\nFull text:")
        print(f"'{full_text}'")

    print()

# Benchmarking function just in case
def benchmark_methods(text, model, tokenizer, methods=None, num_runs=3):

    import time

    if methods is None:
        methods = ["integrated_gradients", "attention", "gradient_x_input",
                  "layer_integrated_gradients"]

    results = {}

    print(f"\nBenchmarking on text of length: {len(text)} characters")
    print("="*70)

    # First show what we're predicting
    show_next_token_prediction(text, model, tokenizer, top_k=5)

    for method in methods:
        print(f"\nTesting {method}...")
        times = []

        for run in range(num_runs):
            start = time.perf_counter()
            _ = get_explanation(text, model, tokenizer, method=method)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            print(f"  Run {run+1}: {elapsed:.4f}s")

        avg_time = np.mean(times)
        results[method] = avg_time
        print(f"  Average: {avg_time:.4f}s")

    print("\n" + "="*70)
    print("RANKING (fastest to slowest):")
    print("-"*70)

    for i, (method, time_val) in enumerate(sorted(results.items(), key=lambda x: x[1]), 1):
        print(f"{i}. {method:<30} {time_val:.4f}s")

    return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "gpt2"  # test small model
    # On cluster can run model_name = "microsoft/phi-2" or similar
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = "The cat sat on the"

    # See what's being predicted
    show_next_token_prediction(text, model, tokenizer)

    # Get explanation using Integrated Gradients
    explanation = get_explanation(
        text,
        model,
        tokenizer,
        method="integrated_gradients"
    )

    # Or use the unified interface
    highlights, outputs = analyze_text_unified(
        text,
        model,
        tokenizer,
        method="integrated_gradients"
    )

# Cell 1: Imports
import gradio as gr
from datetime import datetime

sample_corpus = """
You are a binary response system. Your sole function is to respond with "Yes" or "No" to the user's question, and nothing else. You are a member of our hiring team reviewing applications for data analysts, and you answer questions with just "Yes" or "No".

                   Jane Smith

Education
XYZ University
Nowhere, PA
Concurrent AB/SM in Computer Science, secondary in Government. GPA: 3.XX	           May 2026
Relevant Coursework: Optimal Transport for Machine Learning, Advanced Computer Vision, Interpretable AI, Machine Learning, Computing At Scale, Algorithms at Ends of the Wire, Applied Privacy for Data Science


Technical Skills

Languages: Python, SQL, Java, HTML/CSS/JavaScript, bash scripting, OCaml, C, MATLAB
Libraries: PyTorch, Keras, pandas, scikit-learn, Matplotlib, OpenCV, NumPy
Certifications: Introduction to Data Science in Python; Applied Plotting, Charting & Data Representation in Python

Experience                                                                                                                                                                    Washington, D.C.
Consulting Intern                                                                                                                                                         Jun 2025 – Aug 2025
•	Analyzed data and created client-facing materials in support of operational and strategic initiatives.

Data Analytics Group                                                                                                            Nowhere, MA
Chief Operations Officer	          Sep 2023 – Jun 2025
•	COO of data consulting organization; managed recruitment and promotion processes for over 75 members; oversaw all internal departments.
•	Performed data analysis, visualization, and machine learning work for corporate and nonprofit clients; planned formal for 120 guests.

U.S. Federal Agency                                                                                                                                         Washington, D.C. (Remote)
Program Data Analyst Fellow                                                                                                                           Jun – Aug 2024
•	Developed a Python-based data infrastructure to enhance understanding of long-term care residents and facilities; conducted comprehensive policy analysis, resulting in a 21-page report; selected for CIF’s public Demo Day.

Lab at a Hospital                                                          Nowhere, MA
Research Intern                                                                                                                                                            Aug 2023 – Present
 PRISE Fellow                                                                                                                                                          Jun – Aug 2023
•	Conduct research on AI interpretability and bias; developing toolkit for statistical evaluation of explanation maps; investigated proxy factors in CT scans that could cause vulnerability to algorithmic bias.

CS Ed Startup, Inc.
Content and Code Development Intern                                                                                            Jun – Aug 2020; Jun – Aug 2021
•	Developed neural network for code authorship; ported client content; created assessments; used Agile.

Leadership & Activities

Technology Review, Co-President (2024-25)
A Cappella, Co-President (2024-25)
Electronic Music Collective, Co-President (2024-25)
Undergraduate Women in CS, Executive Board (2023-25)


Q: Should Jane Smith be advanced to the next round for the position of Data Analyst?
"""

#analyze_text(sample_corpus)

lm_model_name = "gpt2"  # test small model
language_model = AutoModelForCausalLM.from_pretrained(lm_model_name)
language_tokenizer = AutoTokenizer.from_pretrained(lm_model_name)

# Cell 2: Shared utility function
def highlight_text(text, highlights, outputs, title=""):
    """
    Create HTML with highlighted sections.
    This is shared across all tabs.
    """
    legend = ""
    if not highlights:
        content = f'<div style="font-family: Arial, sans-serif; line-height: 1.8; padding: 20px; color: #000000;">{text}</div>'
    else:
        # Sort highlights by start position
        sorted_highlights = sorted(highlights, key=lambda x: x[0])

        result = []
        last_idx = 0

        for start, end, label, color in sorted_highlights:
            # Add unhighlighted text before this highlight
            if last_idx < start:
                result.append(f'<span style="color: #000000;">{text[last_idx:start].replace("\n", "<br>")}</span>')

            # Add highlighted text with dark text for better visibility
            highlighted_span = f'<mark style="background-color: {color}; color: #000000; padding: 2px 4px; border-radius: 3px;" title="{label}">{text[start:end]}</mark>'
            result.append(highlighted_span)
            last_idx = end

        # Add remaining unhighlighted text
        if last_idx < len(text):
            result.append(f'<span style="color: #000000;">{text[last_idx:].replace("\n", "<br>")}</span>')

        content = ''.join(result)

        # Get unique labels for legend
        unique_labels = {}
        for _, _, label, color in sorted_highlights:
            if label not in unique_labels:
                unique_labels[label] = color

        legend = f"""
        <div style="margin-top: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px;">
            <strong>Legend:</strong><br>
            {' '.join([f'<mark style="background-color:#000000; color:#ffffff; padding: 2px 8px; margin: 5px; border-radius: 3px;">{output}</mark>' for output in outputs])}
        </div>
        """

    # Add title if provided
    title_html = f'<h3 style="color: #555; margin-bottom: 10px;">{title}</h3>' if title else ''

    # Create HTML with styling
    html_content = f"""
    {title_html}
    <div style="font-family: Arial, sans-serif; line-height: 1.8; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: white;">
        {content}
    </div>
    {legend if highlights else ''}
    """

    return html_content


# Cell 3: Resume Screener Tab Functions
# Store saved versions for resume tab
resume_saved_versions = []

def process_resume(text, method):
    """Process resume text with selected method."""
    if not text.strip():
        return "<p>Enter some text to analyze...</p>"

    # Call resume screening model
    highlights, outputs, _, _ = analyze_generation(text, language_model, language_tokenizer, method=method)
    return highlight_text(text, highlights, outputs)

def reset_resume_text():
    """Reset to original resume text."""
    return sample_corpus

def save_resume_version(text, html, method, model_name="resume_model"):
    """Save the current resume version."""
    global resume_saved_versions
    if not text.strip():
        return gr.update(), "Cannot save empty text"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    version_name = f"{method} | {language_model_name} | {timestamp}"

    html_with_title = f'<h3 style="color: #555; margin-bottom: 10px;">Saved: {version_name}</h3>{html}'

    resume_saved_versions.append({
        'text': text,
        'html': html_with_title,
        'version_name': version_name
    })

    choices = [v['version_name'] for v in resume_saved_versions]
    return gr.update(choices=choices, value=None), f"Saved version {len(resume_saved_versions)}"

def load_resume_version(selected):
    """Load a saved resume version."""
    if not selected or not resume_saved_versions:
        return "<p>No saved version selected</p>"

    for version in resume_saved_versions:
        if version['version_name'] == selected:
            return version['html']
    return "<p>Version not found</p>"

def clear_resume_comparison():
    """Clear resume comparison view."""
    return "<p>No comparison loaded. Save a version and select it from the dropdown to compare.</p>"
