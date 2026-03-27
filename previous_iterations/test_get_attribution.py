"""
test_get_attribution.py
-----------------------
Smoke-test for the generalised get_attribution() function.
Loads EleutherAI/gpt-neo-125M, runs every supported Inseq/Captum method,
prints a formatted attribution table for each one, and exits non-zero if
any method fails.

Requirements:
    pip install torch transformers inseq captum

Run:
    python test_get_attribution.py
    python test_get_attribution.py --model gpt2 --text "The quick brown fox"
"""

import argparse
import sys
import traceback
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import inseq
from transformers import AutoModelForCausalLM, AutoTokenizer

# ══════════════════════════════════════════════════════════════════════════════
# ANSI helpers  (degrade gracefully if the terminal doesn't support colour)
# ══════════════════════════════════════════════════════════════════════════════

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"

_BAR_CHARS = " ▏▎▍▌▋▊▉█"   # 8 levels for the sparkline


# ══════════════════════════════════════════════════════════════════════════════
# Shared data contract  (copy of the class used in resume_utility.py)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class UnifiedExplanation:
    """SHAP-compatible output structure used throughout the pipeline."""
    values: np.ndarray       # (1, n_tokens, vocab_size)
    data: List[List[str]]    # (1, n_tokens)  — token strings
    base_values: np.ndarray  # (1, vocab_size) — baseline prediction
    output_names: List[str]  # top-k predicted next tokens

    def __getitem__(self, key):
        if key == 0:
            return self
        raise IndexError(f"Index {key} out of range")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers  (same signatures as resume_utility.py)
# ══════════════════════════════════════════════════════════════════════════════

def _get_top_k_tokens(logits: torch.Tensor, tokenizer, k: int = 10):
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=k)
    top_tokens = [tokenizer.decode([i]) for i in top_indices]
    return top_tokens, top_probs.cpu().numpy(), top_indices.cpu().numpy()


def _ensure_tokens_for_inseq(model, tokenizer):
    """Guarantee bos/eos/pad tokens exist on both tokenizer and model.config."""
    vocab = tokenizer.get_vocab()
    default_id = None
    for tok in ["<|endoftext|>", "</s>", "<eos>", "<|end_of_text|>"]:
        if tok in vocab:
            default_id = vocab[tok]
            break
    if default_id is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
        default_id = tokenizer.eos_token_id

    for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
        if getattr(tokenizer, attr) is None:
            setattr(tokenizer, attr, default_id)

    if hasattr(model, "config"):
        for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
            if getattr(model.config, attr, None) is None:
                setattr(model.config, attr, getattr(tokenizer, attr))

    for id_attr, str_attr in [
        ("eos_token_id", "eos_token"),
        ("bos_token_id", "bos_token"),
        ("pad_token_id", "pad_token"),
    ]:
        if getattr(tokenizer, str_attr) is None:
            setattr(tokenizer, str_attr,
                    tokenizer.convert_ids_to_tokens(getattr(tokenizer, id_attr)))
    return tokenizer


# ══════════════════════════════════════════════════════════════════════════════
# Method registry
# ══════════════════════════════════════════════════════════════════════════════

INSEQ_METHOD_REGISTRY: dict = {
    # ── gradient-based ────────────────────────────────────────────────────────
    "integrated_gradients":            {"needs_baseline": True,  "needs_n_steps": True},
    "layer_integrated_gradients":      {"needs_baseline": True,  "needs_n_steps": True},
    "sequential_integrated_gradients": {"needs_baseline": True,  "needs_n_steps": True},
    "saliency":                        {"needs_baseline": False, "needs_n_steps": False},
    "input_x_gradient":                {"needs_baseline": False, "needs_n_steps": False},
    "gradient_shap":                   {"needs_baseline": True,  "needs_n_steps": False},
    "deep_lift":                       {"needs_baseline": True,  "needs_n_steps": False},
    # ── attention ─────────────────────────────────────────────────────────────
    "attention":                       {"needs_baseline": False, "needs_n_steps": False},
}

# Short-hand aliases accepted by get_attribution()
_METHOD_ALIASES: dict = {
    "ig":   "integrated_gradients",
    "lig":  "layer_integrated_gradients",
    "sig":  "sequential_integrated_gradients",
    "gxi":  "input_x_gradient",
    "grad": "saliency",
    "attn": "attention",
    "shap": "gradient_shap",
    "dl":   "deep_lift",
}


# ══════════════════════════════════════════════════════════════════════════════
# Core function under test
# ══════════════════════════════════════════════════════════════════════════════

def get_attribution(
    text: str,
    model,
    tokenizer,
    inseq_model,
    method: str = "integrated_gradients",
    target_token_id: Optional[int] = None,
    position: int = -1,
    n_steps: int = 50,
    baselines: Optional[torch.Tensor] = None,
    stdevs: float = 0.0,
    multiply_by_inputs: Optional[bool] = None,
    **extra_attribution_args,
) -> UnifiedExplanation:
    """
    Generalised attribution function compatible with every Inseq/Captum method.

    Replaces all per-method helpers (_get_integrated_gradients_causal, etc.).
    Adding support for a new Captum method requires only a new entry in
    INSEQ_METHOD_REGISTRY — no new wrapper function.

    Args:
        text:                  Input text to attribute.
        model:                 HuggingFace AutoModelForCausalLM.
        tokenizer:             Corresponding tokenizer.
        inseq_model:           Pre-initialised inseq AttributionModel.
        method:                Any key in INSEQ_METHOD_REGISTRY or _METHOD_ALIASES.
        target_token_id:       Token to explain. None → greedy argmax.
        position:              Sequence position to predict from (-1 = last).
        n_steps:               Integration steps (IG / LIG / SIG only).
        baselines:             Baseline tensor. None → Inseq default (zeros).
        stdevs:                Noise std-dev for GradientSHAP.
        multiply_by_inputs:    Override the method's multiply_by_inputs flag.
        **extra_attribution_args:
                               Any additional kwargs forwarded verbatim into
                               Inseq's attribution_args dict.

    Returns:
        UnifiedExplanation — same contract as the old per-method helpers.
    """
    # 1. Resolve alias → canonical method name
    canonical_method = _METHOD_ALIASES.get(method.lower(), method.lower())
    method_meta = INSEQ_METHOD_REGISTRY.get(canonical_method, {})

    # 2. Forward pass to determine target token and metadata
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        logits = model(**inputs).logits
        next_token_logits = logits[0, position, :]   # (vocab_size,)

    if target_token_id is None:
        target_token_id = int(next_token_logits.argmax())

    # 3. Build kwargs for inseq_model.attribute()
    #
    #    Inseq splits method configuration into two distinct namespaces:
    #
    #      • Top-level kwargs on .attribute()  — things Inseq itself consumes,
    #        such as n_steps (controls the integration loop) and n_target_steps.
    #
    #      • attribution_args dict             — passed verbatim to the
    #        underlying Captum method (baselines, stdevs, internal_batch_size…).
    #
    #    Putting n_steps inside attribution_args causes Captum to receive an
    #    unexpected keyword and either silently ignore it or crash mid-run.

    # Top-level Inseq kwargs
    inseq_kwargs: dict = {"show_progress": False, "method": canonical_method}

    if method_meta.get("needs_n_steps"):
        inseq_kwargs["n_steps"] = n_steps

    # attribution_args — Captum-level kwargs only
    attribution_args: dict = {}

    if method_meta.get("needs_baseline") and baselines is not None:
        attribution_args["baselines"] = baselines

    if stdevs and canonical_method == "gradient_shap":
        attribution_args["stdevs"] = stdevs

    if multiply_by_inputs is not None:
        attribution_args["multiply_by_inputs"] = multiply_by_inputs

    # Forward any extra caller-supplied kwargs straight to Captum
    attribution_args.update(extra_attribution_args)

    if attribution_args:
        inseq_kwargs["attribution_args"] = attribution_args

    # 4. Run attribution via Inseq
    #
    #    Without generated_texts, Inseq free-generates up to max_new_tokens
    #    continuation tokens and attributes all of them.  The prompt tokens in
    #    those extra steps receive NaN attributions because there is no gradient
    #    signal at padding positions.
    #
    #    Fix: pass generated_texts = input + target_token so Inseq attributes
    #    exactly ONE generation step (the token we already chose above), giving
    #    clean finite scores for every prompt token and nothing else.
    target_token_str = tokenizer.decode([target_token_id])
    out = inseq_model.attribute(
        input_texts=text,
        generated_texts=text + target_token_str,
        **inseq_kwargs,
    )

    # 5. Extract per-token scores from Inseq output
    #
    #    Inseq uses different attribution fields depending on architecture:
    #
    #      • Encoder-decoder (e.g. T5, BART):
    #          seq_out.source_attributions  → scores over encoder (source) tokens
    #          seq_out.target_attributions  → scores over decoder (target) tokens
    #
    #      • Decoder-only (e.g. GPT-2, GPT-Neo, LLaMA):
    #          seq_out.source_attributions  → None  (no separate encoder)
    #          seq_out.target_attributions  → scores over all prompt tokens
    #
    #    We therefore prefer target_attributions and fall back to
    #    source_attributions for encoder-decoder models.
    seq_out = out[0]

    raw_attr = seq_out.target_attributions
    if raw_attr is None:
        raw_attr = seq_out.source_attributions
    if raw_attr is None:
        raise ValueError(
            "Both source_attributions and target_attributions are None. "
            "Check that the Inseq method supports this model architecture."
        )

    attr_array = raw_attr.detach().cpu().numpy()

    # Token strings — for decoder-only models seq_out.target is the full
    # sequence: prompt tokens + the one generated token.  Drop the generated
    # token from the token list; the attr_array strip is handled per-method
    # below because attention needs the full tensor before reducing.
    token_sequence = seq_out.target if seq_out.target_attributions is not None else seq_out.source
    tokens = [str(t.token) for t in token_sequence][:-1]   # drop generated token

    # Collapse attr_array to a 1-D vector of length n_prompt_tokens.
    #
    #   Gradient methods (IG, saliency, GxI, GradSHAP):
    #     Inseq returns shape (n_prompt_tokens + 1, n_target_steps[, hidden_dim])
    #     where axis-0 = full sequence including the generated token.
    #     We strip the last row with [:-1] then reduce.
    #
    #       2-D (seq, target_steps)              — scalar methods (saliency, GxI)
    #       3-D (seq, target_steps, hidden_dim)  — scalar but with hidden axis (IG)
    #
    #   Layer-IG is different: Inseq returns (n_target_steps, n_prompt_tokens, hidden_dim).
    #   Axes 0 and 1 are swapped relative to the other gradient methods.
    #   We transpose first, then apply the same reduction.
    #
    #   Attention returns a 4-D tensor (seq, target_steps, n_heads, seq).
    #   Both seq axes span the *full* sequence (prompt + generated token).
    #   We must NOT pre-strip attr_array here because:
    #     - we need the last query row (index -1) which IS the generated token position
    #       — that row holds the attention weights FROM the generated token TO each
    #       context token, which is exactly the signal we want
    #     - the keys axis (last) also has length n_prompt+1; we strip that after
    #   So for attention we extract, then strip the last key position.

    # ── DEBUG: print raw tensor layout for methods that are still failing ──────
    # Remove once LIG and attention are confirmed working.
    if canonical_method in ("layer_integrated_gradients", "attention"):
        print(f"\n  [DEBUG] canonical_method = {canonical_method!r}")
        print(f"  [DEBUG] attr_array.shape  = {attr_array.shape}")
        print(f"  [DEBUG] attr_array.dtype  = {attr_array.dtype}")
        flat = attr_array.ravel()
        finite = flat[np.isfinite(flat)]
        print(f"  [DEBUG] finite elements   = {len(finite)} / {len(flat)}")
        if len(finite):
            print(f"  [DEBUG] finite range      = [{finite.min():.6f}, {finite.max():.6f}]")
        if attr_array.ndim == 4:
            # Show per-layer NaN count for the last-query row
            for li in range(attr_array.shape[0]):
                row = attr_array[li, 0, -1, :-1]
                nan_count = np.sum(~np.isfinite(row))
                print(f"  [DEBUG] layer {li:2d} last-query row: "
                      f"nan={nan_count}/{len(row)}  "
                      f"range=[{np.nanmin(row):.4f}, {np.nanmax(row):.4f}]")
    # ─────────────────────────────────────────────────────────────────────────

    if attr_array.ndim == 4:
        # Attention: Inseq returns (n_layers, target_steps, seq, seq)
        #   - axis 0: transformer layers
        #   - axis 1: target steps (1 because we pinned generated_texts)
        #   - axis 2: query positions (full seq including generated token)
        #   - axis 3: key positions   (full seq including generated token)
        #
        # Use nanmean so that any NaN-containing layers don't poison the average.
        # Take the last query row (generated token → context attention) and strip
        # the final key column (generated token attending to itself).
        token_scores = np.nanmean(attr_array[:, 0, -1, :-1], axis=0)
    elif attr_array.ndim == 3:
        # Shape is (seq, target_steps, hidden_dim) for both IG-family and LIG.
        # For LIG the hidden scores may be tiny (near-zero) but are real values.
        token_scores = attr_array[:-1, 0, :].sum(axis=-1)
    elif attr_array.ndim == 2:
        # scalar gradient: (seq, target_steps)
        token_scores = attr_array[:-1, 0]
    else:
        token_scores = attr_array[:-1]

    # 6. Pack into UnifiedExplanation
    #
    #    We store token_scores in two places:
    #      a) values[0, :, target_token_id]  — sparse vocab-dim array for
    #         downstream SHAP-compatible consumers that index by vocab column
    #      b) values[0, :, 0]               — dense column 0 as a reliable
    #         read-back path inside this script (avoids index mismatch if
    #         target_token_id shifts between the forward pass and the lookup)
    n_tokens   = len(tokens)
    base_probs = torch.softmax(next_token_logits, dim=-1).cpu().numpy()
    vocab_size = base_probs.shape[0]

    values = np.zeros((1, n_tokens, vocab_size), dtype=np.float32)
    values[0, :, target_token_id] = token_scores   # canonical SHAP-compat column
    values[0, :, 0]               = token_scores   # dense copy at col-0 for easy readback

    top_tokens, _, _ = _get_top_k_tokens(next_token_logits, tokenizer)

    return UnifiedExplanation(
        values=values,
        data=[tokens],
        base_values=base_probs.reshape(1, -1),
        output_names=top_tokens,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Pretty-printer
# ══════════════════════════════════════════════════════════════════════════════

def _bar(value: float, width: int = 18) -> str:
    """Signed Unicode bar centred at zero."""
    clamped = max(-1.0, min(1.0, value))
    half    = width // 2
    filled  = min(int(abs(clamped) * half), half)
    if value >= 0:
        bar = " " * half + _GREEN + ("█" * filled).ljust(half) + _RESET
    else:
        bar = _RED + ("█" * filled).rjust(half) + _RESET + " " * half
    return f"[{bar}]"


def _sparkline(scores: np.ndarray, width: int = 40) -> str:
    """One-line Unicode sparkline summarising the full attribution vector."""
    # Guard: drop NaNs before computing range so one bad value doesn't
    # poison the entire sparkline.
    clean = scores[np.isfinite(scores)]
    if len(clean) == 0:
        return "  (all NaN — no finite attribution scores)"
    lo, hi = clean.min(), clean.max()
    span   = hi - lo if hi != lo else 1.0
    step   = max(1, len(scores) // width)
    chars  = []
    for i in range(0, len(scores), step):
        bucket = scores[i : i + step]
        mean   = float(np.nanmean(bucket))
        if not np.isfinite(mean):
            chars.append("?")
            continue
        norm = (mean - lo) / span
        idx  = int(norm * (len(_BAR_CHARS) - 1))
        chars.append(_BAR_CHARS[idx])
    return "".join(chars)


def print_attribution_table(
    explanation: UnifiedExplanation,
    method_name: str,
    target_token: str,
) -> None:
    """
    Print a formatted attribution table, e.g.:

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  METHOD  integrated_gradients            TARGET → ' Yes'            ║
    ╚══════════════════════════════════════════════════════════════════════╝

      #   Token                  Score     Bar  (neg ◄── 0 ──► pos)
     ──────────────────────────────────────────────────────────────────────
      1    The                  +0.3821   [          ███████             ]
      2    hiring               -0.1043   [   ███                       ]
      ...

     Sparkline : ▁▂▄▆█▇▄▂▁
     Stats     : min=-0.2103  max=+0.3821  mean=+0.0412  std=0.1837
     Top-5 next: Yes | No | Absolutely | Sure | Perhaps
    """
    tokens = explanation.data[0]

    # Col-0 holds the dense copy of token_scores written by get_attribution.
    # This is more reliable than searching for the non-zero vocab column because
    # target_token_id=0 would otherwise be ambiguous.
    scores      = explanation.values[0, :, 0].astype(float)
    finite_mask = np.isfinite(scores)
    abs_max     = np.abs(scores[finite_mask]).max() if finite_mask.any() else 1.0
    norm_scores = np.where(finite_mask, scores / abs_max if abs_max > 0 else scores, 0.0)

    # ── header ────────────────────────────────────────────────────────────────
    W = 72
    print(f"\n{_BOLD}╔{'═' * W}╗{_RESET}")
    left  = f"  METHOD  {_CYAN}{method_name}{_RESET}"
    right = f"TARGET → {_YELLOW}{repr(target_token)}{_RESET}"
    # measure printable length (strip ANSI for padding calc)
    raw   = f"  METHOD  {method_name}  TARGET → {repr(target_token)}"
    gap   = W - len(raw) - 2
    print(f"{_BOLD}║{_RESET}  {left}{'  ' + ' ' * max(gap, 0)}{right}  {_BOLD}║{_RESET}")
    print(f"{_BOLD}╚{'═' * W}╝{_RESET}")

    # ── column headers ────────────────────────────────────────────────────────
    print(f"\n  {_DIM}{'#':>4}   {'Token':<22}  {'Score':>9}   "
          f"Bar  (neg ◄── 0 ──► pos){_RESET}")
    print(f"  {'─' * (W - 2)}")

    # ── one row per token ─────────────────────────────────────────────────────
    for i, (tok, raw_score, norm) in enumerate(zip(tokens, scores, norm_scores), 1):
        display = (tok
                   .replace("Ġ", " ")
                   .replace("Ċ", "↵")
                   .replace("▁", " "))
        sign    = "+" if raw_score >= 0 else ""
        score_s = f"{sign}{raw_score:.4f}" if np.isfinite(raw_score) else "   NaN"

        # Bold the token with the largest absolute attribution
        emphasis = _BOLD if np.isfinite(raw_score) and abs(raw_score) == abs_max else ""
        print(f"  {emphasis}{i:>4}   {display:<22}  {score_s:>9}   {_bar(norm)}{_RESET}")

    # ── footer ────────────────────────────────────────────────────────────────
    print(f"  {'─' * (W - 2)}")
    print(f"\n  {_DIM}Sparkline :{_RESET}  {_sparkline(scores)}")
    fin = scores[np.isfinite(scores)]
    stats_str = (
        f"min={np.nanmin(fin):+.4f}  max={np.nanmax(fin):+.4f}  "
        f"mean={np.nanmean(fin):+.4f}  std={np.nanstd(fin):.4f}"
        if len(fin) else "  (no finite scores)"
    )
    print(f"  {_DIM}Stats     :{_RESET}  {stats_str}")
    top5 = " | ".join(
        t.replace("Ġ", " ").replace("▁", " ").strip()
        for t in explanation.output_names[:5]
    )
    print(f"  {_DIM}Top-5 next:{_RESET}  {_YELLOW}{top5}{_RESET}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Test harness
# ══════════════════════════════════════════════════════════════════════════════

# Methods to exercise — extend this list to test new registry entries
METHODS_TO_TEST = [
    # (method_name,               extra_kwargs)
    ("integrated_gradients",      {"n_steps": 20}),
    ("saliency",                  {}),
    ("input_x_gradient",          {}),
    ("layer_integrated_gradients",{"n_steps": 20}),
    ("gradient_shap",             {"stdevs": 0.1}),
    ("attention",                 {}),
    # Aliases — each should resolve to its canonical method
    ("ig",                        {"n_steps": 10}),
    ("gxi",                       {}),
    ("attn",                      {}),
]


def run_attribution_tests(
    text: str = (
        "The hiring committee reviewed the application and decided the candidate was"
    ),
    model_name: str = "EleutherAI/gpt-neo-125M",
) -> tuple:
    """
    Load model, run every method in METHODS_TO_TEST, print tables.

    Returns:
        results  : dict[method_name -> UnifiedExplanation]   (passed)
        failures : dict[method_name -> error_string]          (failed)
    """
    # ── load ──────────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{'═' * 74}{_RESET}")
    print(f"{_BOLD}  Loading model: {model_name}{_RESET}")
    print(f"{'═' * 74}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = _ensure_tokens_for_inseq(model, tokenizer)
    model.eval()

    # Single inseq_model instance — method= is overridden per .attribute() call
    inseq_model = inseq.load_model(model, "integrated_gradients", tokenizer=tokenizer)
    print(f"{_GREEN}✓ Model loaded{_RESET}\n")

    # Determine the greedy next token once so every method explains the same target
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        logits = model(**inputs).logits[0, -1, :]
    target_id  = int(logits.argmax())
    target_tok = tokenizer.decode([target_id])

    print(f"  Input : {_CYAN}{repr(text)}{_RESET}")
    print(f"  Target: {_YELLOW}{repr(target_tok)}{_RESET}  (token id {target_id})\n")

    results:  dict = {}
    failures: dict = {}

    # ── run each method ───────────────────────────────────────────────────────
    for method_name, kwargs in METHODS_TO_TEST:
        print(f"{_DIM}{'─' * 74}{_RESET}")
        print(f"  Running  {_BOLD}{method_name}{_RESET} …", end="", flush=True)
        try:
            expl = get_attribution(
                text, model, tokenizer, inseq_model,
                method=method_name,
                target_token_id=target_id,
                **kwargs,
            )

            # ── sanity assertions ─────────────────────────────────────────────
            assert expl.values.shape[0] == 1, \
                f"batch dim should be 1, got {expl.values.shape[0]}"
            assert expl.values.shape[2] == logits.shape[0], \
                f"vocab dim mismatch: {expl.values.shape[2]} vs {logits.shape[0]}"
            assert len(expl.data[0]) == expl.values.shape[1], \
                f"token count mismatch: {len(expl.data[0])} vs {expl.values.shape[1]}"

            # Soft check: warn if all finite scores are zero (can happen for
            # layer-level methods on short prompts/small models), but only hard-
            # fail if every single value is NaN (method produced no signal at all).
            scores  = expl.values[0, :, 0]
            finite  = scores[np.isfinite(scores)]
            all_nan = len(finite) == 0
            all_zero_finite = len(finite) > 0 and np.all(finite == 0)
            assert not all_nan, \
                "all attribution scores are NaN — method produced no signal"
            if all_zero_finite:
                print(f"  {_YELLOW}⚠ warning:{_RESET} all finite scores are zero "
                      f"(expected for layer methods on short prompts)")
            max_str = f"{np.abs(finite).max():.4f}" if len(finite) else "NaN"
            print(f"  {_GREEN}✓ passed{_RESET}  "
                  f"shape={expl.values.shape}  "
                  f"n_tokens={len(expl.data[0])}  "
                  f"|scores|_max={max_str}")

            print_attribution_table(expl, method_name, target_tok)
            results[method_name] = expl

        except Exception as exc:
            print(f"  {_RED}✗ FAILED{_RESET}")
            traceback.print_exc()
            failures[method_name] = str(exc)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{_BOLD}{'═' * 74}{_RESET}")
    print(f"{_BOLD}  SUMMARY  —  {len(METHODS_TO_TEST)} methods tested  "
          f"({len(results)} passed, {len(failures)} failed){_RESET}")
    print(f"{'═' * 74}")

    for method_name, _ in METHODS_TO_TEST:
        if method_name in results:
            s      = results[method_name].values[0, :, 0]   # dense col-0 copy
            finite = s[np.isfinite(s)]
            if len(finite):
                print(f"  {_GREEN}✓{_RESET}  {method_name:<40} "
                      f"max={np.nanmax(finite):+.4f}  min={np.nanmin(finite):+.4f}")
            else:
                print(f"  {_GREEN}✓{_RESET}  {method_name:<40} (all NaN)")
        else:
            msg = failures.get(method_name, "unknown error")
            print(f"  {_RED}✗{_RESET}  {method_name:<40} {msg}")

    print(f"{'═' * 74}\n")
    return results, failures


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test get_attribution() across all methods.")
    parser.add_argument(
        "--model", default="EleutherAI/gpt-neo-125M",
        help="HuggingFace model name (default: EleutherAI/gpt-neo-125M)"
    )
    parser.add_argument(
        "--text",
        default="The hiring committee reviewed the application and decided the candidate was",
        help="Input text to attribute",
    )
    args = parser.parse_args()

    _, failures = run_attribution_tests(text=args.text, model_name=args.model)
    sys.exit(len(failures))   # non-zero exit if any method failed
