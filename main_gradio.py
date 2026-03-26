import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import gradio as gr
import json

# Helper functions for version changes and exports
def on_resume_version_change(selected, batch_enabled, batch_results):
    show_selected = gr.update(visible=bool(selected))
    show_csv = gr.update(visible=(batch_enabled and bool(batch_results)))
    from resume_utility import load_resume_version
    return load_resume_version(selected), show_selected, show_csv

def on_img_version_change(selected, batch_enabled, res1, res2):
    show_selected = gr.update(visible=bool(selected))
    show_csv = gr.update(visible=(batch_enabled and (bool(res1) or bool(res2))))
    from image_utility import load_image_version
    return load_image_version(selected), show_selected, show_csv
from resume_utility import (
    sample_corpus,
    process_resume,
    process_batch_resume,
    generate_nl_variations_code,
    explain_resume,
    explain_batch_variation,
    reset_resume_text,
    save_resume_version,
    load_resume_version,
    clear_resume_comparison,
    initialize_model,
    initialize_llama_model,
    llama_model_name,
    get_suggested_anchor,
    GENDER_PRESET_TEMPLATE,
    CUSTOM_EXTENDED_TEMPLATE,
    validate_and_load_resume_model,  # NEW — HF URL loader
    run_general_lm,                  # NEW — general LM runner
)
from image_utility import (
    blip_generate_caption_only,
    blip_analyze_image,
    blip_occlude_then_analyze,
    blip_batch_caption_images,
    save_image_version,
    load_image_version,
    clear_image_comparison,
    get_image_version_choices,
    switch_image_model,    
    validate_and_load_image_model,   # NEW — HF URL loader
)
from credit_utility import (
    sample_credit_data,
    predict_credit_risk,
    get_feature_importance,
    compare_scenarios,
    reset_credit_data,
    validate_and_load_credit_model,
    save_credit_version,
    export_selected_credit_html,
    export_all_credit_html,
    switch_credit_model,
)
from settings_manager import export_settings, import_settings

global has_explanation
has_explanation = True


theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Open Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
)

# NEW: Function to split sample_corpus into three parts
def split_resume_corpus(corpus_text):
    """
    Split the sample corpus into lead prompt, main body, and end prompt.
    This is a heuristic split - adjust the logic based on your actual corpus structure.
    """
    lines = corpus_text.strip().split('\n')
    
    # Example heuristic: 
    # - Lead prompt: first 2 lines (or lines before "Education:" or similar section headers)
    # - Main body: middle section with actual resume content
    # - End prompt: last line or two (instructions to model)
    
    # Simple approach: split by looking for key markers
    # Adjust these indices/logic based on your actual sample_corpus structure
    
    # For now, using a simple split:
    # Lead: first 10% of lines
    # Body: middle 80% of lines
    # End: last 10% of lines
    
    total_lines = len(lines)
    lead_end = max(1, total_lines // 10)
    body_start = lead_end
    body_end = total_lines - max(1, total_lines // 10)
    
    lead_prompt = '\n'.join(lines[:lead_end])
    main_body = '\n'.join(lines[body_start:body_end])
    end_prompt = '\n'.join(lines[body_end:])
    
    return lead_prompt, main_body, end_prompt


# NEW: Function to combine the three parts back into full text
def combine_resume_parts(lead, body, end):
    """Combine the three parts back into a single resume text."""
    parts = []
    if lead.strip():
        parts.append(lead.strip())
    if body.strip():
        parts.append(body.strip())
    if end.strip():
        parts.append(end.strip())
    return '\n'.join(parts)


# NEW: Functions to revert individual sections
def revert_lead_prompt():
    """Revert lead prompt to original."""
    lead, _, _ = split_resume_corpus(sample_corpus)
    return lead

def revert_main_body():
    """Revert main body to original."""
    _, body, _ = split_resume_corpus(sample_corpus)
    return body

def revert_end_prompt():
    """Revert end prompt to original."""
    _, _, end = split_resume_corpus(sample_corpus)
    return end


# NEW: Modified process_resume wrapper to handle three inputs and batch analysis
def process_resume_split(lead, body, end, method, temp, batch_enabled, batch_token, batch_num_vars, batch_dimension, variations_code, progress=gr.Progress()):
    """Process resume from three separate text inputs, with optional batch analysis."""
    full_text = combine_resume_parts(lead, body, end)
    
    if batch_enabled:
        # Call batch processing function with Faker parameters or custom code
        # NEW: returns (html, batch_results, None, model_display)
        html, results, _, display = process_batch_resume(full_text, method, temp, batch_token, batch_num_vars, batch_dimension, variations_code, progress=progress)
        return html, results, None, None, display, gr.update(visible=False), gr.update(visible=False), gr.update(visible=True, open=True), gr.update(maximum=len(results)-1 if results else 0)
    else:
        # Call regular processing function
        # returns (html, continuation, full_text, model_display)
        html, cont, full, display = process_resume(full_text, method, temp)
        return html, [], cont, full, display, gr.update(visible=True, interactive=True), gr.update(visible=True), gr.update(visible=False), gr.update(maximum=0)


# NEW: Modified explain_resume wrapper to handle three inputs
def explain_resume_split(lead, body, end, continuation_state, fulltext_state, method, rank_order_json):
    """Explain resume from three separate text inputs."""
    full_text = combine_resume_parts(lead, body, end)
    import json
    try:
        rank_order = json.loads(rank_order_json)
    except:
        rank_order = ["Fidelity", "Simplicity", "Robustness"]
    # Always use calibrated mode when a rank order is present — triggers ROC-weighted method selection
    effective_method = "calibrated" if rank_order else method
    return explain_resume(full_text, continuation_state, fulltext_state, effective_method, rank_order=rank_order)


def create_legend():
    fig, ax = plt.subplots(figsize=(8, 1.5))
    
    # Create colorbar
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=0, vmax=1)
    
    # Create colorbar
    cb = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation='horizontal'
    )
    
    # Set labels
    cb.set_label('Impact on Response', fontsize=12)
    cb.ax.set_xticks([0, 1])
    cb.ax.set_xticklabels(['More Negative', 'More Positive']) 
    
    plt.tight_layout()
    return fig


# Initialize the split parts from sample_corpus
initial_lead, initial_body, initial_end = split_resume_corpus(sample_corpus)


# Cell 5: Create the Gradio Interface
custom_css = """
/* Calibration Widget Styles */
#sortable-list {
    list-style-type: none;
    padding: 0;
}
.sortable-item {
    background-color: #f1f1f1 !important;
    border: 1px solid #ccc;
    padding: 10px;
    margin-bottom: 5px;
    cursor: move;
    border-radius: 4px;
    color: #333 !important;
    font-weight: 500;
}
.sortable-item:hover {
    background-color: #e1e1e1 !important;
}
.sortable-item.dragging {
    opacity: 0.5;
}
.item-subtitle {
    font-size: 0.85em;
    color: #444 !important; /* Darker for better contrast */
    font-weight: normal;
    margin-top: 2px;
    line-height: 1.25;
}
.sortable-item strong {
    color: #111 !important; /* Very dark for main tags */
    display: block;
    margin-bottom: 2px;
}
/* Ensure model results and variation text is dark enough */
.dark-text-contrast {
    color: #222 !important;
}
.dark-text-contrast strong {
    color: #000 !important;
}
.btn-orange {
    background-color: #f39c12 !important;
    color: white !important;
}
.btn-green {
    background-color: #27ae60 !important;
    color: white !important;
}
/* Contrast fix for dark mode */
.dark-text-container {
    color: #2c3e50 !important;
}
.dark-text-container * {
    color: #2c3e50 !important;
}
/* Calibration rank container: lock to light background + dark text */
#calibration-rank-container {
    background-color: #f9f9f9 !important;
    color: #333 !important;
}
#calibration-rank-container .draggable {
    background-color: #e0e0e0 !important;
    color: #333 !important;
    border: 1px solid #b0b0b0 !important;
}
#calibration-rank-container .draggable strong {
    color: #111 !important;
}
#calibration-rank-container .item-subtitle {
    color: #444 !important;
}
.revert-btn-styled {
    background-color: #2c3e50 !important;
    color: #ffffff !important;
    border: 1px solid #444 !important;
}
.revert-btn-styled:hover {
    background-color: #3e5871 !important;
}
.revert-btn-styled:disabled {
    background-color: #bdc3c7 !important;
    color: #7f8c8d !important;
    cursor: not-allowed;
}
"""


CALIBRATION_WIDGET_HTML = """
<div id="calibration-rank-container">
    <div class="draggable" draggable="true" data-value="Fidelity">
        <strong>Fidelity</strong>
        <div class="item-subtitle">the explanation will adhere to exactly what the model actually considers</div>
    </div>
    <div class="draggable" draggable="true" data-value="Simplicity">
        <strong>Simplicity</strong>
        <div class="item-subtitle">easy for humans to understand, not overcomplicated by too many features</div>
    </div>
    <div class="draggable" draggable="true" data-value="Robustness">
        <strong>Robustness</strong>
        <div class="item-subtitle">not vulnerable to <a href="https://arxiv.org/abs/1806.08049" target="_blank" style="color: #2980b9; text-decoration: underline;">adversarial perturbation attacks</a> (click for paper)</div>
    </div>
</div>
"""

LEGEND_HTML = """
<div style="margin-top: 10px; display: flex; flex-direction: column; gap: 4px; font-family: sans-serif; font-size: 0.85em; max-width: 400px;">
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="color: #666; font-weight: bold;">Impact on Response:</span>
        <div style="flex-grow: 1; height: 8px; background: linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725); border-radius: 4px;"></div>
    </div>
    <div style="display: flex; justify-content: space-between; color: #888; padding-left: 85px;">
        <span>More Negative</span>
        <span>More Positive</span>
    </div>
</div>
"""
with gr.Blocks(title="Algorithmic Audit Toolkit", css=custom_css, theme=theme) as demo:
    with gr.Column(visible=True) as intro_page:
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=3):
                gr.Markdown("<h1 style='text-align: center;'>Algorithmic Audit Toolkit</h1>")
                gr.Markdown("<h3 style='text-align: center;'>Prototype Auditing Interface</h3>")
                gr.Markdown("The following interface is meant to be used for gray-box auditing with access to an explanation. Each tab represents a different type of functionality (resume screening, image captioning, and credit risk) for which you can try out multiple models and interpretability methods.")
                
                gr.Markdown("### Instructions")
                gr.Markdown("Choose to audit language, vision-language, or supervised models. You can load any HuggingFace model by pasting its URL or ID in Stage 1. Then you can modify inputs, run analyses in Stage 2, and compare or save results in Stage 3.")
                
                gr.Markdown("<br>")
                with gr.Row():
                    intro_resume_btn = gr.Button("Language", variant="primary", size="lg")
                    intro_image_btn = gr.Button("Vision-Language", variant="primary", size="lg", interactive=True)
                    intro_credit_btn = gr.Button("Supervised", variant="primary", size="lg", interactive=True)
            with gr.Column(scale=1):
                pass

    with gr.Row(visible=False) as main_app_layout:
        with gr.Column(scale=1, min_width=200) as sidebar:
            gr.Markdown("### Navigation")
            nav_home_btn = gr.Button("Home", variant="secondary")
            gr.Markdown("---")
            nav_resume_btn = gr.Button("Language", variant="primary")
            nav_image_btn = gr.Button("Vision-Language", variant="secondary", interactive=True)
            nav_credit_btn = gr.Button("Supervised", variant="secondary", interactive=True)

        def navigate(choice):
            show_intro = (choice == "home")
            show_main = not show_intro
            return [
                gr.update(visible=show_intro),
                gr.update(visible=show_main),
                gr.update(visible=(choice == "resume")),
                gr.update(visible=(choice == "image")),
                gr.update(visible=(choice == "credit")),
                gr.update(variant="primary" if choice == "resume" else "secondary"),
                gr.update(variant="primary" if choice == "image" else "secondary"),
                gr.update(variant="primary" if choice == "credit" else "secondary"),
            ]

        with gr.Column(scale=5) as content_area:

            # Section 1: Resume Screener
            with gr.Column(visible=True) as resume_section:
                continuation_state = gr.State()
                fulltext_state = gr.State()
                # Track the active inner step (0=Input, 1=Results, 2=Compare)
                screener_tab_state = gr.State(value=0)
                # Track whether an analysis has been run (gates the first Next button)
                analysis_done_state = gr.State(value=False)
    
    
    
                # ── Inner step tabs ──────────────────────────────────────────────
                with gr.Tabs(selected=0) as screener_steps:
    
                    # ── Step 0: Start Here ──────────────────────────────
                    with gr.Tab("Step 0: Start Here", id=-1):
                        gr.Markdown("### Welcome to the Algorithmic Audit Toolkit")
                        gr.Markdown("You can start a new audit session or import a previously saved configuration.")
                        
                        with gr.Group():
                            gr.Markdown("#### Import or Create New")
                            with gr.Row():
                                resume_settings_import_btn = gr.File(label="📂 Import Settings File (.json)", file_count="single", file_types=[".json"])
                            
                            gr.Markdown("<br>")
                            with gr.Row():
                                start_new_audit_btn = gr.Button("Start New Language Audit", variant="primary")
                        
                        gr.Markdown("---")
                        gr.Markdown("Proceed to **Step 1** once you have loaded your configuration or are ready to set up a new audit.")
                        start_new_audit_btn.click(fn=lambda: gr.update(selected=0), outputs=screener_steps)

                    # ── Step 1: Input & Configuration ───────────────────────────
                    with gr.Tab("Step 1: Input & Configuration", id=0):
                        gr.Markdown("Configure the model and choose your use case. Paste a HuggingFace model URL or ID to load any compatible causal or masked language model.")
    
                        # ── 1. Configure Model ───────────────────────────────────
                        gr.Markdown("### 1. Configure Model")
                        with gr.Row():
                            resume_model_input = gr.Textbox(
                                label="HuggingFace Model ID or URL",
                                placeholder="e.g. EleutherAI/gpt-neo-125M  or  https://huggingface.co/gpt2",
                                value="EleutherAI/gpt-neo-125M",
                                scale=5,
                                interactive=True,
                            )
                            resume_load_btn = gr.Button("Load Model", variant="primary", scale=1)
                        resume_model_status = gr.Markdown("", elem_id="resume_model_status")
                        gr.Markdown(
                            "*Supports any causal LM (text-generation) or masked LM (fill-mask) from HuggingFace.*",
                        )

                        with gr.Accordion("Configure Hyperparameters", open=False):
                            temperature_slider = gr.Number(
                                label="Model Temperature",
                                value=0.45,
                                precision=2,
                                info="Controls randomness. Lower values are more deterministic; higher values (up to 1) increase variability.",
                            )

                        gr.Markdown("---")

                        # ── 2. Select Use Case ───────────────────────────────────
                        gr.Markdown("### 2. Select Use Case")
                        resume_usecase_radio = gr.Radio(
                            choices=["General (Causal / Masked LM)", "Resume Screening"],
                            value="General (Causal / Masked LM)",
                            label="",
                            info="Choose the analysis context. Resume Screening uses a structured hiring prompt; General lets you run any prompt.",
                        )

                        gr.Markdown("---")
                        gr.Markdown("### 3. Edit Input")

                        # ── General LM panel ─────────────────────────────────────
                        with gr.Column(visible=True) as general_lm_panel:
                            gr.Markdown("#### Prompt")
                            general_prompt_input = gr.Textbox(
                                lines=8,
                                label="Prompt",
                                placeholder="Enter your prompt here. For masked LMs, add [MASK] where you want the model to predict.",
                                value="The quick brown fox jumps over the",
                            )

                        # ── Resume Screening panel ───────────────────────────────
                        with gr.Column(visible=False) as resume_panel:
                            # Lead Prompt Section
                            gr.Markdown("#### Lead Prompt")
                            gr.Markdown("*The part of the prompt given to the model before the resume content.*")
                            resume_lead_input = gr.Textbox(
                                value=initial_lead,
                                lines=3,
                                label="Lead Prompt",
                                placeholder="Leading instructions or context...",
                            )
                            with gr.Row():
                                revert_lead_btn = gr.Button("revert to original", variant="secondary", size="sm", interactive=False, elem_classes=["revert-btn-styled"])

                            # Main Body Section
                            gr.Markdown("#### Main Resume Body")
                            resume_body_input = gr.Textbox(
                                value=initial_body,
                                lines=12,
                                label="Resume Content",
                                placeholder="Paste the main resume content here...",
                            )
                            with gr.Row():
                                revert_body_btn = gr.Button("revert to original", variant="secondary", size="sm", interactive=False, elem_classes=["revert-btn-styled"])

                            # End Prompt Section
                            gr.Markdown("#### End Prompt")
                            gr.Markdown("*The part of the prompt given to the model after the resume content.*")
                            resume_end_input = gr.Textbox(
                                value=initial_end,
                                lines=3,
                                label="End Prompt",
                                placeholder="Closing instructions...",
                            )
                            with gr.Row():
                                revert_end_btn = gr.Button("revert to original", variant="secondary", size="sm", interactive=False, elem_classes=["revert-btn-styled"])

                        gr.Markdown("---")
                        gr.Markdown("### Enable Batch Analysis")
                        gr.Markdown("**Analyze many variations to understand statistical patterns across demographics**")
                        batch_analysis_toggle = gr.Checkbox(
                            label="Enable Batch Audit Mode",
                            value=False,
                            info="Run analysis with LLM or Preset variations (only available in Resume Screening mode)"
                        )
    
                        batch_preset_selection = gr.Dropdown(
                            choices=["Preset: Gender", "Natural Language (LLM)", "Custom Extended"],
                            value="Preset: Gender",
                            label="Variation Content Strategy",
                            visible=False
                        )
    
                        batch_num_variations = gr.Number(
                            label="Number of Variations",
                            value=20,
                            precision=0,
                            visible=False,
                            info="How many variations to test."
                        )
    
                        batch_token_input = gr.Textbox(
                            label="Token to Vary (Anchor) - *this token will be changed for each item in the batch*",
                            placeholder="e.g., Jane",
                            value="Jane",
                            visible=False,
                            info="The word in your resume that will be replaced by variations."
                        )
    
                        batch_nl_input = gr.Textbox(
                            label="What characteristic or content do you want to audit for bias?",
                            placeholder="e.g., 'Vary the names to include different religious backgrounds'.",
                            visible=False,
                            info="Describe how you want to vary the resume."
                        )
    
                        batch_generate_code_btn = gr.Button("Generate Variation Code", variant="secondary", visible=False)
    
                        batch_code_preview = gr.Textbox(
                            label="Variation Code (Review and Edit)",
                            placeholder="Python code will appear here...",
                            lines=10,
                            visible=False,
                            interactive=True
                        )
                        
                        # ── 4. Save These Settings ────────────────────────────────
                        gr.Markdown("### 4. Save These Settings")
                        with gr.Row():
                            resume_settings_export_btn = gr.DownloadButton("💾 Export Audit Settings", variant="secondary")
                            # resume_settings_import_btn moved to Step 0
    
                        gr.Markdown("---")
                        with gr.Row():
                            resume_analyze_btn = gr.Button(
                                "Analyze (Single Run)", 
                                variant="primary", 
                                elem_classes=["btn-orange"],
                                visible=False
                            )
                            general_analyze_btn = gr.Button(
                                "Generate Output", 
                                variant="primary", 
                                elem_classes=["btn-orange"],
                                visible=True
                            )
                            batch_execute_btn = gr.Button(
                                "Execute Batch Audit", 
                                variant="primary", 
                                elem_classes=["btn-orange"],
                                visible=False
                            )
                            resume_reset_all_btn = gr.Button("Revert All", variant="secondary")
    
                        gr.Markdown("---")
                        with gr.Row():
                            step1_next_btn = gr.Button("Next: View Results →", variant="primary", interactive=False)
                            step1_loading_msg = gr.Markdown("⏳ **Analysis in progress... Please wait.**", visible=False)
    

                    # ── Step 2: Results & Interpretation ────────────────────────
                    with gr.Tab("Step 2: Results & Interpretation", id=1, interactive=False) as step2_tab:
                        gr.Markdown("View the model's generated content and explore its decision-making process.")
    
                        resume_current_model_display = gr.Markdown("**Model:** Loading...")
                        pure_html_output = gr.HTML(
                            label="Model Output",
                            value="<p>Click 'Analyze' in Step 1 to see results...</p>",
                            elem_classes=["dark-text-contrast"]
                        )
                        
                        with gr.Column(visible=True) as general_token_panel:
                            gr.Markdown("#### Select a Token to Explain")
                            with gr.Row():
                                general_token_slider = gr.Slider(
                                    minimum=0, maximum=50, value=0, step=1,
                                    label="Token Index",
                                    interactive=True,
                                )
                                general_token_display = gr.Markdown("*Run the model first to see available tokens.*")
                            general_explain_btn = gr.Button("Explain Selected Token", variant="primary", interactive=False)
                        
                        with gr.Accordion("Advanced: Calibrate Explanations", open=False, visible=False) as calibration_accordion:
                            gr.Markdown("### Rank what you value in an explanation")
                            gr.HTML(CALIBRATION_WIDGET_HTML)
                            explanation_rank_state = gr.Textbox(value='["Fidelity", "Simplicity", "Robustness"]', visible=False, label="Calibration Rank Status")
                            ranking_saved_msg = gr.Markdown("**Ranking saved**", visible=False)
                            
                            # JavaScript helper to sync the rank string from the HTML widget to the Gradio textbox
                            demo.load(None, None, None, js="""
                                () => {
                                    window.updateCalibrationRank = (rankStr) => {
                                        const textbox = document.querySelector('textarea[label="Calibration Rank Status"]');
                                        if (textbox) {
                                            textbox.value = rankStr;
                                            textbox.dispatchEvent(new Event('input'));
                                        }
                                        // Show the "Ranking saved" message by triggering the Markdown visibility
                                        // In Gradio, we'll use the change event on the textbox to handle this in Python
                                    };
                                    // Initialize the drag-and-drop functionality
                                    const initDragAndDrop = () => {
                                        const container = document.getElementById('calibration-rank-container');
                                        if (!container) return;

                                        let draggedItem = null;

                                        container.addEventListener('dragstart', (e) => {
                                            draggedItem = e.target;
                                            e.dataTransfer.effectAllowed = 'move';
                                            e.dataTransfer.setData('text/plain', e.target.dataset.value);
                                            setTimeout(() => {
                                                e.target.classList.add('dragging');
                                            }, 0);
                                        });

                                        container.addEventListener('dragover', (e) => {
                                            e.preventDefault();
                                            const afterElement = getDragAfterElement(container, e.clientY);
                                            const currentDragging = document.querySelector('.dragging');
                                            if (currentDragging === null) return; // No item being dragged

                                            if (afterElement == null) {
                                                container.appendChild(currentDragging);
                                            } else {
                                                container.insertBefore(currentDragging, afterElement);
                                            }
                                        });

                                        container.addEventListener('dragend', (e) => {
                                            e.target.classList.remove('dragging');
                                            draggedItem = null;
                                            updateRankAndNotify();
                                        });

                                        const getDragAfterElement = (container, y) => {
                                            const draggableElements = [...container.querySelectorAll('.draggable:not(.dragging)')];
                                            return draggableElements.reduce((closest, child) => {
                                                const box = child.getBoundingClientRect();
                                                const offset = y - box.top - box.height / 2;
                                                if (offset < 0 && offset > closest.offset) {
                                                    return { offset: offset, element: child };
                                                } else {
                                                    return closest;
                                                }
                                            }, { offset: -Number.POSITIVE_INFINITY }).element;
                                        };

                                        const updateRankAndNotify = () => {
                                            const newRank = Array.from(container.children)
                                                                 .filter(child => child.classList.contains('draggable'))
                                                                 .map(child => child.dataset.value);
                                            window.updateCalibrationRank(JSON.stringify(newRank));
                                        };

                                        // Initial update
                                        updateRankAndNotify();
                                    };

                                    // Run initialization after Gradio components are rendered
                                    // Use a MutationObserver to detect when the container is added to the DOM
                                    const observer = new MutationObserver((mutationsList, observer) => {
                                        for (const mutation of mutationsList) {
                                            if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                                                const container = document.getElementById('calibration-rank-container');
                                                if (container) {
                                                    initDragAndDrop();
                                                    observer.disconnect(); // Stop observing once initialized
                                                    break;
                                                }
                                            }
                                        }
                                    });

                                    // Start observing the body for changes
                                    observer.observe(document.body, { childList: true, subtree: true });

                                    // Inject CSS for better contrast and dragging
                                    const style = document.createElement('style');
                                    style.innerHTML = `
                                        #calibration-rank-container {
                                            display: flex;
                                            flex-direction: column;
                                            gap: 8px;
                                            padding: 10px;
                                            border: 1px solid #ccc;
                                            border-radius: 4px;
                                            background-color: #f9f9f9;
                                        }
                                        .draggable {
                                            padding: 10px 15px;
                                            background-color: #e0e0e0; /* Darker grey */
                                            border: 1px solid #b0b0b0; /* Darker border */
                                            border-radius: 4px;
                                            cursor: grab;
                                            font-weight: 500;
                                            color: #333; /* Darker text for contrast */
                                            transition: background-color 0.2s, border-color 0.2s, transform 0.1s;
                                            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                                        }
                                        .draggable:hover {
                                            background-color: #d0d0d0; /* Even darker on hover */
                                            border-color: #909090;
                                        }
                                        .draggable.dragging {
                                            opacity: 0.6;
                                            transform: scale(1.02);
                                            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                                        }
                                    `;
                                    document.head.appendChild(style);
                                }
                            """)

                        # Listen for changes in rank state to show "Ranking saved"
                        explanation_rank_state.change(fn=lambda: gr.update(visible=True), outputs=ranking_saved_msg)

                        resume_method_dropdown = gr.State(value="calibrated")
                        resume_explain_btn = gr.Button("Explain", variant='primary', interactive=has_explanation, visible=False)
                        with gr.Column():
                            explanation_html = gr.HTML(
                                label="Explanation",
                                value="<p>No explanation generated. Click 'Explain' to see results...</p>",
                                visible=has_explanation,
                                elem_classes=["dark-text-contrast"]
                            )
                            gr.HTML(LEGEND_HTML, visible=has_explanation)
    
                        # Batch Interpretation Carousel
                        with gr.Accordion("Explanations by Variation", open=False, visible=has_explanation) as batch_carousel_accordion:
                            gr.Markdown("### Variation Explorer")
                            gr.Markdown("Select a variation from your last batch run to see its specific interpretability highlights.")
    
                            batch_results_state = gr.State([])
    
                            with gr.Row():
                                carousel_index_slider = gr.Slider(
                                    minimum=0, maximum=19, value=0, step=1,
                                    label="Select Variation index",
                                    interactive=True
                                )
    
                            carousel_preview_html = gr.HTML(
                                label="Variation Preview",
                                value="<p>Select a variation to see details...</p>"
                            )
    
                            carousel_explain_btn = gr.Button("Explain Selection", variant="primary", interactive=True)
    
                            with gr.Column():
                                carousel_explanation_html = gr.HTML(
                                    label="Variation Explanation Highlights",
                                    value="<p>Detailed analysis of the selected variation will appear here...</p>",
                                    visible=has_explanation,
                                    elem_classes=["dark-text-contrast"]
                                )
                                gr.HTML(LEGEND_HTML, visible=has_explanation)
    
                        # Manual save button (still available on Step 2)
                        with gr.Row():
                            resume_save_btn = gr.Button("Save Current Version", variant="primary")
                        resume_save_status = gr.Markdown("")
    
                        gr.Markdown("---")
                        with gr.Row():
                            step2_back_btn = gr.Button("← Back to Input", variant="secondary")
                            step2_next_btn = gr.Button("Next: Compare & Track →", variant="primary")
                        autosave_status = gr.Markdown("", visible=False)
    
                    # ── Step 3: Download, Share, and Compare ──────────────────────────────────
                    with gr.Tab("Step 3: Download, Share, and Compare", id=2, interactive=False) as step3_tab:
                        gr.Markdown("Save versions to track your audit progress and compare different model behaviors side-by-side.")
                        gr.Markdown("*Versions are autosaved when you navigate here from Step 2.*")
    
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### Comparison Column A")
                                resume_version_dropdown_a = gr.Dropdown(
                                    choices=[],
                                    label="Select version A",
                                    interactive=True
                                )
                                resume_comparison_output_a = gr.HTML(
                                    label="Saved Highlights A",
                                    value="<p>Select a version to compare.</p>"
                                )
                                resume_download_selected_btn_a = gr.DownloadButton("Download A (HTML)", variant="primary", visible=False, size="sm")

                            with gr.Column():
                                gr.Markdown("#### Comparison Column B")
                                resume_version_dropdown_b = gr.Dropdown(
                                    choices=[],
                                    label="Select version B",
                                    interactive=True
                                )
                                resume_comparison_output_b = gr.HTML(
                                    label="Saved Highlights B",
                                    value="<p>Select a version to compare.</p>"
                                )
                                resume_download_selected_btn_b = gr.DownloadButton("Download B (HTML)", variant="primary", visible=False, size="sm")

                        gr.Markdown("---")
                        with gr.Row():
                            resume_download_all_btn = gr.DownloadButton("Download All Trials (HTML)", variant="secondary")
                            resume_download_csv_btn = gr.DownloadButton("Download Batch Results (CSV)", variant="secondary", visible=False)
                            resume_settings_export_btn_stage3 = gr.DownloadButton("💾 Save These Settings", variant="secondary")
    
                        gr.Markdown("---")
                        with gr.Row():
                            step3_back_btn = gr.Button("← Back to Results", variant="secondary")
    
                # ── Event Handlers ───────────────────────────────────────────────
    
                # ── Model Loading & Use Case Toggles ─────────────────────────────
                
                # Language Tab Model Load
                resume_load_btn.click(
                    fn=validate_and_load_resume_model,
                    inputs=resume_model_input,
                    outputs=[resume_model_status, resume_current_model_display]
                )

                # Language Tab Use Case Toggle
                def toggle_language_usecase(usecase):
                    is_resume = "Resume Screening" in usecase
                    return (
                        gr.update(visible=not is_resume),  # general_lm_panel
                        gr.update(visible=is_resume),      # resume_panel
                        gr.update(visible=is_resume),      # batch_analysis_toggle
                        gr.update(visible=not is_resume),  # general_analyze_btn
                        gr.update(visible=is_resume),      # resume_analyze_btn
                        gr.update(visible=is_resume),      # resume_reset_all_btn
                        gr.update(visible=not is_resume),  # general_token_panel
                        gr.update(visible=is_resume),      # resume_explain_btn
                        gr.update(visible=is_resume),      # calibration_accordion
                        gr.update(visible=is_resume),      # batch_carousel_accordion
                    )
                
                resume_usecase_radio.change(
                    fn=toggle_language_usecase,
                    inputs=resume_usecase_radio,
                    outputs=[
                        general_lm_panel, resume_panel, batch_analysis_toggle,
                        general_analyze_btn, resume_analyze_btn, resume_reset_all_btn,
                        general_token_panel, resume_explain_btn, calibration_accordion,
                        batch_carousel_accordion
                    ]
                )


    
                # Navigation: Step 1 → Step 2
                step1_next_btn.click(
                    fn=lambda: gr.update(selected=1),
                    inputs=None,
                    outputs=screener_steps
                )
    
                # Navigation: Step 2 → Step 1
                step2_back_btn.click(
                    fn=lambda: gr.update(selected=0),
                    inputs=None,
                    outputs=screener_steps
                )
    
                # Navigation: Step 3 → Step 2
                step3_back_btn.click(
                    fn=lambda: gr.update(selected=1),
                    inputs=None,
                    outputs=screener_steps
                )
    
                # Navigation: Step 2 → Step 3 with autosave
                def autosave_and_advance(lead, body, end, html_output, expl_html, batch_enabled, batch_results):
                    """Autosave the current analysis and return the tab switch + updated dropdown."""
                    full_text = combine_resume_parts(lead, body, end)
    
                    if batch_enabled:
                        content = html_output
                        mode_label = f"Batch ({len(batch_results)} vars)" if batch_results else "Batch"
                    else:
                        content = html_output + "<hr>" + expl_html if "No explanation generated" not in expl_html else html_output
                        mode_label = "Single Run"
    
                    from resume_utility import lm_model_name as current_model
                    from datetime import datetime as _dt
                    timestamp = _dt.now().strftime("%Y-%m-%d %H:%M")
                    # Descriptive autosave label
                    auto_label = f"Auto: {current_model} | {timestamp} | {mode_label}"
    
                    save_status, dropdown_update = save_resume_version(full_text, content, auto_label=auto_label)
    
                    return (
                        gr.update(selected=2),            # switch to step 3
                        dropdown_update,                  # refresh version dropdown A
                        dropdown_update,                  # refresh version dropdown B
                        gr.update(value=f"Autosaved: *{auto_label}*", visible=True),  # show autosave notice
                    )
    
                step2_next_btn.click(
                    fn=autosave_and_advance,
                    inputs=[
                        resume_lead_input, resume_body_input, resume_end_input,
                        pure_html_output, explanation_html,
                        batch_analysis_toggle, batch_results_state
                    ],
                    outputs=[screener_steps, resume_version_dropdown_a, resume_version_dropdown_b, autosave_status]
                )
    
                # Unified Batch UI Toggle
                def toggle_batch_visibility(enabled, num_vars, preset):
                    # Visibility for common batch elements
                    visibility = gr.update(visible=enabled)
                    
                    # Determine visibility for code-related elements based on preset
                    is_custom = (preset == "Custom")
                    code_box_visibility = gr.update(visible=enabled) # Show code regardless of preset if batch enabled
                    nl_visibility = gr.update(visible=(enabled and is_custom))
                    
                    return (
                        visibility, # batch_num_variations
                        visibility, # batch_token_input
                        visibility, # batch_preset_selection
                        visibility, # batch_execute_btn
                        gr.update(visible=not enabled), # resume_explain_btn
                        gr.update(visible=not enabled), # explanation_html
                        gr.update(visible=enabled),     # batch_carousel_accordion
                        gr.update(maximum=num_vars),    # carousel_index_slider
                        gr.update(interactive=not enabled), # resume_analyze_btn
                        nl_visibility, # batch_nl_input
                        nl_visibility, # batch_generate_code_btn
                        code_box_visibility # batch_code_preview
                    )
    
                batch_analysis_toggle.change(
                    fn=toggle_batch_visibility,
                    inputs=[batch_analysis_toggle, batch_num_variations, batch_preset_selection],
                    outputs=[
                        batch_num_variations, batch_token_input, batch_preset_selection, 
                        batch_execute_btn, resume_explain_btn, explanation_html, 
                        batch_carousel_accordion, carousel_index_slider, resume_analyze_btn,
                        batch_nl_input, batch_generate_code_btn, batch_code_preview
                    ]
                )
    
                # Update presets/search & vary anchor
                def update_preset_behavior(preset, body, num_vars):
                    anchor = get_suggested_anchor(body)
                    if preset == "Preset: Gender":
                        code = GENDER_PRESET_TEMPLATE.format(num_vars=num_vars, anchor=anchor)
                        return (
                            gr.update(visible=False), # NL input hidden
                            gr.update(visible=False), # Gen code btn hidden
                            gr.update(value=code, visible=True), # Code preview shown
                            gr.update(value=anchor, visible=True) # Anchor pre-filled
                        )
                    elif preset == "Custom Extended":
                        code = CUSTOM_EXTENDED_TEMPLATE.format(num_vars=num_vars)
                        return (
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(value=code, visible=True),
                            gr.update(value=anchor, visible=True)
                        )
                    else: # Natural Language
                        return (
                            gr.update(visible=True), # NL input shown
                            gr.update(visible=True), # Gen code btn shown
                            gr.update(visible=True), # Code preview shown (empty/result)
                            gr.update(visible=True)  # Anchor visible
                        )
    
                batch_preset_selection.change(
                    fn=update_preset_behavior,
                    inputs=[batch_preset_selection, resume_body_input, batch_num_variations],
                    outputs=[batch_nl_input, batch_generate_code_btn, batch_code_preview, batch_token_input]
                )
    
                # Code Generation Handler
                batch_generate_code_btn.click(
                    fn=generate_nl_variations_code,
                    inputs=[batch_nl_input, batch_num_variations],
                    outputs=batch_code_preview
                )
    
                # Individual revert buttons
                revert_lead_btn.click(fn=revert_lead_prompt, inputs=None, outputs=resume_lead_input)
                revert_body_btn.click(fn=revert_main_body, inputs=None, outputs=resume_body_input)
                revert_end_btn.click(fn=revert_end_prompt, inputs=None, outputs=resume_end_input)

                # Enable revert buttons on change
                resume_lead_input.change(fn=lambda: gr.update(interactive=True), outputs=revert_lead_btn)
                resume_body_input.change(fn=lambda: gr.update(interactive=True), outputs=revert_body_btn)
                resume_end_input.change(fn=lambda: gr.update(interactive=True), outputs=revert_end_btn)
    
                def revert_all():
                    lead, body, end = split_resume_corpus(sample_corpus)
                    return lead, body, end
    
                resume_reset_all_btn.click(
                    fn=revert_all,
                    inputs=None,
                    outputs=[resume_lead_input, resume_body_input, resume_end_input]
                )

                def handle_resume_settings_export(model, temp, usecase, prompt, lead, body, end, batch_on, batch_preset, batch_num, batch_token, batch_nl, batch_code):
                    settings = {
                        "model": model, "temp": temp, "usecase": usecase, "prompt": prompt,
                        "lead": lead, "body": body, "end": end, "batch_enabled": batch_on,
                        "batch_preset": batch_preset, "batch_num": batch_num, "batch_token": batch_token,
                        "batch_nl": batch_nl, "batch_code": batch_code
                    }
                    return export_settings(settings)

                def handle_resume_settings_import(file_info):
                    if not file_info: return [gr.update()]*13
                    from settings_manager import import_settings
                    data = import_settings(file_info.name)
                    if not data: return [gr.update()]*13
                    return (
                        data.get("model", ""), data.get("temp", 0.45), data.get("usecase", "General (Causal / Masked LM)"),
                        data.get("prompt", ""), data.get("lead", ""), data.get("body", ""), data.get("end", ""),
                        data.get("batch_enabled", False), data.get("batch_preset", "Preset: Gender"),
                        data.get("batch_num", 20), data.get("batch_token", ""), data.get("batch_nl", ""), data.get("batch_code", "")
                    )

                resume_settings_export_btn.click(
                    fn=handle_resume_settings_export,
                    inputs=[
                        resume_model_input, temperature_slider, resume_usecase_radio, general_prompt_input,
                        resume_lead_input, resume_body_input, resume_end_input, batch_analysis_toggle,
                        batch_preset_selection, batch_num_variations, batch_token_input, batch_nl_input, batch_code_preview
                    ],
                    outputs=resume_settings_export_btn
                )

                resume_settings_export_btn_stage3.click(
                    fn=handle_resume_settings_export,
                    inputs=[
                        resume_model_input, temperature_slider, resume_usecase_radio, general_prompt_input,
                        resume_lead_input, resume_body_input, resume_end_input, batch_analysis_toggle,
                        batch_preset_selection, batch_num_variations, batch_token_input, batch_nl_input, batch_code_preview
                    ],
                    outputs=resume_settings_export_btn_stage3
                )

                resume_settings_import_btn.upload(
                    fn=handle_resume_settings_import,
                    inputs=resume_settings_import_btn,
                    outputs=[
                        resume_model_input, temperature_slider, resume_usecase_radio, general_prompt_input,
                        resume_lead_input, resume_body_input, resume_end_input, batch_analysis_toggle,
                        batch_preset_selection, batch_num_variations, batch_token_input, batch_nl_input, batch_code_preview
                    ]
                )

                # ── General LM Handlers ──────────────────────────────────────────
                def handle_general_analyze(prompt, temp):
                    from resume_utility import run_general_lm
                    html_output, continuation, full_text, model_display, tokens_list = run_general_lm(prompt, 0, temp)
                    
                    if tokens_list:
                        clean_tokens = [t.replace('Ġ', ' ').replace('Â', '').strip() or t for t in tokens_list]
                        tok_str = " | ".join([f"{i}: {t}" for i, t in enumerate(clean_tokens)])
                        tok_display = f"**Tokens:** {tok_str}"
                        slider_update = gr.update(maximum=len(tokens_list)-1, value=0, interactive=True)
                        explain_interactive = True
                    else:
                        tok_display = "*No tokens generated.*"
                        slider_update = gr.update(maximum=0, value=0, interactive=False)
                        explain_interactive = False
                        
                    return (
                        html_output, continuation, full_text, model_display,
                        slider_update, tok_display, gr.update(interactive=explain_interactive),
                        gr.update(visible=True, interactive=True, elem_classes=["btn-green"]),
                        gr.update(visible=False),
                        gr.update(interactive=True), gr.update(interactive=True)
                    )

                def start_analysis_loading():
                    return gr.update(visible=False), gr.update(visible=True)

                general_analyze_btn.click(
                    fn=start_analysis_loading,
                    inputs=None,
                    outputs=[step1_next_btn, step1_loading_msg]
                ).then(
                    fn=handle_general_analyze,
                    inputs=[general_prompt_input, temperature_slider],
                    outputs=[
                        pure_html_output, continuation_state, fulltext_state,
                        resume_current_model_display, general_token_slider, general_token_display, general_explain_btn,
                        step1_next_btn, step1_loading_msg, step2_tab, step3_tab
                    ],
                    show_progress="full"
                )

                def handle_general_explain(prompt, continuation, token_idx, rank_state):
                    from resume_utility import explain_feature_attribution, _model, _tokenizer
                    if not _model or not _tokenizer:
                        return "<p>Model not loaded. Please load a model and run analysis first.</p>"
                    
                    import json
                    try:
                        ranks = json.loads(rank_state)
                    except:
                        ranks = ["Fidelity", "Simplicity", "Robustness"]
                    
                    try:
                        expl_html = explain_feature_attribution(
                            model=_model, tokenizer=_tokenizer,
                            input_text=prompt, target_text=prompt + (continuation if continuation else ""),
                            attribution_method="integrated_gradients",
                            ranks=ranks,
                            explain_token_index=int(token_idx)
                        )
                        return expl_html
                    except Exception as e:
                        return f"<p style='color:red;'><strong>Error computing attribution:</strong> {str(e)}</p>"
                    
                general_explain_btn.click(
                    fn=lambda: gr.update(value="<p><i>Computing attribution... Please wait.</i></p>"),
                    outputs=explanation_html
                ).then(
                    fn=handle_general_explain,
                    inputs=[general_prompt_input, continuation_state, general_token_slider, explanation_rank_state],
                    outputs=explanation_html,
                    show_progress="full"
                ).then(
                    fn=lambda: gr.update(selected=1),
                    outputs=screener_steps
                )
                # ─────────────────────────────────────────────────────────────────
    
                def process_and_enable_next(lead, body, end, rank_order, temp, batch_enabled, batch_token, batch_num_vars, batch_dimension, variations_code):
                    results = process_resume_split(lead, body, end, "integrated_gradients", temp, batch_enabled, batch_token, batch_num_vars, batch_dimension, variations_code)
                    # results: (html, batch_results_state, continuation_state, fulltext_state,
                    #           model_display, explain_btn, explanation_html, batch_carousel, carousel_slider)
                    
                    # Update Next button (make visible/green) and Tab interactivity
                    return results + (
                        gr.update(visible=True, interactive=True, elem_classes=["btn-green"]), # step1_next_btn
                        gr.update(visible=False), # step1_loading_msg
                        gr.update(interactive=True), # step2_tab
                        gr.update(interactive=True)  # step3_tab
                    )
    
                resume_analyze_btn.click(
                    fn=start_analysis_loading,
                    inputs=None,
                    outputs=[step1_next_btn, step1_loading_msg]
                ).then(
                    fn=process_and_enable_next,
                    inputs=[
                        resume_lead_input, resume_body_input, resume_end_input,
                        explanation_rank_state, temperature_slider,
                        gr.State(False),
                        batch_token_input, batch_num_variations,
                        gr.State("Gender"), gr.State(None)
                    ],
                    outputs=[
                        pure_html_output, batch_results_state, continuation_state, fulltext_state,
                        resume_current_model_display, resume_explain_btn, explanation_html,
                        batch_carousel_accordion, carousel_index_slider,
                        step1_next_btn, step1_loading_msg, step2_tab, step3_tab
                    ],
                    show_progress="full"
                ).then(
                    fn=lambda: gr.update(interactive=has_explanation),
                    inputs=None,
                    outputs=[resume_explain_btn]
                )
    
                # Batch Execute Button — also enables the Next button
                batch_execute_btn.click(
                    fn=start_analysis_loading,
                    inputs=None,
                    outputs=[step1_next_btn, step1_loading_msg]
                ).then(
                    fn=process_and_enable_next,
                    inputs=[
                        resume_lead_input, resume_body_input, resume_end_input,
                        explanation_rank_state, temperature_slider,
                        gr.State(True),
                        batch_token_input, batch_num_variations,
                        gr.State("Gender"), batch_code_preview
                    ],
                    outputs=[
                        pure_html_output, batch_results_state, continuation_state, fulltext_state,
                        resume_current_model_display, resume_explain_btn, explanation_html,
                        batch_carousel_accordion, carousel_index_slider,
                        step1_next_btn, step1_loading_msg, step2_tab, step3_tab
                    ],
                    show_progress="full"
                )
    
                # Carousel event handlers
                def update_carousel_preview(results, index):
                    if not results or index >= len(results):
                        return "<p>No variation data available.</p>"
                    res = results[int(index)]
                    sentiment_color = '#27ae60' if res['sentiment'] == 'positive' else '#e74c3c' if res['sentiment'] == 'negative' else '#7f8c8d'
                    
                    # Build runs list for carousel
                    runs_list_html = "<ul style='margin: 10px 0; padding-left: 20px; color: #000;'>"
                    for i, (run, score) in enumerate(zip(res['runs'], res['scores'])):
                        r_sent = 'POSITIVE' if score > 0 else 'NEGATIVE' if score < 0 else 'NEUTRAL'
                        r_col = '#27ae60' if score > 0 else '#e74c3c' if score < 0 else '#7f8c8d'
                        runs_list_html += f"<li>Run {i+1}: \"{run['continuation']}\" — <span style='color: {r_col}; font-weight: bold;'>{r_sent}</span></li>"
                    runs_list_html += "</ul>"

                    html = f"""
                    <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #fff; color: #000;">
                        <p style="color: #000;"><strong>Variation:</strong> {res['variation']}</p>
                        <p style="color: #000;"><strong>Category:</strong> {res['category']}</p>
                        <p style="color: #000;"><strong>Sentiment:</strong> <span style="color: {sentiment_color}; font-weight: bold;">{res['sentiment'].upper()}</span> (Avg: {res['avg_score']:.2f})</p>
                        <hr>
                        <p style="color: #000;"><strong>All {len(res['runs'])} Runs:</strong></p>
                        {runs_list_html}
                    </div>
                    """
                    return html
    
                carousel_index_slider.change(
                    fn=update_carousel_preview,
                    inputs=[batch_results_state, carousel_index_slider],
                    outputs=carousel_preview_html
                )
    
                carousel_explain_btn.click(
                    fn=explain_batch_variation,
                    inputs=[batch_results_state, carousel_index_slider, resume_method_dropdown, explanation_rank_state],
                    outputs=carousel_explanation_html
                )
    
                # Explain button
                resume_explain_btn.click(
                    fn=explain_resume_split,
                    inputs=[resume_lead_input, resume_body_input, resume_end_input, continuation_state, fulltext_state, resume_method_dropdown, explanation_rank_state],
                    outputs=[explanation_html]
                )
    
                # Manual save button (Step 2)
                def save_version_split(lead, body, end, html_output, expl_html, batch_enabled):
                    full_text = combine_resume_parts(lead, body, end)
                    if batch_enabled:
                        content = html_output
                    else:
                        content = html_output + "<hr>" + expl_html if "No explanation generated" not in expl_html else html_output
                    return save_resume_version(full_text, content)
    
                resume_save_btn.click(
                    fn=save_version_split,
                    inputs=[
                        resume_lead_input, resume_body_input, resume_end_input,
                        pure_html_output, explanation_html, batch_analysis_toggle
                    ],
                    outputs=[resume_save_status, resume_version_dropdown_a]
                ).then(
                    fn=lambda x: x,
                    inputs=[resume_version_dropdown_a],
                    outputs=[resume_version_dropdown_b]
                )
    
                resume_version_dropdown_a.change(
                    fn=on_resume_version_change,
                    inputs=[resume_version_dropdown_a, batch_analysis_toggle, batch_results_state],
                    outputs=[resume_comparison_output_a, resume_download_selected_btn_a, resume_download_csv_btn]
                )

                resume_version_dropdown_b.change(
                    fn=on_resume_version_change,
                    inputs=[resume_version_dropdown_b, batch_analysis_toggle, batch_results_state],
                    outputs=[resume_comparison_output_b, resume_download_selected_btn_b, resume_download_csv_btn]
                )
    
                # resume_clear_btn_a.click(
                #     fn=clear_resume_comparison,
                #     inputs=None,
                #     outputs=[resume_comparison_output_a, resume_version_dropdown_a]
                # )

                # resume_clear_btn_b.click(
                #     fn=clear_resume_comparison,
                #     inputs=None,
                #     outputs=[resume_comparison_output_b, resume_version_dropdown_b]
                # )
    

    
    
            # Section 2: Image Captioning — 3-tab flow (mirrors Resume Screener)
            with gr.Column(visible=False) as image_section:
    
                # ── Shared state ────────────────────────────────────────────────
                img_caption_state      = gr.State(value="")
                img_tokens_state       = gr.State(value=[])
                img_original_state     = gr.State(value=None)   # PIL original
                img_attr_state         = gr.State(value=None)   # PIL attribution overlay
                img_batch_results_g1   = gr.State(value=[])
                img_batch_results_g2   = gr.State(value=[])
                img_batch_mode         = gr.State(value=False)
    
                gr.Markdown("## Image Captioning & Interpretability")
                gr.Markdown(
                    "Probe how vision-language models (BLIP) generate captions. "
                    "Upload a single image with optional occlusion, or run batch analysis "
                    "across multiple images and compare across sessions."
                )
    
                # ── Inner step tabs ──────────────────────────────────────────────
                with gr.Tabs(selected=0) as image_steps:
    
                    # ── Step 1: Input & Configuration ────────────────────────────
                    with gr.Tab("Step 1: Input & Configuration", id=0):
                        gr.Markdown(
                            "Choose **Single Image** mode to upload and optionally occlude an image, "
                            "or enable **Batch Mode** to caption multiple images at once."
                        )
    
                        with gr.Row():
                            with gr.Column(scale=1):
                                with gr.Accordion("1. Configure Model", open=True):
                                    gr.Markdown("Replace the default vision-language model with a HuggingFace one.")
                                    image_model_input = gr.Textbox(
                                        label="HuggingFace Model ID or URL",
                                        placeholder="e.g. Salesforce/blip-image-captioning-base",
                                        value="Salesforce/blip-image-captioning-base",
                                    )
                                    image_load_btn = gr.Button("Load Model", variant="primary")
                                    image_model_status = gr.Markdown("")
                                    image_usecase_radio = gr.Radio(
                                        ["General (VL Captioning)"], 
                                        value="General (VL Captioning)", 
                                        label="Use Case", 
                                        visible=False
                                    )

                                with gr.Accordion("2. Edit Input", open=True):
                                    img_batch_toggle = gr.Checkbox(
                                        label="Enable Batch Mode",
                                        value=False,
                                        info="Upload multiple images for batch captioning"
                                    )
        
                                    # Single-image panel
                                    with gr.Column(visible=True) as img_single_panel:
                                        gr.Markdown("#### Single Image Input")
                                        gr.Markdown(
                                            "*Upload an image. Use the brush or eraser to "
                                            "occlude regions and probe model sensitivity before analysing.*"
                                        )
                                        img_editor = gr.ImageEditor(
                                            label="Upload or Edit Image",
                                            type="pil",
                                            brush=gr.Brush(colors=["#000000", "#FF0000", "#FFFFFF"]),
                                            eraser=gr.Eraser(),
                                        )
    
                                # Batch-upload panel
                                with gr.Column(visible=False) as img_batch_panel:
                                    gr.Markdown("#### Batch Image Comparison")
                                    gr.Markdown(
                                        "*Compare two groups of images. Upload or generate them, and "
                                        "in Step 2 view captions, most common words, and run occlusion.*"
                                    )
                                    img_batch_source_mode = gr.Radio(["Upload Images", "Generate Images"], value="Upload Images", label="Sourcing Method", interactive=True)
                                    
                                    with gr.Column(visible=True) as img_batch_upload_group:
                                        img_batch_upload_g1 = gr.File(label="Group 1: Upload Images", file_count="multiple", file_types=["image"])
                                        img_batch_upload_g2 = gr.File(label="Group 2: Upload Images", file_count="multiple", file_types=["image"])
                                        
                                    with gr.Column(visible=False) as img_batch_generate_group:
                                        img_batch_prompt_g1 = gr.Textbox(label="Group 1: Image Prompt", lines=2)
                                        img_batch_prompt_g2 = gr.Textbox(label="Group 2: Image Prompt", lines=2)
                                        img_batch_gen_count = gr.Slider(minimum=1, maximum=5, value=2, step=1, label="Images per group (max 5)")
    
                                    image_current_model_display = gr.Markdown("**Model:** microsoft/git-large-coco")
                                        
                                    gr.Markdown("#### Analysis Settings")
                                    with gr.Accordion("Attribution Settings", open=True):
                                        img_opacity_slider = gr.Slider(
                                            minimum=0.0, maximum=1.0, value=0.6, step=0.05,
                                            label="Attribution Overlay Opacity"
                                        )
                                        img_steps_slider = gr.Slider(
                                            minimum=10, maximum=100, value=50, step=10,
                                            label="Integration Steps (more = slower, more accurate)"
                                        )
    
                        gr.Markdown("---")
                        with gr.Row():
                            img_analyze_btn = gr.Button(
                                "Analyze (Single)",
                                variant="primary",
                                elem_classes=["btn-orange"]
                            )
                            img_batch_btn = gr.Button(
                                "Run Batch Analysis",
                                variant="primary",
                                elem_classes=["btn-orange"],
                                visible=False
                            )
    
                        with gr.Row():
                            img_step1_next_btn = gr.Button(
                                "Next: View Results →",
                                variant="primary",
                                interactive=False
                            )
                            img_step1_loading = gr.Markdown(
                                "⏳ **Analysing… Please wait.**", visible=False
                            )
    
                    # ── Step 2: Results & Interpretation ─────────────────────────
                    with gr.Tab("Step 2: Results & Interpretation", id=1,
                                 interactive=False) as img_step2_tab:
                        gr.Markdown("View the model's output and explore token-level attributions.")
    
                        # --- Single-image results ---
                        with gr.Column(visible=True) as img_single_results:
                            with gr.Row():
                                with gr.Column(scale=1):
                                    img_caption_out = gr.Textbox(
                                        label="Generated Caption", lines=2, interactive=False
                                    )
                                    img_tokens_out = gr.Textbox(
                                        label="Caption Tokens",
                                        lines=6, interactive=False
                                    )
                                    img_token_slider = gr.Slider(
                                        minimum=0, maximum=20, value=0, step=1,
                                        label="Select Token Index to Attribute",
                                        interactive=True
                                    )
                                    img_compute_btn = gr.Button(
                                        "🔬 Compute Attribution for Selected Token",
                                        variant="primary"
                                    )
                                with gr.Column(scale=1):
                                    img_attr_out = gr.Image(
                                        label="Attribution Heatmap", type="pil"
                                    )
                                    img_original_out = gr.Image(
                                        label="Original / Edited Image", type="pil"
                                    )
    
                        # --- Batch results ---
                        with gr.Column(visible=False) as img_batch_results_panel:
                            gr.Markdown("### Batch Comparison Results")
                            img_batch_word_freq_chart = gr.Image(label="Most Common Words Comparison", type="pil", interactive=False)
                            
                            gr.Markdown(
                                "Click a thumbnail in either gallery to select it, "
                                "then load it into the occlusion editor below."
                            )
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("#### Group 1 Gallery")
                                    img_batch_gallery_g1 = gr.Gallery(
                                        label="Group 1", columns=2, height="auto", preview=False, object_fit="contain"
                                    )
                                with gr.Column():
                                    gr.Markdown("#### Group 2 Gallery")
                                    img_batch_gallery_g2 = gr.Gallery(
                                        label="Group 2", columns=2, height="auto", preview=False, object_fit="contain"
                                    )
    
                            with gr.Row():
                                img_batch_group_selector = gr.Radio(["Group 1", "Group 2"], value="Group 1", label="Select Group to Analyse", visible=False)
                                img_batch_idx_slider = gr.Slider(
                                    minimum=0, maximum=10, value=0, step=1,
                                    label="Selected Image Index", visible=False
                                )

                            with gr.Row():
                                with gr.Column(scale=1):
                                    img_batch_caption_out = gr.Textbox(
                                        label="Caption", lines=2, interactive=False
                                    )
                                    img_batch_tokens_out = gr.Textbox(
                                        label="Tokens", lines=5, interactive=False
                                    )
                                    img_batch_token_slider = gr.Slider(
                                        minimum=0, maximum=20, value=0, step=1,
                                        label="Select Token Index to Attribute",
                                        interactive=True
                                    )
                                    img_batch_compute_btn = gr.Button(
                                        "🔬 Compute Attribution for Selected Token",
                                        variant="primary"
                                    )
                                with gr.Column(scale=1):
                                    img_batch_attr_out = gr.Image(
                                        label="Attribution Heatmap", type="pil"
                                    )
                                    img_batch_orig_out = gr.Image(
                                        label="Selected / Edited Image", type="pil"
                                    )

                            with gr.Accordion("Advanced: Occlusion Probe", open=False):
                                gr.Markdown(
                                    "*Paint or erase regions, then click 'Compute Attribution' "
                                    "to re-caption and see how the analysis changes.*"
                                )
                                img_occlude_editor = gr.ImageEditor(
                                    label="Edit / Occlude Loaded Image",
                                    type="pil",
                                    brush=gr.Brush(colors=["#000000", "#FF0000", "#FFFFFF"]),
                                    eraser=gr.Eraser(),
                                )
                                img_batch_recompute_btn = gr.Button(
                                    "🔬 Re-Caption & Compute Attribution",
                                    variant="primary"
                                )
    
                        # --- Save button (shared) ---
                        with gr.Row():
                            img_save_btn = gr.Button("Save Current Version", variant="primary")
                        img_save_status = gr.Markdown("")
    
                        gr.Markdown("---")
                        with gr.Row():
                            img_step2_back_btn = gr.Button("← Back to Input", variant="secondary")
                            img_step2_next_btn = gr.Button(
                                "Next: Compare & Track →", variant="primary"
                            )
                        img_autosave_status = gr.Markdown("", visible=False)
    
                    # ── Step 3: Download, Share, and Compare ───────────────────────────────────
                    with gr.Tab("Step 3: Download, Share, and Compare", id=2, interactive=False) as img_step3_tab:
                        gr.Markdown(
                            "Save versions in Step 2 to track audit progress and compare "
                            "different images or occlusion conditions side-by-side."
                        )
                        gr.Markdown("*Versions are autosaved when you navigate here from Step 2.*")
    
                        img_version_dropdown = gr.Dropdown(
                            choices=[],
                            label="Select saved version to compare",
                            interactive=True
                        )
                        img_clear_btn = gr.Button(
                            "Clear Comparison", variant="secondary", interactive=False
                        )
                        with gr.Row():
                            img_download_selected_btn = gr.DownloadButton("Download Selected Session (HTML)", variant="primary", visible=False)
                            img_download_all_btn = gr.DownloadButton("Download All Sessions (HTML)", variant="secondary")
                            img_download_csv_btn = gr.DownloadButton("Download Batch Results (CSV)", variant="secondary", visible=False)
                        img_comparison_output = gr.HTML(
                            label="Saved Version",
                            value="<p>No comparison loaded. Save a version and select it from the dropdown.</p>"
                        )
    
                        gr.Markdown("---")
                        with gr.Row():
                            img_step3_back_btn = gr.Button("← Back to Results", variant="secondary")
    
                # ── Event Handlers ───────────────────────────────────────────────
    
                # Batch-mode toggle
                def toggle_img_batch(enabled):
                    return (
                        gr.update(visible=not enabled),   # single panel
                        gr.update(visible=enabled),        # batch panel
                        gr.update(visible=not enabled),    # analyze btn
                        gr.update(visible=enabled),        # batch btn
                    )
    
                # Batch-mode source toggle
                def toggle_img_batch_sourcing(mode):
                    is_upload = mode == "Upload Images"
                    return gr.update(visible=is_upload), gr.update(visible=not is_upload)
                    
                img_batch_source_mode.change(
                    fn=toggle_img_batch_sourcing,
                    inputs=img_batch_source_mode,
                    outputs=[img_batch_upload_group, img_batch_generate_group]
                )
    
                img_batch_toggle.change(
                    fn=toggle_img_batch,
                    inputs=img_batch_toggle,
                    outputs=[img_single_panel, img_batch_panel,
                             img_analyze_btn, img_batch_btn]
                )
    
                # Step nav helpers
                def img_start_loading():
                    return gr.update(visible=False), gr.update(visible=True)
    
                # Single analyse → generate caption → enable next
                def img_single_analyse(editor_img, batch_enabled):
                    from image_utility import blip_generate_caption_only
                    # Returns: caption_out, tokens_out, attr_out, orig_out,
                    #          caption_state, tokens_state, orig_state, attr_state,
                    #          next_btn, loading_msg, step2_tab, step3_tab,
                    #          single_results, batch_results_panel
                    cap, toks_str, _, orig, cap_str, toks = blip_generate_caption_only(editor_img)
                    return (
                        cap, toks_str, None, orig,
                        cap_str, toks, orig, None,
                        gr.update(visible=True, interactive=True, elem_classes=["btn-green"]),
                        gr.update(visible=False),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(visible=True),
                        gr.update(visible=False),
                    )
    
                img_analyze_btn.click(
                    fn=img_start_loading,
                    inputs=None,
                    outputs=[img_step1_next_btn, img_step1_loading]
                ).then(
                    fn=img_single_analyse,
                    inputs=[img_editor, img_batch_toggle],
                    outputs=[
                        img_caption_out, img_tokens_out, img_attr_out, img_original_out,
                        img_caption_state, img_tokens_state, img_original_state, img_attr_state,
                        img_step1_next_btn, img_step1_loading,
                        img_step2_tab, img_step3_tab,
                        img_single_results, img_batch_results_panel,
                    ],
                    show_progress="full"
                )
    
                # Batch analyse → caption all images → enable next
                def img_batch_analyse(mode, up_g1, up_g2, pr_g1, pr_g2, gen_count):
                    from image_utility import blip_batch_caption_images, generate_batch_images, generate_word_freq_chart
                    if mode == "Upload Images":
                        if not up_g1 and not up_g2:
                            return ([], [], [], [], None,
                                    gr.update(), gr.update(visible=False),
                                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                                    gr.update(visible=False), gr.update(visible=True))
                        res_g1, gal_g1 = blip_batch_caption_images(up_g1 or [])
                        res_g2, gal_g2 = blip_batch_caption_images(up_g2 or [])
                    else:
                        if (not pr_g1 or not pr_g1.strip()) and (not pr_g2 or not pr_g2.strip()):
                            return ([], [], [], [], None,
                                    gr.update(), gr.update(visible=False),
                                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                                    gr.update(visible=False), gr.update(visible=True))
                        imgs_g1 = generate_batch_images(pr_g1, gen_count) if pr_g1 and pr_g1.strip() else []
                        imgs_g2 = generate_batch_images(pr_g2, gen_count) if pr_g2 and pr_g2.strip() else []
                        res_g1, gal_g1 = blip_batch_caption_images(imgs_g1)
                        res_g2, gal_g2 = blip_batch_caption_images(imgs_g2)
                    
                    chart_img = generate_word_freq_chart(res_g1, res_g2)
                    
                    return (
                        res_g1,
                        res_g2,
                        gal_g1,
                        gal_g2,
                        chart_img,
                        gr.update(visible=True, interactive=True, elem_classes=["btn-green"]),
                        gr.update(visible=False),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(visible=False),    # hide single results
                        gr.update(visible=True),     # show batch results
                    )
    
                img_batch_btn.click(
                    fn=img_start_loading,
                    inputs=None,
                    outputs=[img_step1_next_btn, img_step1_loading]
                ).then(
                    fn=img_batch_analyse,
                    inputs=[img_batch_source_mode, img_batch_upload_g1, img_batch_upload_g2, img_batch_prompt_g1, img_batch_prompt_g2, img_batch_gen_count],
                    outputs=[
                        img_batch_results_g1, img_batch_results_g2,
                        img_batch_gallery_g1, img_batch_gallery_g2,
                        img_batch_word_freq_chart,
                        img_step1_next_btn, img_step1_loading,
                        img_step2_tab, img_step3_tab,
                        img_single_results, img_batch_results_panel,
                    ],
                    show_progress="full"
                )
    
                # Gallery click → load image data and reset analysis
                def on_gallery_select(results_g1, results_g2, group, evt: gr.SelectData):
                    results = results_g1 if group == "Group 1" else results_g2
                    idx = evt.index
                    if not results or idx >= len(results):
                        return idx, group, None, "", "", None, []
                    
                    item = results[idx]
                    toks_str = "\n".join([f"{i}: {t}" for i, t in enumerate(item["tokens"])])
                    
                    return (
                        idx,
                        group,
                        item["image"],   # selected/orig out
                        item["caption"], # caption out
                        toks_str,        # tokens out
                        item["image"],   # occlude editor
                        None,            # clear heatmap
                        item["caption"], # state
                        item["tokens"],  # state
                    )
    
                img_batch_gallery_g1.select(
                    fn=on_gallery_select,
                    # We need to know which group was clicked, but SelectData doesn't tell us.
                    # We'll use a wrapper or separate functions that call the unified one.
                    inputs=[img_batch_results_g1, img_batch_results_g2, gr.State("Group 1")],
                    outputs=[
                        img_batch_idx_slider, img_batch_group_selector,
                        img_batch_orig_out, img_batch_caption_out, img_batch_tokens_out,
                        img_occlude_editor, img_batch_attr_out,
                        img_caption_state, img_tokens_state
                    ]
                )
                img_batch_gallery_g2.select(
                    fn=on_gallery_select,
                    inputs=[img_batch_results_g1, img_batch_results_g2, gr.State("Group 2")],
                    outputs=[
                        img_batch_idx_slider, img_batch_group_selector,
                        img_batch_orig_out, img_batch_caption_out, img_batch_tokens_out,
                        img_occlude_editor, img_batch_attr_out,
                        img_caption_state, img_tokens_state
                    ]
                )
    
                # (Remaining click handlers are fine as is)
    
                # Batch: Compute Attribution for selected image (from results)
                img_batch_compute_btn.click(
                    fn=blip_analyze_image,
                    inputs=[
                        img_batch_orig_out, img_opacity_slider, img_steps_slider,
                        img_batch_token_slider, img_caption_state, img_tokens_state
                    ],
                    outputs=[
                        img_batch_caption_out, img_batch_tokens_out,
                        img_batch_attr_out, img_batch_orig_out,
                        img_caption_state, img_tokens_state
                    ]
                )

                # Batch: Re-caption from occluded image, then compute attribution
                img_batch_recompute_btn.click(
                    fn=blip_occlude_then_analyze,
                    inputs=[
                        img_occlude_editor, img_opacity_slider, img_steps_slider,
                        img_batch_token_slider
                    ],
                    outputs=[
                        img_batch_caption_out, img_batch_tokens_out,
                        img_batch_attr_out, img_batch_orig_out,
                        img_caption_state, img_tokens_state
                    ]
                )
    
                # Image Content Handlers
                image_load_btn.click(
                    fn=validate_and_load_image_model,
                    inputs=image_model_input,
                    outputs=[image_model_status, image_current_model_display]
                )

                # Step navigation
                img_step1_next_btn.click(
                    fn=lambda: gr.update(selected=1),
                    inputs=None, outputs=image_steps
                )
                img_step2_back_btn.click(
                    fn=lambda: gr.update(selected=0),
                    inputs=None, outputs=image_steps
                )
                img_step3_back_btn.click(
                    fn=lambda: gr.update(selected=1),
                    inputs=None, outputs=image_steps
                )
    

    
                # Autosave → Step 3
                def img_autosave_and_advance(
                    caption, attr_state, orig_state, batch_mode, batch_graph, 
                    batch_results_g1, batch_results_g2, group, selected_idx
                ):
                    from image_utility import save_image_version
                    if batch_mode:
                        results = batch_results_g1 if group == "Group 1" else batch_results_g2
                        # selected_idx might be None if nothing clicked yet
                        idx = int(selected_idx) if selected_idx is not None else 0
                        item = results[idx] if results and idx < len(results) else {}
                        cap = item.get("caption", caption)
                        orig = item.get("image", orig_state)
                    else:
                        cap = caption
                        orig = orig_state
                        
                    from datetime import datetime as _dt
                    label = f"Auto: {_dt.now().strftime('%Y-%m-%d %H:%M')} | {'Batch' if batch_mode else 'Single'}"
                    _, dropdown_update = save_image_version(
                        cap, attr_state, orig, 
                        batch_mode=batch_mode, 
                        batch_graph=batch_graph,
                        results_g1=batch_results_g1,
                        results_g2=batch_results_g2,
                        auto_label=label
                    )
                    return (
                        gr.update(selected=2),
                        dropdown_update,
                        gr.update(value=f"Autosaved: *{label}*", visible=True),
                    )
    
                img_step2_next_btn.click(
                    fn=img_autosave_and_advance,
                    inputs=[
                        img_caption_state, img_attr_state, img_original_state,
                        img_batch_mode, img_batch_word_freq_chart,
                        img_batch_results_g1, img_batch_results_g2, 
                        img_batch_group_selector, img_batch_idx_slider
                    ],
                    outputs=[image_steps, img_version_dropdown, img_autosave_status]
                )
    
                # Manual save
                def img_manual_save(
                    caption, attr_state, orig_state, 
                    batch_mode, batch_graph, batch_results_g1, batch_results_g2
                ):
                    return save_image_version(
                        caption, attr_state, orig_state,
                        batch_mode=batch_mode,
                        batch_graph=batch_graph,
                        results_g1=batch_results_g1,
                        results_g2=batch_results_g2
                    )
    
                img_save_btn.click(
                    fn=img_manual_save,
                    inputs=[
                        img_caption_state, img_attr_state, img_original_state,
                        img_batch_mode, img_batch_word_freq_chart, 
                        img_batch_results_g1, img_batch_results_g2
                    ],
                    outputs=[img_save_status, img_version_dropdown]
                )
    
                # Step 3: Load comparison
                img_version_dropdown.change(
                    fn=on_img_version_change,
                    inputs=[img_version_dropdown, img_batch_mode, img_batch_results_g1, img_batch_results_g2],
                    outputs=[img_comparison_output, img_download_selected_btn, img_download_csv_btn]
                )
    
                # Step 3: Clear
                img_clear_btn.click(
                    fn=clear_image_comparison,
                    inputs=None,
                    outputs=[img_comparison_output, img_version_dropdown]
                )
    
            # Section 3: Credit Risk Analyzer
            with gr.Column(visible=False) as credit_section:
                gr.Markdown("### Credit Risk Model Auditor")
                gr.Markdown("Analyze credit risk predictions and understand which factors influence the model's decisions.")
                
                with gr.Tabs(selected=0) as credit_steps:
                    # ── Step 1: Input & Configuration ────────────────────────────
                    with gr.Tab("Step 1: Input & Configuration", id=0):
                        with gr.Row():
                            with gr.Column(scale=1):
                                with gr.Accordion("1. Configure Model", open=True):
                                    credit_model_input = gr.Textbox(
                                        label="HuggingFace Model ID or URL",
                                        placeholder="e.g. Alfazril/credit-risk-prediction",
                                        value="Alfazril/credit-risk-prediction",
                                    )
                                    credit_load_btn = gr.Button("Load Model", variant="primary")
                                    credit_model_status = gr.Markdown("")
                                    gr.Markdown("*Requires a scikit-learn compatible tabular model (.pkl or .joblib).*")
                                
                                with gr.Accordion("2. Applicant Information", open=True):
                                    credit_age = gr.Slider(minimum=18, maximum=80, value=35, step=1, label="Age")
                                    credit_income = gr.Slider(minimum=10000, maximum=500000, value=50000, step=5000, label="Annual Income ($)")
                                    credit_score = gr.Slider(minimum=300, maximum=850, value=650, step=10, label="Credit Score")
                                    credit_debt_ratio = gr.Slider(minimum=0, maximum=100, value=30, step=1, label="Debt-to-Income Ratio (%)")
                                    credit_employment_years = gr.Slider(minimum=0, maximum=40, value=5, step=1, label="Years at Current Job")
                                    credit_loan_amount = gr.Slider(minimum=1000, maximum=100000, value=15000, step=1000, label="Loan Amount Requested ($)")
                                    credit_num_accounts = gr.Slider(minimum=0, maximum=20, value=3, step=1, label="Number of Credit Accounts")
                                    credit_delinquencies = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Past Delinquencies")
                                    
                                    with gr.Accordion("🔧 Advanced: Manual Feature Overrides (JSON)", open=False):
                                        gr.Markdown("Directly override any model feature. Example: `{\"person_income\": 150000, \"age\": 40}`")
                                        credit_feature_overrides = gr.Textbox(placeholder="{}", label="Overrides (JSON)", lines=2)
                                    
                                    with gr.Row():
                                        credit_predict_btn = gr.Button("Predict Risk", variant="primary")
                                        credit_reset_btn = gr.Button("Reset to Default", variant="secondary")
                                        credit_settings_export_btn = gr.DownloadButton("💾 Export Settings", variant="secondary")
                                        credit_settings_import_btn = gr.File(label="📂 Import Settings", file_count="single", file_types=[".json"])
                                    
                                    credit_method_dropdown = gr.Dropdown(
                                        choices=["shap", "lime", "integrated_gradients"],
                                        value="shap",
                                        label="Explanation Method",
                                        interactive=True
                                    )

                    # ── Step 2: Risk Assessment ──────────────────────────────────
                    with gr.Tab("Step 2: Risk Assessment", id=1, interactive=False) as credit_step2_tab:
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown("### Risk Assessment")
                                credit_current_model_display = gr.Markdown("**Model:** Loading...")
                                credit_risk_output = gr.HTML(
                                    label="Risk Prediction",
                                    value="<p>Predict risk in Step 1 to see results...</p>"
                                )
                                credit_feature_plot = gr.Plot(label="Feature Importance")
                            
                            with gr.Column(scale=1):
                                gr.Markdown("### Scenario Analysis")
                                credit_scenario_feature = gr.Dropdown(
                                    choices=["Age", "Income", "Credit Score", "Debt Ratio", "Employment Years"],
                                    value="Credit Score",
                                    label="Feature to Vary",
                                    interactive=True
                                )
                                credit_compare_btn = gr.Button("Compare Scenarios", variant="primary")
                                credit_scenario_plot = gr.Plot(label="Scenario Comparison")
                        
                        with gr.Row():
                            credit_step2_back_btn = gr.Button("← Back to Input", variant="secondary")
                            credit_step2_next_btn = gr.Button("Next: Download & Compare →", variant="primary")
                        credit_autosave_status = gr.Markdown("", visible=False)

                    # ── Step 3: Download, Share, and Compare ──────────────────────
                    with gr.Tab("Step 3: Download, Share, and Compare", id=2, interactive=False) as credit_step3_tab:
                        gr.Markdown("Save versions to track your audit progress and compare different model behaviors side-by-side.")
                        
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### Comparison Column A")
                                credit_version_dropdown_a = gr.Dropdown(choices=[], label="Select version A", interactive=True)
                                with gr.Row():
                                    credit_clear_btn_a = gr.Button("Clear A", variant="secondary", size="sm")
                                    credit_download_selected_btn_a = gr.DownloadButton("Download A (HTML)", variant="primary", visible=False, size="sm")
                                credit_comparison_output_a = gr.HTML(value="<p>Select a version to compare.</p>")

                            with gr.Column():
                                gr.Markdown("#### Comparison Column B")
                                credit_version_dropdown_b = gr.Dropdown(choices=[], label="Select version B", interactive=True)
                                with gr.Row():
                                    credit_clear_btn_b = gr.Button("Clear B", variant="secondary", size="sm")
                                    credit_download_selected_btn_b = gr.DownloadButton("Download B (HTML)", variant="primary", visible=False, size="sm")
                                credit_comparison_output_b = gr.HTML(value="<p>Select a version to compare.</p>")
                        
                        with gr.Row():
                            credit_download_all_btn = gr.DownloadButton("Download All Saved (HTML)", variant="secondary")
                            credit_step3_back_btn = gr.Button("← Back to Assessment", variant="secondary")

                gr.Markdown("""
                ### How to Use:
                - Adjust sliders in **Step 1** and click "Predict Risk".
                - View detailed metrics and scenario analysis in **Step 2**.
                - Save and compare your findings in **Step 3**.
                """)
                # event handlers:
    
                # Event Handlers for Credit Risk Tab
                def handle_predict_and_save(age, income, credit_score, debt_ratio, employment_years, 
                                          loan_amount, num_accounts, delinquencies, method, 
                                          feature_overrides_str):
                    # 1. Parse overrides
                    overrides = {}
                    if feature_overrides_str and feature_overrides_str.strip():
                        try:
                            overrides = json.loads(feature_overrides_str)
                        except Exception as e:
                            print(f"Error parsing feature overrides: {e}")
                    
                    # 2. Predict
                    html, fig, model_disp = predict_credit_risk(
                        age, income, credit_score, debt_ratio, employment_years, 
                        loan_amount, num_accounts, delinquencies, method, feature_overrides=overrides
                    )
                    
                    # 3. Autosave
                    choices, msg = save_credit_version(
                        age, income, credit_score, debt_ratio, employment_years, 
                        loan_amount, num_accounts, delinquencies, html
                    )
                    
                    # 3. Update UI
                    return (
                        html, fig, model_disp,
                        gr.update(visible=True, interactive=True), # Step 2 tab
                        gr.update(selected=1),                     # Switch to Step 2
                        gr.update(choices=choices),                # Update Dropdown A
                        gr.update(choices=choices),                # Update Dropdown B
                    )

                def handle_credit_settings_export(age, income, score, debt, emp, loan, accounts, delinq, model_id, method, overrides):
                    settings = {
                        "age": age, "income": income, "credit_score": score, "debt_ratio": debt,
                        "employment_years": emp, "loan_amount": loan, "num_accounts": accounts,
                        "delinquencies": delinq, "model_id": model_id, "method": method, "overrides": overrides
                    }
                    return export_settings(settings)
                
                def handle_credit_settings_import(file_info):
                    if not file_info: return [gr.update()]*11
                    from settings_manager import import_settings
                    data = import_settings(file_info.name)
                    if not data: return [gr.update()]*11
                    return (
                        data.get("age", 35), data.get("income", 50000), data.get("credit_score", 650),
                        data.get("debt_ratio", 30), data.get("employment_years", 5), data.get("loan_amount", 15000),
                        data.get("num_accounts", 3), data.get("delinquencies", 0), data.get("model_id", ""),
                        data.get("method", "shap"), data.get("overrides", "{}")
                    )

                credit_predict_btn.click(
                    fn=handle_predict_and_save,
                    inputs=[
                        credit_age, credit_income, credit_score, credit_debt_ratio,
                        credit_employment_years, credit_loan_amount, credit_num_accounts,
                        credit_delinquencies, credit_method_dropdown, credit_feature_overrides
                    ],
                    outputs=[
                        credit_risk_output, credit_feature_plot, credit_current_model_display,
                        credit_step2_tab, credit_steps, credit_version_dropdown_a, credit_version_dropdown_b
                    ]
                )
                
                credit_settings_export_btn.click(
                    fn=handle_credit_settings_export,
                    inputs=[
                        credit_age, credit_income, credit_score, credit_debt_ratio,
                        credit_employment_years, credit_loan_amount, credit_num_accounts,
                        credit_delinquencies, credit_model_input, credit_method_dropdown, credit_feature_overrides
                    ],
                    outputs=credit_settings_export_btn
                )
                
                credit_settings_import_btn.upload(
                    fn=handle_credit_settings_import,
                    inputs=credit_settings_import_btn,
                    outputs=[
                        credit_age, credit_income, credit_score, credit_debt_ratio,
                        credit_employment_years, credit_loan_amount, credit_num_accounts,
                        credit_delinquencies, credit_model_input, credit_method_dropdown, credit_feature_overrides
                    ]
                )

                credit_step2_next_btn.click(
                    fn=lambda: (gr.update(selected=2), gr.update(interactive=True)),
                    outputs=[credit_steps, credit_step3_tab]
                )

                credit_step2_back_btn.click(fn=lambda: gr.update(selected=0), outputs=credit_steps)
                credit_step3_back_btn.click(fn=lambda: gr.update(selected=1), outputs=credit_steps)

                def on_credit_version_change(selected):
                    from credit_utility import _saved_credit_versions
                    if not selected or selected not in _saved_credit_versions:
                        return "<p>Select a version.</p>", gr.update(visible=False)
                    data = _saved_credit_versions[selected]
                    return data['html'], gr.update(visible=True)

                credit_version_dropdown_a.change(
                    fn=on_credit_version_change,
                    inputs=credit_version_dropdown_a,
                    outputs=[credit_comparison_output_a, credit_download_selected_btn_a]
                )
                credit_version_dropdown_b.change(
                    fn=on_credit_version_change,
                    inputs=credit_version_dropdown_b,
                    outputs=[credit_comparison_output_b, credit_download_selected_btn_b]
                )

                credit_download_selected_btn_a.click(fn=export_selected_credit_html, inputs=credit_version_dropdown_a, outputs=credit_download_selected_btn_a)
                credit_download_selected_btn_b.click(fn=export_selected_credit_html, inputs=credit_version_dropdown_b, outputs=credit_download_selected_btn_b)
                credit_download_all_btn.click(fn=export_all_credit_html, outputs=credit_download_all_btn)

                credit_clear_btn_a.click(fn=lambda: ("<p>Select a version.</p>", gr.update(visible=False)), outputs=[credit_comparison_output_a, credit_download_selected_btn_a])
                credit_clear_btn_b.click(fn=lambda: ("<p>Select a version.</p>", gr.update(visible=False)), outputs=[credit_comparison_output_b, credit_download_selected_btn_b])

                credit_reset_btn.click(
                    fn=reset_credit_data,
                    inputs=None,
                    outputs=[
                        credit_age, credit_income, credit_score, credit_debt_ratio,
                        credit_employment_years, credit_loan_amount, credit_num_accounts,
                        credit_delinquencies
                    ]
                )

                credit_compare_btn.click(
                    fn=compare_scenarios,
                    inputs=[
                        credit_age, credit_income, credit_score, credit_debt_ratio,
                        credit_employment_years, credit_loan_amount, credit_num_accounts,
                        credit_delinquencies, credit_scenario_feature
                    ],
                    outputs=credit_scenario_plot
                )
    
                gr.Markdown("""
                ### How to Use:
                - Adjust the sliders to input applicant information
                - Click "Predict Risk" to see the model's credit risk assessment
                - View feature importance to understand which factors most influenced the decision
                - Use "Scenario Analysis" to explore how changing specific features affects the prediction
                - Export a detailed report for documentation and compliance purposes
                """)
    
            # --- Global Export Handlers ---
            from resume_utility import export_all_html as res_export_all, export_selected_html as res_export_selected, export_batch_csv as res_export_csv
            from image_utility import export_all_html as img_export_all, export_selected_html as img_export_selected, export_batch_csv as img_export_csv
            
            # Resume Exports
            resume_download_all_btn.click(fn=res_export_all, inputs=None, outputs=resume_download_all_btn)
            
            resume_download_selected_btn_a.click(fn=res_export_selected, inputs=resume_version_dropdown_a, outputs=resume_download_selected_btn_a)
            resume_download_selected_btn_b.click(fn=res_export_selected, inputs=resume_version_dropdown_b, outputs=resume_download_selected_btn_b)
            resume_download_csv_btn.click(fn=res_export_csv, inputs=batch_results_state, outputs=resume_download_csv_btn)
            
            # Image Exports
            img_download_all_btn.click(fn=img_export_all, inputs=None, outputs=img_download_all_btn)
            
            img_download_selected_btn.click(fn=img_export_selected, inputs=img_version_dropdown, outputs=img_download_selected_btn)
            img_download_csv_btn.click(fn=img_export_csv, inputs=[img_batch_results_g1, img_batch_results_g2], outputs=img_download_csv_btn)
            
            # Credit Model Load
            credit_load_btn.click(
                fn=validate_and_load_credit_model,
                inputs=credit_model_input,
                outputs=[credit_model_status, credit_current_model_display]
            )

    
    # Navbar Handlers
    nav_outputs = [intro_page, main_app_layout, resume_section, image_section, credit_section, nav_resume_btn, nav_image_btn, nav_credit_btn]
    
    intro_resume_btn.click(fn=lambda: navigate("resume"), outputs=nav_outputs)
    intro_image_btn.click(fn=lambda: navigate("image"), outputs=nav_outputs)
    intro_credit_btn.click(fn=lambda: navigate("credit"), outputs=nav_outputs)
    
    nav_home_btn.click(fn=lambda: navigate("home"), outputs=nav_outputs)
    nav_resume_btn.click(fn=lambda: navigate("resume"), outputs=nav_outputs)
    nav_image_btn.click(fn=lambda: navigate("image"), outputs=nav_outputs)
    nav_credit_btn.click(fn=lambda: navigate("credit"), outputs=nav_outputs)

# Cell 6: Launch the interface
demo.launch(debug=True, share=True, show_error=True)