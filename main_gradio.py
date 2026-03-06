import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients
import gradio as gr
import io
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
    get_resume_model_choices,      # NEW
    switch_resume_model,
    initialize_model,              # NEW
    initialize_llama_model,        # NEW
    llama_model_name,              # NEW
    get_suggested_anchor,          # NEW
    GENDER_PRESET_TEMPLATE,        # NEW
    CUSTOM_EXTENDED_TEMPLATE       # NEW
)
from image_utility import (
    # BLIP / LVLM-Interpret functions
    blip_generate_caption_only,
    blip_analyze_image,
    blip_occlude_then_analyze,
    blip_batch_caption_images,
    save_image_version,
    load_image_version,
    clear_image_comparison,
    get_image_version_choices,
)
from credit_utility import (
    sample_credit_data, 
    predict_credit_risk, 
    get_feature_importance, 
    compare_scenarios, 
    reset_credit_data, 
    export_credit_report,
    get_credit_model_choices,      # NEW
    switch_credit_model            # NEW
)

global has_explanation
has_explanation = False

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
def explain_resume_split(lead, body, end, continuation_state, fulltext_state, method):
    """Explain resume from three separate text inputs."""
    full_text = combine_resume_parts(lead, body, end)
    return explain_resume(full_text, continuation_state, fulltext_state, method)


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
    cb.set_label('Importance', fontsize=12)
    cb.ax.set_xticks([0, 1])
    cb.ax.set_xticklabels(['Less Important', 'More Important'])
    
    plt.tight_layout()
    return fig


# Initialize the split parts from sample_corpus
initial_lead, initial_body, initial_end = split_resume_corpus(sample_corpus)


# Cell 5: Create the Gradio Interface
custom_css = """
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
"""
with gr.Blocks(title="Algorithmic Audit Toolkit", css=custom_css) as demo:
    with gr.Column(visible=True) as intro_page:
        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=2):
                gr.Markdown("<h1 style='text-align: center;'>Algorithmic Audit Toolkit</h1>")
                gr.Markdown("<h3 style='text-align: center;'>Prototype Auditing Interface</h3>")
                gr.Markdown("The following interface is meant to be used for gray-box auditing with access to an explanation. Each tab represents a different type of functionality (resume screening, image captioning, and credit risk) for which you can try out multiple models and interpretability methods.")
                
                gr.Markdown("### Instructions")
                gr.Markdown("To assess the model, select an interface below, edit the input on the left side, and press 'Analyze' to see the output. In the middle column, if applicable, use the 'Explain' feature to see likely attributions. Then press 'Save Current Version' to enable comparison between inputs or models on the right column.")
                
                gr.Markdown("<br>")
                with gr.Row():
                    intro_resume_btn = gr.Button("Resume Screener", variant="primary", size="lg")
                    intro_image_btn = gr.Button("Image Captioning", variant="primary", size="lg")
                    intro_credit_btn = gr.Button("Credit Risk", variant="primary", size="lg")
            with gr.Column(scale=1):
                pass

    with gr.Row(visible=False) as main_app_layout:
        with gr.Column(scale=1, min_width=200) as sidebar:
            gr.Markdown("### Navigation")
            nav_home_btn = gr.Button("🏠 Home", variant="secondary")
            gr.Markdown("---")
            nav_resume_btn = gr.Button("Resume Screener", variant="primary")
            nav_image_btn = gr.Button("Image Captioning", variant="secondary", interactive=True)
            nav_credit_btn = gr.Button("Credit Risk", variant="secondary", interactive=False)

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
    
                    # ── Step 1: Input & Configuration ───────────────────────────
                    with gr.Tab("Step 1: Input & Configuration", id=0):
                        gr.Markdown("Modify the resume content to identify potential bias across different characteristics (e.g., gender, race, age). Then configure the analysis parameters below.")
    
                        with gr.Row():
                            with gr.Column(scale=1):
                                # Lead Prompt Section
                                with gr.Accordion("Lead Prompt", open=False):
                                    gr.Markdown("*The part of the prompt given to the model before the resume content.*")
                                    resume_lead_input = gr.Textbox(
                                        value=initial_lead,
                                        lines=3,
                                        label="Lead Prompt",
                                        placeholder="Leading instructions or context..."
                                    )
                                    with gr.Row():
                                        revert_lead_btn = gr.Button("Revert Lead", variant="secondary", size="sm")
    
                                # Main Body Section
                                gr.Markdown("#### Main Resume Body")
                                resume_body_input = gr.Textbox(
                                    value=initial_body,
                                    lines=12,
                                    label="Resume Content",
                                    placeholder="Paste the main resume content here..."
                                )
                                with gr.Row():
                                    revert_body_btn = gr.Button("Revert Body", variant="secondary", size="sm")
    
                                # End Prompt Section
                                with gr.Accordion("End Prompt", open=False):
                                    gr.Markdown("*The part of the prompt given to the model after the resume content.*")
                                    resume_end_input = gr.Textbox(
                                        value=initial_end,
                                        lines=3,
                                        label="End Prompt",
                                        placeholder="Closing instructions..."
                                    )
                                    with gr.Row():
                                        revert_end_btn = gr.Button("Revert End", variant="secondary", size="sm")
    
                            with gr.Column(scale=1):
                                # Model selection (moved here)
                                with gr.Row():
                                    resume_model_dropdown = gr.Dropdown(
                                        choices=get_resume_model_choices(),
                                        value=get_resume_model_choices()[0],
                                        label="Select Resume Screening Model",
                                        interactive=True,
                                        scale=3
                                    )
                                    resume_model_status = gr.Markdown("")
    
                                temperature_slider = gr.Slider(
                                    minimum=0, maximum=1, value=0.45, step=0.01, 
                                    label="Model Temperature",
                                    info="Controls randomness. Lower values are more deterministic; higher values (up to 1) increase variability."
                                )
    
                                gr.Markdown("---")
                                gr.Markdown("### Enable Batch Analysis")
                                gr.Markdown("**Analyze many variations to understand statistical patterns across demographics**")
                                batch_analysis_toggle = gr.Checkbox(
                                    label="Enable Batch Audit Mode",
                                    value=False,
                                    info="Run analysis with LLM or Preset variations"
                                )
    
                                batch_preset_selection = gr.Dropdown(
                                    choices=["Preset: Gender", "Natural Language (LLM)", "Custom Extended"],
                                    value="Preset: Gender",
                                    label="Variation Content Strategy",
                                    visible=False
                                )
    
                                batch_num_variations = gr.Slider(
                                    minimum=3, maximum=100, value=20, step=1,
                                    label="Number of Variations",
                                    visible=False,
                                    info="How many variations to test."
                                )
    
                                batch_token_input = gr.Textbox(
                                    label="Token to Vary (Anchor)",
                                    placeholder="e.g., John",
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
    
                                gr.Markdown("---")
                                with gr.Row():
                                    resume_analyze_btn = gr.Button(
                                        "Analyze (Single Run)", 
                                        variant="primary", 
                                        elem_classes=["btn-orange"]
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
                            value="<p>Click 'Analyze' in Step 1 to see results...</p>"
                        )
                        resume_method_dropdown = gr.Dropdown(
                            choices=["integrated_gradients", "layer_integrated_gradients", "shap"],
                            value="integrated_gradients",
                            label="Explanation Method",
                            interactive=has_explanation
                        )
                        resume_explain_btn = gr.Button("Explain - Button Disabled??", variant='primary', interactive=False, visible=has_explanation)
                        explanation_html = gr.HTML(
                            label="Explanation",
                            value="<p>No explanation generated. Click 'Explain' to see results...</p>",
                            visible=has_explanation
                        )
    
                        # Batch Interpretation Carousel
                        with gr.Accordion("Explanations by Variation (Currently Disabled)", open=False, visible=has_explanation) as batch_carousel_accordion:
                            gr.Markdown("### Variation Explorer")
                            gr.Markdown("Select a variation from your last batch run to see its specific interpretability highlights.")
    
                            batch_results_state = gr.State([])
    
                            with gr.Row():
                                carousel_index_slider = gr.Slider(
                                    minimum=0, maximum=19, value=0, step=1,
                                    label="Select Variation index",
                                    interactive=False
                                )
    
                            carousel_preview_html = gr.HTML(
                                label="Variation Preview",
                                value="<p>Select a variation to see details...</p>"
                            )
    
                            carousel_explain_btn = gr.Button("Explain Selection", variant="primary", interactive=has_explanation)
    
                            carousel_explanation_html = gr.HTML(
                                label="Variation Explanation Highlights",
                                value="<p>Detailed analysis of the selected variation will appear here...</p>",
                                visible=has_explanation
                            )
    
                        # Manual save button (still available on Step 2)
                        with gr.Row():
                            resume_save_btn = gr.Button("Save Current Version", variant="primary")
                        resume_save_status = gr.Markdown("")
    
                        gr.Markdown("---")
                        with gr.Row():
                            step2_back_btn = gr.Button("← Back to Input", variant="secondary")
                            step2_next_btn = gr.Button("Next: Compare & Track →", variant="primary")
                        autosave_status = gr.Markdown("", visible=False)
    
                    # ── Step 3: Compare & Track ──────────────────────────────────
                    with gr.Tab("Step 3: Compare & Track", id=2, interactive=False) as step3_tab:
                        gr.Markdown("Save versions to track your audit progress and compare different model behaviors side-by-side.")
                        gr.Markdown("*Versions are autosaved when you navigate here from Step 2.*")
    
                        resume_version_dropdown = gr.Dropdown(
                            choices=[],
                            label="Select saved version to compare",
                            interactive=True
                        )
                        resume_clear_btn = gr.Button("Clear Comparison", variant="secondary", interactive=False)
                        resume_comparison_output = gr.HTML(
                            label="Saved Highlights",
                            value="<p>No comparison loaded. Save a version and select it from the dropdown to compare.</p>"
                        )
    
                        gr.Markdown("---")
                        with gr.Row():
                            step3_back_btn = gr.Button("← Back to Results", variant="secondary")
    
                # ── Event Handlers ───────────────────────────────────────────────
    
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
                        dropdown_update,                  # refresh version dropdown
                        gr.update(value=f"Autosaved: *{auto_label}*", visible=True),  # show autosave notice
                    )
    
                step2_next_btn.click(
                    fn=autosave_and_advance,
                    inputs=[
                        resume_lead_input, resume_body_input, resume_end_input,
                        pure_html_output, explanation_html,
                        batch_analysis_toggle, batch_results_state
                    ],
                    outputs=[screener_steps, resume_version_dropdown, autosave_status]
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
    
                def revert_all():
                    lead, body, end = split_resume_corpus(sample_corpus)
                    return lead, body, end
    
                resume_reset_all_btn.click(
                    fn=revert_all,
                    inputs=None,
                    outputs=[resume_lead_input, resume_body_input, resume_end_input]
                )
    
                def start_analysis_loading():
                    return gr.update(visible=False), gr.update(visible=True)
    
                def process_and_enable_next(lead, body, end, method, temp, batch_enabled, batch_token, batch_num_vars, batch_dimension, variations_code):
                    results = process_resume_split(lead, body, end, method, temp, batch_enabled, batch_token, batch_num_vars, batch_dimension, variations_code)
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
                        resume_method_dropdown, temperature_slider,
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
                        resume_method_dropdown, temperature_slider,
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
                    html = f"""
                    <div style="padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #fff; color: #000;">
                        <p style="color: #000;"><strong>Variation:</strong> {res['variation']}</p>
                        <p style="color: #000;"><strong>Category:</strong> {res['category']}</p>
                        <p style="color: #000;"><strong>Sentiment:</strong> <span style="color: {sentiment_color}; font-weight: bold;">{res['sentiment'].upper()}</span></p>
                        <hr>
                        <p style="color: #000;"><strong>Continuation:</strong> "{res['continuation']}"</p>
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
                    inputs=[batch_results_state, carousel_index_slider, resume_method_dropdown],
                    outputs=carousel_explanation_html
                )
    
                # Explain button
                resume_explain_btn.click(
                    fn=explain_resume_split,
                    inputs=[resume_lead_input, resume_body_input, resume_end_input, continuation_state, fulltext_state, resume_method_dropdown],
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
                    outputs=[resume_save_status, resume_version_dropdown]
                )
    
                resume_version_dropdown.change(
                    fn=load_resume_version,
                    inputs=resume_version_dropdown,
                    outputs=resume_comparison_output
                )
    
                resume_clear_btn.click(
                    fn=clear_resume_comparison,
                    inputs=None,
                    outputs=[resume_comparison_output, resume_version_dropdown]
                )
    
                resume_model_dropdown.change(
                    fn=switch_resume_model,
                    inputs=resume_model_dropdown,
                    outputs=[resume_model_status, resume_current_model_display]
                )
    
    
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
    
                            with gr.Column(scale=1):
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
                                img_batch_group_selector = gr.Radio(["Group 1", "Group 2"], value="Group 1", label="Select Group to Occlude")
                                img_batch_idx_slider = gr.Slider(
                                    minimum=0, maximum=10, value=0, step=1,
                                    label="Selected Image Index"
                                )
                                img_load_occlude_btn = gr.Button(
                                    "Load Selected for Occlusion",
                                    variant="secondary"
                                )
    
                            gr.Markdown("#### Occlusion Editor (loaded image)")
                            gr.Markdown(
                                "*Paint or erase regions, then click 'Compute Attribution' "
                                "to see how the caption changes.*"
                            )
                            img_occlude_editor = gr.ImageEditor(
                                label="Edit / Occlude Loaded Image",
                                type="pil",
                                brush=gr.Brush(colors=["#000000", "#FF0000", "#FFFFFF"]),
                                eraser=gr.Eraser(),
                            )
                            img_batch_token_slider = gr.Slider(
                                minimum=0, maximum=20, value=0, step=1,
                                label="Select Token Index to Attribute",
                                interactive=True
                            )
                            img_batch_compute_btn = gr.Button(
                                "🔬 Re-Caption & Compute Attribution",
                                variant="primary"
                            )
                            img_batch_caption_out = gr.Textbox(
                                label="Caption (updated from occluded image)", lines=2, interactive=False
                            )
                            img_batch_tokens_out = gr.Textbox(
                                label="Tokens", lines=5, interactive=False
                            )
                            with gr.Row():
                                img_batch_attr_out = gr.Image(
                                    label="Attribution Heatmap", type="pil"
                                )
                                img_batch_orig_out = gr.Image(
                                    label="Loaded / Edited Image", type="pil"
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
    
                    # ── Step 3: Compare & Track ───────────────────────────────────
                    with gr.Tab("Step 3: Compare & Track", id=2, interactive=False) as img_step3_tab:
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
                        img_comparison_out = gr.HTML(
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
    
                # Gallery click → update index slider and group
                def handle_gallery_select_g1(evt: gr.SelectData):
                    return evt.index, "Group 1"
                def handle_gallery_select_g2(evt: gr.SelectData):
                    return evt.index, "Group 2"
    
                img_batch_gallery_g1.select(
                    fn=handle_gallery_select_g1,
                    inputs=None,
                    outputs=[img_batch_idx_slider, img_batch_group_selector]
                )
                img_batch_gallery_g2.select(
                    fn=handle_gallery_select_g2,
                    inputs=None,
                    outputs=[img_batch_idx_slider, img_batch_group_selector]
                )
    
                # Load batch image into occlusion editor
                def load_for_occlusion(results_g1, results_g2, group, idx):
                    results = results_g1 if group == "Group 1" else results_g2
                    if not results or int(idx) >= len(results):
                        return None, "", "", "", []
                    item = results[int(idx)]
                    toks_str = "\n".join([f"{i}: {t}" for i, t in enumerate(item["tokens"])])
                    return (
                        item["image"],   # into editor
                        item["caption"], # caption textbox
                        toks_str,        # tokens textbox
                        item["caption"], # caption state
                        item["tokens"],  # tokens state
                    )
    
                img_load_occlude_btn.click(
                    fn=load_for_occlusion,
                    inputs=[img_batch_results_g1, img_batch_results_g2, img_batch_group_selector, img_batch_idx_slider],
                    outputs=[
                        img_occlude_editor,
                        img_batch_caption_out, img_batch_tokens_out,
                        img_caption_state, img_tokens_state,
                    ]
                )
    
                # Single: Compute Attribution
                img_compute_btn.click(
                    fn=blip_analyze_image,
                    inputs=[
                        img_editor, img_opacity_slider, img_steps_slider,
                        img_token_slider, img_caption_state, img_tokens_state
                    ],
                    outputs=[
                        img_caption_out, img_tokens_out, img_attr_out, img_original_out,
                        img_caption_state, img_tokens_state
                    ]
                ).then(
                    fn=lambda attr, orig: (attr, orig),
                    inputs=[img_attr_out, img_original_out],
                    outputs=[img_attr_state, img_original_state]
                )
    
                # Batch: Re-caption from occluded image, then compute attribution
                img_batch_compute_btn.click(
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
                    caption, attr_state, orig_state, batch_mode, batch_results_g1, batch_results_g2, group, selected_idx
                ):
                    from image_utility import save_image_version
                    if batch_mode:
                        results = batch_results_g1 if group == "Group 1" else batch_results_g2
                        idx = int(selected_idx)
                        item = results[idx] if idx < len(results) else {}
                        cap = item.get("caption", caption)
                        orig = item.get("image", orig_state)
                    else:
                        cap = caption
                        orig = orig_state
                    from datetime import datetime as _dt
                    label = f"Auto: {_dt.now().strftime('%Y-%m-%d %H:%M')} | {'Batch' if batch_mode else 'Single'}"
                    _, dropdown_update = save_image_version(cap, attr_state, orig, auto_label=label)
                    return (
                        gr.update(selected=2),
                        dropdown_update,
                        gr.update(value=f"Autosaved: *{label}*", visible=True),
                    )
    
                img_step2_next_btn.click(
                    fn=img_autosave_and_advance,
                    inputs=[
                        img_caption_state, img_attr_state, img_original_state,
                        img_batch_mode, img_batch_results_g1, img_batch_results_g2, img_batch_group_selector, img_batch_idx_slider
                    ],
                    outputs=[image_steps, img_version_dropdown, img_autosave_status]
                )
    
                # Manual save
                def img_manual_save(caption, attr_state, orig_state):
                    return save_image_version(caption, attr_state, orig_state)
    
                img_save_btn.click(
                    fn=img_manual_save,
                    inputs=[img_caption_state, img_attr_state, img_original_state],
                    outputs=[img_save_status, img_version_dropdown]
                )
    
                # Step 3: Load comparison
                img_version_dropdown.change(
                    fn=load_image_version,
                    inputs=img_version_dropdown,
                    outputs=img_comparison_out
                )
    
                # Step 3: Clear
                img_clear_btn.click(
                    fn=clear_image_comparison,
                    inputs=None,
                    outputs=[img_comparison_out, img_version_dropdown]
                )
    
            # Section 3: Credit Risk Analyzer
            with gr.Column(visible=False) as credit_section:
                gr.Markdown("### Credit Risk Model Auditor")
                gr.Markdown("Analyze credit risk predictions and understand which factors influence the model's decisions.")
                
                # ADDED: Model selection dropdown
                with gr.Row():
                    credit_model_dropdown = gr.Dropdown(
                        choices=get_credit_model_choices(),
                        value=get_credit_model_choices()[0],
                        label="Select Credit Risk Model",
                        interactive=True,
                        scale=3
                    )
                    credit_model_status = gr.Markdown("")
    
                with gr.Row():
                    # Left column - Input features
                    with gr.Column(scale=1):
                        gr.Markdown("### Applicant Information")
                        credit_age = gr.Slider(minimum=18, maximum=80, value=35, step=1, label="Age")
                        credit_income = gr.Slider(minimum=10000, maximum=500000, value=50000, step=5000, label="Annual Income ($)")
                        credit_score = gr.Slider(minimum=300, maximum=850, value=650, step=10, label="Credit Score")
                        credit_debt_ratio = gr.Slider(minimum=0, maximum=100, value=30, step=1, label="Debt-to-Income Ratio (%)")
                        credit_employment_years = gr.Slider(minimum=0, maximum=40, value=5, step=1, label="Years at Current Job")
                        credit_loan_amount = gr.Slider(minimum=1000, maximum=100000, value=15000, step=1000, label="Loan Amount Requested ($)")
                        credit_num_accounts = gr.Slider(minimum=0, maximum=20, value=3, step=1, label="Number of Credit Accounts")
                        credit_delinquencies = gr.Slider(minimum=0, maximum=10, value=0, step=1, label="Past Delinquencies")
                        
                        with gr.Row():
                            credit_predict_btn = gr.Button("Predict Risk", variant="primary")
                            credit_reset_btn = gr.Button("Reset to Default", variant="secondary")
                        
                        credit_method_dropdown = gr.Dropdown(
                            choices=["shap", "lime", "integrated_gradients"],
                            value="shap",
                            label="Explanation Method",
                            interactive=True
                        )
    
                    # Middle column - Prediction results
                    with gr.Column(scale=1):
                        gr.Markdown("### Risk Assessment")
                        # ADDED: Show current model
                        credit_current_model_display = gr.Markdown("**Model:** Loading...")
                        credit_risk_output = gr.HTML(
                            label="Risk Prediction",
                            value="<p>Enter applicant information and click 'Predict Risk'...</p>"
                        )
                        credit_feature_plot = gr.Plot(label="Feature Importance")
                        
                    # Right column - Scenario comparison
                    with gr.Column(scale=1):
                        gr.Markdown("### Scenario Analysis")
                        gr.Markdown("Compare how changes to specific features affect the prediction")
                        credit_scenario_feature = gr.Dropdown(
                            choices=["Age", "Income", "Credit Score", "Debt Ratio", "Employment Years"],
                            value="Credit Score",
                            label="Feature to Vary",
                            interactive=True
                        )
                        credit_compare_btn = gr.Button("Compare Scenarios", variant="primary")
                        credit_scenario_plot = gr.Plot(label="Scenario Comparison")
                        credit_export_btn = gr.Button("Export Analysis Report", variant="secondary")
                        credit_export_status = gr.Markdown("")
                # event handlers:
    
                # MODIFIED: Added credit_current_model_display to outputs
                credit_predict_btn.click(
                    fn=predict_credit_risk,
                    inputs=[
                        credit_age, credit_income, credit_score, credit_debt_ratio,
                        credit_employment_years, credit_loan_amount, credit_num_accounts,
                        credit_delinquencies, credit_method_dropdown
                    ],
                    outputs=[credit_risk_output, credit_feature_plot, credit_current_model_display]  # ADDED 3rd output
                )
    
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
    
                credit_export_btn.click(
                    fn=export_credit_report,
                    inputs=[
                        credit_age, credit_income, credit_score, credit_debt_ratio,
                        credit_employment_years, credit_loan_amount, credit_num_accounts,
                        credit_delinquencies, credit_risk_output, credit_feature_plot
                    ],
                    outputs=credit_export_status
                )
    
                # ADDED: Handle model changes
                credit_model_dropdown.change(
                    fn=switch_credit_model,
                    inputs=credit_model_dropdown,
                    outputs=[credit_model_status, credit_current_model_display]
                )
    
                gr.Markdown("""
                ### How to Use:
                - Adjust the sliders to input applicant information
                - Click "Predict Risk" to see the model's credit risk assessment
                - View feature importance to understand which factors most influenced the decision
                - Use "Scenario Analysis" to explore how changing specific features affects the prediction
                - Export a detailed report for documentation and compliance purposes
                """)
    
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
demo.launch(debug=True, share=True)