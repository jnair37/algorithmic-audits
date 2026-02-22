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
    llama_model_name               # NEW
)
from image_utility import (
    test_img, 
    run_integrated_gradients, 
    generate_caption_only,
    run_gradcam_analysis,
    compare_multiple_images,
    get_sample_image_choices,
    get_sample_image_by_index,
    get_image_model_choices,       # NEW
    switch_image_model
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
def process_resume_split(lead, body, end, method, temp, batch_enabled, batch_token, batch_num_vars, batch_dimension, variations_code):
    """Process resume from three separate text inputs, with optional batch analysis."""
    full_text = combine_resume_parts(lead, body, end)
    
    if batch_enabled:
        # Call batch processing function with Faker parameters or custom code
        # NEW: returns (html, batch_results, None, model_display)
        html, results, _, display = process_batch_resume(full_text, method, temp, batch_token, batch_num_vars, batch_dimension, variations_code)
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
with gr.Blocks(title="Test Auditing Interface") as demo:
    gr.Markdown("# Prototype Auditing Interface")
    gr.Markdown("The following interface is meant to be used for gray-box auditing with access to an explanation. Each tab represents a different type of functionality (resume screening, image captioning, and credit risk) for which you can try out multiple models and interpretability methods.")
    gr.Markdown("# Instructions")
    gr.Markdown("To assess the model, edit the input on the left side and press 'Analyze' to see the output. In the middle column, if applicable, use the 'Explain' feature to see likely attributions. Then press 'Save Current Version' to enable comparison between inputs or models on the right column.")

    with gr.Tabs():

        # Tab 1: Resume Screener
        with gr.Tab("Resume Screener"):
            continuation_state = gr.State()
            fulltext_state = gr.State()
            # Track the active inner step (0=Input, 1=Results, 2=Compare)
            screener_tab_state = gr.State(value=0)
            # Track whether an analysis has been run (gates the first Next button)
            analysis_done_state = gr.State(value=False)

            gr.Markdown("### Resume Screener")

            # Model selection row (always visible above the step tabs)
            with gr.Row():
                resume_model_dropdown = gr.Dropdown(
                    choices=get_resume_model_choices(),
                    value=get_resume_model_choices()[0],
                    label="Select Resume Screening Model",
                    interactive=True,
                    scale=3
                )
                resume_model_status = gr.Markdown("")

            # ── Inner step tabs ──────────────────────────────────────────────
            with gr.Tabs(selected=0) as screener_steps:

                # ── Step 1: Input & Configuration ───────────────────────────
                with gr.Tab("Step 1: Input & Configuration", id=0):
                    gr.Markdown("Modify the resume content to identify potential bias across different characteristics (e.g., gender, race, age). Then configure the analysis parameters below.")

                    # Lead Prompt Section
                    gr.Markdown("#### Lead Prompt (scaffolding before resume)")
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
                        lines=8,
                        label="Resume Content",
                        placeholder="Paste the main resume content here..."
                    )
                    with gr.Row():
                        revert_body_btn = gr.Button("Revert Body", variant="secondary", size="sm")

                    # End Prompt Section
                    gr.Markdown("#### End Prompt (scaffolding after resume)")
                    resume_end_input = gr.Textbox(
                        value=initial_end,
                        lines=3,
                        label="End Prompt",
                        placeholder="Closing instructions..."
                    )
                    with gr.Row():
                        revert_end_btn = gr.Button("Revert End", variant="secondary", size="sm")

                    temperature_slider = gr.Slider(minimum=0, maximum=1, value=0.45, step=0.01, label="Model Temperature")

                    gr.Markdown("---")
                    gr.Markdown("### Analysis Settings")
                    gr.Markdown("Choose between a single-run analysis or a batch audit of multiple variations.")
                    batch_analysis_toggle = gr.Checkbox(
                        label="Enable Batch Analysis",
                        value=False,
                        info="Run analysis with LLM-generated variations"
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
                        info="Enter the token from your resume that you want to vary."
                    )

                    batch_nl_input = gr.Textbox(
                        label="What characteristic or content do you want to audit for bias?",
                        placeholder="e.g., 'Vary the names to include different religious backgrounds', 'Vary the graduation year to test for age bias'.",
                        visible=False,
                        lines=2,
                        info="Describe how you want to vary the resume to uncover potential discriminatory patterns."
                    )

                    batch_generate_code_btn = gr.Button("Generate Variation Code", variant="secondary", visible=False)

                    batch_code_preview = gr.Textbox(
                        label="Variation Code (Review and Edit)",
                        placeholder="Generated Python code will appear here...",
                        lines=8,
                        visible=False,
                        interactive=True
                    )

                    with gr.Row():
                        resume_analyze_btn = gr.Button("Analyze (Single Run)", variant="secondary")
                        batch_execute_btn = gr.Button("Execute Batch Audit", variant="primary", visible=False)
                        resume_reset_all_btn = gr.Button("Revert All", variant="secondary")

                    gr.Markdown("---")
                    with gr.Row():
                        step1_next_btn = gr.Button("Next: View Results →", variant="primary", interactive=False)

                # ── Step 2: Results & Interpretation ────────────────────────
                with gr.Tab("Step 2: Results & Interpretation", id=1):
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
                        interactive=False
                    )
                    resume_explain_btn = gr.Button("Explain", variant='primary', interactive=False, visible=False)
                    explanation_html = gr.HTML(
                        label="Explanation",
                        value="<p>No explanation generated. Click 'Explain' to see results...</p>"
                    )

                    # Batch Interpretation Carousel
                    with gr.Accordion("Explanations by Variation (Currently Disabled)", open=False, visible=False) as batch_carousel_accordion:
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

                        carousel_explain_btn = gr.Button("Explain Selection", variant="primary", interactive=False)

                        carousel_explanation_html = gr.HTML(
                            label="Variation Explanation Highlights",
                            value="<p>Detailed analysis of the selected variation will appear here...</p>"
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
                with gr.Tab("Step 3: Compare & Track", id=2):
                    gr.Markdown("Save versions to track your audit progress and compare different model behaviors side-by-side.")
                    gr.Markdown("*Versions are autosaved when you navigate here from Step 2.*")

                    resume_version_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select saved version to compare",
                        interactive=True
                    )
                    resume_clear_btn = gr.Button("Clear Comparison", variant="secondary")
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

            # Batch UI toggle
            def update_batch_ui(enabled, num_vars):
                return (
                    gr.update(visible=enabled),           # batch_token_input
                    gr.update(visible=enabled),           # batch_nl_input
                    gr.update(visible=enabled),           # batch_generate_code_btn
                    gr.update(visible=enabled),           # batch_code_preview
                    gr.update(visible=enabled),           # batch_execute_btn
                    gr.update(visible=enabled),           # batch_num_variations
                    gr.update(visible=not enabled),       # resume_explain_btn
                    gr.update(visible=not enabled),       # explanation_html
                    gr.update(visible=enabled),           # batch_carousel_accordion
                    gr.update(maximum=num_vars),          # carousel_index_slider
                    gr.update(interactive=not enabled)    # resume_analyze_btn
                )

            batch_analysis_toggle.change(
                fn=update_batch_ui,
                inputs=[batch_analysis_toggle, batch_num_variations],
                outputs=[
                    batch_token_input, batch_nl_input, batch_generate_code_btn,
                    batch_code_preview, batch_execute_btn, batch_num_variations,
                    resume_explain_btn, explanation_html, batch_carousel_accordion, carousel_index_slider,
                    resume_analyze_btn
                ]
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

            # Analyze button (Single Run) — also enables the Next button and marks analysis as done
            def process_and_enable_next(lead, body, end, method, temp, batch_enabled, batch_token, batch_num_vars, batch_dimension, variations_code):
                results = process_resume_split(lead, body, end, method, temp, batch_enabled, batch_token, batch_num_vars, batch_dimension, variations_code)
                # results = (html, batch_results_state, continuation_state, fulltext_state,
                #            model_display, explain_btn, explanation_html, batch_carousel, carousel_slider)
                # Append the enabled Next button state
                return results + (gr.update(interactive=True),)

            resume_analyze_btn.click(
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
                    step1_next_btn   # unlock Next button after first analysis
                ]
            )

            # Batch Execute Button — also enables the Next button
            batch_execute_btn.click(
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
                    step1_next_btn
                ]
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


        # Tab 2: Image Captioning
        with gr.Tab("Image Captioning", interactive=False):
            gr.Markdown("### Vision Model Interpretability")
            gr.Markdown("Analyze how vision-language models generate image captions and which visual features they attend to.")
            
            # ADDED: Model selection dropdown
            with gr.Row():
                image_model_dropdown = gr.Dropdown(
                    choices=get_image_model_choices(),
                    value=get_image_model_choices()[0],
                    label="Select Vision Model",
                    interactive=True,
                    scale=3
                )
                image_model_status = gr.Markdown("")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Image Input")
                    image_source_dropdown = gr.Dropdown(
                        choices=get_sample_image_choices(),
                        value=get_sample_image_choices()[0],
                        label="Select Sample Image",
                        interactive=True
                    )
                    image_upload = gr.Image(type="pil", label="Or Upload Your Own Image")
                    analysis_method = gr.Radio(
                        choices=["Integrated Gradients", "GradCAM", "Multi-Image Comparison"],
                        value="Integrated Gradients",
                        label="Explanation Method"
                    )
                    num_tokens_slider = gr.Slider(
                        minimum=1, 
                        maximum=20, 
                        value=5, 
                        step=1,
                        label="Number of tokens to explain (Integrated Gradients only)"
                    )
                    run_analysis_btn = gr.Button("Run Full Analysis", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### Analysis Results")
                    # ADDED: Show current model
                    image_current_model_display = gr.Markdown("**Model:** Loading...")
                    caption_output = gr.Textbox(label="Generated Caption", lines=2)
                    viz_output = gr.Image(label="Visual Explanation", type="pil")

            # Sample image selection handler
            def update_image_from_dropdown(choice):
                idx = get_sample_image_choices().index(choice)
                img = get_sample_image_by_index(idx)
                return img

            image_source_dropdown.change(
                fn=update_image_from_dropdown,
                inputs=image_source_dropdown,
                outputs=image_upload
            )

            # Full analysis button
            def run_selected_analysis(uploaded_img, img_source, method, num_tokens):
                if method == "Integrated Gradients":
                    caption, viz = run_integrated_gradients(uploaded_img, img_source, num_tokens)
                elif method == "GradCAM":
                    caption, viz = run_gradcam_analysis(uploaded_img, img_source)
                elif method == "Multi-Image Comparison":
                    caption, viz = compare_multiple_images(num_images=3)
                else:
                    return "Unknown method", None
                from image_utility import get_current_model_name
                model_display = f"**Model:** {get_current_model_name()}"
                return caption, viz, model_display  # ADDED 3rd return value

            run_analysis_btn.click(
                fn=run_selected_analysis,
                inputs=[image_upload, image_source_dropdown, analysis_method, num_tokens_slider],
                outputs=[caption_output, viz_output, image_current_model_display]
            )

            # ADDED: Handle model changes
            image_model_dropdown.change(
                fn=switch_image_model,
                inputs=image_model_dropdown,
                outputs=[image_model_status, image_current_model_display]
            )

            gr.Markdown("""
            ### Explanation Methods:
            - **Integrated Gradients**: Token-by-token attribution showing which pixels influenced each word
            - **GradCAM**: Spatial attention map showing overall important regions
            - **Multi-Image Comparison**: Compare captions and attributions across multiple images
            
            ### How to Use:
            1. Select an image source (sample dataset images or upload your own)
            2. Choose an explanation method
            3. Click "Run Full Analysis" to see the caption with visual explanations
            4. For Integrated Gradients, adjust the number of tokens to explain
            """)

        # Tab 3: Credit Risk Analyzer
        with gr.Tab("Credit Risk Analyzer", interactive=False):
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

# Cell 6: Launch the interface
demo.launch(debug=True, share=True)