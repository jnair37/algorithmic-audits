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
    reset_resume_text, 
    save_resume_version, 
    load_resume_version, 
    clear_resume_comparison,
    get_resume_model_choices,      # NEW
    switch_resume_model
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


# Cell 5: Create the Gradio Interface
with gr.Blocks(title="Test Auditing Interface") as demo:
    gr.Markdown("# Prototype Auditing Interface")
    gr.Markdown("The following interface is meant to be used for gray-box auditing with access to an explanation. Each tab represents a different type of functionality (resume screening, image captioning, and credit risk) for which you can try out multiple models and interpretability methods.")

    with gr.Tabs():
        # Tab 1: Resume Screener
        with gr.Tab("Resume Screener"):
            gr.Markdown("### Resume Screener")
            # ADDED: Model selection dropdown
            with gr.Row():
                resume_model_dropdown = gr.Dropdown(
                    choices=get_resume_model_choices(),
                    value=get_resume_model_choices()[0],
                    label="Select Resume Screening Model",
                    interactive=True,
                    scale=3
                )
                resume_model_status = gr.Markdown("")
    
            with gr.Row():
                # Left column - Text editor
                with gr.Column(scale=1):
                    gr.Markdown("### Modify Input Resume")
                    resume_text_input = gr.Textbox(
                        value=sample_corpus,
                        lines=15,
                        label="Edit Resume to Assess",
                        placeholder="Paste or type your text here..."
                    )
                    resume_method_dropdown = gr.Dropdown(
                        choices=["integrated_gradients", "layer_integrated_gradients", "shap"],
                        value="integrated_gradients",
                        label="Explanation Method",
                        interactive=True
                    )
                    with gr.Row():
                        resume_analyze_btn = gr.Button("Analyze", variant="primary")
                        resume_reset_btn = gr.Button("Revert to Original", variant="secondary")

                # Middle column - Current analysis
                with gr.Column(scale=1):
                    gr.Markdown("### Current Analysis")
                    # ADDED: Show current model being used
                    resume_current_model_display = gr.Markdown("**Model:** Loading...")
                    resume_html_output = gr.HTML(
                        label="Current Highlights",
                        value="<p>Click 'Analyze' to see results...</p>"
                    )
                    with gr.Row():
                        resume_save_btn = gr.Button("Save Current Version", variant="primary")
                    resume_save_status = gr.Markdown("")

                # Right column - Saved version
                with gr.Column(scale=1):
                    gr.Markdown("### Saved Version (Comparison)")
                    resume_version_dropdown = gr.Dropdown(
                        choices=[],
                        label="Select saved version to compare",
                        interactive=True
                    )
                    resume_clear_btn = gr.Button("🗑️ Clear Comparison", variant="secondary")
                    resume_comparison_output = gr.HTML(
                        label="Saved Highlights",
                        value="<p>No comparison loaded. Save a version and select it from the dropdown to compare.</p>"
                    )

            # MODIFIED: Added resume_current_model_display to outputs
            resume_analyze_btn.click(
                fn=process_resume,
                inputs=[resume_text_input, resume_method_dropdown],
                outputs=[resume_html_output, resume_current_model_display]  # ADDED 2nd output
            )

            resume_reset_btn.click(
                fn=reset_resume_text,
                inputs=None,
                outputs=resume_text_input
            )

            resume_save_btn.click(
                fn=save_resume_version,
                inputs=[resume_text_input, resume_html_output, resume_method_dropdown],
                outputs=[resume_version_dropdown, resume_save_status]
            )

            resume_version_dropdown.change(
                fn=load_resume_version,
                inputs=resume_version_dropdown,
                outputs=resume_comparison_output
            )

            resume_clear_btn.click(
                fn=clear_resume_comparison,
                inputs=None,
                outputs=resume_comparison_output
            )

            # ADDED: Handle model selection changes
            resume_model_dropdown.change(
                fn=switch_resume_model,
                inputs=resume_model_dropdown,
                outputs=[resume_model_status, resume_current_model_display]
            )

        # Tab 2: Image Captioner
        with gr.Tab("Image Captioner"):
            gr.Markdown("# Image Captioner with Interpretability")
            gr.Markdown("Analyze image captioning models with multiple explanation methods and image sources.")

            # ADDED: Model selection dropdown
            with gr.Row():
                image_model_dropdown = gr.Dropdown(
                    choices=get_image_model_choices(),
                    value=get_image_model_choices()[0],
                    label="Select Image Captioning Model",
                    interactive=True,
                    scale=3
                )
                image_model_status = gr.Markdown("")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Image Selection")
                    image_source_dropdown = gr.Dropdown(
                        choices=get_sample_image_choices(),
                        value="Sample 1",
                        label="Select Image Source",
                        interactive=True
                    )
                    image_upload = gr.Image(
                        label="Upload Custom Image (when 'Upload' is selected)",
                        type="numpy",
                        interactive=True
                    )
                    image_preview = gr.Image(
                        value=test_img,
                        label="Current Image",
                        interactive=False
                    )
                    
                    gr.Markdown("### Analysis Options")
                    analysis_method = gr.Radio(
                        choices=["Integrated Gradients", "GradCAM", "Multi-Image Comparison"],
                        value="Integrated Gradients",
                        label="Explanation Method"
                    )
                    num_tokens_slider = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Number of Tokens to Explain (IG only)",
                        visible=True
                    )
                    
                    with gr.Row():
                        run_caption_btn = gr.Button("Generate Caption Only", variant="secondary")
                        run_analysis_btn = gr.Button("Run Full Analysis", variant="primary")

                with gr.Column(scale=2):
                    gr.Markdown("### Results")
                    # ADDED: Show current model
                    image_current_model_display = gr.Markdown("**Model:** Loading...")
                    caption_output = gr.Textbox(label="Generated Caption", lines=2)
                    viz_output = gr.Image(label="Visualization")

            # Event handlers for image selection
            def update_preview(choice, uploaded_img):
                if choice == "Upload":
                    if uploaded_img is not None:
                        return Image.fromarray(uploaded_img.astype(np.uint8))
                    return None
                else:
                    return get_sample_image_by_index(choice)
            
            image_source_dropdown.change(
                fn=update_preview,
                inputs=[image_source_dropdown, image_upload],
                outputs=image_preview
            )
            
            image_upload.change(
                fn=update_preview,
                inputs=[image_source_dropdown, image_upload],
                outputs=image_preview
            )

            # Toggle num_tokens slider visibility
            def toggle_slider(method):
                return gr.update(visible=(method == "Integrated Gradients"))
            
            analysis_method.change(
                fn=toggle_slider,
                inputs=analysis_method,
                outputs=num_tokens_slider
            )

            # ADDED: New wrapper function to include model display
            def caption_with_model(uploaded_img, img_source):
                caption = generate_caption_only(uploaded_img, img_source)
                from image_utility import get_current_model_name
                model_display = f"**Model:** {get_current_model_name()}"
                return caption, model_display

            # MODIFIED: Added model_display to outputs
            run_caption_btn.click(
                fn=caption_with_model,
                inputs=[image_upload, image_source_dropdown],
                outputs=[caption_output, image_current_model_display]  # ADDED 2nd output
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
                outputs=[caption_output, viz_output]
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
        with gr.Tab("Credit Risk Analyzer"):
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
                    credit_export_btn = gr.Button("📄 Export Analysis Report", variant="secondary")
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