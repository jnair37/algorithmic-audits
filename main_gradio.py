import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captum.attr import IntegratedGradients
import gradio as gr
import io
from resume_utility import sample_corpus, process_resume, reset_resume_text, save_resume_version, load_resume_version, clear_resume_comparison
from image_utility import test_img, run_integrated_gradients
from credit_utility import sample_credit_data, predict_credit_risk, get_feature_importance, compare_scenarios, reset_credit_data, export_credit_report


# Cell 5: Create the Gradio Interface
with gr.Blocks(title="AI Auditor Tool") as demo:
    gr.Markdown("# AI Auditor Tool")
    gr.Markdown("Analyze and audit different AI models with interpretable explanations.")

    with gr.Tabs():
        # Tab 1: Resume Screener
        with gr.Tab("Resume Screener"):
            gr.Markdown("### Resume Screener")
            with gr.Row():
                # Left column - Text editor
                with gr.Column(scale=1):
                    gr.Markdown("### Text Editor")
                    resume_text_input = gr.Textbox(
                        value=sample_corpus,
                        lines=15,
                        label="Edit Resume to Assess",
                        placeholder="Paste or type your text here..."
                    )
                    resume_method_dropdown = gr.Dropdown(
                        choices=["integrated_gradients", "layer_integrated_gradients", "shap"],
                        value="integrated_gradients",
                        label="Explanation Method (note: SHAP is doable but slow!)",
                        interactive=True
                    )
                    with gr.Row():
                        resume_analyze_btn = gr.Button("Analyze", variant="primary")
                        resume_reset_btn = gr.Button("Revert to Original", variant="secondary")
                    with gr.Row():
                        resume_save_btn = gr.Button("Save Current Version", variant="primary")
                    resume_save_status = gr.Markdown("")

                # Middle column - Current analysis
                with gr.Column(scale=1):
                    gr.Markdown("### Current Analysis")
                    resume_html_output = gr.HTML(
                        label="Current Highlights",
                        value="<p>Click 'Analyze' to see results...</p>"
                    )

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

            # Resume tab event handlers
            resume_analyze_btn.click(
                fn=process_resume,
                inputs=[resume_text_input, resume_method_dropdown],
                outputs=resume_html_output
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

        # Tab 2: Image Captioner
        with gr.Tab("Image Captioner"):
            gr.Markdown("# Image Captioner")
            gr.Markdown("Click the button to generate a caption and visualize which image regions influenced each token.")

            with gr.Row():
                with gr.Column():
                    test_image_display = gr.Image(value=test_img, label="Test Image", interactive=False)
                    run_btn = gr.Button("Generate Caption + Explain w/ Integrated Gradients", variant="primary")

                with gr.Column():
                    caption_output = gr.Textbox(label="Generated Caption", lines=2)
                    viz_output = gr.Image(label="Integrated Gradients Visualization")

            run_btn.click(
                fn=run_integrated_gradients,
                inputs=[test_image_display],
                outputs=[caption_output, viz_output]
            )

            gr.Markdown("""
            ### Explanation:
            - **Top row**: Original image with each explained token labeled
            - **Bottom row**: Attribution heatmaps showing which pixels were most important (red = high importance)
            - The visualization shows the first 3 content tokens from the generated caption
            """)

        # Tab 3: Credit Risk Analyzer
        with gr.Tab("Credit Risk Analyzer"):
            gr.Markdown("### Credit Risk Model Auditor")
            gr.Markdown("Analyze credit risk predictions and understand which factors influence the model's decisions.")
            
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

            # Credit tab event handlers
            credit_predict_btn.click(
                fn=predict_credit_risk,
                inputs=[
                    credit_age, credit_income, credit_score, credit_debt_ratio,
                    credit_employment_years, credit_loan_amount, credit_num_accounts,
                    credit_delinquencies, credit_method_dropdown
                ],
                outputs=[credit_risk_output, credit_feature_plot]
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

            gr.Markdown("""
            ### How to Use:
            - Adjust the sliders to input applicant information
            - Click "Predict Risk" to see the model's credit risk assessment
            - View feature importance to understand which factors most influenced the decision
            - Use "Scenario Analysis" to explore how changing specific features affects the prediction
            - Export a detailed report for documentation and compliance purposes
            """)

# Cell 6: Launch the interface
demo.launch(debug=True)