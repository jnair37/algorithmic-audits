# -*- coding: utf-8 -*-
"""

Algorithmic Audits: Interface Demo for Resume Screening Language Models

Experimentation file is located at
    https://colab.research.google.com/drive/1A2jDDfmlqQSFY7blInVOukQ6oizqiyqX
"""

#### START HERE: Updated SHAP for text generation with current libraries

from transformers import AutoModelForCausalLM, AutoTokenizer
import shap
import numpy as np
import torch
import matplotlib.pyplot as plt

# Use a modern open-source model
# Choose based on your GPU memory:
# model_name = "meta-llama/Llama-3.2-1B"  # 1B params - runs on most GPUs
# model_name = "meta-llama/Llama-3.2-3B"  # 3B params - better quality
model_name = "microsoft/phi-2"  # 2.7B params - good performance
# model_name = "mistralai/Mistral-7B-v0.1"  # 7B params - needs ~16GB VRAM
# model_name = "Qwen/Qwen2.5-7B"  # 7B params - strong performance

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Configure generation parameters
# Modern approach: pass these directly to generate() or set as defaults
model.generation_config.max_length = 50
model.generation_config.temperature = 0.7
model.generation_config.top_k = 50
model.generation_config.do_sample = True
model.generation_config.no_repeat_ngram_size = 2

# Set pad token if not already set (required for some models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# limit response length
model.generation_config.max_length = 50
model.generation_config.max_new_tokens = 4

def get_shap_values(input_text):
  # Create SHAP explainer
  explainer = shap.Explainer(model, tokenizer)
  shap_values = explainer(input_text)
  return shap_values

# Vis task 3: pass the highlight vals for that one word into highlight_func

import matplotlib.colors as mcolors

def analyze_text(text):

  # Take in all strings and all values
  shap_values = get_shap_values([text])
  all_values = shap_values.values
  base_values = shap_values.base_values
  data_values = shap_values.data
  outputs = shap_values.output_names

  #print("outputs sanity check")
  #print(outputs)
  # PICK ONE COLUMN
  #print(np.shape(all_values[0][0]))

  # TODO: how to ensure the column is always right
  all_values = all_values[0][:, -1] # Last token for now.... not really the best though....

  min_val = min(all_values)
  max_val = max(all_values)
  #print(min_val)
  #print(max_val)
  normalized = [(v - min_val) / (max_val - min_val) for v in all_values]
  #print(normalized)
  # Choose a colormap
  cmap = plt.cm.seismic
  precomp = [mcolors.to_hex(cmap(norm_val)) for norm_val in normalized]

  # Data Values is text arr that corresponds to values. so basically each match.start() is a location of data_values[i] + len(data_values[i])?
  # print("DATA VALUES being cycled through (INPUT) to SHAP")
  # print(data_values)
  current_pos = 0
  highlights = []
  for i in range(len(data_values[0])):
    word = data_values[0][i]
    color = precomp[i]
    value = all_values[i]
    #print("WORD:")
    #print(word)
    start = text.find(word, current_pos)
    if start != -1:
        end = start + len(word)
        label = f"{value:.2f}"
        highlights.append((start, end, label, color))
        current_pos = end

  return highlights, outputs[len(data_values[0]):]


# Sample Corpus...

sample_corpus = """

You are a member of our hiring team reviewing applications for data analysts, and you answer questions with just one word.

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

Q: Based on the above resume, should Jane Smith be advanced to the next round for the position of Data Analyst?
"""

analyze_text(sample_corpus)

# AI Generated Experienced Hire corpus

sample_corpus_experienced = """
                   Alex Johnson

Education
ABC University
Somewhere, NY
B.S. in Computer Science, Minor in Economics	                                                                              May 2000
Relevant Coursework: Distributed Systems, Artificial Intelligence, Algorithms, Database Systems, Computer Networks, Machine Learning, Econometrics


Technical Skills

Languages: Python, SQL, Java, C++, JavaScript/TypeScript, bash scripting, MATLAB, R
Libraries & Frameworks: PyTorch, TensorFlow, scikit-learn, Hadoop/Spark ecosystem, Flask, Django, React, pandas, NumPy, OpenCV
Tools: Docker, Kubernetes, Git, AWS/GCP/Azure, Tableau, Airflow
Certifications: AWS Certified Solutions Architect; Google Professional Data Engineer; Certified Scrum Master

Experience                                                                                                                                                                    San Francisco, CA
Senior Data Science Manager	                                                                                                                   Jan 2018 – Present
•	Lead a team of 12 data scientists and ML engineers delivering large-scale predictive analytics, experimentation, and AI-driven product features across a Fortune 500 technology firm.
•	Designed and deployed end-to-end machine learning systems supporting >50M users; improved model reliability and interpretability through MLOps pipelines and monitoring frameworks.
•	Partnered with product and engineering leadership to define strategy, roadmap, hiring, and cross-functional collaboration.

Global Analytics Consulting Firm	                                                                                             New York, NY
Principal Data Scientist	                                                                                                            Aug 2010 – Dec 2017
•	Directed analytics engagements for government, healthcare, and financial services clients, translating business problems into actionable data solutions.
•	Built predictive models, causal inference analyses, and operational dashboards that informed multimillion-dollar policy and investment decisions.
•	Mentored junior staff, created internal training curriculum, and led firm-wide initiatives in responsible AI and bias evaluation.

U.S. Federal Research Agency	                                                                                               Washington, D.C.
Senior Research Fellow (Data & Policy)	                                                                                    Jun 2006 – Jul 2010
•	Developed national-scale data pipelines and statistical models to evaluate long-term care, public health outcomes, and emergency response systems.
•	Published multiple reports and co-authored peer-reviewed articles on algorithmic fairness, health informatics, and socioeconomic data integration.
•	Advised cross-agency working groups; presented findings to congressional staff and federal leadership.

Healthcare AI Lab at Major Hospital	                                                                                         Boston, MA
Machine Learning Researcher	                                                                                                 Jul 2003 – May 2006
•	Conducted research on computer vision for medical imaging and early interpretability methods for diagnostic AI systems.
•	Developed prototypes for real-time decision-support tools used in pilot clinical studies.
•	Managed collaborations between physicians, statisticians, and software teams.

EdTech Startup, Inc.	                                                                                                          Remote / New York, NY
Software Engineer → Senior Engineer	                                                                                 Jun 2000 – Jun 2003
•	Built educational analytics tools and interactive coding curricula for early-stage venture-backed startup.
•	Led migration from monolithic to modular architecture; improved code reliability and deployment workflows.
•	Developed content generation tools, assessment engines, and data visualization modules used by thousands of students.

Leadership & Activities

Applied AI Alliance, Steering Committee Member (2018–Present)
Women in Technology Network, Board Member (2015–2021)
Open-Source ML Community, Contributor & Maintainer (2012–Present)
Tech for Social Good Initiative, Program Lead (2010–2014)


"""


import gradio as gr
import re
from datetime import datetime

def highlight_text(text, highlights, outputs, title=""):
    """
    Create HTML with highlighted sections.
    """
    if not highlights:
        content = f'<div style="font-family: Arial, sans-serif; line-height: 1.8; padding: 20px;">{text}</div>'
    else:
        # Sort highlights by start position
        sorted_highlights = sorted(highlights, key=lambda x: x[0])

        result = []
        last_idx = 0

        for start, end, label, color in sorted_highlights:
            # Add unhighlighted text before this highlight
            if last_idx < start:
                result.append(text[last_idx:start].replace('\n', '<br>'))

            # Add highlighted text
            highlighted_span = f'<mark style="background-color: {color}; padding: 2px 4px; border-radius: 3px;" title="{label}">{text[start:end]}</mark>'
            result.append(highlighted_span)
            last_idx = end

        # Add remaining unhighlighted text
        if last_idx < len(text):
            result.append(text[last_idx:].replace('\n', '<br>'))

        content = ''.join(result)

        # Get unique labels for legend
        unique_labels = {}
        for _, _, label, color in sorted_highlights:
            if label not in unique_labels:
                unique_labels[label] = color

        legend = f"""
        <div style="margin-top: 20px; padding: 10px; background-color: #f5f5f5; border-radius: 5px;">
            <strong>Legend:</strong><br>
            {' '.join([f'<mark style="background-color:#000000; padding: 2px 8px; margin: 5px; border-radius: 3px;">{output}</mark>' for output in outputs])}
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


def process_text(text):
    """
    Main function that processes text and returns highlighted HTML.
    This is called whenever the text is edited.
    """
    if not text.strip():
        return "<p>Enter some text to analyze...</p>"

    # Run your analysis function
    highlights, outputs = analyze_text(text)

    # Generate highlighted HTML
    return highlight_text(text, highlights, outputs)


# Sample resume text
sample_text = sample_corpus

# Store original text and saved versions
original_text = sample_text
saved_versions = []

def reset_text():
    """Reset to original text."""
    return original_text, process_text(original_text)

def save_current_version(text, html):
    """Save the current text and ALREADY-RENDERED analysis."""
    global saved_versions
    if not text.strip():
        return gr.update(), "⚠️ Cannot save empty text"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Use the pre-rendered HTML instead of re-analyzing
    html_with_title = f'<h3 style="color: #555; margin-bottom: 10px;">Saved: {timestamp}</h3>{html}'

    saved_versions.append({
        'text': text,
        'html': html_with_title,
        'timestamp': timestamp
    })

    choices = [f"Version {i+1}: {v['timestamp']}" for i, v in enumerate(saved_versions)]

    return gr.update(choices=choices, value=None), f"✅ Saved version {len(saved_versions)}"

def load_saved_version(selected):
    """Load a saved version for comparison."""
    if not selected or not saved_versions:
        return "<p>No saved version selected</p>"

    # Extract version number from selection
    version_num = int(selected.split(":")[0].replace("Version ", "")) - 1
    return saved_versions[version_num]['html']

def clear_comparison():
    """Clear the comparison view."""
    return "<p>No comparison loaded. Save a version and select it from the dropdown to compare.</p>"

# Create Gradio interface with Blocks for more control
with gr.Blocks(title="Resume Screener Auditor Tool") as demo:
    gr.Markdown("# Resume Screener Auditor Tool")
    gr.Markdown("Test out this interpretable resume screener algorithm on different inputs, save, and compare the results.")

    with gr.Row():
        # Left column - Text editor
        with gr.Column(scale=1):
            gr.Markdown("### Text Editor")
            text_input = gr.Textbox(
                value=sample_text,
                lines=20,
                label="Edit Resume to Assess",
                placeholder="Paste or type your text here..."
            )
            with gr.Row():
                reset_btn = gr.Button("🔄 Revert to Original", variant="secondary")
                save_btn = gr.Button("💾 Save Current Version", variant="primary")
            save_status = gr.Markdown("")

        # Middle column - Current analysis (updates in real-time)
        with gr.Column(scale=1):
            gr.Markdown("### Current Analysis (Live)")
            html_output = gr.HTML(label="Current Highlights", value=process_text(sample_text))

        # Right column - Saved version for comparison
        with gr.Column(scale=1):
            gr.Markdown("### Saved Version (Comparison)")
            version_dropdown = gr.Dropdown(
                choices=[],
                label="Select saved version to compare",
                interactive=True
            )
            clear_btn = gr.Button("🗑️ Clear Comparison", variant="secondary")
            comparison_output = gr.HTML(
                label="Saved Highlights",
                value="<p>No comparison loaded. Save a version and select it from the dropdown to compare.</p>"
            )

    # Update current highlights when text changes
    text_input.change(
        fn=process_text,
        inputs=text_input,
        outputs=html_output
    )

    # Reset button functionality
    reset_btn.click(
        fn=reset_text,
        inputs=None,
        outputs=[text_input, html_output]
    )

    # Save button functionality
    save_btn.click(
        fn=save_current_version,
        inputs=[text_input, html_output],  # Pass both text AND html
        outputs=[version_dropdown, save_status]
    )

    # Load saved version for comparison
    version_dropdown.change(
        fn=load_saved_version,
        inputs=version_dropdown,
        outputs=comparison_output
    )

    # Clear comparison button
    clear_btn.click(
        fn=clear_comparison,
        inputs=None,
        outputs=comparison_output
    )

# Launch the interface
demo.launch(debug=True)
