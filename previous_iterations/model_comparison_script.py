"""
model_comparison_script.py

Systematic comparison of different models and explanation methods across all three use cases.
Tests each combination and saves results to organized directories for manual comparison.

Usage:
    python model_comparison_script.py
"""

import os
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Import utility modules
from resume_utility import (
    process_resume,
    switch_resume_model,
    sample_corpus,
    get_resume_model_choices
)
from image_utility import (
    run_integrated_gradients,
    run_gradcam_analysis,
    switch_image_model,
    get_sample_image_by_index,
    get_image_model_choices
)
from credit_utility import (
    predict_credit_risk,
    compare_scenarios,
    switch_credit_model,
    get_credit_model_choices
)

# ============================================================================
# CONFIGURATION: Define models and methods to test
# ============================================================================

# Resume screening models to test
RESUME_MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "microsoft/deberta-v3-base"
]

# Resume explanation methods to test
RESUME_METHODS = [
    "integrated_gradients",
    "layer_integrated_gradients",
    # "shap"  # Uncomment if you want to test SHAP (slow)
]

# Image captioning models to test
IMAGE_MODELS = [
    "microsoft/git-large-coco",
    "Salesforce/blip-image-captioning-base",
    # "Salesforce/blip-image-captioning-large",  # Uncomment for more thorough testing
    "nlpconnect/vit-gpt2-image-captioning"
]

# Image explanation methods to test
IMAGE_METHODS = [
    "integrated_gradients",
    "gradcam"
]

# Credit risk models to test
CREDIT_MODELS = [
    "Alfazril/credit-risk-prediction",
    "Random Forest Baseline",
    "Gradient Boosting",
    "Logistic Regression"
]

# Credit explanation methods to test
CREDIT_METHODS = [
    "shap",
    # "lime",  # Uncomment if LIME is implemented
    # "integrated_gradients"  # Uncomment if IG is implemented for credit
]

# Test cases for each domain
RESUME_TEST_CASES = [
    {
        "name": "standard_resume",
        "text": sample_corpus
    },
    {
        "name": "minimal_experience",
        "text": """Jane Doe
Junior Developer

EXPERIENCE
Junior Developer at StartupCo (2023-Present)
- Basic web development
- Learning Python

EDUCATION
Bachelor of Science in Computer Science
State University (2019-2023)

SKILLS
Python, HTML, CSS
"""
    }
]

IMAGE_TEST_CASES = [
    {"name": "sample_1", "source": "Sample 1"},
    {"name": "sample_2", "source": "Sample 2"},
    {"name": "sample_3", "source": "Sample 3"}
]

CREDIT_TEST_CASES = [
    {
        "name": "low_risk_applicant",
        "age": 45,
        "income": 120000,
        "credit_score": 800,
        "debt_ratio": 15,
        "employment_years": 15,
        "loan_amount": 15000,
        "num_accounts": 8,
        "delinquencies": 0
    },
    {
        "name": "high_risk_applicant",
        "age": 28,
        "income": 35000,
        "credit_score": 580,
        "debt_ratio": 65,
        "employment_years": 1,
        "loan_amount": 40000,
        "num_accounts": 2,
        "delinquencies": 3
    },
    {
        "name": "medium_risk_applicant",
        "age": 35,
        "income": 60000,
        "credit_score": 680,
        "debt_ratio": 35,
        "employment_years": 5,
        "loan_amount": 20000,
        "num_accounts": 4,
        "delinquencies": 1
    }
]

# Output directory
OUTPUT_DIR = "model_comparison_results"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_output_directories():
    """Create organized directory structure for results"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(OUTPUT_DIR) / timestamp
    
    dirs = {
        'resume': base_dir / "resume_screening",
        'image': base_dir / "image_captioning",
        'credit': base_dir / "credit_risk"
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return base_dir, dirs

def safe_filename(text):
    """Convert text to safe filename"""
    return "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in text)

def save_html_result(html_content, filepath):
    """Save HTML content to file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Analysis Result</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        {html_content}
    </div>
</body>
</html>
""")

def save_metadata(metadata, filepath):
    """Save test metadata as JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

# ============================================================================
# RESUME SCREENING TESTS
# ============================================================================

def test_resume_screening(output_dir):
    """Test all resume screening model and method combinations"""
    print("\n" + "="*80)
    print("TESTING RESUME SCREENING")
    print("="*80)
    
    results_summary = []
    
    for test_case in RESUME_TEST_CASES:
        case_name = test_case['name']
        text = test_case['text']
        
        print(f"\nTest Case: {case_name}")
        print("-" * 80)
        
        for model in RESUME_MODELS:
            print(f"  Testing model: {model}")
            
            # Switch to model
            try:
                switch_resume_model(model)
            except Exception as e:
                print(f"    ❌ Error switching to model: {e}")
                continue
            
            for method in RESUME_METHODS:
                print(f"    - Method: {method}")
                
                try:
                    # Run analysis
                    html_output, model_display = process_resume(text, method)
                    
                    # Create filename
                    safe_model = safe_filename(model)
                    filename = f"{case_name}__model_{safe_model}__method_{method}.html"
                    filepath = output_dir / filename
                    
                    # Save HTML result
                    save_html_result(html_output, filepath)
                    
                    # Save metadata
                    metadata = {
                        'test_case': case_name,
                        'model': model,
                        'method': method,
                        'timestamp': datetime.now().isoformat(),
                        'text_length': len(text)
                    }
                    metadata_path = filepath.with_suffix('.json')
                    save_metadata(metadata, metadata_path)
                    
                    results_summary.append({
                        'domain': 'resume',
                        'test_case': case_name,
                        'model': model,
                        'method': method,
                        'status': 'success',
                        'file': str(filepath.name)
                    })
                    
                    print(f"      ✓ Saved to {filepath.name}")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    results_summary.append({
                        'domain': 'resume',
                        'test_case': case_name,
                        'model': model,
                        'method': method,
                        'status': 'error',
                        'error': str(e)
                    })
    
    return results_summary

# ============================================================================
# IMAGE CAPTIONING TESTS
# ============================================================================

def test_image_captioning(output_dir):
    """Test all image captioning model and method combinations"""
    print("\n" + "="*80)
    print("TESTING IMAGE CAPTIONING")
    print("="*80)
    
    results_summary = []
    
    for test_case in IMAGE_TEST_CASES:
        case_name = test_case['name']
        image_source = test_case['source']
        
        print(f"\nTest Case: {case_name} ({image_source})")
        print("-" * 80)
        
        # Get the test image
        try:
            test_image = get_sample_image_by_index(image_source)
            if test_image is None:
                print(f"  ❌ Could not load image for {image_source}")
                continue
            
            # Save original image for reference
            orig_image_path = output_dir / f"{case_name}__original.png"
            test_image.save(orig_image_path)
            
        except Exception as e:
            print(f"  ❌ Error loading image: {e}")
            continue
        
        for model in IMAGE_MODELS:
            print(f"  Testing model: {model}")
            
            # Switch to model
            try:
                switch_image_model(model)
            except Exception as e:
                print(f"    ❌ Error switching to model: {e}")
                continue
            
            for method in IMAGE_METHODS:
                print(f"    - Method: {method}")
                
                try:
                    # Run analysis based on method
                    if method == "integrated_gradients":
                        caption, viz_image = run_integrated_gradients(
                            None, image_source, num_tokens=3
                        )
                    elif method == "gradcam":
                        caption, viz_image = run_gradcam_analysis(
                            None, image_source
                        )
                    else:
                        print(f"      ⚠ Unknown method: {method}")
                        continue
                    
                    # Create filename
                    safe_model = safe_filename(model)
                    base_filename = f"{case_name}__model_{safe_model}__method_{method}"
                    
                    # Save visualization
                    viz_path = output_dir / f"{base_filename}.png"
                    if viz_image is not None:
                        viz_image.save(viz_path)
                    
                    # Save caption and metadata
                    metadata = {
                        'test_case': case_name,
                        'model': model,
                        'method': method,
                        'caption': caption,
                        'timestamp': datetime.now().isoformat(),
                        'image_source': image_source
                    }
                    metadata_path = output_dir / f"{base_filename}.json"
                    save_metadata(metadata, metadata_path)
                    
                    # Create HTML report
                    html_content = f"""
                    <h2>Image Captioning Result</h2>
                    <p><strong>Model:</strong> {model}</p>
                    <p><strong>Method:</strong> {method}</p>
                    <p><strong>Test Case:</strong> {case_name}</p>
                    <h3>Generated Caption:</h3>
                    <p style="font-size: 18px; background-color: #e9ecef; padding: 15px; border-radius: 5px;">
                        {caption}
                    </p>
                    <h3>Original Image:</h3>
                    <img src="{case_name}__original.png" style="max-width: 400px;">
                    <h3>Attribution Visualization:</h3>
                    <img src="{base_filename}.png" style="max-width: 100%;">
                    """
                    html_path = output_dir / f"{base_filename}.html"
                    save_html_result(html_content, html_path)
                    
                    results_summary.append({
                        'domain': 'image',
                        'test_case': case_name,
                        'model': model,
                        'method': method,
                        'caption': caption,
                        'status': 'success',
                        'file': str(viz_path.name)
                    })
                    
                    print(f"      ✓ Saved to {base_filename}.*")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    results_summary.append({
                        'domain': 'image',
                        'test_case': case_name,
                        'model': model,
                        'method': method,
                        'status': 'error',
                        'error': str(e)
                    })
    
    return results_summary

# ============================================================================
# CREDIT RISK TESTS
# ============================================================================

def test_credit_risk(output_dir):
    """Test all credit risk model and method combinations"""
    print("\n" + "="*80)
    print("TESTING CREDIT RISK ANALYSIS")
    print("="*80)
    
    results_summary = []
    
    for test_case in CREDIT_TEST_CASES:
        case_name = test_case['name']
        
        print(f"\nTest Case: {case_name}")
        print("-" * 80)
        
        for model in CREDIT_MODELS:
            print(f"  Testing model: {model}")
            
            # Switch to model
            try:
                switch_credit_model(model)
            except Exception as e:
                print(f"    ❌ Error switching to model: {e}")
                continue
            
            for method in CREDIT_METHODS:
                print(f"    - Method: {method}")
                
                try:
                    # Run prediction
                    html_output, feature_plot, model_display = predict_credit_risk(
                        age=test_case['age'],
                        income=test_case['income'],
                        credit_score=test_case['credit_score'],
                        debt_ratio=test_case['debt_ratio'],
                        employment_years=test_case['employment_years'],
                        loan_amount=test_case['loan_amount'],
                        num_accounts=test_case['num_accounts'],
                        delinquencies=test_case['delinquencies'],
                        method=method
                    )
                    
                    # Create filename
                    safe_model = safe_filename(model)
                    base_filename = f"{case_name}__model_{safe_model}__method_{method}"
                    
                    # Save HTML output
                    html_path = output_dir / f"{base_filename}.html"
                    save_html_result(html_output, html_path)
                    
                    # Save feature importance plot
                    plot_path = output_dir / f"{base_filename}__features.png"
                    feature_plot.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close(feature_plot)
                    
                    # Save scenario comparison (optional, for Credit Score)
                    try:
                        scenario_plot = compare_scenarios(
                            age=test_case['age'],
                            income=test_case['income'],
                            credit_score=test_case['credit_score'],
                            debt_ratio=test_case['debt_ratio'],
                            employment_years=test_case['employment_years'],
                            loan_amount=test_case['loan_amount'],
                            num_accounts=test_case['num_accounts'],
                            delinquencies=test_case['delinquencies'],
                            feature_to_vary="Credit Score"
                        )
                        scenario_path = output_dir / f"{base_filename}__scenario.png"
                        scenario_plot.savefig(scenario_path, dpi=150, bbox_inches='tight')
                        plt.close(scenario_plot)
                    except Exception as e:
                        print(f"      ⚠ Could not generate scenario plot: {e}")
                    
                    # Save metadata
                    metadata = {
                        'test_case': case_name,
                        'model': model,
                        'method': method,
                        'timestamp': datetime.now().isoformat(),
                        'applicant_data': {k: v for k, v in test_case.items() if k != 'name'}
                    }
                    metadata_path = output_dir / f"{base_filename}.json"
                    save_metadata(metadata, metadata_path)
                    
                    results_summary.append({
                        'domain': 'credit',
                        'test_case': case_name,
                        'model': model,
                        'method': method,
                        'status': 'success',
                        'file': str(html_path.name)
                    })
                    
                    print(f"      ✓ Saved to {base_filename}.*")
                    
                except Exception as e:
                    print(f"      ❌ Error: {e}")
                    results_summary.append({
                        'domain': 'credit',
                        'test_case': case_name,
                        'model': model,
                        'method': method,
                        'status': 'error',
                        'error': str(e)
                    })
    
    return results_summary

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_summary_report(base_dir, all_results):
    """Generate a comprehensive summary report"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    # Count successes and failures
    total_tests = len(all_results)
    successful = sum(1 for r in all_results if r['status'] == 'success')
    failed = total_tests - successful
    
    # Group by domain
    by_domain = {}
    for result in all_results:
        domain = result['domain']
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(result)
    
    # Create HTML summary
    html_content = f"""
    <h1>Model Comparison Test Results</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>Summary</h2>
    <ul>
        <li><strong>Total Tests:</strong> {total_tests}</li>
        <li><strong>Successful:</strong> {successful} ({successful/total_tests*100:.1f}%)</li>
        <li><strong>Failed:</strong> {failed} ({failed/total_tests*100:.1f}%)</li>
    </ul>
    
    <h2>Results by Domain</h2>
    """
    
    for domain, results in by_domain.items():
        domain_success = sum(1 for r in results if r['status'] == 'success')
        html_content += f"""
        <h3>{domain.upper()}</h3>
        <p>Tests: {len(results)} | Success: {domain_success} | Failed: {len(results) - domain_success}</p>
        <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
            <tr style="background-color: #e9ecef;">
                <th>Test Case</th>
                <th>Model</th>
                <th>Method</th>
                <th>Status</th>
                <th>File</th>
            </tr>
        """
        
        for result in results:
            status_color = "#28a745" if result['status'] == 'success' else "#dc3545"
            file_link = f"<a href='{domain}/{result.get('file', 'N/A')}'>{result.get('file', 'N/A')}</a>" if result.get('file') else 'N/A'
            
            html_content += f"""
            <tr>
                <td>{result['test_case']}</td>
                <td>{result['model']}</td>
                <td>{result['method']}</td>
                <td style="color: {status_color};"><strong>{result['status'].upper()}</strong></td>
                <td>{file_link}</td>
            </tr>
            """
        
        html_content += "</table><br>"
    
    # Save summary report
    summary_path = base_dir / "SUMMARY_REPORT.html"
    save_html_result(html_content, summary_path)
    
    # Save JSON summary
    json_summary_path = base_dir / "summary_results.json"
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'total_tests': total_tests,
        'successful': successful,
        'failed': failed,
        'results': all_results
    }
    save_metadata(summary_data, json_summary_path)
    
    print(f"\n✓ Summary report saved to: {summary_path}")
    print(f"✓ JSON results saved to: {json_summary_path}")
    
    return summary_path

def main():
    """Main execution function"""
    print("="*80)
    print("MODEL COMPARISON TESTING SCRIPT")
    print("="*80)
    print(f"\nStarting comprehensive model comparison at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create output directories
    base_dir, dirs = create_output_directories()
    print(f"\nResults will be saved to: {base_dir}")
    
    # Run all tests
    all_results = []
    
    try:
        # Test resume screening
        resume_results = test_resume_screening(dirs['resume'])
        all_results.extend(resume_results)
    except Exception as e:
        print(f"\n❌ Error in resume screening tests: {e}")
    
    try:
        # Test image captioning
        image_results = test_image_captioning(dirs['image'])
        all_results.extend(image_results)
    except Exception as e:
        print(f"\n❌ Error in image captioning tests: {e}")
    
    try:
        # Test credit risk
        credit_results = test_credit_risk(dirs['credit'])
        all_results.extend(credit_results)
    except Exception as e:
        print(f"\n❌ Error in credit risk tests: {e}")
    
    # Generate summary report
    summary_path = generate_summary_report(base_dir, all_results)
    
    # Print final summary
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"\nTotal tests run: {len(all_results)}")
    print(f"Successful: {sum(1 for r in all_results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in all_results if r['status'] == 'error')}")
    print(f"\nAll results saved to: {base_dir}")
    print(f"Open {summary_path} to view detailed results")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()