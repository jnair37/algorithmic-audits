"""
credit_utility.py
Utility functions for Credit Risk Model analysis in Gradio interface
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from huggingface_hub import hf_hub_download
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import io
from datetime import datetime
import tempfile
import os

# ADDED: Track current model
_current_model_name = "Alfazril/credit-risk-prediction"

# ADDED: Model registry
CREDIT_MODELS = {
    "Alfazril/credit-risk-prediction": {
        "repo_id": "Alfazril/credit-risk-prediction",
        "filename": "credit_risk_xgboost_model.pkl",
        "type": "xgboost",
        "description": "XGBoost ensemble model"
    },
    # TODO: CHANGE MODELS
    "Random Forest Baseline": {
        "repo_id": None,
        "filename": None,
        "type": "random_forest",
        "description": "Synthetic Random Forest model"
    },
    "Gradient Boosting": {
        "repo_id": None,
        "filename": None,
        "type": "gradient_boosting",
        "description": "Gradient Boosting Classifier"
    },
    "Logistic Regression": {
        "repo_id": None,
        "filename": None,
        "type": "logistic",
        "description": "Interpretable linear model"
    }
}


# Set style for visualizations
sns.set_style("whitegrid")

# Features from the model (can be updated dynamically)
FEATURE_NAMES = [
    'person_age', 'person_income', 'person_emp_length', 'loan_amnt',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'grade_risk_score', 'combined_risk_score', 'income_to_age_ratio',
    'loan_to_income_ratio', 'credit_utilization', 'person_income_log',
    'loan_amnt_log', 'loan_grade_encoded', 'employment_stability_encoded',
    'debt_to_income_category_encoded', 'income_category_encoded',
    'credit_history_category_encoded', 'loan_intent_target_encoded',
    'person_home_ownership_OTHER', 'person_home_ownership_OWN',
    'person_home_ownership_RENT', 'cb_person_default_on_file_Y',
    'age_group_Millennial', 'age_group_Gen X', 'age_group_Boomer',
    'age_group_Senior', 'age_group_Elder'
]

# Global model and scaler
_model = None
_scaler = None
_explainer = None
_feature_mapping = {} # Store mapping of original names to clean names

def _initialize_model():
    """Initialize or load the credit risk model"""
    global _model, _scaler, _explainer
    
    if _model is not None:
        return
    
    print(f"Initializing credit risk model: {_current_model_name}...")
    
    model_config = CREDIT_MODELS[_current_model_name]
    
    if model_config["repo_id"] is not None:
        # Try to download from HuggingFace
        try:
            model_path = hf_hub_download(
                repo_id=model_config["repo_id"],
                filename=model_config["filename"]
            )
            
            try:
                _model = joblib.load(model_path)
            except:
                with open(model_path, "rb") as f:
                    _model = pickle.load(f)
            
            # ATTEMPT DYNAMIC FEATURE DETECTION
            global FEATURE_NAMES
            detected_features = None
            if hasattr(_model, "feature_names_in_"):
                detected_features = list(_model.feature_names_in_)
            elif hasattr(_model, "feature_names"):
                detected_features = list(_model.feature_names)
            
            if detected_features:
                FEATURE_NAMES = detected_features
            # Else keep defaults for the credit model if it's the default repo
            
            print(f"Model loaded from HuggingFace: {_current_model_name}")
            
            # Initialize SHAP explainer
            _explainer = shap.TreeExplainer(_model)
            print("SHAP explainer initialized")
            
            return
            
        except Exception as e:
            print(f"Could not load HuggingFace model: {e}")
    
    # Create synthetic models based on type
    print(f"Creating synthetic {model_config['type']} model...")
    _scaler = StandardScaler()
    
    np.random.seed(42)
    n_samples = 1000
    X_synthetic = np.random.randn(n_samples, 29)
    y_synthetic = ((X_synthetic[:, 3] > 0.5) |  # High loan amount
                   (X_synthetic[:, 10] > 0.7) |  # High loan to income
                   (X_synthetic[:, 8] > 0.6)).astype(int)  # High combined risk
    
    if model_config["type"] == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        _model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_config["type"] == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        _model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_config["type"] == "logistic":
        from sklearn.linear_model import LogisticRegression
        _model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        # Default to Random Forest
        from sklearn.ensemble import RandomForestClassifier
        _model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    X_synthetic_scaled = _scaler.fit_transform(X_synthetic)
    _model.fit(X_synthetic_scaled, y_synthetic)
    print(f"Synthetic {model_config['type']} model created")

    # Initialize SHAP explainer
    _explainer = shap.TreeExplainer(_model)
    print("SHAP explainer initialized")



# ADDED: Model management functions
def get_credit_model_choices():
    """Return list of available credit risk models"""
    return list(CREDIT_MODELS.keys())

def get_current_model_name():
    """Return the currently selected model name"""
    return _current_model_name

def switch_credit_model(model_name):
    """Switch to a different credit risk model"""
    global _model, _scaler, _explainer, _current_model_name
    
    if model_name not in CREDIT_MODELS:
        return f"❌ Unknown model: {model_name}", f"**Model:** {_current_model_name}"
    
    # Reset current model
    _model = None
    _scaler = None
    _explainer = None
    _current_model_name = model_name
    
    # Initialize new model
    _initialize_model()
    _explainer = shap.TreeExplainer(_model)
    
    model_info = CREDIT_MODELS[model_name]
    status_msg = f"✅ Loaded: {model_name}\n\n*{model_info['description']}*"
    model_display = f"**Model:** {model_name}"
    
    return status_msg, model_display


# Accepted pipeline / model types for supervised credit auditing
_CREDIT_PIPELINE_TAGS = {
    "tabular-classification", "text-classification", "tabular-regression",
    "structured-data-classification",
}
_CREDIT_MODEL_TYPES = {"xgboost", "lightgbm", "sklearn", "random_forest", "gradient_boosting"}


def validate_and_load_credit_model(hf_model_id: str):
    """
    Validate and load a HuggingFace supervised / tabular model by URL or ID.

    Returns:
        (status_markdown, model_display_markdown)
    """
    global _model, _scaler, _explainer, _current_model_name

    if not hf_model_id or not hf_model_id.strip():
        return "⚠️ Please enter a HuggingFace model ID.", f"**Model:** {_current_model_name}"

    model_id = hf_model_id.strip()
    if model_id.startswith("https://huggingface.co/"):
        model_id = model_id[len("https://huggingface.co/"):]
    model_id = model_id.rstrip("/")

    # Check if it's one of our built-in synthetic model names
    if model_id in CREDIT_MODELS and CREDIT_MODELS[model_id].get("repo_id") is None:
        _current_model_name = model_id
        _model = None  # force reload
        _initialize_model()
        return f"✅ **Loaded synthetic model:** `{model_id}`", f"**Model:** {model_id}"

    try:
        from huggingface_hub import model_info as hf_model_info
        from huggingface_hub import list_repo_files, hf_hub_download
        from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
        import joblib, pickle, shap

        try:
            info = hf_model_info(model_id)
        except RepositoryNotFoundError:
            return (
                f"❌ **Repository not found:** `{model_id}`\n\nPlease check the model ID or URL.",
                f"**Model:** {_current_model_name}",
            )
        except GatedRepoError:
            return (
                f"❌ **Gated repository:** `{model_id}`\n\nAccept the model license on HuggingFace first.",
                f"**Model:** {_current_model_name}",
            )

        pipeline_tag = (info.pipeline_tag or "").lower()
        model_type = ""
        if info.config and isinstance(info.config, dict):
            model_type = info.config.get("model_type", "").lower()

        accepted = pipeline_tag in _CREDIT_PIPELINE_TAGS or model_type in _CREDIT_MODEL_TYPES

        if not accepted and pipeline_tag and pipeline_tag not in ("", "null"):
            return (
                f"❌ **Unsupported model type:** `{model_id}`\n\n"
                f"Detected pipeline: `{pipeline_tag}` / architecture: `{model_type or 'unknown'}`\n\n"
                "Only tabular-classification / structured-data models are supported for credit auditing.",
                f"**Model:** {_current_model_name}",
            )

        # Attempt to download and load
        files = list(list_repo_files(model_id))
        
        # 1. Prefer known filename from registry if available
        model_file = None
        if model_id in CREDIT_MODELS:
            model_file = CREDIT_MODELS[model_id].get("filename")
            if model_file not in files:
                model_file = None # Registry file not found in repo
        
        # 2. Heuristic: Prefer XGBoost, then LightGBM, then any .pkl/.joblib
        if model_file is None:
            # Try to find XGBoost then LightGBM specifically to avoid CatBoost issues
            candidates = ["xgboost", "lightgbm"]
            for cand in candidates:
                model_file = next((f for f in files if cand in f.lower() and (f.endswith(".pkl") or f.endswith(".joblib"))), None)
                if model_file: break
                
        # 3. Last resort: any .pkl or .joblib that isn't catboost
        if model_file is None:
            model_file = next((f for f in files if "catboost" not in f.lower() and (f.endswith(".pkl") or f.endswith(".joblib"))), None)
            
        # 4. Absolute last resort: just pick the first one
        if model_file is None:
            model_file = next((f for f in files if f.endswith(".pkl") or f.endswith(".joblib")), None)

        if model_file is None:
            return (
                f"⚠️ **No .pkl/.joblib file found in `{model_id}`.**\n\n"
                "The supervised model tab requires a pickled scikit-learn compatible model.",
                f"**Model:** {_current_model_name}",
            )

        print(f"Loading model file: {model_file} from {model_id}...")
        path = hf_hub_download(repo_id=model_id, filename=model_file)
        try:
            loaded_model = joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                loaded_model = pickle.load(f)

        _model = loaded_model
        _current_model_name = model_id
        _scaler = None # Reset scaler for external models

        # ATTEMPT DYNAMIC FEATURE DETECTION
        global FEATURE_NAMES
        detected_features = None
        if hasattr(_model, "feature_names_in_"):
            detected_features = list(_model.feature_names_in_)
        elif hasattr(_model, "feature_names"):
            detected_features = list(_model.feature_names)
        elif hasattr(_model, "get_booster"): # XGBoost specific
            try:
                detected_features = _model.get_booster().feature_names
            except: pass
            
        if detected_features:
            FEATURE_NAMES = list(detected_features)

        try:
            _explainer = shap.TreeExplainer(_model)
        except Exception:
            _explainer = None  # Non-tree model — SHAP explainer may not work

        status_msg = f"✅ **Loaded:** `{model_id}` (file: `{model_file}`)"
        model_display = f"**Model:** {model_id}"
        return status_msg, model_display

    except Exception as e:
        return (
            f"❌ **Error loading `{model_id}`:** {str(e)}",
            f"**Model:** {_current_model_name}",
        )

def sample_credit_data():
    """Returns default sample credit applicant data"""
    return {
        'age': 35,
        'income': 50000,
        'credit_score': 650,
        'debt_ratio': 30,
        'employment_years': 5,
        'loan_amount': 15000,
        'num_accounts': 3,
        'delinquencies': 0
    }


def _map_inputs_to_features(age, income, credit_score, debt_ratio, 
                           employment_years, loan_amount, num_accounts, delinquencies,
                           feature_names=None):
    """Map Gradio inputs to model features, handles both standard credit and general models"""
    
    # If feature_names is likely the default credit features (29 features), use standard mapping
    if feature_names is not None and len(feature_names) != 29:
        return _map_general_inputs(age, income, credit_score, debt_ratio,
                                  employment_years, loan_amount, num_accounts, delinquencies,
                                  feature_names)

    # Standard Credit Mapping (legacy)
    # Calculate derived features
    loan_percent_income = loan_amount / income
    income_to_age_ratio = income / age
    loan_to_income_ratio = loan_amount / income
    credit_utilization = debt_ratio / 100  # Convert percentage to ratio
    person_income_log = np.log1p(income)
    loan_amnt_log = np.log1p(loan_amount)
    
    # Employment length in months
    person_emp_length = employment_years
    
    # Credit history length (estimate based on age)
    cb_person_cred_hist_length = max(1, age - 18)
    
    # Loan interest rate (estimate based on credit score)
    if credit_score >= 750:
        loan_int_rate = 5.5 + np.random.uniform(-0.5, 0.5)
    elif credit_score >= 700:
        loan_int_rate = 7.5 + np.random.uniform(-0.5, 0.5)
    elif credit_score >= 650:
        loan_int_rate = 9.5 + np.random.uniform(-0.5, 0.5)
    else:
        loan_int_rate = 12.5 + np.random.uniform(-0.5, 0.5)
    
    # Risk scores
    grade_risk_score = min(5, max(0, int((850 - credit_score) / 100)))
    combined_risk_score = (debt_ratio / 100 + loan_percent_income) / 2
    
    # Encoded features (simplified mapping)
    loan_grade_encoded = grade_risk_score
    employment_stability_encoded = min(4, employment_years // 3)
    debt_to_income_category_encoded = min(3, int(debt_ratio / 20))
    income_category_encoded = min(3, int(income / 40000))
    credit_history_category_encoded = min(4, cb_person_cred_hist_length // 10)
    loan_intent_target_encoded = 0  # Default
    
    # One-hot: home ownership (assume RENT for simplicity)
    person_home_ownership_OTHER = 0
    person_home_ownership_OWN = 1 if income > 80000 else 0
    person_home_ownership_RENT = 1 - person_home_ownership_OWN
    
    # Default on file
    cb_person_default_on_file_Y = 1 if delinquencies > 2 else 0
    
    # Age groups
    age_group_Millennial = 1 if 25 <= age < 40 else 0
    age_group_GenX = 1 if 40 <= age < 55 else 0
    age_group_Boomer = 1 if 55 <= age < 70 else 0
    age_group_Senior = 1 if 70 <= age < 80 else 0
    age_group_Elder = 1 if age >= 80 else 0
    
    # Create feature vector in correct order
    features = np.array([
        age, income, person_emp_length, loan_amount,
        loan_int_rate, loan_percent_income, cb_person_cred_hist_length,
        grade_risk_score, combined_risk_score, income_to_age_ratio,
        loan_to_income_ratio, credit_utilization, person_income_log,
        loan_amnt_log, loan_grade_encoded, employment_stability_encoded,
        debt_to_income_category_encoded, income_category_encoded,
        credit_history_category_encoded, loan_intent_target_encoded,
        person_home_ownership_OTHER, person_home_ownership_OWN,
        person_home_ownership_RENT, cb_person_default_on_file_Y,
        age_group_Millennial, age_group_GenX, age_group_Boomer,
        age_group_Senior, age_group_Elder
    ]).reshape(1, -1)
    
    return features

def _map_general_inputs(age, income, credit_score, debt_ratio,
                        employment_years, loan_amount, num_accounts, delinquencies,
                        feature_names):
    """Attempt to map the 8 standard UI sliders to arbitrary feature names"""
    # Simple mapping dictionary for fuzzy matching
    mapping = {
        'age': age, 'income': income, 'score': credit_score, 'debt': debt_ratio,
        'employment': employment_years, 'loan': loan_amount, 'account': num_accounts,
        'delinquency': delinquencies, 'default': delinquencies
    }
    
    feature_vector = []
    for f in feature_names:
        val = 0.0 # Default fallback
        f_lower = f.lower()
        # Try to find a match in the mapping
        found = False
        for key, value in mapping.items():
            if key in f_lower:
                val = float(value)
                found = True
                break
        feature_vector.append(val)
        
    return np.array(feature_vector).reshape(1, -1)


def predict_credit_risk(age, income, credit_score, debt_ratio, 
                       employment_years, loan_amount, num_accounts, 
                       delinquencies, method, feature_overrides=None):
    """
    Predict risk for any supervised model.
    Accepts standard 8 inputs + optional feature_overrides dict.
    """
    _initialize_model()
    
    global FEATURE_NAMES
    
    # Map inputs to features
    if len(FEATURE_NAMES) == 29 and "person_age" in FEATURE_NAMES:
        X = _map_inputs_to_features(age, income, credit_score, debt_ratio,
                                    employment_years, loan_amount, num_accounts, delinquencies,
                                    feature_names=FEATURE_NAMES)
    else:
        # General mapping for arbitrary models
        X = _map_general_inputs(age, income, credit_score, debt_ratio,
                                 employment_years, loan_amount, num_accounts, delinquencies,
                                 FEATURE_NAMES)
    
    # Apply overrides if provided (useful for "compatible with any supervised modeling")
    if feature_overrides:
        for i, f in enumerate(FEATURE_NAMES):
            if f in feature_overrides:
                try:
                    X[0, i] = float(feature_overrides[f])
                except: pass

    # Scale if using synthetic model (identified by _scaler being present)
    if _scaler is not None:
        X_scaled = _scaler.transform(X)
    else:
        X_scaled = X
    
    # Make prediction
    prediction = _model.predict(X_scaled)[0]
    probabilities = _model.predict_proba(X_scaled)[0]
    risk_prob = probabilities[1] * 100
    
    # Determine risk level
    if risk_prob < 30:
        risk_level = "LOW RISK"
        color = "#28a745"
    elif risk_prob < 60:
        risk_level = "MEDIUM RISK"
        color = "#ffc107"
    else:
        risk_level = "HIGH RISK"
        color = "#dc3545"
    
    decision = "APPROVED" if prediction == 0 else "DENIED"
    decision_color = "#28a745" if decision == "APPROVED" else "#dc3545"
    
    # Create HTML output with model name
    html_output = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f8f9fa;">
        <div style="font-size: 11px; color: #666; margin-bottom: 10px; text-align: center;">
            Using model: <strong>{_current_model_name}</strong>
        </div>
        <h2 style="text-align: center; color: {color};">{risk_level}</h2>
        <div style="text-align: center; margin: 20px 0;">
            <div style="font-size: 48px; font-weight: bold; color: {color};">
                {risk_prob:.1f}%
            </div>
            <div style="font-size: 14px; color: #666;">Default Risk Probability</div>
        </div>
        <div style="text-align: center; padding: 15px; background-color: {decision_color}20; 
                    border-radius: 5px; margin-top: 20px;">
            <div style="font-size: 24px; font-weight: bold; color: {decision_color};">
                {decision}
            </div>
        </div>
        <div style="margin-top: 20px; padding: 15px; background-color: white; border-radius: 5px;">
            <h4>Application Summary:</h4>
            <ul style="list-style: none; padding: 0;">
                <li>👤 Age: {age} years</li>
                <li>💰 Annual Income: ${income:,}</li>
                <li>📊 Credit Score: {credit_score}</li>
                <li>💳 Loan Amount: ${loan_amount:,}</li>
                <li>📈 Debt-to-Income Ratio: {debt_ratio}%</li>
                <li>💼 Employment Years: {employment_years}</li>
            </ul>
        </div>
    </div>
    """
    
    print('getting ftr importance')
    # Generate feature importance plot
    fig = get_feature_importance(X_scaled, method)
    
    # Model display
    model_display = f"**Model:** {_current_model_name}"
    
    return html_output, fig, model_display


def get_feature_importance(X_scaled, method):
    """Calculate and plot feature importance using specified method"""
    _initialize_model()
    print('initialized again?')
    print(_explainer)

    # Calculate SHAP values
    shap_values = _explainer.shap_values(X_scaled)
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values_risk = shap_values[1]
    else:
        shap_values_risk = shap_values
    
    shap_values_risk = np.asarray(shap_values_risk, dtype=float)
    
    if shap_values_risk.ndim == 3:
        shap_values_risk = shap_values_risk[:, :, 1]
    
    # Get top 10 most important features
    mean_abs_shap = np.abs(shap_values_risk).mean(axis=0).ravel()
    
    # Create simplified feature names for display
    simplified_names = {
        'person_age': 'Age',
        'person_income': 'Income',
        'loan_amnt': 'Loan Amount',
        'loan_int_rate': 'Interest Rate',
        'loan_percent_income': 'Loan % of Income',
        'grade_risk_score': 'Risk Grade',
        'combined_risk_score': 'Combined Risk',
        'credit_utilization': 'Credit Utilization',
        'debt_to_income_category_encoded': 'Debt-to-Income Category',
        'cb_person_cred_hist_length': 'Credit History Length'
    }
    
    # Get top 10 features
    top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
    top_features = [FEATURE_NAMES[i] for i in top_indices]
    top_values = mean_abs_shap[top_indices]
    
    # Simplify names for display
    display_names = [simplified_names.get(f, f.replace('_', ' ').title()) for f in top_features]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_values)))
    
    bars = ax.barh(range(len(top_values)), top_values, color=colors)
    ax.set_yticks(range(len(top_values)))
    ax.set_yticklabels(display_names)
    ax.set_xlabel('Mean Absolute SHAP Value (Impact on Risk)', fontsize=12)
    ax.set_title(f'Top 10 Most Important Features\nMethod: {method.upper()}', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_values)):
        ax.text(val, i, f' {val:.3f}', va='center', fontsize=10)
    
    plt.tight_layout()
    
    return fig


def compare_scenarios(age, income, credit_score, debt_ratio, 
                     employment_years, loan_amount, num_accounts, 
                     delinquencies, feature_to_vary):
    """
    Compare how varying a specific feature affects risk prediction
    """
    _initialize_model()
    
    # Define ranges for each feature
    feature_ranges = {
        'Age': (18, 80, 'age'),
        'Income': (20000, 200000, 'income'),
        'Credit Score': (300, 850, 'credit_score'),
        'Debt Ratio': (0, 100, 'debt_ratio'),
        'Employment Years': (0, 40, 'employment_years')
    }
    
    if feature_to_vary not in feature_ranges:
        feature_to_vary = 'Credit Score'
    
    min_val, max_val, param_name = feature_ranges[feature_to_vary]
    
    # Create range of values
    num_points = 20
    test_values = np.linspace(min_val, max_val, num_points)
    risk_probabilities = []
    
    # Original values
    base_params = {
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'debt_ratio': debt_ratio,
        'employment_years': employment_years,
        'loan_amount': loan_amount,
        'num_accounts': num_accounts,
        'delinquencies': delinquencies
    }
    
    # Vary the selected feature
    for val in test_values:
        params = base_params.copy()
        params[param_name] = val
        
        X = _map_inputs_to_features(**params)
        
        if _scaler is not None:
            X_scaled = _scaler.transform(X)
        else:
            X_scaled = X
        
        prob = _model.predict_proba(X_scaled)[0][1] * 100
        risk_probabilities.append(prob)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the curve
    ax.plot(test_values, risk_probabilities, linewidth=2.5, color='#1f77b4')
    
    # Mark the current value
    current_val = base_params[param_name]
    current_X = _map_inputs_to_features(**base_params)
    if _scaler is not None:
        current_X_scaled = _scaler.transform(current_X)
    else:
        current_X_scaled = current_X
    current_risk = _model.predict_proba(current_X_scaled)[0][1] * 100
    
    ax.scatter([current_val], [current_risk], color='red', s=200, zorder=5, 
               label='Current Value', marker='o', edgecolors='darkred', linewidths=2)
    
    # Add risk zones
    ax.axhspan(0, 30, alpha=0.1, color='green', label='Low Risk Zone')
    ax.axhspan(30, 60, alpha=0.1, color='yellow', label='Medium Risk Zone')
    ax.axhspan(60, 100, alpha=0.1, color='red', label='High Risk Zone')
    
    ax.set_xlabel(feature_to_vary, fontsize=12, fontweight='bold')
    ax.set_ylabel('Default Risk Probability (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Impact of {feature_to_vary} on Credit Risk', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    return fig


# Global storage for credit risk versions
_saved_credit_versions = {}

def reset_credit_data():
    """Reset all input fields to default values"""
    defaults = sample_credit_data()
    return (
        defaults['age'],
        defaults['income'],
        defaults['credit_score'],
        defaults['debt_ratio'],
        defaults['employment_years'],
        defaults['loan_amount'],
        defaults['num_accounts'],
        defaults['delinquencies']
    )


def _fig_to_base64(fig):
    """Convert a Matplotlib figure to a base64 PNG data URI."""
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


def get_credit_report_html(data):
    """
    Generate a full HTML report for a credit risk analysis version.
    """
    timestamp = data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    age = data.get('age')
    income = data.get('income')
    credit_score = data.get('credit_score')
    debt_ratio = data.get('debt_ratio')
    employment_years = data.get('employment_years')
    loan_amount = data.get('loan_amount')
    num_accounts = data.get('num_accounts')
    delinquencies = data.get('delinquencies')
    risk_output = data.get('risk_output', 'N/A')
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .section {{ background-color: #f9f9f9; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #3498db; }}
            .risk-low {{ color: #27ae60; font-weight: bold; font-size: 1.2em; }}
            .risk-med {{ color: #f39c12; font-weight: bold; font-size: 1.2em; }}
            .risk-high {{ color: #c0392b; font-weight: bold; font-size: 1.2em; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Credit Risk Analysis Report</h1>
            <p>Generated: {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>Risk Assessment</h2>
            <div>{risk_output}</div>
        </div>
        
        <div class="section">
            <h2>Feature Attributions</h2>
            <p>The following chart identifies which features most heavily influenced the model's prediction of risk.</p>
            {f'<img src="{data["graph_b64"]}" style="max-width:100%; border-radius:8px; border:1px solid #ddd;"/>' if data.get("graph_b64") else "<p><em>No explanation graph available.</em></p>"}
        </div>
        
        <div class="section">
            <h2>Applicant Information</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Age</td><td>{age} years</td></tr>
                <tr><td>Annual Income</td><td>${income:,}</td></tr>
                <tr><td>Credit Score</td><td>{credit_score}</td></tr>
                <tr><td>Debt-to-Income Ratio</td><td>{debt_ratio}%</td></tr>
                <tr><td>Years at Current Job</td><td>{employment_years}</td></tr>
                <tr><td>Loan Amount Requested</td><td>${loan_amount:,}</td></tr>
                <tr><td>Number of Credit Accounts</td><td>{num_accounts}</td></tr>
                <tr><td>Past Delinquencies</td><td>{delinquencies}</td></tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Note on Methodology</h2>
            <p>This report uses local model explanations (SHAP/LIME/Integrated Gradients) to identify the key factors contributing to the predicted credit risk. The results should be used for auditing purposes to ensure algorithmic fairness and compliance.</p>
        </div>
    </body>
    </html>
    """
    return html


def save_credit_version(age, income, credit_score, debt_ratio, 
                        employment_years, loan_amount, num_accounts, 
                        delinquencies, risk_output, version_name=None,
                        explanation_fig=None):
    """
    Save the current credit analysis version to global storage.
    """
    global _saved_credit_versions
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not version_name:
        # Better, more descriptive naming
        model_snippet = _current_model_name.split('/')[-1] if '/' in _current_model_name else _current_model_name
        version_name = f"📊 {model_snippet} | Score: {credit_score}, Income: ${income:,} | {timestamp}"
        
    data = {
        'timestamp': timestamp,
        'age': age,
        'income': income,
        'credit_score': credit_score,
        'debt_ratio': debt_ratio,
        'employment_years': employment_years,
        'loan_amount': loan_amount,
        'num_accounts': num_accounts,
        'delinquencies': delinquencies,
        'risk_output': risk_output,
        'graph_b64': None
    }

    if explanation_fig:
        try:
            data['graph_b64'] = _fig_to_base64(explanation_fig)
        except Exception as e:
            print(f"Error converting credit figure to base64: {e}")
    
    # Generate HTML content for this version
    data['html'] = get_credit_report_html(data)
    
    _saved_credit_versions[version_name] = data
    
    # Return updated list of version names for dropdown
    return list(_saved_credit_versions.keys()), f"✅ Saved: {version_name}"


def export_selected_credit_html(selected):
    """Export a single selected version as an HTML file."""
    global _saved_credit_versions
    if not selected or selected not in _saved_credit_versions:
        return None
    
    html = _saved_credit_versions[selected]['html']
    temp_dir = tempfile.mkdtemp()
    safe_label = "".join([c if c.isalnum() else "_" for c in selected])[:40]
    path = os.path.join(temp_dir, f"credit_audit_{safe_label}.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path


def export_all_credit_html():
    """Export all saved versions consolidated into one HTML file."""
    global _saved_credit_versions
    if not _saved_credit_versions:
        return None
    
    html = "<html><body><h1>All Saved Credit Risk Audits</h1><hr>"
    for version_name, data in _saved_credit_versions.items():
        html += f"<h2>{version_name}</h2>"
        html += data['html']
        html += "<hr>"
    html += "</body></html>"
    
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, "all_credit_risk_audits.html")
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    return path

