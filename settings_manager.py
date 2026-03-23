import json
import os
import tempfile
import gradio as gr
from datetime import datetime

def export_settings(settings_dict):
    """
    Export current experiment settings to a JSON file.
    settings_dict: Dictionary of {label/id: value}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"audit_settings_{timestamp}.json"
    
    # Clean settings_dict (ensure JSON serializable)
    export_data = {}
    for k, v in settings_dict.items():
        if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
            export_data[k] = v
        else:
            export_data[k] = str(v)
            
    temp_dir = tempfile.mkdtemp()
    path = os.path.join(temp_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=4)
    return path

def import_settings(file_path):
    """
    Load settings from a JSON file.
    Returns a dictionary of settings.
    """
    if not file_path:
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error importing settings: {e}")
        return None

def apply_settings_to_ui(settings, component_map):
    """
    Map imported settings to Gradio components.
    component_map: Dictionary of {key_in_json: gr.Component}
    Returns a list of gr.update() for the components.
    """
    updates = []
    for key, component in component_map.items():
        if key in settings:
            updates.append(gr.update(value=settings[key]))
        else:
            updates.append(gr.update()) # No change
    return updates
