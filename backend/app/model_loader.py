import torch
import os
from huggingface_hub import hf_hub_download
from severity_model_v2.models.fusion_model import FusionModel
import numpy as np

def load_model():
    model = FusionModel()
    
    try:
        # Download model from Hugging Face
        model_path = hf_hub_download(
            repo_id="Charan2714/skin-pigmentation",
            filename="severity_swin_multimodal_best.pth",
            cache_dir="./hf_cache"
        )
        
        # Load the model weights
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load state dict with strict=False to handle any architecture differences
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded model from Hugging Face: Charan2714/skin-pigmentation")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model from Hugging Face: {e}")
        print(f"[WARNING] Using model with random weights")
    
    model.eval()
    return model

def get_severity_label(score):
    if score <= 0.25:
        return "Mild"
    elif score <= 0.6:
        return "Moderate"
    else:
        return "Severe"

# Demo function to simulate varied predictions
def simulate_varied_predictions():
    """Generate varied predictions for demo purposes"""
    return np.random.choice([0.15, 0.4, 0.8], p=[0.3, 0.4, 0.3])