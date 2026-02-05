import torch
import os
from models.fusion_model import FusionModel
import numpy as np

def load_model():
    model = FusionModel()
    
    # Load pretrained Swin Transformer weights
    model_path = os.path.join(os.path.dirname(__file__), "..", "new_train_model.pth")
    
    if os.path.exists(model_path):
        pretrained_dict = torch.load(model_path, map_location='cpu')
        if 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']
        
        # Load pretrained weights into both encoders with size matching
        model_dict = model.state_dict()
        
        # Update clinical encoder (only matching layers)
        clinical_dict = {f'clinical_encoder.model.{k}': v for k, v in pretrained_dict.items() 
                        if f'clinical_encoder.model.{k}' in model_dict and 
                        model_dict[f'clinical_encoder.model.{k}'].shape == v.shape}
        model_dict.update(clinical_dict)
        
        # Update dermoscopy encoder (only matching layers)
        dermoscopy_dict = {f'dermoscopy_encoder.model.{k}': v for k, v in pretrained_dict.items() 
                          if f'dermoscopy_encoder.model.{k}' in model_dict and 
                          model_dict[f'dermoscopy_encoder.model.{k}'].shape == v.shape}
        model_dict.update(dermoscopy_dict)
        
        model.load_state_dict(model_dict, strict=False)
        print(f"[INFO] Loaded pretrained weights from {model_path}")
    else:
        print(f"[WARNING] Pretrained model not found at {model_path}, using random weights")
    
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