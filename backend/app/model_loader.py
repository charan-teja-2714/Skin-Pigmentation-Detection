import torch
from models.fusion_model import FusionModel

def load_model():
    model = FusionModel()
    model.eval()
    return model

def get_severity_label(score):
    if score <= 0.25:
        return "Mild"
    elif score <= 0.6:
        return "Moderate"
    else:
        return "Severe"