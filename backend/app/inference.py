import torch
from .utils import preprocess_image
from .model_loader import get_severity_label

def run_inference(model, clinical_image, dermoscopy_image=None, multispectral_image=None):
    with torch.no_grad():
        clinical_tensor = preprocess_image(clinical_image)
        
        dermoscopy_tensor = None
        if dermoscopy_image:
            dermoscopy_tensor = preprocess_image(dermoscopy_image)
            
        multispectral_tensor = None
        if multispectral_image:
            multispectral_tensor = preprocess_image(multispectral_image)
        
        score = model(clinical_tensor, dermoscopy_tensor, multispectral_tensor)
        score_value = float(score.squeeze().item())
        severity = get_severity_label(score_value)
        
        return {
            "score": round(score_value, 3),
            "severity": severity
        }