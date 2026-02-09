import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import io

# ============================================
# FIX: Resize and compress images before inference
# ============================================
def preprocess_image(image_bytes):
    # Open image
    image = Image.open(image_bytes).convert('RGB')
    
    # FIX: Check file size and compress if > 1MB
    if hasattr(image_bytes, 'seek'):
        image_bytes.seek(0, 2)  # Seek to end
        size_mb = image_bytes.tell() / (1024 * 1024)
        image_bytes.seek(0)  # Reset
        
        if size_mb > 1.0:
            # Compress image
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=85, optimize=True)
            output.seek(0)
            image = Image.open(output).convert('RGB')
    
    # FIX: Resize to 224x224 BEFORE inference (already in transform but explicit)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image).unsqueeze(0)
    return tensor