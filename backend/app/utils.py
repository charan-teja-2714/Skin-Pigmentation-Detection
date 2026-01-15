import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_bytes).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return tensor