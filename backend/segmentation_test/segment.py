import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Image preprocessing (U-Net style)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def run_segmentation(image_path, model):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

    # output: (1, 1, H, W) or (1, 2, H, W)
    if output.shape[1] > 1:
        mask = torch.argmax(output, dim=1)
    else:
        mask = (output > 0.5).long()

    mask = mask.squeeze().cpu().numpy()
    return image, mask
