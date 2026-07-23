import torch
from PIL import Image
from torchvision import transforms
from severity_model_v2.models.dual_modal_fusion import DualModalFusionModel

MODEL_PATH = r"I:\Final Year Projects\Skin-Pigmentation-Detection - Copy\backend\severity_model_v2\checkpoints_dual_modal\best_model.pth"
TEST_IMG = r"I:\Final Year Projects\Skin-Pigmentation-Detection - Copy\backend\testing\ISIC_0034274.jpg"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = DualModalFusionModel(model_name="swin_tiny_patch4_window7_224", pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

img = transform(Image.open(TEST_IMG).convert('RGB')).unsqueeze(0)

with torch.no_grad():
    score = model(img, img).item()

severity = "Mild" if score <= 0.25 else "Moderate" if score <= 0.6 else "Severe"

print(f"Image: {TEST_IMG.split('\\')[-1]}")
print(f"Score: {score:.4f}")
print(f"Severity: {severity}")
