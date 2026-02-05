import torch
from torchvision import transforms
from PIL import Image
import yaml
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.fusion_model import FusionModel


# -----------------------------
# LOAD CONFIG
# -----------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

USE_GPU = config["device"]["use_gpu"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")

MODEL_PATH = config["training"]["save_path"]


# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = transform(image)
    return image.unsqueeze(0)  # (1, 3, 224, 224)


# -----------------------------
# SEVERITY MAPPING WITH CONFIDENCE
# -----------------------------
def score_to_severity_with_confidence(score):
    if score < 0.25:
        severity = "Mild"
        # Confidence based on distance from boundary
        confidence = 1.0 - (0.25 - score) / 0.25
    elif score < 0.6:
        severity = "Moderate"
        # Distance from nearest boundary
        dist_to_mild = score - 0.25
        dist_to_severe = 0.6 - score
        min_dist = min(dist_to_mild, dist_to_severe)
        confidence = min_dist / 0.175  # 0.175 = (0.6-0.25)/2
    else:
        severity = "Severe"
        # Confidence based on distance from boundary
        confidence = (score - 0.6) / 0.4
    
    # Ensure confidence is between 0.5 and 1.0
    confidence = max(0.5, min(1.0, confidence))
    return severity, confidence


# -----------------------------
# MAIN INFERENCE
# -----------------------------
def main():
    print(f"[INFO] Using device: {DEVICE}")

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    model = FusionModel().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"[INFO] Loaded model from {MODEL_PATH}")

    # -----------------------------
    # PROVIDE IMAGE PATHS
    # -----------------------------
    clinical_image_path = "../backend/data/clinical/images/example.png"
    dermoscopy_image_path = "../backend/data/dermoscopy/images/example.png"

    clinical_img = load_image(clinical_image_path).to(DEVICE)
    dermo_img = load_image(dermoscopy_image_path).to(DEVICE)

    # -----------------------------
    # INFERENCE
    # -----------------------------
    with torch.no_grad():
        score = model(
            clinical_img=clinical_img,
            dermoscopy_img=dermo_img
        )

    score_value = score.item()
    severity, confidence = score_to_severity_with_confidence(score_value)

    print("\n========== RESULT ==========")
    print(f"Pigmentation Score  : {score_value:.4f}")
    print(f"Severity Level     : {severity}")
    print(f"Confidence         : {confidence:.2%}")
    print("============================\n")


if __name__ == "__main__":
    main()
