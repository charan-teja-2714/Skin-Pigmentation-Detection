import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cust_datasets.dermoscopy_dataset import DermoscopyDataset
from cust_datasets.clinical_dataset import ClinicalDataset
from cust_datasets.multimodal_dataset import MultiModalDataset
from models.fusion_model import FusionModel


# -----------------------------
# LOAD CONFIG
# -----------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DERMO_PATH = config["dataset"]["dermoscopy_path"]
CLINICAL_PATH = config["dataset"]["clinical_path"]
# MODEL_PATH = config["training"]["save_path"]
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model_swinvit.pt")
BATCH_SIZE = config["training"]["batch_size"]
NUM_WORKERS = config["training"]["num_workers"]

USE_GPU = config["device"]["use_gpu"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")


# -----------------------------
# VALIDATION FUNCTION
# -----------------------------
def validate():
    print(f"[INFO] Using device: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # -----------------------------
    # DATASETS
    # -----------------------------
    dermo_dataset = DermoscopyDataset(DERMO_PATH, transform)
    clinical_dataset = ClinicalDataset(CLINICAL_PATH, transform)

    full_dataset = MultiModalDataset(
        dermoscopy_ds=dermo_dataset,
        clinical_ds=clinical_dataset
    )

    # Use validation split (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    _, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    model = FusionModel().to(DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Load with strict=False to handle missing keys
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"[INFO] Loaded model from {MODEL_PATH}")
    print(f"[INFO] Validation samples: {len(val_dataset)}")

    # -----------------------------
    # VALIDATION LOOP
    # -----------------------------
    criterion = nn.MSELoss()
    val_loss = 0.0
    num_batches = 0

    val_bar = tqdm(val_loader, desc="Validating")

    with torch.no_grad():
        for clinical_img, dermo_img, label in val_bar:
            clinical_img = clinical_img.to(DEVICE)
            dermo_img = dermo_img.to(DEVICE)
            label = label.to(DEVICE)

            preds = model(
                clinical_img=clinical_img,
                dermoscopy_img=dermo_img
            )

            loss = criterion(preds, label)
            val_loss += loss.item()
            num_batches += 1

            val_bar.set_postfix(loss=loss.item())

    avg_val_loss = val_loss / num_batches

    print(f"\n[RESULT] Average Validation Loss: {avg_val_loss:.4f}")
    print("[DONE] Validation complete.")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run train.py first.")

    validate()