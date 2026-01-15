import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.dermoscopy_dataset import DermoscopyDataset
from datasets.clinical_dataset import ClinicalDataset
from datasets.multispectral_dataset import MultispectralDataset
from datasets.multimodal_dataset import MultiModalDataset
from models.fusion_model import FusionModel


# -----------------------------
# LOAD CONFIG
# -----------------------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DERMO_PATH = config["dataset"]["dermoscopy_path"]
CLINICAL_PATH = config["dataset"]["clinical_path"]
MODEL_PATH = config["training"]["save_path"]

BATCH_SIZE = config["training"]["batch_size"]
NUM_WORKERS = config["training"]["num_workers"]

USE_GPU = config["device"]["use_gpu"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# SEVERITY MAPPING
# -----------------------------
def score_to_severity(score):
    if score < 0.25:
        return "Mild"
    elif score < 0.6:
        return "Moderate"
    else:
        return "Severe"


# -----------------------------
# MAIN EVALUATION
# -----------------------------
def evaluate():
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
    # DATASETS (VALIDATION ONLY)
    # -----------------------------
    dermo_dataset = DermoscopyDataset(DERMO_PATH, transform)
    clinical_dataset = ClinicalDataset(CLINICAL_PATH, transform)
    multispectral_dataset = MultispectralDataset(CLINICAL_PATH, transform)

    full_dataset = MultiModalDataset(
        dermoscopy_ds=dermo_dataset,
        clinical_ds=clinical_dataset,
        multispectral_ds=multispectral_dataset
    )

    # Use same 80/20 logic, but only evaluate on VAL
    val_size = int(0.2 * len(full_dataset))
    _, val_dataset = torch.utils.data.random_split(
        full_dataset, [len(full_dataset) - val_size, val_size]
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
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"[INFO] Loaded model from {MODEL_PATH}")

    preds_list = []
    labels_list = []
    severities = []

    mse_loss = torch.nn.MSELoss()
    total_loss = 0.0

    # -----------------------------
    # EVALUATION LOOP
    # -----------------------------
    with torch.no_grad():
        for clinical_img, dermo_img, multi_img, label in val_loader:
            clinical_img = clinical_img.to(DEVICE)
            dermo_img = dermo_img.to(DEVICE)
            multi_img = multi_img.to(DEVICE)
            label = label.to(DEVICE)

            preds = model(
                clinical_img=clinical_img,
                dermoscopy_img=dermo_img,
                multispectral_img=multi_img
            )

            loss = mse_loss(preds, label)
            total_loss += loss.item()

            preds_np = preds.cpu().numpy().flatten()
            labels_np = label.cpu().numpy().flatten()

            preds_list.extend(preds_np)
            labels_list.extend(labels_np)

            for s in preds_np:
                severities.append(score_to_severity(s))

    # -----------------------------
    # METRICS
    # -----------------------------
    preds_arr = np.array(preds_list)
    labels_arr = np.array(labels_list)

    mse = np.mean((preds_arr - labels_arr) ** 2)
    mae = np.mean(np.abs(preds_arr - labels_arr))
    rmse = np.sqrt(mse)

    print("\n========== EVALUATION RESULTS ==========")
    print(f"MSE  : {mse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print("=======================================\n")

    # -----------------------------
    # SAVE CSV REPORT
    # -----------------------------
    df = pd.DataFrame({
        "prediction_score": preds_arr,
        "true_score": labels_arr,
        "severity": severities
    })

    csv_path = os.path.join(OUTPUT_DIR, "metrics.csv")
    df.to_csv(csv_path, index=False)

    # -----------------------------
    # SCORE DISTRIBUTION PLOT
    # -----------------------------
    plt.figure()
    plt.hist(preds_arr, bins=30)
    plt.title("Pigmentation Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "score_distribution.png"))
    plt.close()

    # -----------------------------
    # SEVERITY DISTRIBUTION PLOT
    # -----------------------------
    severity_counts = pd.Series(severities).value_counts()

    plt.figure()
    severity_counts.plot(kind="bar")
    plt.title("Severity Distribution")
    plt.xlabel("Severity")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "severity_distribution.png"))
    plt.close()

    print("[DONE] Evaluation complete.")
    print(f"[DONE] Results saved in `{OUTPUT_DIR}/`")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run train.py first.")

    evaluate()
