import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error

from datasets.severity_dataset import SeverityDataset
from models.fusion_model import FusionModel

# ================= CONFIG =================
DATA_ROOT = "data"
CHECKPOINT_PATH = r"I:\Final Year Projects\Skin-Pigmentation-Detection\backend\checkpoints\new_train_model.pth"
BATCH_SIZE = 8
VAL_SPLIT = 0.2

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= DATASET =================
full_dataset = SeverityDataset(DATA_ROOT, transform)

dataset_size = len(full_dataset)
val_size = int(dataset_size * VAL_SPLIT)
train_size = dataset_size - val_size

_, val_dataset = random_split(
    full_dataset, [train_size, val_size]
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Total samples     : {dataset_size}")
print(f"Validation samples: {val_size}")

# ================= MODEL =================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FusionModel().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

# ================= EVALUATION =================
all_preds = []
all_labels = []

with torch.no_grad():
    loop = tqdm(val_loader, desc="Evaluating")
    for images, metadata, labels in loop:
        images = images.to(device)
        metadata = metadata.to(device)

        preds = model(images, metadata).squeeze(1).cpu().numpy()
        labels = labels.numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

# ================= METRICS =================
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

mae = mean_absolute_error(all_labels, all_preds)
mse = mean_squared_error(all_labels, all_preds)
rmse = np.sqrt(mse)

# Severity-level accuracy (rule-based bins)
def to_severity(score):
    if score < 0.3:
        return "Mild"
    elif score < 0.6:
        return "Moderate"
    else:
        return "Severe"

pred_levels = [to_severity(s) for s in all_preds]
true_levels = [to_severity(s) for s in all_labels]

severity_acc = np.mean(
    [p == t for p, t in zip(pred_levels, true_levels)]
)

# ================= RESULTS =================
print("\n===== Evaluation Results =====")
print(f"MAE   : {mae:.4f}")
print(f"MSE   : {mse:.4f}")
print(f"RMSE  : {rmse:.4f}")
print(f"Severity Accuracy : {severity_acc * 100:.2f}%")
print("==============================")
