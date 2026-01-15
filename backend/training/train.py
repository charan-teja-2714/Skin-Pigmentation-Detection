# import os
# import yaml
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms

# from datasets.dermoscopy_dataset import DermoscopyDataset
# from datasets.clinical_dataset import ClinicalDataset
# from datasets.multimodal_dataset import MultiModalDataset
# from models.fusion_model import FusionModel
# from datasets.multispectral_dataset import MultispectralDataset


# # -----------------------------
# # LOAD CONFIG
# # -----------------------------
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# DERMO_PATH = config["dataset"]["dermoscopy_path"]
# CLINICAL_PATH = config["dataset"]["clinical_path"]

# BATCH_SIZE = config["training"]["batch_size"]
# EPOCHS = config["training"]["epochs"]
# LEARNING_RATE = config["training"]["learning_rate"]
# NUM_WORKERS = config["training"]["num_workers"]
# SAVE_PATH = config["training"]["save_path"]

# USE_GPU = config["device"]["use_gpu"]
# DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")


# # -----------------------------
# # TRAINING FUNCTION
# # -----------------------------
# def train():
#     print(f"[INFO] Using device: {DEVICE}")

#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])

#     dermo_dataset = DermoscopyDataset(DERMO_PATH, transform)
#     clinical_dataset = ClinicalDataset(CLINICAL_PATH, transform)

#     full_dataset = MultiModalDataset(
#         dermoscopy_ds=dermo_dataset,
#         clinical_ds=clinical_dataset
#     )

#     # -----------------------------
#     # TRAIN / VAL SPLIT (80 / 20)
#     # -----------------------------
#     train_size = int(0.8 * len(full_dataset))
#     val_size = len(full_dataset) - train_size

#     train_dataset, val_dataset = random_split(
#         full_dataset, [train_size, val_size]
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=True
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=True
#     )

#     print(f"[INFO] Dermoscopy images: {len(dermo_dataset)}")
#     print(f"[INFO] Clinical images: {len(clinical_dataset)}")
#     print(f"[INFO] Train samples: {len(train_dataset)}")
#     print(f"[INFO] Val samples: {len(val_dataset)}")

#     # -----------------------------
#     # MODEL, LOSS, OPTIMIZER
#     # -----------------------------
#     model = FusionModel().to(DEVICE)
#     criterion = nn.MSELoss()

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=1e-4
#     )

#     best_val_loss = float("inf")

#     # -----------------------------
#     # TRAINING LOOP
#     # -----------------------------
#     for epoch in range(EPOCHS):
#         # -------- TRAIN --------
#         model.train()
#         train_loss = 0.0

#         for clinical_img, dermo_img, label in train_loader:
#             clinical_img = clinical_img.to(DEVICE)
#             dermo_img = dermo_img.to(DEVICE)
#             label = label.to(DEVICE)

#             preds = model(
#                 clinical_img=clinical_img,
#                 dermoscopy_img=dermo_img,
#                 multispectral_img=None
#             )

#             loss = criterion(preds, label)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()

#         avg_train_loss = train_loss / len(train_loader)

#         # -------- VALIDATE --------
#         model.eval()
#         val_loss = 0.0

#         with torch.no_grad():
#             for clinical_img, dermo_img, label in val_loader:
#                 clinical_img = clinical_img.to(DEVICE)
#                 dermo_img = dermo_img.to(DEVICE)
#                 label = label.to(DEVICE)

#                 preds = model(
#                     clinical_img=clinical_img,
#                     dermoscopy_img=dermo_img,
#                     multispectral_img=None
#                 )

#                 loss = criterion(preds, label)
#                 val_loss += loss.item()

#         avg_val_loss = val_loss / len(val_loader)

#         print(
#             f"Epoch [{epoch+1}/{EPOCHS}] | "
#             f"Train Loss: {avg_train_loss:.4f} | "
#             f"Val Loss: {avg_val_loss:.4f}"
#         )

#         # -------- SAVE BEST MODEL --------
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), SAVE_PATH)
#             print(f"[INFO] Best model saved (val loss={best_val_loss:.4f})")

#     print("[DONE] Training + Validation complete.")
#     print(f"[DONE] Best model saved at: {SAVE_PATH}")


# # -----------------------------
# # ENTRY POINT
# # -----------------------------
# if __name__ == "__main__":
#     if not os.path.exists(DERMO_PATH):
#         raise FileNotFoundError(f"Dermoscopy path not found: {DERMO_PATH}")

#     if not os.path.exists(CLINICAL_PATH):
#         raise FileNotFoundError(f"Clinical path not found: {CLINICAL_PATH}")

#     train()


# import os
# import yaml
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms

# import sys
# from pathlib import Path

# # Add parent directory to path
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from datasets.dermoscopy_dataset import DermoscopyDataset
# from datasets.clinical_dataset import ClinicalDataset
# from datasets.multispectral_dataset import MultispectralDataset
# from datasets.multimodal_dataset import MultiModalDataset
# from models.fusion_model import FusionModel


# # -----------------------------
# # LOAD CONFIG
# # -----------------------------
# with open("config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# DERMO_PATH = config["dataset"]["dermoscopy_path"]
# CLINICAL_PATH = config["dataset"]["clinical_path"]

# BATCH_SIZE = config["training"]["batch_size"]
# EPOCHS = config["training"]["epochs"]
# LEARNING_RATE = config["training"]["learning_rate"]
# NUM_WORKERS = config["training"]["num_workers"]
# SAVE_PATH = config["training"]["save_path"]

# USE_GPU = config["device"]["use_gpu"]
# DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")


# # -----------------------------
# # TRAINING FUNCTION
# # -----------------------------
# def train():
#     print(f"[INFO] Using device: {DEVICE}")

#     # Image preprocessing
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225]
#         )
#     ])

#     # -----------------------------
#     # DATASETS
#     # -----------------------------
#     dermo_dataset = DermoscopyDataset(DERMO_PATH, transform)
#     clinical_dataset = ClinicalDataset(CLINICAL_PATH, transform)

#     # Multispectral simulated from clinical images
#     multispectral_dataset = MultispectralDataset(
#         CLINICAL_PATH, transform
#     )

#     full_dataset = MultiModalDataset(
#         dermoscopy_ds=dermo_dataset,
#         clinical_ds=clinical_dataset,
#         multispectral_ds=multispectral_dataset
#     )

#     # -----------------------------
#     # TRAIN / VAL SPLIT (80 / 20)
#     # -----------------------------
#     train_size = int(0.8 * len(full_dataset))
#     val_size = len(full_dataset) - train_size

#     train_dataset, val_dataset = random_split(
#         full_dataset, [train_size, val_size]
#     )

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=True,
#         num_workers=NUM_WORKERS,
#         pin_memory=True
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=BATCH_SIZE,
#         shuffle=False,
#         num_workers=NUM_WORKERS,
#         pin_memory=True
#     )

#     print(f"[INFO] Dermoscopy images     : {len(dermo_dataset)}")
#     print(f"[INFO] Clinical images      : {len(clinical_dataset)}")
#     print(f"[INFO] Multispectral images : {len(multispectral_dataset)}")
#     print(f"[INFO] Train samples        : {len(train_dataset)}")
#     print(f"[INFO] Val samples          : {len(val_dataset)}")

#     # -----------------------------
#     # MODEL, LOSS, OPTIMIZER
#     # -----------------------------
#     model = FusionModel().to(DEVICE)
#     criterion = nn.MSELoss()

#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=LEARNING_RATE,
#         weight_decay=1e-4
#     )

#     best_val_loss = float("inf")

#     # -----------------------------
#     # TRAINING LOOP
#     # -----------------------------
#     for epoch in range(EPOCHS):

#         # -------- TRAIN --------
#         model.train()
#         train_loss = 0.0

#         for clinical_img, dermo_img, multi_img, label in train_loader:
#             clinical_img = clinical_img.to(DEVICE)
#             dermo_img = dermo_img.to(DEVICE)
#             multi_img = multi_img.to(DEVICE)
#             label = label.to(DEVICE)

#             preds = model(
#                 clinical_img=clinical_img,
#                 dermoscopy_img=dermo_img,
#                 multispectral_img=multi_img
#             )

#             loss = criterion(preds, label)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             train_loss += loss.item()

#         avg_train_loss = train_loss / len(train_loader)

#         # -------- VALIDATE --------
#         model.eval()
#         val_loss = 0.0

#         with torch.no_grad():
#             for clinical_img, dermo_img, multi_img, label in val_loader:
#                 clinical_img = clinical_img.to(DEVICE)
#                 dermo_img = dermo_img.to(DEVICE)
#                 multi_img = multi_img.to(DEVICE)
#                 label = label.to(DEVICE)

#                 preds = model(
#                     clinical_img=clinical_img,
#                     dermoscopy_img=dermo_img,
#                     multispectral_img=multi_img
#                 )

#                 loss = criterion(preds, label)
#                 val_loss += loss.item()

#         avg_val_loss = val_loss / len(val_loader)

#         print(
#             f"Epoch [{epoch+1}/{EPOCHS}] | "
#             f"Train Loss: {avg_train_loss:.4f} | "
#             f"Val Loss: {avg_val_loss:.4f}"
#         )

#         # -------- SAVE BEST MODEL --------
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), SAVE_PATH)
#             print(f"[INFO] Best model saved (val loss={best_val_loss:.4f})")

#     print("[DONE] Training + Validation complete.")
#     print(f"[DONE] Best model saved at: {SAVE_PATH}")


# # -----------------------------
# # ENTRY POINT
# # -----------------------------
# if __name__ == "__main__":

#     if not os.path.exists(DERMO_PATH):
#         raise FileNotFoundError(f"Dermoscopy path not found: {DERMO_PATH}")

#     if not os.path.exists(CLINICAL_PATH):
#         raise FileNotFoundError(f"Clinical path not found: {CLINICAL_PATH}")

#     train()


import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import sys
from pathlib import Path
from tqdm import tqdm   # ⭐ NEW

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

BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LEARNING_RATE = config["training"]["learning_rate"]
NUM_WORKERS = config["training"]["num_workers"]
SAVE_PATH = config["training"]["save_path"]

USE_GPU = config["device"]["use_gpu"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() and USE_GPU else "cpu")


# -----------------------------
# TRAINING FUNCTION
# -----------------------------
def train():
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
    multispectral_dataset = MultispectralDataset(CLINICAL_PATH, transform)

    full_dataset = MultiModalDataset(
        dermoscopy_ds=dermo_dataset,
        clinical_ds=clinical_dataset,
        multispectral_ds=multispectral_dataset
    )

    # -----------------------------
    # TRAIN / VAL SPLIT
    # -----------------------------
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"[INFO] Dermoscopy images     : {len(dermo_dataset)}")
    print(f"[INFO] Clinical images      : {len(clinical_dataset)}")
    print(f"[INFO] Multispectral images : {len(multispectral_dataset)}")
    print(f"[INFO] Train samples        : {len(train_dataset)}")
    print(f"[INFO] Val samples          : {len(val_dataset)}")

    # -----------------------------
    # MODEL, LOSS, OPTIMIZER
    # -----------------------------
    model = FusionModel().to(DEVICE)
    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    best_val_loss = float("inf")

    # -----------------------------
    # TRAINING LOOP
    # -----------------------------
    for epoch in range(EPOCHS):

        # -------- TRAIN --------
        model.train()
        train_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]",
            leave=False
        )

        for clinical_img, dermo_img, multi_img, label in train_bar:
            clinical_img = clinical_img.to(DEVICE)
            dermo_img = dermo_img.to(DEVICE)
            multi_img = multi_img.to(DEVICE)
            label = label.to(DEVICE)

            preds = model(
                clinical_img=clinical_img,
                dermoscopy_img=dermo_img,
                multispectral_img=multi_img
            )

            loss = criterion(preds, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # -------- VALIDATE --------
        model.eval()
        val_loss = 0.0

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]",
            leave=False
        )

        with torch.no_grad():
            for clinical_img, dermo_img, multi_img, label in val_bar:
                clinical_img = clinical_img.to(DEVICE)
                dermo_img = dermo_img.to(DEVICE)
                multi_img = multi_img.to(DEVICE)
                label = label.to(DEVICE)

                preds = model(
                    clinical_img=clinical_img,
                    dermoscopy_img=dermo_img,
                    multispectral_img=multi_img
                )

                loss = criterion(preds, label)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )

        # -------- SAVE BEST MODEL --------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"[INFO] Best model saved (val loss={best_val_loss:.4f})")

    print("[DONE] Training + Validation complete.")
    print(f"[DONE] Best model saved at: {SAVE_PATH}")


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":

    if not os.path.exists(DERMO_PATH):
        raise FileNotFoundError(f"Dermoscopy path not found: {DERMO_PATH}")

    if not os.path.exists(CLINICAL_PATH):
        raise FileNotFoundError(f"Clinical path not found: {CLINICAL_PATH}")

    train()
