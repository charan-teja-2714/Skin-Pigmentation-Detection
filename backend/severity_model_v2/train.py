# import os
# import sys
# import torch
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from torchvision import transforms

# # Add current directory to path
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# from datasets.severity_dataset import SeverityDataset
# from models.fusion_model import FusionModel

# # ---------------- CONFIG ----------------
# DATA_ROOT = r"I:\Final Year Projects\Skin-Pigmentation-Detection\backend\data"
# BATCH_SIZE = 8
# EPOCHS = 15
# LR = 1e-4
# CKPT_DIR = "checkpoints"

# os.makedirs(CKPT_DIR, exist_ok=True)

# # ---------------- DATA ----------------
# # transform = transforms.Compose([
# #     transforms.ToTensor(),
# #     transforms.Resize((224, 224))
# # ])

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])


# dataset = SeverityDataset(DATA_ROOT, transform)
# print(f"Dataset size: {len(dataset)}")

# if len(dataset) == 0:
#     print(f"ERROR: No data found in {DATA_ROOT}")
#     print("Expected structure:")
#     print("data/")
#     print("  clinical/")
#     print("    metadata.csv")
#     print("    images/")
#     print("  dermoscopy/")
#     print("    metadata.csv")
#     print("    images/")
#     exit(1)

# loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# # ---------------- MODEL ----------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = FusionModel().to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# criterion = torch.nn.MSELoss()

# best_loss = float("inf")

# # ---------------- TRAIN ----------------
# for epoch in range(1, EPOCHS + 1):
#     model.train()
#     running_loss = 0.0

#     loop = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
#     for images, metadata, labels in loop:
#         images = images.to(device)
#         metadata = metadata.to(device)
#         labels = labels.to(device).unsqueeze(1)

#         optimizer.zero_grad()
#         preds = model(images, metadata)
#         loss = criterion(preds, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         loop.set_postfix(loss=loss.item())

#     avg_loss = running_loss / len(loader)

#     # Save last epoch
#     torch.save(model.state_dict(), f"{CKPT_DIR}/new_train_last_epoch.pth")

#     # Save best model
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         torch.save(model.state_dict(), f"{CKPT_DIR}/new_train_model.pth")

#     print(f"Epoch {epoch} finished | Avg Loss: {avg_loss:.4f}")




# import os
# import torch
# from tqdm import tqdm
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms

# from datasets.severity_dataset import SeverityDataset
# from models.fusion_model import FusionModel

# # ================= CONFIG =================
# DATA_ROOT = "data"
# BATCH_SIZE = 8
# EPOCHS = 15
# LR = 1e-4
# VAL_SPLIT = 0.2
# CHECKPOINT_DIR = "checkpoints"
# RESUME = True   # 🔁 set False to start fresh

# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # ================= TRANSFORMS =================
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])

# # ================= DATASET =================
# full_dataset = SeverityDataset(DATA_ROOT, transform)
# dataset_size = len(full_dataset)

# val_size = int(dataset_size * VAL_SPLIT)
# train_size = dataset_size - val_size

# train_dataset, val_dataset = random_split(
#     full_dataset, [train_size, val_size]
# )

# train_loader = DataLoader(
#     train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
# )

# val_loader = DataLoader(
#     val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
# )

# print(f"Total samples     : {dataset_size}")
# print(f"Training samples  : {train_size}")
# print(f"Validation samples: {val_size}")

# # ================= MODEL =================
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = FusionModel().to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# criterion = torch.nn.MSELoss()

# start_epoch = 1
# best_val_loss = float("inf")

# # ================= RESUME =================
# resume_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")

# if RESUME and os.path.exists(resume_path):
#     checkpoint = torch.load(resume_path, map_location=device)
#     model.load_state_dict(checkpoint["model_state"])
#     optimizer.load_state_dict(checkpoint["optimizer_state"])
#     start_epoch = checkpoint["epoch"] + 1
#     best_val_loss = checkpoint["best_val_loss"]

#     print(f"🔁 Resuming from epoch {start_epoch}")
# else:
#     print("🆕 Starting training from scratch")

# # ================= TRAINING =================
# for epoch in range(start_epoch, EPOCHS + 1):

#     # ---------- TRAIN ----------
#     model.train()
#     train_loss = 0.0

#     train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]")
#     for images, metadata, labels in train_loop:
#         images = images.to(device)
#         metadata = metadata.to(device)
#         labels = labels.to(device).unsqueeze(1)

#         optimizer.zero_grad()
#         preds = model(images, metadata)
#         loss = criterion(preds, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()
#         train_loop.set_postfix(loss=loss.item())

#     avg_train_loss = train_loss / len(train_loader)

#     # ---------- VALIDATION ----------
#     model.eval()
#     val_loss = 0.0

#     with torch.no_grad():
#         val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [VAL]")
#         for images, metadata, labels in val_loop:
#             images = images.to(device)
#             metadata = metadata.to(device)
#             labels = labels.to(device).unsqueeze(1)

#             preds = model(images, metadata)
#             loss = criterion(preds, labels)

#             val_loss += loss.item()
#             val_loop.set_postfix(loss=loss.item())

#     avg_val_loss = val_loss / len(val_loader)

#     # ---------- SAVE BEST FOR THIS EPOCH ----------
#     best_epoch_path = os.path.join(
#         CHECKPOINT_DIR, f"best_epoch_{epoch:02d}.pth"
#     )

#     torch.save({
#         "epoch": epoch,
#         "model_state": model.state_dict(),
#         "optimizer_state": optimizer.state_dict(),
#         "val_loss": avg_val_loss
#     }, best_epoch_path)

#     # ---------- SAVE LAST CHECKPOINT (RESUME) ----------
#     torch.save({
#         "epoch": epoch,
#         "model_state": model.state_dict(),
#         "optimizer_state": optimizer.state_dict(),
#         "best_val_loss": min(best_val_loss, avg_val_loss)
#     }, resume_path)

#     # ---------- TRACK GLOBAL BEST ----------
#     if avg_val_loss < best_val_loss:
#         best_val_loss = avg_val_loss
#         torch.save(
#             model.state_dict(),
#             os.path.join(CHECKPOINT_DIR, "best_model_overall.pth")
#         )
#         improved = "✅"
#     else:
#         improved = "—"

#     # ---------- LOG ----------
#     print(
#         f"\nEpoch {epoch} Summary:"
#         f"\n  Train Loss: {avg_train_loss:.6f}"
#         f"\n  Val Loss  : {avg_val_loss:.6f} {improved}"
#         f"\n  Best Val  : {best_val_loss:.6f}\n"
#     )

# print("🎉 Training complete")




import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from datasets.severity_dataset import SeverityDataset
from models.image_only_model import ImageOnlyModel

# ================= CONFIG =================
DATA_ROOT = r"I:\Final Year Projects\Skin-Pigmentation-Detection\backend\data"
BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
VAL_SPLIT = 0.2
CHECKPOINT_DIR = "checkpoints_image_only"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================= TRANSFORMS =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ================= DATA =================
dataset = SeverityDataset(DATA_ROOT, transform)

if len(dataset) == 0:
    print("ERROR: No images found in data directories!")
    print("Please ensure you have images in:")
    print(f"  {DATA_ROOT}/clinical/images/")
    print(f"  {DATA_ROOT}/dermoscopy/images/")
    exit(1)

print(f"Dataset size: {len(dataset)}")

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# ================= MODEL =================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ImageOnlyModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

best_val_loss = float("inf")

# ================= TRAIN =================
for epoch in range(1, EPOCHS + 1):

    # ---- TRAIN ----
    model.train()
    train_loss = 0

    for images, _, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]"):
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(images)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train = train_loss / len(train_loader)

    # ---- VAL ----
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, _, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [VAL]"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            preds = model(images)
            loss = criterion(preds, labels)
            val_loss += loss.item()

    avg_val = val_loss / len(val_loader)

    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/last_epoch.pth")

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")
        flag = "✅"
    else:
        flag = "—"

    print(
        f"\nEpoch {epoch} | Train: {avg_train:.4f} | Val: {avg_val:.4f} {flag}\n"
    )
