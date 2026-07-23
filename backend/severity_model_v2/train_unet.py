"""
Train the U-Net segmentation model on ISIC dermoscopy images + masks.

Usage (run from the backend directory):
    python -m severity_model_v2.train_unet

Outputs:
    severity_model_v2/checkpoints_unet/best_unet.pth   <- used at inference
    severity_model_v2/checkpoints_unet/last_unet.pth
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Ensure imports work from the backend root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from severity_model_v2.models.unet import UNet
from severity_model_v2.datasets.unet_dataset import ISICSegmentationDataset

# ── Configuration ─────────────────────────────────────────────
DATA_ROOT    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "data", "dermoscopy")
IMAGES_DIR   = os.path.join(DATA_ROOT, "images")
MASKS_DIR    = os.path.join(DATA_ROOT, "masks")

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "checkpoints_unet")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE   = 8
EPOCHS       = 20
LR           = 1e-4
VAL_SPLIT    = 0.15
NUM_WORKERS  = 0      # set to 4 if you have a multi-core CPU with enough RAM
RESUME       = False  # set True to resume from last_unet.pth
# ─────────────────────────────────────────────────────────────


# ── Loss: BCE + Dice ──────────────────────────────────────────
def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    pred   = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce  = nn.functional.binary_cross_entropy(pred, target)
    dice = dice_loss(pred, target)
    return 0.5 * bce + 0.5 * dice


# ── Training loop ─────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[UNET] Device: {device}")

    # Dataset
    full_dataset = ISICSegmentationDataset(IMAGES_DIR, MASKS_DIR, augment=True)
    n_val  = int(len(full_dataset) * VAL_SPLIT)
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    # Val dataset without augmentation
    val_ds.dataset.augment = False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)

    print(f"[UNET] Train: {n_train}  |  Val: {n_val}")

    # Model
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    start_epoch = 1
    best_val_loss = float("inf")

    if RESUME:
        ckpt = os.path.join(CHECKPOINT_DIR, "last_unet.pth")
        if os.path.exists(ckpt):
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_epoch = state.get("epoch", 1) + 1
            best_val_loss = state.get("best_val_loss", float("inf"))
            print(f"[UNET] Resumed from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch:02d}/{EPOCHS} [Train]",
                         unit="batch", leave=False)
        for imgs, masks in train_bar:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss  = combined_loss(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_loader)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0
        val_bar = tqdm(val_loader,
                       desc=f"Epoch {epoch:02d}/{EPOCHS} [Val]  ",
                       unit="batch", leave=False)
        with torch.no_grad():
            for imgs, masks in val_bar:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                batch_loss = combined_loss(preds, masks).item()
                val_loss  += batch_loss

                # IoU
                pred_bin = (preds > 0.5).float()
                inter = (pred_bin * masks).sum()
                union = (pred_bin + masks).clamp(0, 1).sum()
                batch_iou = (inter / (union + 1e-6)).item()
                val_iou  += batch_iou
                val_bar.set_postfix(loss=f"{batch_loss:.4f}", iou=f"{batch_iou:.4f}")

        val_loss /= len(val_loader)
        val_iou  /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch [{epoch:02d}/{EPOCHS}]  "
              f"Train Loss: {train_loss:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  |  "
              f"Val IoU: {val_iou:.4f}")

        # Save checkpoints
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
        }
        torch.save(state, os.path.join(CHECKPOINT_DIR, "last_unet.pth"))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save only the model weights for inference
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINT_DIR, "best_unet.pth"))
            print(f"  -> Best model saved (val_loss={best_val_loss:.4f})")

    print("[UNET] Training complete.")
    print(f"[UNET] Best model -> {os.path.join(CHECKPOINT_DIR, 'best_unet.pth')}")


if __name__ == "__main__":
    train()
