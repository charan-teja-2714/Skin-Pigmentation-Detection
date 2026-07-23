import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from datasets.dual_modal_dataset import DualModalDataset
from models.dual_modal_fusion import DualModalFusionModel

# ================= CONFIG =================
DATA_ROOT = r"I:\Final Year Projects\Skin-Pigmentation-Detection - Copy\backend\data"
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
VAL_SPLIT = 0.2
CHECKPOINT_DIR = "checkpoints_dual_modal"
RESUME = False  # Set True to resume from last checkpoint

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ================= TRANSFORMS =================
# Note: Masking is applied in the dataset __getitem__ method
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ================= DATASET =================
print("Loading dual-modal dataset...")
dataset = DualModalDataset(DATA_ROOT, transform)

if len(dataset) == 0:
    print("ERROR: No paired samples found!")
    print("Expected structure:")
    print(f"  {DATA_ROOT}/clinical/images/")
    print(f"  {DATA_ROOT}/dermoscopy/images/")
    print(f"  {DATA_ROOT}/dermoscopy/masks/")
    exit(1)

print(f"\nDataset size: {len(dataset)}")

# Train/Val split
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples: {train_size}")
print(f"Validation samples: {val_size}")

# ================= MODEL =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nUsing device: {device}")

model = DualModalFusionModel(
    model_name="swin_tiny_patch4_window7_224",
    pretrained=True
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

start_epoch = 1
best_val_loss = float("inf")

# ================= RESUME =================
resume_path = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")

if RESUME and os.path.exists(resume_path):
    print(f"\nResuming from {resume_path}")
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))
    print(f"Resuming from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")
else:
    # Auto-resume from latest epoch checkpoint
    epoch_files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("epoch_") and f.endswith(".pth")]
    if epoch_files:
        latest_epoch = max([int(f.split("_")[1].split(".")[0]) for f in epoch_files])
        latest_path = os.path.join(CHECKPOINT_DIR, f"epoch_{latest_epoch:02d}.pth")
        print(f"\nAuto-resuming from latest epoch: {latest_path}")
        model.load_state_dict(torch.load(latest_path, map_location=device))
        start_epoch = latest_epoch + 1
        if os.path.exists(resume_path):
            checkpoint = torch.load(resume_path, map_location=device)
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"Continuing from epoch {start_epoch}")
    else:
        print("\nStarting training from scratch")

# ================= TRAINING =================
print(f"\nTraining for {EPOCHS} epochs...\n")

for epoch in range(start_epoch, EPOCHS + 1):

    # ========== TRAIN ==========
    model.train()
    train_loss = 0.0
    train_batches = 0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [TRAIN]")
    for batch in train_bar:
        clinical = batch['clinical'].to(device)
        dermoscopy = batch['dermoscopy'].to(device)
        severity = batch['severity'].to(device).unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        preds = model(clinical, dermoscopy)
        loss = criterion(preds, severity)

        # Backward pass
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1

        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = train_loss / train_batches

    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [VAL]")
        for batch in val_bar:
            clinical = batch['clinical'].to(device)
            dermoscopy = batch['dermoscopy'].to(device)
            severity = batch['severity'].to(device).unsqueeze(1)

            # Forward pass
            preds = model(clinical, dermoscopy)
            loss = criterion(preds, severity)

            val_loss += loss.item()
            val_batches += 1

            val_bar.set_postfix(loss=loss.item())

    avg_val_loss = val_loss / val_batches

    # ========== SAVE CHECKPOINTS ==========
    # Save last checkpoint (for resuming)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": min(best_val_loss, avg_val_loss),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss
    }, resume_path)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        flag = "✅ NEW BEST"
    else:
        flag = "—"

    # Save epoch checkpoint
    epoch_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch:02d}.pth")
    torch.save(model.state_dict(), epoch_path)

    # ========== LOGGING ==========
    print(
        f"\n{'='*60}\n"
        f"Epoch {epoch}/{EPOCHS} Summary:\n"
        f"  Train Loss: {avg_train_loss:.6f}\n"
        f"  Val Loss  : {avg_val_loss:.6f} {flag}\n"
        f"  Best Val  : {best_val_loss:.6f}\n"
        f"{'='*60}\n"
    )

print("\n🎉 Training complete!")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Best model saved to: {os.path.join(CHECKPOINT_DIR, 'best_model.pth')}")
