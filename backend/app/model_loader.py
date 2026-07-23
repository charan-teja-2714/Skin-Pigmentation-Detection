import torch
import os
from severity_model_v2.models.dual_modal_fusion import DualModalFusionModel
from severity_model_v2.models.unet import UNet

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    _BASE, "severity_model_v2", "checkpoints_dual_modal", "best_model.pth"
)
UNET_PATH = os.path.join(
    _BASE, "severity_model_v2", "checkpoints_unet", "best_unet.pth"
)

_model = None
_unet  = None


# ── Dual-modal model (required) ───────────────────────────────
def load_model():
    global _model

    model = DualModalFusionModel(
        model_name="swin_tiny_patch4_window7_224",
        pretrained=False
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"[ERROR] Trained model not found at: {MODEL_PATH}\n"
            "Please run train_dual_modal.py first."
        )

    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"[INFO] Loaded dual-modal model from {MODEL_PATH}")

    model.eval()
    _model = model
    return model


def get_model():
    if _model is None:
        raise RuntimeError(
            "[ERROR] Model not loaded. Ensure load_model() was called at startup."
        )
    return _model


# ── U-Net segmentation model (optional) ──────────────────────
def load_unet():
    """
    Load the U-Net segmentation model if its checkpoint exists.
    Returns the loaded model, or None if the checkpoint is not found
    (falls back to GrabCut in that case).
    """
    global _unet

    if not os.path.exists(UNET_PATH):
        print(f"[INFO] U-Net checkpoint not found at {UNET_PATH}.")
        print("[INFO] Segmentation will use GrabCut (CV-based) instead.")
        print("[INFO] Run 'python -m severity_model_v2.train_unet' to train the U-Net.")
        _unet = None
        return None

    model = UNet(in_channels=3, out_channels=1)
    state_dict = torch.load(UNET_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    _unet = model
    print(f"[INFO] Loaded U-Net segmentation model from {UNET_PATH}")
    return model


def get_unet():
    """Returns the U-Net model, or None if not loaded (GrabCut will be used)."""
    return _unet


# ── Shared utility ────────────────────────────────────────────
def get_severity_label(score):
    if score <= 0.25:
        return "Mild"
    elif score <= 0.6:
        return "Moderate"
    else:
        return "Severe"
