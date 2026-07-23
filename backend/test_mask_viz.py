"""
Quick visual test for generate_inference_mask.
Run from the backend directory:
    python test_mask_viz.py path/to/your_leg_image.jpg

Saves three images in the same folder as the input:
    _original.png   - resized 224x224 input
    _mask.png       - white = pigmented region, black = rest
    _overlay.png    - red highlight on pigmented spots
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image

# So we can import app modules without starting Flask
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from app.inference import generate_inference_mask


def visualize(image_path: str):
    # ── Load & resize ──────────────────────────────────────
    pil_img = Image.open(image_path).convert("RGB")
    pil_224 = pil_img.resize((224, 224), Image.BILINEAR)
    img_rgb  = np.array(pil_224, dtype=np.uint8)

    # ── Generate mask ──────────────────────────────────────
    binary_mask = generate_inference_mask(img_rgb)          # (224,224) float32

    mask_u8  = (binary_mask * 255).astype(np.uint8)
    coverage = float(np.mean(binary_mask) * 100)
    print(f"[INFO] Mask coverage: {coverage:.1f}%  (pixels marked as pigmented)")

    # ── Build images ──────────────────────────────────────
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Red overlay
    red_layer          = np.zeros_like(img_rgb)
    red_layer[mask_u8 > 0] = [255, 0, 0]
    overlay = cv2.addWeighted(img_rgb, 0.65, red_layer, 0.35, 0)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    # ── Save ──────────────────────────────────────────────
    base  = os.path.splitext(image_path)[0]
    out_orig    = base + "_original.png"
    out_mask    = base + "_mask.png"
    out_overlay = base + "_overlay.png"

    cv2.imwrite(out_orig,    img_bgr)
    cv2.imwrite(out_mask,    mask_u8)
    cv2.imwrite(out_overlay, overlay_bgr)

    print(f"[SAVED] {out_orig}")
    print(f"[SAVED] {out_mask}   (white = pigmented area)")
    print(f"[SAVED] {out_overlay} (red highlight = pigmented area)")
    print()
    print("Expected: the pigmented/darker spot on the leg should be WHITE in the mask image")
    print("          and highlighted RED in the overlay image.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_mask_viz.py <path_to_image>")
        sys.exit(1)
    visualize(sys.argv[1])
