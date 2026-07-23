# import torch
# import cv2
# import numpy as np
# import base64
# from io import BytesIO
# from PIL import Image
# from .utils import preprocess_image
# from .model_loader import get_severity_label
# from .pigmentation_analyzer import PigmentationAnalyzer, rule_based_severity

# def generate_masks(clinical_tensor):
#     """Generate robust skin and pigmentation masks using advanced techniques"""
#     analyzer = PigmentationAnalyzer()
#     img = analyzer.tensor_to_image(clinical_tensor)
    
#     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
#     # === ROBUST SKIN DETECTION ===
#     # Multi-space skin detection
#     # HSV-based detection
#     hsv_skin1 = cv2.inRange(hsv, np.array([0, 20, 60]), np.array([20, 150, 255]))
#     hsv_skin2 = cv2.inRange(hsv, np.array([160, 20, 60]), np.array([180, 150, 255]))
#     hsv_skin = cv2.bitwise_or(hsv_skin1, hsv_skin2)
    
#     # RGB-based detection (Kovac rule)
#     r, g, b = cv2.split(img)
#     rgb_skin = ((r > 95) & (g > 40) & (b > 20) & 
#                 ((r.max() - r.min()) > 15) & 
#                 (r > g) & (r > b)).astype(np.uint8) * 255
    
#     # YCrCb-based detection
#     ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
#     ycrcb_skin = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    
#     # Combine all skin detection methods
#     skin_mask = cv2.bitwise_or(hsv_skin, cv2.bitwise_or(rgb_skin, ycrcb_skin))
    
#     # Morphological operations to clean skin mask
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
#     skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
#     # Fill holes in skin mask
#     contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         if cv2.contourArea(contour) > 1000:  # Only large skin regions
#             cv2.fillPoly(skin_mask, [contour], 255)
    
#     # === ROBUST PIGMENTATION DETECTION ===
#     # Multi-approach pigmentation detection
    
#     # Approach 1: Statistical analysis within skin regions
#     skin_region = gray[skin_mask > 0]
#     if len(skin_region) > 0:
#         skin_mean = np.mean(skin_region)
#         skin_std = np.std(skin_region)
#         # Adaptive threshold based on skin statistics
#         pigment_threshold = max(60, skin_mean - 2.0 * skin_std)
#     else:
#         pigment_threshold = 80
    
#     # Approach 2: LAB color space (better for pigmentation)
#     l_channel = lab[:, :, 0]
#     a_channel = lab[:, :, 1]
#     b_channel = lab[:, :, 2]
    
#     # Dark spots in L channel
#     lab_pigment = (l_channel < pigment_threshold).astype(np.uint8) * 255
    
#     # Brown/dark pigmentation in A-B channels
#     ab_pigment = cv2.inRange(lab, np.array([0, 120, 120]), np.array([120, 140, 140]))
    
#     # Approach 3: HSV-based specific pigmentation
#     hsv_pigment = cv2.inRange(hsv, np.array([5, 50, 20]), np.array([25, 255, 100]))
    
#     # Approach 4: Texture-based detection using Local Binary Patterns
#     # Simplified texture analysis
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     texture_diff = cv2.absdiff(gray, blurred)
#     texture_pigment = (texture_diff > 10).astype(np.uint8) * 255
    
#     # Combine pigmentation detection methods
#     pigment_combined = cv2.bitwise_or(lab_pigment, cv2.bitwise_or(ab_pigment, hsv_pigment))
#     pigment_combined = cv2.bitwise_and(pigment_combined, texture_pigment)
    
#     # Only consider pigmentation within skin areas
#     pigment_in_skin = cv2.bitwise_and(pigment_combined, skin_mask)
    
#     # Advanced morphological operations for pigmentation
#     # Remove noise and small artifacts
#     kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     pigment_in_skin = cv2.morphologyEx(pigment_in_skin, cv2.MORPH_OPEN, kernel_small)
    
#     # Connect nearby pigmentation areas
#     kernel_connect = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     pigment_in_skin = cv2.morphologyEx(pigment_in_skin, cv2.MORPH_CLOSE, kernel_connect)
    
#     # Filter by area - remove very small detections
#     contours, _ = cv2.findContours(pigment_in_skin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filtered_pigment = np.zeros_like(pigment_in_skin)
#     for contour in contours:
#         if cv2.contourArea(contour) > 50:  # Minimum pigmentation area
#             cv2.fillPoly(filtered_pigment, [contour], 255)
    
#     pigment_in_skin = filtered_pigment
    
#     # Convert masks to base64 for frontend
#     def mask_to_base64(mask, color, alpha=0.4):
#         colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
#         colored_mask[mask > 0] = color
        
#         # Create better overlay with transparency
#         overlay = cv2.addWeighted(img, 1-alpha, colored_mask, alpha, 0)
        
#         _, buffer = cv2.imencode('.png', overlay)
#         img_base64 = base64.b64encode(buffer).decode('utf-8')
#         return f"data:image/png;base64,{img_base64}"
    
#     return {
#         'skin_mask': mask_to_base64(skin_mask, [0, 255, 0], 0.3),  # Green for skin
#         'pigmentation_mask': mask_to_base64(pigment_in_skin, [255, 0, 0], 0.5)  # Red for pigmentation
#     }

# def run_inference(model, clinical_image):
#     with torch.no_grad():
#         clinical_tensor = preprocess_image(clinical_image)
        
#         # Extract meaningful features from clinical image
#         analyzer = PigmentationAnalyzer()
#         features = analyzer.extract_features(clinical_tensor)
        
#         # Generate masks
#         masks = generate_masks(clinical_tensor)
        
#         # Use rule-based severity assessment
#         score_value, severity = rule_based_severity(features)
        
#         return {
#             "score": round(score_value, 3),
#             "severity": severity,
#             "features": {
#                 "pigmented_area": round(features['pigmented_area_pct'], 1),
#                 "avg_intensity": round(features['avg_intensity'], 3),
#                 "contrast": round(features['contrast'], 3)
#             },
#             "masks": masks
#         }


import torch
import cv2
import numpy as np
import base64
from PIL import Image
from torchvision import transforms

from .model_loader import get_model, get_unet, get_severity_label
from .pigmentation_analyzer import PigmentationAnalyzer, rule_based_severity
from .llm_advisor import LLMAdvisor


# ─────────────────────────────────────────────────────────────
# Preprocessing transform — must exactly match training
# ─────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ─────────────────────────────────────────────────────────────
# Helper: RGB numpy array → base64 PNG data-URI
# ─────────────────────────────────────────────────────────────
def _np_to_base64(img_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")


# =========================================================
# Generate Skin & Pigmentation Masks (Classical Vision)
# =========================================================
# def generate_masks(image_tensor):
#     analyzer = PigmentationAnalyzer()
#     img = analyzer.tensor_to_image(image_tensor)

#     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#     # -----------------------------
#     # SKIN DETECTION
#     # -----------------------------
#     skin_lower1 = np.array([0, 15, 50])
#     skin_upper1 = np.array([25, 180, 255])
#     skin_lower2 = np.array([160, 15, 50])
#     skin_upper2 = np.array([180, 180, 255])

#     mask1 = cv2.inRange(hsv, skin_lower1, skin_upper1)
#     mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
#     skin_mask = cv2.bitwise_or(mask1, mask2)

#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
#     skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

#     # Fill large skin regions
#     contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         if cv2.contourArea(cnt) > 1000:
#             cv2.fillPoly(skin_mask, [cnt], 255)

#     # -----------------------------
#     # PIGMENTATION DETECTION
#     # -----------------------------
#     skin_region = gray[skin_mask > 0]
#     if len(skin_region) > 0:
#         skin_mean = np.mean(skin_region)
#         skin_std = np.std(skin_region)
#         pigment_thresh = max(80, skin_mean - 1.5 * skin_std)
#     else:
#         pigment_thresh = 80

#     # LAB-based dark pigmentation
#     lab_pigment = (lab[:, :, 0] < pigment_thresh).astype(np.uint8) * 255

#     # HSV pigmentation range (brown/dark)
#     hsv_pigment = cv2.inRange(hsv, np.array([5, 50, 20]), np.array([25, 255, 120]))

#     # Texture difference
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     texture = cv2.absdiff(gray, blurred)
#     texture_mask = (texture > 10).astype(np.uint8) * 255

#     pigment_mask = cv2.bitwise_and(lab_pigment, hsv_pigment)
#     pigment_mask = cv2.bitwise_and(pigment_mask, texture_mask)
#     pigment_mask = cv2.bitwise_and(pigment_mask, skin_mask)

#     kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     pigment_mask = cv2.morphologyEx(pigment_mask, cv2.MORPH_OPEN, kernel_small)

#     # Remove tiny noise
#     contours, _ = cv2.findContours(pigment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     clean_pigment = np.zeros_like(pigment_mask)
#     for cnt in contours:
#         if cv2.contourArea(cnt) > 50:
#             cv2.fillPoly(clean_pigment, [cnt], 255)

#     return {
#         "skin_mask": mask_to_base64(img, skin_mask, color=(0, 255, 0), alpha=0.3),
#         "pigmentation_mask": mask_to_base64(img, clean_pigment, color=(255, 0, 0), alpha=0.5)
#     }


# ─────────────────────────────────────────────────────────────
# Binary mask generation — GrabCut-based segmentation
# Falls back to LAB dark-spot detection if GrabCut yields nothing.
# When the trained U-Net is available, generate_inference_mask is
# bypassed entirely (see run_image_inference).
# ─────────────────────────────────────────────────────────────
def _grabcut_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Use OpenCV GrabCut to isolate the foreground lesion/pigmented area.

    GrabCut iteratively fits Gaussian Mixture Models to foreground
    (lesion) and background (normal skin / surroundings) and produces
    a clean binary segmentation without any training.

    Returns: uint8 (224, 224) — 255 = lesion, 0 = background
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # ── Initial rect: central 70 % of the image ──
    margin_y, margin_x = int(h * 0.15), int(w * 0.15)
    rect = (margin_x, margin_y,
            w - 2 * margin_x,
            h - 2 * margin_y)

    mask_gc  = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(img_bgr, mask_gc, rect,
                    bgd_model, fgd_model,
                    iterCount=5,
                    mode=cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((h, w), np.uint8)

    # Pixels labelled GC_FGD (1) or GC_PR_FGD (3) = foreground
    fg = np.where((mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
                  255, 0).astype(np.uint8)
    return fg


def _lab_darkspot_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    LAB-based fallback: find pixels whose L (lightness) is significantly
    below the skin median — i.e. the pigmented / darker spots.

    Returns: uint8 (H, W) — 255 = dark pigmented spot
    """
    lab  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    L    = lab[:, :, 0].astype(np.float32)

    # Skin ROI: anything above V=20 in HSV (very loose, avoids true black BG)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    roi = (hsv[:, :, 2] > 20).astype(np.uint8) * 255
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, k11)
    roi = cv2.dilate(roi, k11, iterations=2)

    roi_L = L[roi > 0]
    if len(roi_L) < 200:
        median_L = float(np.median(L))
    else:
        median_L = float(np.median(roi_L))

    # Pigmented spot = noticeably darker than the median skin lightness
    thresh = max(median_L * 0.78, 35.0)
    dark   = (L < thresh).astype(np.uint8) * 255
    dark   = cv2.bitwise_and(dark, roi)

    # Cleanup
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k5)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    cnts, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean   = np.zeros_like(dark)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 30:
            cv2.fillPoly(clean, [cnt], 255)

    # Absolute fallback: bottom-20 % darkest pixels globally
    if np.count_nonzero(clean) < 80:
        q20   = float(np.percentile(gray.astype(np.float32), 20))
        clean = (gray.astype(np.float32) < q20).astype(np.uint8) * 255

    return clean


def _unet_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Run the trained U-Net to produce a binary segmentation mask.
    Returns float32 (H, W) with 1.0 = lesion, 0.0 = background,
    or None if the U-Net is not loaded.
    """
    unet = get_unet()
    if unet is None:
        return None

    tensor = _transform(Image.fromarray(img_rgb)).unsqueeze(0)  # (1,3,224,224)
    with torch.no_grad():
        prob = unet(tensor)                # (1,1,224,224) sigmoid output
    binary = (prob.squeeze().numpy() > 0.5).astype(np.float32)
    return binary


def generate_inference_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Primary mask generator (used when U-Net is NOT loaded).

    1. Try GrabCut — good segmentation on most clinical photos.
    2. If GrabCut finds too little (< 3 % of image) fall back to
       LAB-based dark-spot detection.

    Returns: float32 (224, 224), 1.0 = pigmented, 0.0 = background
    """
    gc_mask = _grabcut_mask(img_rgb)

    coverage = np.count_nonzero(gc_mask) / gc_mask.size
    if coverage < 0.03 or coverage > 0.90:
        # GrabCut grabbed everything or nothing — use LAB fallback
        mask_u8 = _lab_darkspot_mask(img_rgb)
    else:
        mask_u8 = gc_mask

    return (mask_u8 > 0).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# Build comparison images (original / masked / overlay)
# ─────────────────────────────────────────────────────────────
def _build_comparison(img_rgb: np.ndarray, binary_mask: np.ndarray) -> dict:
    mask_u8   = (binary_mask * 255).astype(np.uint8)
    mask_3ch  = np.stack([binary_mask] * 3, axis=-1)               # (224,224,3)
    masked_rgb = (img_rgb.astype(np.float32) * mask_3ch).astype(np.uint8)

    # Red overlay on original to show detected pigmented region
    red_layer = np.zeros_like(img_rgb)
    red_layer[mask_u8 > 0] = [255, 0, 0]
    overlay = cv2.addWeighted(img_rgb, 0.65, red_layer, 0.35, 0)

    return {
        "original": _np_to_base64(img_rgb),    # what the clinical encoder sees
        "masked":   _np_to_base64(masked_rgb),  # what the dermoscopy encoder sees
        "overlay":  _np_to_base64(overlay),     # highlighted pigmented region
    }


# ─────────────────────────────────────────────────────────────
# Build legacy mask dict (frontend backward-compatibility)
# ─────────────────────────────────────────────────────────────
def _build_legacy_masks(img_rgb: np.ndarray, binary_mask: np.ndarray) -> dict:
    mask_u8 = (binary_mask * 255).astype(np.uint8)

    def _overlay(mask, color, alpha):
        layer = np.zeros_like(img_rgb)
        layer[mask > 0] = color
        blended = cv2.addWeighted(img_rgb, 1 - alpha, layer, alpha, 0)
        return _np_to_base64(blended)

    return {
        "skin_mask":         _overlay(mask_u8, [0, 255, 0], 0.3),
        "pigmentation_mask": _overlay(mask_u8, [255, 0, 0], 0.5),
    }


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────
def run_inference(clinical_image=None, manual_data=None):
    if clinical_image is None and manual_data:
        return run_manual_inference(manual_data)
    if clinical_image:
        return run_image_inference(clinical_image, manual_data)
    raise ValueError("Either clinical_image or manual_data must be provided")


# ─────────────────────────────────────────────────────────────
# Compute features directly from the segmentation mask.
# This ensures the score and the visualization are based on the
# SAME detected region (GrabCut / U-Net).
# ─────────────────────────────────────────────────────────────
def _features_from_mask(img_rgb: np.ndarray,
                        binary_mask: np.ndarray) -> dict:
    """
    All features measured inside the detected pigmented region.

    Returns dict with:
      pigmented_area_pct  — % of total image area that is pigmented
      avg_intensity       — darkness of the region (0 = white, 1 = black)
      contrast            — std-dev of pixel brightness inside region
      asymmetry           — 0 (symmetric) to 1 (highly asymmetric)
      border_irregularity — 0 (round) to 1 (jagged)
      color_variance      — spread of hues inside the lesion
    """
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask_u8 = (binary_mask * 255).astype(np.uint8)
    total   = float(binary_mask.size)
    n_px    = int(np.count_nonzero(mask_u8))

    # ── Area ───────────────────────────────────────────────────
    pigmented_area_pct = (n_px / total) * 100.0

    # ── Intensity & contrast (within masked region) ────────────
    if n_px > 0:
        region_px      = gray[mask_u8 > 0].astype(np.float32)
        avg_brightness = float(np.mean(region_px))
        avg_intensity  = max(0.05, 1.0 - avg_brightness / 255.0)
        contrast       = float(np.std(region_px)) / 255.0
    else:
        avg_intensity = 0.05
        contrast      = 0.0

    # ── Asymmetry (ABCDE — A) ──────────────────────────────────
    flipped       = cv2.flip(mask_u8, 1)          # horizontal mirror
    diff          = cv2.bitwise_xor(mask_u8, flipped)
    asymmetry     = float(np.count_nonzero(diff)) / max(n_px * 2, 1)

    # ── Border irregularity (ABCDE — B) ───────────────────────
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest     = max(cnts, key=cv2.contourArea)
        area_cnt    = cv2.contourArea(largest)
        perimeter   = cv2.arcLength(largest, True)
        # Circularity: 1 = perfect circle, 0 = very irregular
        circularity = (4 * np.pi * area_cnt / (perimeter ** 2 + 1e-6))
        border_irregularity = max(0.0, 1.0 - float(circularity))
    else:
        border_irregularity = 0.0

    # ── Color variance (ABCDE — C) ─────────────────────────────
    if n_px > 0:
        hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        hue     = hsv_img[:, :, 0].astype(np.float32)
        hue_in  = hue[mask_u8 > 0]
        color_variance = float(np.std(hue_in)) / 180.0
    else:
        color_variance = 0.0

    return {
        "pigmented_area_pct":  pigmented_area_pct,
        "avg_intensity":       avg_intensity,
        "contrast":            contrast,
        "asymmetry":           round(asymmetry, 3),
        "border_irregularity": round(border_irregularity, 3),
        "color_variance":      round(color_variance, 3),
        "is_skin":             True,
    }


# ─────────────────────────────────────────────────────────────
# Image-based inference
# Scoring   : features derived from GrabCut/U-Net mask
# Dual-modal: runs for the masked-input visualization only
# ─────────────────────────────────────────────────────────────
def run_image_inference(clinical_image, manual_data=None):
    # ── Load & resize ──
    pil_img     = Image.open(clinical_image).convert("RGB")
    pil_resized = pil_img.resize((224, 224), Image.BILINEAR)
    img_rgb     = np.array(pil_resized, dtype=np.uint8)   # (224,224,3)

    # ── Normalized tensor for dual-modal model ──
    clinical_tensor = _transform(pil_resized).unsqueeze(0)   # (1,3,224,224)

    # ── Stage 1: Validate this is a skin image ──
    analyzer = PigmentationAnalyzer(debug=False)
    raw_img  = analyzer.tensor_to_image(clinical_tensor.squeeze(0))
    if not analyzer.is_skin_image(raw_img):
        return {
            "score":    0.0,
            "severity": "Not a skin image",
            "features": {"input_method": "image", "is_skin": False},
            "masks":    {},
            "advisory": "Please upload a clear photo of the affected skin area.",
        }

    # ── Stage 2: Segmentation — generates the pigmentation mask ──
    # Priority 1: U-Net  (trained — most accurate)
    # Priority 2: GrabCut + LAB fallback (immediate, no training)
    binary_mask = _unet_mask(img_rgb)
    if binary_mask is None:
        binary_mask = generate_inference_mask(img_rgb)

    # ── Stage 3: Compute ALL features from the mask region ──
    # The score is now always consistent with the visualised mask.
    features = _features_from_mask(img_rgb, binary_mask)

    # ── No-pigmentation check ─────────────────────────────────
    # GrabCut often captures the ENTIRE face/hand as "foreground".
    # When the mask is very large (> 55 %) AND the region is not
    # noticeably darker than normal skin (avg_intensity < 0.55),
    # there is no specific pigmented lesion in the image.
    # Threshold 0.55 covers light→medium skin tones (avg_intensity ~0.25–0.52).
    # Real dark lesions typically read avg_intensity > 0.55.
    mask_coverage = float(np.mean(binary_mask))
    no_dark_lesion = (
        mask_coverage > 0.55 and features["avg_intensity"] < 0.55
    ) or mask_coverage < 0.02

    if no_dark_lesion and not manual_data:
        return {
            "score":    0.0,
            "severity": "No Pigmentation Detected",
            "features": {
                "pigmented_area_pct": round(features["pigmented_area_pct"], 2),
                "avg_intensity":      round(features["avg_intensity"], 3),
                "is_skin":            True,
                "input_method":       "image",
            },
            "masks":      _build_legacy_masks(img_rgb, binary_mask),
            "comparison": _build_comparison(img_rgb, binary_mask),
            "advisory": (
                "No significant pigmentation abnormality was detected. "
                "If you have a specific area of concern, photograph it "
                "closer up so it fills most of the frame."
            ),
        }

    score, severity = rule_based_severity(features)

    # ── Stage 4: Run dual-modal model (masked input for visualization) ──
    mask_t            = torch.from_numpy(binary_mask).unsqueeze(0).unsqueeze(0)
    dermoscopy_tensor = clinical_tensor * mask_t
    model = get_model()
    with torch.no_grad():
        model(clinical_tensor, dermoscopy_tensor)   # output not used for scoring

    # ── Optional manual blend ──
    if manual_data:
        manual_score = _manual_score(manual_data)
        score = max(0.0, min(1.0, 0.7 * score + 0.3 * manual_score))
        severity = get_severity_label(score)

    area_pct      = features["pigmented_area_pct"]
    comparison    = _build_comparison(img_rgb, binary_mask)
    legacy_masks  = _build_legacy_masks(img_rgb, binary_mask)
    advisory_text = LLMAdvisor().get_llm_advice(
        severity_score=score,
        severity_level=severity,
        area_pct=area_pct,
        contrast=features["contrast"],
    )

    result = {
        "score":    round(score, 3),
        "severity": severity,
        "features": {
            "pigmented_area_pct":  round(area_pct, 2),
            "avg_intensity":       round(features["avg_intensity"], 3),
            "contrast":            round(features["contrast"], 3),
            "asymmetry":           features["asymmetry"],
            "border_irregularity": features["border_irregularity"],
            "color_variance":      features["color_variance"],
            "is_skin":             True,
            "input_method":        "image" + (" + manual" if manual_data else ""),
        },
        "masks":      legacy_masks,
        "comparison": comparison,
        "advisory":   advisory_text,
    }

    if manual_data:
        result["manual_inputs"] = {
            k: v for k, v in manual_data.items() if v not in (None, "")
        }

    return result


# ─────────────────────────────────────────────────────────────
# Manual-only inference (rule-based, no model needed)
# ─────────────────────────────────────────────────────────────
def run_manual_inference(manual_data):
    score  = 0.0
    filled = 0

    if manual_data.get("age"):
        try:
            age = int(manual_data["age"])
            score += 0.05 if age < 30 else (0.1 if age < 50 else 0.15)
            filled += 1
        except ValueError:
            pass

    for key, mapping in [
        ("affected_area",         {"small_spots": 0.1, "several_small_spots": 0.2, "large_patches": 0.4, "most_of_area": 0.6}),
        ("pigmentation_intensity",{"light": 0.1, "medium": 0.2, "dark": 0.3}),
        ("duration",              {"less_than_1_month": 0.05, "one_to_six_months": 0.1, "more_than_six_months": 0.15, "several_years": 0.2}),
        ("progression",           {"stable": 0.05, "slowly_increasing": 0.1, "rapidly_increasing": 0.2}),
        ("sun_exposure",          {"low": 0.02, "moderate": 0.05, "high": 0.1}),
        ("sunscreen_use",         {"regularly": 0.0, "occasionally": 0.05, "never": 0.1}),
        ("user_concern",          {"not_concerned": 0.0, "somewhat_concerned": 0.05, "very_concerned": 0.1}),
    ]:
        if manual_data.get(key):
            score += mapping.get(manual_data[key], 0)
            filled += 1

    for symptom in ("itching", "burning", "pain"):
        if manual_data.get(symptom) == "yes":
            score += 0.1
            filled += 1

    if filled < 3:
        score = max(score, 0.2)
    score    = max(0.0, min(1.0, score))
    severity = get_severity_label(score)

    advisory_text = LLMAdvisor().get_llm_advice(
        severity_score=score, severity_level=severity, area_pct=0, contrast=0.5
    )

    return {
        "score":    round(score, 3),
        "severity": severity,
        "features": {
            "input_method":    "manual",
            "fields_provided": filled,
            **{k: v for k, v in manual_data.items() if v not in (None, "")},
        },
        "advisory": advisory_text,
    }


# ─────────────────────────────────────────────────────────────
# Internal helper: manual score for blending
# ─────────────────────────────────────────────────────────────
def _manual_score(manual_data) -> float:
    score = 0.0

    if manual_data.get("age"):
        try:
            age = int(manual_data["age"])
            score += 0.05 if age < 30 else (0.1 if age < 50 else 0.15)
        except ValueError:
            pass

    score += {"small_spots": 0.1, "several_small_spots": 0.2,
              "large_patches": 0.4, "most_of_area": 0.6}.get(
                  manual_data.get("affected_area"), 0)
    score += {"light": 0.1, "medium": 0.2, "dark": 0.3}.get(
                  manual_data.get("pigmentation_intensity"), 0)
    score += {"less_than_1_month": 0.05, "one_to_six_months": 0.1,
              "more_than_six_months": 0.15, "several_years": 0.2}.get(
                  manual_data.get("duration"), 0)
    score += {"stable": 0.05, "slowly_increasing": 0.1,
              "rapidly_increasing": 0.2}.get(manual_data.get("progression"), 0)

    for s in ("itching", "burning", "pain"):
        if manual_data.get(s) == "yes":
            score += 0.1

    return max(0.0, min(1.0, score))
