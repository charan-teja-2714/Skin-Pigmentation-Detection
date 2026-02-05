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
from io import BytesIO

from .utils import preprocess_image
from .pigmentation_analyzer import PigmentationAnalyzer, rule_based_severity
from .llm_advisor import LLMAdvisor


# =========================================================
# Utility: Convert mask overlay to base64 (for frontend)
# =========================================================
def mask_to_base64(original_img, mask, color=(255, 0, 0), alpha=0.5):
    """
    Overlay mask on image and return base64 encoded PNG
    """
    overlay = original_img.copy()
    color_mask = np.zeros_like(original_img)
    color_mask[mask > 0] = color

    overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)

    _, buffer = cv2.imencode(".png", overlay)
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


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


def generate_masks(image_tensor):
    """
    Generate VISUALIZATION-FRIENDLY skin and pigmentation masks.
    High recall for pigmentation display, conservative logic is handled elsewhere.
    """
    analyzer = PigmentationAnalyzer()
    img = analyzer.tensor_to_image(image_tensor)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # =====================================================
    # SKIN DETECTION (same as before, reliable)
    # =====================================================
    skin_lower1 = np.array([0, 15, 50])
    skin_upper1 = np.array([25, 180, 255])
    skin_lower2 = np.array([160, 15, 50])
    skin_upper2 = np.array([180, 180, 255])

    mask1 = cv2.inRange(hsv, skin_lower1, skin_upper1)
    mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
    skin_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    # Fill large skin regions
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            cv2.fillPoly(skin_mask, [cnt], 255)

    # =====================================================
    # PIGMENTATION DETECTION (UPDATED – HIGH RECALL)
    # =====================================================

    skin_region = gray[skin_mask > 0]
    skin_mean = np.mean(skin_region) if len(skin_region) > 0 else 120
    skin_std = np.std(skin_region) if len(skin_region) > 0 else 20

    # 🔹 WIDER HSV pigmentation range (critical fix)
    hsv_pigment = cv2.inRange(
        hsv,
        np.array([0, 30, 15]),     # wider hue & saturation
        np.array([35, 255, 160])
    )

    # 🔹 SOFTER darkness cue (do NOT over-restrict)
    dark_mask = (gray < skin_mean - 0.8 * skin_std).astype(np.uint8) * 255

    # 🔹 IMPORTANT: OR instead of AND
    pigment_mask = cv2.bitwise_or(hsv_pigment, dark_mask)

    # Restrict to skin only
    pigment_mask = cv2.bitwise_and(pigment_mask, skin_mask)

    # =====================================================
    # CLEANUP (light, do NOT over-clean)
    # =====================================================
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pigment_mask = cv2.morphologyEx(pigment_mask, cv2.MORPH_OPEN, kernel_small)
    pigment_mask = cv2.morphologyEx(pigment_mask, cv2.MORPH_CLOSE, kernel_small)

    # Remove very tiny noise
    contours, _ = cv2.findContours(pigment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_pigment = np.zeros_like(pigment_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 30:   # LOWER threshold for visibility
            cv2.fillPoly(clean_pigment, [cnt], 255)

    # =====================================================
    # BASE64 OVERLAYS FOR FRONTEND
    # =====================================================
    def mask_to_base64(mask, color, alpha):
        overlay = img.copy()
        color_mask = np.zeros_like(img)
        color_mask[mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1 - alpha, color_mask, alpha, 0)
        _, buffer = cv2.imencode(".png", overlay)
        return f"data:image/png;base64,{base64.b64encode(buffer).decode()}"

    return {
        "skin_mask": mask_to_base64(skin_mask, (0, 255, 0), 0.3),
        "pigmentation_mask": mask_to_base64(clean_pigment, (255, 0, 0), 0.5),
    }


# =========================================================
# MAIN INFERENCE FUNCTION
# =========================================================
def run_inference(model, clinical_image):
    """
    Main inference pipeline
    - preprocess image
    - extract pigmentation features
    - generate masks
    - rule-based severity estimation
    - LLM advisory (post-processing)
    """

    with torch.no_grad():
        # -----------------------------
        # PREPROCESS
        # -----------------------------
        clinical_tensor = preprocess_image(clinical_image)

        analyzer = PigmentationAnalyzer()

        # -----------------------------
        # FEATURE EXTRACTION
        # -----------------------------
        features = analyzer.extract_features(clinical_tensor)

        # -----------------------------
        # MASK GENERATION
        # -----------------------------
        masks = generate_masks(clinical_tensor)

        # -----------------------------
        # SEVERITY ESTIMATION
        # -----------------------------
        score, severity = rule_based_severity(features)

        # -----------------------------
        # LLM ADVISORY (POST-PROCESSING)
        # -----------------------------
        llm_advisor = LLMAdvisor()
        advisory_text = llm_advisor.get_llm_advice(
            severity_score=score,
            severity_level=severity,
            area_pct=features['pigmented_area_pct'],
            contrast=features['contrast']
        )

        # -----------------------------
        # RESPONSE
        # -----------------------------
        return {
            "score": round(score, 3),
            "severity": severity,
            "features": {
                "pigmented_area_pct": round(features["pigmented_area_pct"], 2),
                "avg_intensity": round(features["avg_intensity"], 3),
                "contrast": round(features["contrast"], 3),
                "is_skin": features.get("is_skin", True)
            },
            "masks": masks,
            "advisory": advisory_text
        }
