# import torch
# import torch.nn.functional as F
# import cv2
# import numpy as np
# from PIL import Image

# class PigmentationAnalyzer:
#     """Extract meaningful pigmentation features for rule-based severity"""
    
#     def __init__(self):
#         # More specific HSV ranges for actual pigmentation (darker spots)
#         self.pigment_lower = np.array([5, 50, 20])   # Specific brown/dark pigmentation
#         self.pigment_upper = np.array([20, 255, 120])  # Narrower range
        
#         # Skin tone detection range
#         self.skin_lower = np.array([0, 10, 30])
#         self.skin_upper = np.array([40, 200, 255])
    
#     def extract_features(self, image_tensor):
#         """Extract pigmentation features from image tensor"""
#         # Convert tensor to numpy image
#         img = self.tensor_to_image(image_tensor)
        
#         # Check if image contains skin
#         if not self.is_skin_image(img):
#             return {
#                 'pigmented_area_pct': 0.0,
#                 'avg_intensity': 1.0,  # High intensity = no pigmentation
#                 'contrast': 0.0,
#                 'is_skin': False
#             }
        
#         # Feature 1: Pigmented area percentage
#         pigmented_area = self.calculate_pigmented_area(img)
        
#         # Feature 2: Average pigmentation intensity
#         avg_intensity = self.calculate_avg_intensity(img)
        
#         # Feature 3: Color contrast
#         contrast = self.calculate_contrast(img)
        
#         return {
#             'pigmented_area_pct': pigmented_area,
#             'avg_intensity': avg_intensity,
#             'contrast': contrast,
#             'is_skin': True
#         }
    
#     def is_skin_image(self, img):
#         """Balanced skin detection - accepts skin images, rejects obvious non-skin"""
#         hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
#         # Broader HSV ranges for skin detection
#         skin_lower1 = np.array([0, 15, 50])   
#         skin_upper1 = np.array([25, 180, 255])
        
#         skin_lower2 = np.array([160, 15, 50])  
#         skin_upper2 = np.array([180, 180, 255])
        
#         mask1 = cv2.inRange(hsv, skin_lower1, skin_upper1)
#         mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
#         skin_mask = cv2.bitwise_or(mask1, mask2)
        
#         skin_percentage = (np.sum(skin_mask > 0) / (img.shape[0] * img.shape[1])) * 100
        
#         # Basic color analysis
#         rgb_mean = np.mean(img, axis=(0, 1))
        
#         # Only reject obvious non-skin scenarios
#         very_blue = rgb_mean[2] > rgb_mean[0] + 50 and rgb_mean[2] > rgb_mean[1] + 50  # Very blue
#         very_green = rgb_mean[1] > rgb_mean[0] + 50 and rgb_mean[1] > rgb_mean[2] + 50  # Very green
        
#         # Edge density - only reject extremely high edge density
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         edges = cv2.Canny(gray, 50, 150)
#         edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1]) * 100
        
#         print(f"DEBUG: skin_pct={skin_percentage:.1f}%, very_blue={very_blue}, very_green={very_green}, edge_density={edge_density:.1f}%")
        
#         # Lenient validation - only reject obvious non-skin
#         is_skin = (skin_percentage > 10 and  # Low threshold
#                   not very_blue and 
#                   not very_green and
#                   edge_density < 35)  # High edge tolerance
        
#         return is_skin
    
#     def tensor_to_image(self, tensor):
#         """Convert normalized tensor back to image"""
#         # Denormalize
#         mean = np.array([0.485, 0.456, 0.406])
#         std = np.array([0.229, 0.224, 0.225])
        
#         img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
#         img = img * std + mean
#         img = np.clip(img * 255, 0, 255).astype(np.uint8)
#         return img
    
#     def calculate_pigmented_area(self, img):
#         """Calculate percentage of pigmented area with refined detection"""
#         hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
#         # Get skin mask first
#         skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
#         # Refined pigmentation detection
#         # Approach 1: HSV-based specific pigmentation
#         pigment_mask1 = cv2.inRange(hsv, self.pigment_lower, self.pigment_upper)
        
#         # Approach 2: Relative darkness within skin (more selective)
#         skin_region = gray[skin_mask > 0]
#         if len(skin_region) > 0:
#             skin_mean = np.mean(skin_region)
#             skin_std = np.std(skin_region)
#             # Only consider pixels significantly darker than average skin
#             dark_threshold = max(80, skin_mean - 1.5 * skin_std)
#             dark_mask = gray < dark_threshold
#         else:
#             dark_mask = gray < 80
        
#         # Approach 3: Edge-based pigmentation detection
#         edges = cv2.Canny(gray, 30, 100)
#         kernel = np.ones((3,3), np.uint8)
#         edge_dilated = cv2.dilate(edges, kernel, iterations=1)
        
#         # Combine all approaches with AND operation for precision
#         pigment_mask = cv2.bitwise_and(pigment_mask1, dark_mask.astype(np.uint8) * 255)
#         pigment_mask = cv2.bitwise_and(pigment_mask, cv2.bitwise_not(edge_dilated))  # Remove edges
        
#         # Only consider pigmentation within skin areas
#         pigment_in_skin = cv2.bitwise_and(pigment_mask, skin_mask)
        
#         # Remove small noise
#         kernel = np.ones((2,2), np.uint8)
#         pigment_in_skin = cv2.morphologyEx(pigment_in_skin, cv2.MORPH_OPEN, kernel)
        
#         skin_pixels = np.sum(skin_mask > 0)
#         pigmented_pixels = np.sum(pigment_in_skin > 0)
        
#         print(f"DEBUG PIGMENTATION: skin_pixels={skin_pixels}, pigmented_pixels={pigmented_pixels}")
        
#         if skin_pixels == 0:
#             return 0.0
        
#         return (pigmented_pixels / skin_pixels) * 100
    
#     def calculate_avg_intensity(self, img):
#         """Calculate average intensity of pigmented regions with improved detection"""
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
#         skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
#         # Multiple pigmentation detection approaches
#         pigment_mask1 = cv2.inRange(hsv, self.pigment_lower, self.pigment_upper)
#         dark_mask = gray < 120
#         pigment_mask = cv2.bitwise_or(pigment_mask1, dark_mask.astype(np.uint8) * 255)
        
#         pigment_in_skin = cv2.bitwise_and(pigment_mask, skin_mask)
        
#         if np.sum(pigment_in_skin) == 0:
#             # Check overall skin darkness as fallback
#             skin_region = gray[skin_mask > 0]
#             if len(skin_region) > 0:
#                 avg_skin_brightness = np.mean(skin_region)
#                 intensity = 1.0 - (avg_skin_brightness / 255.0)
#                 print(f"DEBUG INTENSITY: No pigmentation found, using skin brightness: {intensity:.3f}")
#                 return max(0.1, intensity)  # Minimum intensity for skin images
#             return 0.1
        
#         pigmented_intensities = gray[pigment_in_skin > 0]
#         intensity = 1.0 - (float(np.mean(pigmented_intensities)) / 255.0)
#         print(f"DEBUG INTENSITY: Pigmentation found, intensity: {intensity:.3f}")
#         return intensity
    
#     def calculate_contrast(self, img):
#         """Calculate color contrast in skin regions"""
#         hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#         skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
#         if np.sum(skin_mask) == 0:
#             return 0.0
        
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#         skin_region = gray[skin_mask > 0]
        
#         return float(np.std(skin_region)) / 255.0

# def rule_based_severity(features):
#     """Enhanced severity assessment with better severe case detection"""
    
#     # Check if it's a skin image
#     if not features.get('is_skin', True):
#         return 0.0, "Not a skin image - Please upload a clear skin/face image"
    
#     area = features['pigmented_area_pct']
#     intensity = features['avg_intensity']
#     contrast = features['contrast']
    
#     print(f"DEBUG: area={area:.1f}%, intensity={intensity:.3f}, contrast={contrast:.3f}")
    
#     # Enhanced severity scoring with better severe detection
#     # Combine area and intensity for more accurate assessment
#     severity_score = (area * 0.006) + (intensity * 0.7) + (contrast * 0.3)
    
#     # Rule-based classification with improved thresholds
#     if severity_score < 0.15 or (area < 1 and intensity < 0.2):
#         return 0.1, "Mild"
#     elif severity_score < 0.35 or (area < 5 and intensity < 0.4):
#         return 0.25, "Mild"
#     elif severity_score < 0.55 or (area < 15 and intensity < 0.6):
#         return 0.45, "Moderate"
#     elif severity_score < 0.75 or (area < 30 and intensity < 0.8):
#         return 0.65, "Moderate"
#     else:
#         # Severe cases: high area OR high intensity OR both moderate
#         if area > 40 or intensity > 0.85 or (area > 20 and intensity > 0.7):
#             return 0.9, "Severe"
#         else:
#             return 0.75, "Severe"

import cv2
import numpy as np


class PigmentationAnalyzer:
    """
    Extract explainable pigmentation features for rule-based severity estimation.
    This module DOES NOT perform medical diagnosis.
    """

    def __init__(self, debug: bool = True):  # Enable debug by default
        self.debug = debug

        # HSV ranges (tuned for brown / dark pigmentation)
        self.pigment_lower = np.array([5, 50, 20])
        self.pigment_upper = np.array([25, 255, 120])

        # Broad skin tone range (HSV)
        self.skin_lower = np.array([0, 15, 50])
        self.skin_upper = np.array([40, 200, 255])

    # ======================================================
    # PUBLIC API
    # ======================================================
    def extract_features(self, image_tensor):
        """
        Main feature extractor.
        Returns pigmentation-related features for severity estimation.
        """
        img = self.tensor_to_image(image_tensor)

        # Validate skin presence
        is_skin = self.is_skin_image(img)

        if not is_skin:
            return {
                "pigmented_area_pct": 0.0,
                "avg_intensity": 1.0,
                "contrast": 0.0,
                "is_skin": False,
            }

        pigmented_area = self.calculate_pigmented_area(img)
        avg_intensity = self.calculate_avg_intensity(img)
        contrast = self.calculate_contrast(img)

        return {
            "pigmented_area_pct": pigmented_area,
            "avg_intensity": avg_intensity,
            "contrast": contrast,
            "is_skin": True,
        }

    # ======================================================
    # SKIN VALIDATION
    # ======================================================
    # def is_skin_image(self, img):
    #     """
    #     Balanced skin detection with strong non-skin rejection.
    #     """
    #     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    #     skin_lower1 = np.array([0, 15, 50])
    #     skin_upper1 = np.array([25, 180, 255])
    #     skin_lower2 = np.array([160, 15, 50])
    #     skin_upper2 = np.array([180, 180, 255])

    #     mask1 = cv2.inRange(hsv, skin_lower1, skin_upper1)
    #     mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
    #     skin_mask = cv2.bitwise_or(mask1, mask2)

    #     skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

    #     # Color sanity checks
    #     rgb_mean = np.mean(img, axis=(0, 1))
    #     too_blue = rgb_mean[2] > rgb_mean[0] + 40 and rgb_mean[2] > rgb_mean[1] + 40
    #     too_green = rgb_mean[1] > rgb_mean[0] + 40 and rgb_mean[1] > rgb_mean[2] + 40
    #     too_dark = np.mean(rgb_mean) < 50
    #     too_bright = np.mean(rgb_mean) > 220

    #     # Edge density check (reject buildings, charts)
    #     gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     edges = cv2.Canny(gray, 50, 150)
    #     edge_density = np.sum(edges > 0) / edges.size

    #     if self.debug:
    #         print(
    #             f"[SKIN DEBUG] skin_ratio={skin_ratio:.2f}, "
    #             f"edge_density={edge_density:.2f}, "
    #             f"too_blue={too_blue}, too_green={too_green}"
    #         )

    #     return (
    #         skin_ratio > 0.12
    #         and not too_blue
    #         and not too_green
    #         and not too_dark
    #         and not too_bright
    #         and edge_density < 0.30
    #     )

    def is_skin_image(self, img):
        """
        Enhanced skin detection with room/object rejection
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # HSV skin detection
        skin_lower1 = np.array([0, 15, 50])
        skin_upper1 = np.array([25, 180, 255])
        skin_lower2 = np.array([160, 15, 50])
        skin_upper2 = np.array([180, 180, 255])

        mask1 = cv2.inRange(hsv, skin_lower1, skin_upper1)
        mask2 = cv2.inRange(hsv, skin_lower2, skin_upper2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

        # Color checks
        rgb_mean = np.mean(img, axis=(0, 1))
        very_blue = rgb_mean[2] > rgb_mean[0] + 50 and rgb_mean[2] > rgb_mean[1] + 50
        very_green = rgb_mean[1] > rgb_mean[0] + 50 and rgb_mean[1] > rgb_mean[2] + 50
        too_dark = np.mean(rgb_mean) < 40
        too_bright = np.mean(rgb_mean) > 240

        # Edge density
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # NEW: Texture analysis to detect rooms/objects
        # Calculate local binary pattern-like texture
        kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        texture_response = cv2.filter2D(gray, -1, kernel)
        high_texture_ratio = np.sum(np.abs(texture_response) > 20) / texture_response.size
        
        # NEW: Color uniformity check
        color_std = np.std(img.reshape(-1, 3), axis=0)
        avg_color_std = np.mean(color_std)
        
        # NEW: Saturation check - skin has moderate saturation
        s_channel = hsv[:,:,1]
        avg_saturation = np.mean(s_channel)
        
        if self.debug:
            print(f"[SKIN] skin_ratio={skin_ratio:.3f}, edge_density={edge_density:.3f}")
            print(f"[SKIN] high_texture_ratio={high_texture_ratio:.3f}, avg_color_std={avg_color_std:.1f}")
            print(f"[SKIN] avg_saturation={avg_saturation:.1f}, brightness={np.mean(rgb_mean):.1f}")

        # Enhanced validation with adjusted thresholds
        is_skin = (
            skin_ratio > 0.15 and          # At least 15% skin-like pixels
            not very_blue and              # Not predominantly blue
            not very_green and             # Not predominantly green
            not too_dark and               # Not too dark
            not too_bright and             # Not too bright (room was 206.4)
            edge_density < 0.35 and        # Not too many edges
            high_texture_ratio < 0.30 and  # Allow more texture for pigmented skin
            15 < avg_color_std < 80 and    # Moderate color variation
            20 < avg_saturation < 120 and  # Reasonable saturation for skin
            np.mean(rgb_mean) < 200        # Reject very bright images (rooms)
        )
        
        return is_skin

    # ======================================================
    # IMAGE UTILS
    # ======================================================
    def tensor_to_image(self, tensor):
        """
        Convert normalized PyTorch tensor to uint8 RGB image.
        """
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        img = tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
        img = (img * std + mean) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    # ======================================================
    # FEATURE 1: PIGMENTED AREA
    # ======================================================
    def calculate_pigmented_area(self, img):
        """
        FIXED: Better pigmentation detection with multiple approaches
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Get skin mask
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        
        # Multiple pigmentation detection approaches
        # 1. HSV-based brown/dark pigmentation (FIXED ranges)
        hsv_pigment = cv2.inRange(hsv, np.array([0, 30, 20]), np.array([30, 255, 150]))
        
        # 2. LAB-based dark spot detection (L channel)
        l_channel = lab[:, :, 0]
        lab_dark = l_channel < 100  # Dark spots in L channel
        
        # 3. Relative darkness within skin regions
        skin_pixels = gray[skin_mask > 0]
        if len(skin_pixels) > 0:
            skin_mean = np.mean(skin_pixels)
            skin_std = np.std(skin_pixels)
            # More sensitive threshold for pigmentation
            dark_threshold = max(60, skin_mean - 1.0 * skin_std)
        else:
            dark_threshold = 80
            
        relative_dark = gray < dark_threshold
        
        # 4. Color-based detection (brown/black pigmentation)
        r, g, b = cv2.split(img)
        # Brown pigmentation: low blue, moderate red/green
        brown_mask = (b < 100) & (r > 50) & (g > 50) & (r + g > b + 50)
        
        # Combine all approaches with OR (union) for better detection
        pigment_combined = np.logical_or(hsv_pigment > 0, lab_dark)
        pigment_combined = np.logical_or(pigment_combined, relative_dark)
        pigment_combined = np.logical_or(pigment_combined, brown_mask)
        
        # Only within skin areas
        pigment_in_skin = np.logical_and(pigment_combined, skin_mask > 0)
        
        # Light cleanup (don't over-clean)
        pigment_mask = pigment_in_skin.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        pigment_mask = cv2.morphologyEx(pigment_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate percentage
        skin_pixels_count = np.sum(skin_mask > 0)
        pigmented_pixels = np.sum(pigment_mask > 0)
        
        if self.debug:
            print(f"[PIGMENTATION] skin_pixels={skin_pixels_count}, pigmented={pigmented_pixels}")
            print(f"[PIGMENTATION] dark_threshold={dark_threshold:.1f}, skin_mean={skin_mean:.1f}")
            
        if skin_pixels_count == 0:
            return 0.0
            
        return (pigmented_pixels / skin_pixels_count) * 100.0

    # ======================================================
    # FEATURE 2: AVERAGE INTENSITY
    # ======================================================
    def calculate_avg_intensity(self, img):
        """
        Normalized darkness of pigmented regions.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)
        pigment_mask = cv2.inRange(hsv, self.pigment_lower, self.pigment_upper)

        pigment_in_skin = cv2.bitwise_and(pigment_mask, skin_mask)

        if np.sum(pigment_in_skin) == 0:
            skin_region = gray[skin_mask > 0]
            if len(skin_region) == 0:
                return 0.1
            return max(0.1, 1.0 - np.mean(skin_region) / 255.0)

        pigment_intensity = gray[pigment_in_skin > 0]
        return 1.0 - np.mean(pigment_intensity) / 255.0

    # ======================================================
    # FEATURE 3: CONTRAST
    # ======================================================
    def calculate_contrast(self, img):
        """
        Contrast within skin regions.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        skin_mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)

        if np.sum(skin_mask) == 0:
            return 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        skin_pixels = gray[skin_mask > 0]

        return float(np.std(skin_pixels)) / 255.0


# ======================================================
# RULE-BASED SEVERITY
# ======================================================
def rule_based_severity(features):
    """
    FIXED: Better severity classification for clear skin
    """

    if not features.get("is_skin", True):
        return 0.0, "Not a skin image"

    area = features["pigmented_area_pct"]
    intensity = features["avg_intensity"]
    contrast = features["contrast"]
    
    print(f"[SEVERITY] area={area:.1f}%, intensity={intensity:.3f}, contrast={contrast:.3f}")

    # Clear skin detection - very low pigmentation
    if area < 3 and intensity < 0.3:
        return 0.1, "Mild"
    
    # Light pigmentation
    if area < 8 or (area < 15 and intensity < 0.4):
        return 0.2, "Mild"
    
    # Moderate pigmentation
    if area < 25 or (area < 35 and intensity < 0.6):
        return 0.5, "Moderate"
    
    # Significant pigmentation
    if area < 50 or intensity < 0.8:
        return 0.7, "Moderate"
    
    # Severe pigmentation
    return 0.9, "Severe"
