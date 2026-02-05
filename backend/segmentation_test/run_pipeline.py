import os
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# ----------------------------
# DEVICE
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ----------------------------
# MODEL
# ----------------------------
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

print("[INFO] Loading model...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device)
model.eval()

# ----------------------------
# TEST FOLDER
# ----------------------------
TEST_FOLDER = r"I:\Final Year Projects\Skin-Pigmentation-Detection\backend\data\testing"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

print("\n[INFO] Running batch prediction...\n")

# ----------------------------
# LOOP
# ----------------------------
for file_name in os.listdir(TEST_FOLDER):

    if not file_name.lower().endswith(IMAGE_EXTS):
        continue

    image_path = os.path.join(TEST_FOLDER, file_name)

    # ----------------------------
    # LOAD IMAGE
    # ----------------------------
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # ----------------------------
    # SKIN DETECTION (GATE 1)
    # ----------------------------
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)

    skin_mask = cv2.inRange(
        hsv,
        np.array([0, 20, 70]),
        np.array([25, 255, 255])
    )

    skin_ratio = np.sum(skin_mask > 0) / skin_mask.size

    if skin_ratio < 0.15:
        print(f"Image: {file_name}")
        print("  ❌ Not a skin image")
        print("-" * 40)
        continue

    # ----------------------------
    # SEGMENTATION
    # ----------------------------
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    mask = torch.argmax(outputs.logits, dim=1).squeeze().cpu().numpy()

    binary_mask = (mask > 0).astype(np.uint8)

    binary_mask = cv2.resize(
        binary_mask,
        (image_np.shape[1], image_np.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    lesion_pixels = np.sum(binary_mask == 1)
    total_pixels = binary_mask.size
    area_percentage = (lesion_pixels / total_pixels) * 100

    # ----------------------------
    # PIGMENTATION VALIDATION (GATE 2)
    # ----------------------------
    if lesion_pixels == 0:
        print(f"Image: {file_name}")
        print("  Severity   : Normal / No Pigmentation Detected")
        print("-" * 40)
        continue

    lesion_intensity = gray[binary_mask == 1].mean()
    background_intensity = gray[binary_mask == 0].mean()

    intensity_diff = background_intensity - lesion_intensity

    # ----------------------------
    # FINAL SEVERITY LOGIC
    # ----------------------------
    if intensity_diff < 10:
        severity = "Normal / No Significant Pigmentation"
    elif area_percentage < 10:
        severity = "Mild"
    elif area_percentage < 30:
        severity = "Moderate"
    else:
        severity = "Severe"

    # ----------------------------
    # OUTPUT
    # ----------------------------
    print(f"Image: {file_name}")
    print(f"  Skin Ratio : {skin_ratio:.2f}")
    print(f"  Area %     : {area_percentage:.2f}")
    print(f"  Intensity Δ: {intensity_diff:.2f}")
    print(f"  Severity   : {severity}")
    print("-" * 40)

# import torch
# import numpy as np
# import cv2
# from PIL import Image
# from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# # ----------------------------
# # DEVICE (GPU SUPPORT)
# # ----------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"[INFO] Using device: {device}")

# # ----------------------------
# # LOAD MODEL
# # ----------------------------
# MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
# # MODEL_NAME = "google/derm-foundation"

# print("[INFO] Loading model...")
# processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
# model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME)
# model.to(device)
# model.eval()

# # ----------------------------
# # LOAD IMAGE
# # ----------------------------
# image = Image.open(r"C:\Users\charan27\Downloads\archive (7)\HAM10000_images_part_2\ISIC_0034181.jpg").convert("RGB")
# inputs = processor(images=image, return_tensors="pt").to(device)

# # ----------------------------
# # INFERENCE
# # ----------------------------
# print("[INFO] Running segmentation...")
# with torch.no_grad():
#     outputs = model(**inputs)

# logits = outputs.logits
# mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

# # ----------------------------
# # POST-PROCESS (BINARY MASK)
# # ----------------------------
# # treat non-zero regions as "affected"
# binary_mask = (mask > 0).astype(np.uint8)

# # ----------------------------
# # SEVERITY FEATURES
# # ----------------------------
# image_np = np.array(image)
# gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

# binary_mask = (mask > 0).astype(np.uint8)
# binary_mask = cv2.resize(
#     binary_mask,
#     (image_np.shape[1], image_np.shape[0]),
#     interpolation=cv2.INTER_NEAREST
# )

# lesion_pixels = np.sum(binary_mask == 1)
# total_pixels = binary_mask.size
# area_percentage = (lesion_pixels / total_pixels) * 100

# mean_intensity = gray[binary_mask == 1].mean() if lesion_pixels > 0 else 0



# # If affected area is unrealistically high for pigmentation, ignore
# if area_percentage > 50:
#     severity = "Normal / No Pigmentation Detected"
# else:
#     if area_percentage < 10:
#         severity = "Mild"
#     elif area_percentage < 30:
#         severity = "Moderate"
#     else:
#         severity = "Severe"


# # ----------------------------
# # OUTPUT
# # ----------------------------
# print("\n====== RESULTS ======")
# print(f"Affected Area (%) : {area_percentage:.2f}")
# print(f"Mean Intensity    : {mean_intensity:.2f}")
# print(f"Severity          : {severity}")
# print("=====================\n")
