import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def extract_pigmentation_features(img):
    """
    Extract pigmentation-related features from an image.
    Returns area %, intensity, contrast.
    """

    # Resize for consistency
    img = cv2.resize(img, (256, 256))

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Normalize L channel
    L_norm = L / 255.0

    # Pigmented area: darker than mean threshold
    threshold = np.mean(L_norm) - 0.05
    pigmented_mask = L_norm < threshold

    pigmented_area_pct = (np.sum(pigmented_mask) / pigmented_mask.size) * 100

    # Average intensity (darkness-based)
    avg_intensity = 1.0 - np.mean(L_norm)

    # Contrast (intensity variation)
    contrast = np.std(L_norm)

    return pigmented_area_pct, avg_intensity, contrast


def analyze_dataset(dataset_name, image_dir, output_csv):
    records = []

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    for img_name in tqdm(image_files, desc=f"Processing {dataset_name}"):
        img_path = os.path.join(image_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w, _ = img.shape
        area, intensity, contrast = extract_pigmentation_features(img)

        records.append({
            "dataset": dataset_name,
            "image_name": img_name,
            "height": h,
            "width": w,
            "pigmented_area_pct": area,
            "avg_intensity": intensity,
            "contrast": contrast
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    print(f"\n📌 {dataset_name} SUMMARY")
    print("Total images:", len(df))
    print("Avg resolution:", int(df.height.mean()), "x", int(df.width.mean()))
    print("Mean pigmented area %:", round(df.pigmented_area_pct.mean(), 2))
    print("Mean intensity:", round(df.avg_intensity.mean(), 3))
    print("Mean contrast:", round(df.contrast.mean(), 3))

    return df


# -------- RUN ANALYSIS --------

isic_df = analyze_dataset(
    "ISIC_2018_Dermoscopy",
    "data/dermoscopy/images",
    "isic2018_metadata.csv"
)

pad_df = analyze_dataset(
    "PAD_UFES_20_Clinical",
    "data/clinical/images",
    "pad_ufes_20_metadata.csv"
)
