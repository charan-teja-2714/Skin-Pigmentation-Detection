import numpy as np
import cv2

def compute_features(image, mask):
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    lesion_pixels = np.sum(mask == 1)
    total_pixels = mask.size

    area_percentage = (lesion_pixels / total_pixels) * 100

    if lesion_pixels > 0:
        mean_intensity = gray[mask == 1].mean()
    else:
        mean_intensity = 0

    return area_percentage, mean_intensity


def compute_severity(area_percentage):
    if area_percentage < 10:
        return "Mild"
    elif area_percentage < 30:
        return "Moderate"
    else:
        return "Severe"
