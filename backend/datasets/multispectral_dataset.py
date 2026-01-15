import os
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MultispectralDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

    def __len__(self):
        return len(self.images)

    def simulate_multispectral(self, image):
        """
        Simulate multispectral image using HSV + LAB channels
        """
        image = np.array(image)

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Select pseudo spectral channels
        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        l = lab[:, :, 0]

        multispectral = np.stack([h, s, l], axis=-1)
        multispectral = Image.fromarray(multispectral.astype(np.uint8))

        return multispectral

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        image = self.simulate_multispectral(image)

        if self.transform:
            image = self.transform(image)

        return image
