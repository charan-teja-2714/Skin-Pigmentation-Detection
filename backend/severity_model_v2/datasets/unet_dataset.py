"""
Dataset for U-Net segmentation training.

Pairs ISIC dermoscopy images (data/dermoscopy/images/) with their
binary segmentation masks (data/dermoscopy/masks/).

Mask naming: ISIC_0000000.jpg  ->  ISIC_0000000_segmentation.png
"""

import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


IMAGE_SIZE = 224   # must match U-Net training / inference size

# ImageNet normalisation (same as the dual-modal model)
_img_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_mask_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),          # -> [0, 1] float
])

# ── light augmentation for training ──
_aug_img_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

_aug_mask_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])


class ISICSegmentationDataset(Dataset):
    """
    Loads paired ISIC dermoscopy image + binary mask.

    Args:
        images_dir  : folder containing  ISIC_XXXXXXX.jpg
        masks_dir   : folder containing  ISIC_XXXXXXX_segmentation.png
        augment     : whether to apply random augmentation (use for training)
    """

    def __init__(self, images_dir: str, masks_dir: str, augment: bool = False):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.augment    = augment

        # Collect only images that have a matching mask
        self.pairs = []
        for fname in sorted(os.listdir(images_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem  = os.path.splitext(fname)[0]          # e.g. ISIC_0000000
            mask_fname = stem + "_segmentation.png"
            mask_path  = os.path.join(masks_dir, mask_fname)
            if os.path.exists(mask_path):
                self.pairs.append((
                    os.path.join(images_dir, fname),
                    mask_path,
                ))

        print(f"[ISICSegmentationDataset] Found {len(self.pairs)} image-mask pairs.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.pairs[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load mask — convert to grayscale, threshold to binary
        mask_arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_arr = (mask_arr > 127).astype(np.uint8) * 255
        mask = Image.fromarray(mask_arr)

        if self.augment:
            # Apply identical random seed so img + mask get the same transform
            seed = np.random.randint(0, 2**31)

            import torch, random
            torch.manual_seed(seed);  random.seed(seed)
            img  = _aug_img_transform(img)

            torch.manual_seed(seed);  random.seed(seed)
            mask = _aug_mask_transform(mask)
        else:
            img  = _img_transform(img)
            mask = _mask_transform(mask)

        # Binarise mask tensor: 0 or 1
        mask = (mask > 0.5).float()

        return img, mask
