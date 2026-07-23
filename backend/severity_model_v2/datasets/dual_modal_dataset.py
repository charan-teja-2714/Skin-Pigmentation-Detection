import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import random


class DualModalDataset(Dataset):
    """
    Dataset for dual-modal fusion:
    - Clinical images: used as-is (no masking)
    - Dermoscopy images: masked using binary segmentation masks
    """

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir: Path to data directory containing clinical/ and dermoscopy/
            transform: Transform to apply (resize, normalize, etc.)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Load clinical images
        clinical_dir = os.path.join(root_dir, "clinical", "images")
        clinical_files = []
        if os.path.exists(clinical_dir):
            clinical_files = [
                os.path.join(clinical_dir, f)
                for f in os.listdir(clinical_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]
            print(f"Found {len(clinical_files)} clinical images")

        # Load dermoscopy images and masks
        dermo_dir = os.path.join(root_dir, "dermoscopy", "images")
        mask_dir = os.path.join(root_dir, "dermoscopy", "masks")
        dermo_samples = []

        if os.path.exists(dermo_dir) and os.path.exists(mask_dir):
            for img_file in os.listdir(dermo_dir):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    continue

                # Construct mask filename: ISIC_0000000.jpg -> ISIC_0000000_segmentation.png
                base_name = os.path.splitext(img_file)[0]
                mask_file = f"{base_name}_segmentation.png"

                img_path = os.path.join(dermo_dir, img_file)
                mask_path = os.path.join(mask_dir, mask_file)

                # Only include if mask exists
                if os.path.exists(mask_path):
                    dermo_samples.append({
                        'image': img_path,
                        'mask': mask_path
                    })

            print(f"Found {len(dermo_samples)} dermoscopy images with masks")

        # Create paired samples (random pairing)
        # Use the smaller of the two datasets as the base
        if len(clinical_files) == 0 or len(dermo_samples) == 0:
            raise ValueError(
                f"Need both clinical and dermoscopy images!\n"
                f"Clinical: {len(clinical_files)}, Dermoscopy: {len(dermo_samples)}"
            )

        # Create samples by pairing each dermoscopy with a random clinical
        for dermo_sample in dermo_samples:
            clinical_path = random.choice(clinical_files)

            # Generate synthetic severity score based on filename hash
            hash_val = hash(os.path.basename(dermo_sample['image'])) % 1000

            self.samples.append({
                'clinical_path': clinical_path,
                'dermoscopy_path': dermo_sample['image'],
                'mask_path': dermo_sample['mask'],
                'area': (hash_val % 50) + 10,  # 10-60%
                'intensity': (hash_val % 100) / 100.0,  # 0-1
                'contrast': ((hash_val * 7) % 100) / 100.0  # 0-1
            })

        print(f"Created {len(self.samples)} dual-modal samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load clinical image (no mask)
        clinical_img = Image.open(sample['clinical_path']).convert('RGB')

        # Load dermoscopy image and mask
        dermo_img = Image.open(sample['dermoscopy_path']).convert('RGB')
        mask_img = Image.open(sample['mask_path']).convert('L')  # Grayscale

        # Apply transforms (resize, toTensor, normalize)
        if self.transform:
            clinical_img = self.transform(clinical_img)
            dermo_img = self.transform(dermo_img)

            # Transform mask separately (only resize and toTensor, no normalization)
            from torchvision import transforms
            mask_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
            mask_tensor = mask_transform(mask_img)

        # Apply masking to dermoscopy image (element-wise multiplication)
        # Mask is binary: 1 for lesion area, 0 for background
        # Binarize mask (threshold at 0.5)
        mask_binary = (mask_tensor > 0.5).float()

        # Apply mask to dermoscopy image (3 channels)
        masked_dermo = dermo_img * mask_binary

        # Calculate severity score (rule-based)
        severity_score = (
            sample['area'] * 0.006 +
            sample['intensity'] * 0.7 +
            sample['contrast'] * 0.3
        )
        severity_score = min(max(severity_score, 0.0), 1.0)

        return {
            'clinical': clinical_img,
            'dermoscopy': masked_dermo,  # Already masked
            'severity': torch.tensor(severity_score, dtype=torch.float32)
        }
