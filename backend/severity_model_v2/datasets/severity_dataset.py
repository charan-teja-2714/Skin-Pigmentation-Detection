# import os
# import torch
# import numpy as np
# from torch.utils.data import Dataset
# from PIL import Image

# class SeverityDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         """
#         root_dir = data/
#         """
#         self.transform = transform
#         self.samples = []
        
#         # Load images directly from directories
#         for modality in ["clinical", "dermoscopy"]:
#             img_dir = os.path.join(root_dir, modality, "images")
            
#             if not os.path.exists(img_dir):
#                 print(f"Warning: {img_dir} not found")
#                 continue
                
#             # Get all image files
#             image_files = [f for f in os.listdir(img_dir) 
#                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
#             print(f"Found {len(image_files)} images in {modality}")
            
#             for img_file in image_files:
#                 img_path = os.path.join(img_dir, img_file)
#                 # Generate synthetic metadata based on filename hash for consistency
#                 hash_val = hash(img_file) % 1000
#                 self.samples.append({
#                     "image_path": img_path,
#                     "area": (hash_val % 50) + 10,  # 10-60%
#                     "intensity": (hash_val % 100) / 100.0,  # 0-1
#                     "contrast": ((hash_val * 7) % 100) / 100.0  # 0-1
#                 })
        
#         print(f"Total samples loaded: {len(self.samples)}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         s = self.samples[idx]

#         # Load image using PIL for better compatibility with transforms
#         image = Image.open(s["image_path"]).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         metadata = torch.tensor([
#             s["area"] / 100.0,
#             s["intensity"],
#             s["contrast"]
#         ], dtype=torch.float32)

#         # Rule-based severity score (same logic everywhere)
#         severity_score = (
#             s["area"] * 0.006 +
#             s["intensity"] * 0.7 +
#             s["contrast"] * 0.3
#         )
#         severity_score = min(max(severity_score, 0.0), 1.0)

#         return image, metadata, torch.tensor(severity_score, dtype=torch.float32)


import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class SeverityDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir = data/
        """
        self.transform = transform
        self.samples = []
        
        # Load images directly from directories
        for modality in ["dermoscopy"]:
            img_dir = os.path.join(root_dir, modality, "images")
            
            if not os.path.exists(img_dir):
                print(f"Warning: {img_dir} not found")
                continue
                
            # Get all image files
            image_files = [f for f in os.listdir(img_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            print(f"Found {len(image_files)} images in {modality}")
            
            for img_file in image_files:
                img_path = os.path.join(img_dir, img_file)
                # Generate synthetic metadata based on filename hash for consistency
                hash_val = hash(img_file) % 1000
                self.samples.append({
                    "image_path": img_path,
                    "area": (hash_val % 50) + 10,  # 10-60%
                    "intensity": (hash_val % 100) / 100.0,  # 0-1
                    "contrast": ((hash_val * 7) % 100) / 100.0  # 0-1
                })
        
        print(f"Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load image using PIL for better compatibility with transforms
        image = Image.open(s["image_path"]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        metadata = torch.tensor([
            s["area"] / 100.0,
            s["intensity"],
            s["contrast"]
        ], dtype=torch.float32)

        # Rule-based severity score (same logic everywhere)
        severity_score = (
            s["area"] * 0.006 +
            s["intensity"] * 0.7 +
            s["contrast"] * 0.3
        )
        severity_score = min(max(severity_score, 0.0), 1.0)

        return image, metadata, torch.tensor(severity_score, dtype=torch.float32)
