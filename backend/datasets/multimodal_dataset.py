# import torch
# from torch.utils.data import Dataset
# import random

# class MultiModalDataset(Dataset):
#     def __init__(self, dermoscopy_dataset, clinical_dataset):
#         self.dermo_ds = dermoscopy_dataset
#         self.clinical_ds = clinical_dataset
#         self.length = max(len(self.dermo_ds), len(self.clinical_ds))

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         dermo_idx = random.randint(0, len(self.dermo_ds) - 1)
#         clinical_idx = random.randint(0, len(self.clinical_ds) - 1)

#         dermo_img = self.dermo_ds[dermo_idx]
#         clinical_img = self.clinical_ds[clinical_idx]

#         # Pseudo pigmentation severity label
#         label = torch.rand(1)

#         return {
#             "clinical": clinical_img,
#             "dermoscopy": dermo_img,
#             "label": label
#         }


import torch
import random
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    def __init__(self, dermoscopy_ds, clinical_ds, multispectral_ds):
        self.dermo = dermoscopy_ds
        self.clinical = clinical_ds
        self.multi = multispectral_ds
        self.length = max(len(self.dermo), len(self.clinical))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dermo_img = self.dermo[random.randint(0, len(self.dermo) - 1)]
        clinical_img = self.clinical[random.randint(0, len(self.clinical) - 1)]
        multi_img = self.multi[random.randint(0, len(self.multi) - 1)]

        label = torch.rand(1)  # pseudo pigmentation score

        return clinical_img, dermo_img, multi_img, label
