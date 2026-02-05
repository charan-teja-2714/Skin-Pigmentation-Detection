import torch
import random
from torch.utils.data import Dataset

class MultiModalDataset(Dataset):
    def __init__(self, dermoscopy_ds, clinical_ds):
        self.dermo = dermoscopy_ds
        self.clinical = clinical_ds
        self.length = max(len(self.dermo), len(self.clinical))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dermo_img = self.dermo[random.randint(0, len(self.dermo) - 1)]
        clinical_img = self.clinical[random.randint(0, len(self.clinical) - 1)]
        
        # Simulate realistic pigmentation labels (not random)
        label_choices = [0.1, 0.15, 0.2, 0.35, 0.45, 0.55, 0.7, 0.8, 0.9]
        label = torch.tensor([random.choice(label_choices)])
        
        return clinical_img, dermo_img, label