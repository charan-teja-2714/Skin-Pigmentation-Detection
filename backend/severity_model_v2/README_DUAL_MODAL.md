# Dual-Modal Fusion Model for Skin Pigmentation Detection

## Overview
This dual-modal fusion model combines clinical and dermoscopy images for improved severity prediction. The key feature is **selective masking** where dermoscopy images are masked during preprocessing while clinical images remain unmasked.

## Architecture

### Model Components
1. **Clinical Swin Encoder**: Processes full, unmasked clinical images
2. **Dermoscopy Swin Encoder**: Processes masked dermoscopy images (lesion regions only)
3. **Cross-Attention Fusion**: Fuses features from both modalities
4. **Prediction Head**: Outputs severity score (0-1)

### Why Two Separate Encoders?
- Clinical and dermoscopy images have different characteristics
- Clinical: Full skin surface, natural lighting, broader context
- Dermoscopy: Close-up, magnified, masked lesion regions
- Separate encoders allow each modality to learn optimal representations

## Dataset

### Data Structure
```
data/
├── clinical/
│   └── images/           # Unmasked clinical images
└── dermoscopy/
    ├── images/           # Dermoscopy images
    └── masks/            # Binary segmentation masks (e.g., ISIC_0000000_segmentation.png)
```

### Preprocessing Pipeline

#### Clinical Images (No Masking)
1. Resize to 224×224
2. Convert to tensor
3. Normalize (ImageNet stats)

#### Dermoscopy Images (With Masking)
1. Resize image to 224×224
2. Resize mask to 224×224
3. Convert both to tensors
4. Binarize mask (threshold > 0.5)
5. **Apply mask**: `masked_image = image × mask` (element-wise multiplication)
6. Normalize (ImageNet stats)

### Dataset Statistics
- Clinical images: 2,298
- Dermoscopy images with masks: 2,594
- Total paired samples: 2,594

## Training

### Quick Start
```bash
cd backend/severity_model_v2
python train_dual_modal.py
```

### Configuration
Edit `train_dual_modal.py` to modify:
- `BATCH_SIZE`: Default 8
- `EPOCHS`: Default 20
- `LR`: Default 1e-4
- `VAL_SPLIT`: Default 0.2 (20% validation)
- `CHECKPOINT_DIR`: Default "checkpoints_dual_modal"

### Resume Training
Set `RESUME = True` in `train_dual_modal.py` to continue from last checkpoint.

### Checkpoints
- `best_model.pth`: Best model based on validation loss
- `last_checkpoint.pth`: Latest epoch (for resuming)
- `epoch_XX.pth`: Checkpoint for each epoch

## Files Created

### Dataset
- `datasets/dual_modal_dataset.py`: Dataset loader with selective masking

### Model
- `models/dual_modal_fusion.py`: Dual-modal fusion model with two Swin encoders

### Training
- `train_dual_modal.py`: Main training script
- `test_dual_modal.py`: Test suite (has emoji encoding issues on Windows, use manual tests)

## Testing

### Quick Manual Test
```python
# Test dataset
python -c "
from torchvision import transforms
from datasets.dual_modal_dataset import DualModalDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = DualModalDataset('data', transform)
print(f'Loaded {len(dataset)} samples')
sample = dataset[0]
print(f'Clinical: {sample[\"clinical\"].shape}')
print(f'Dermoscopy: {sample[\"dermoscopy\"].shape}')
"
```

### Test Model
```python
# Test model forward pass
python -c "
import torch
from models.dual_modal_fusion import DualModalFusionModel

model = DualModalFusionModel(pretrained=True)
clinical = torch.randn(2, 3, 224, 224)
dermoscopy = torch.randn(2, 3, 224, 224)

output = model(clinical, dermoscopy)
print(f'Output shape: {output.shape}')
"
```

## Key Implementation Details

### Masking Strategy
- **Dermoscopy**: Masked using binary segmentation masks
  - Background pixels → 0 (black)
  - Lesion pixels → original values
  - Focuses encoder on relevant lesion regions

- **Clinical**: No masking
  - Full image preserved
  - Captures broader skin context

### Normalization
Both modalities use ImageNet normalization:
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

Applied AFTER masking to ensure consistency.

### Loss Function
- MSE Loss (Mean Squared Error)
- Predicts continuous severity score [0, 1]

### Optimizer
- Adam optimizer
- Learning rate: 1e-4

## Expected Performance
The model should converge with:
- Training loss decreasing steadily
- Validation loss stabilizing after ~10-15 epochs
- Outputs in range [0, 1] due to Sigmoid activation in prediction head

## Troubleshooting

### Issue: No masks found
- Ensure masks are in `data/dermoscopy/masks/`
- Check naming: `ISIC_0000000.jpg` → `ISIC_0000000_segmentation.png`

### Issue: CUDA out of memory
- Reduce `BATCH_SIZE` in `train_dual_modal.py`
- Try batch size 4 or 2

### Issue: Model not improving
- Check if masks are correctly applied (dermoscopy should have black regions)
- Verify both modalities are being used (check forward pass)
- Try lowering learning rate to 5e-5

## Next Steps
After training:
1. Evaluate on test set using `evaluate.py` (create if needed)
2. Visualize attention maps to see what the model learns
3. Compare with single-modality baselines
4. Fine-tune hyperparameters based on validation performance
