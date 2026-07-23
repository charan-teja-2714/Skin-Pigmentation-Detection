"""
Test script to verify dual-modal fusion model and dataset setup.
Run this before starting training to ensure everything works correctly.
"""

import os
import sys
import torch
from torchvision import transforms

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.dual_modal_dataset import DualModalDataset
from models.dual_modal_fusion import DualModalFusionModel


def test_dataset():
    """Test dataset loading and preprocessing"""
    print("="*60)
    print("Testing Dual-Modal Dataset")
    print("="*60)

    DATA_ROOT = r"I:\Final Year Projects\Skin-Pigmentation-Detection - Copy\backend\data"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        dataset = DualModalDataset(DATA_ROOT, transform)
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Total samples: {len(dataset)}")

        # Test loading one sample
        print("\n📥 Loading sample batch...")
        sample = dataset[0]

        print(f"\n   Clinical image shape: {sample['clinical'].shape}")
        print(f"   Dermoscopy image shape: {sample['dermoscopy'].shape}")
        print(f"   Severity score: {sample['severity'].item():.4f}")

        # Check if masking was applied correctly
        print("\n🎭 Checking masking:")
        print(f"   Dermoscopy image min: {sample['dermoscopy'].min():.4f}")
        print(f"   Dermoscopy image max: {sample['dermoscopy'].max():.4f}")
        print(f"   Clinical image min: {sample['clinical'].min():.4f}")
        print(f"   Clinical image max: {sample['clinical'].max():.4f}")

        # Check if dermoscopy has black regions (from masking)
        black_pixels = (sample['dermoscopy'] == 0).sum().item()
        total_pixels = sample['dermoscopy'].numel()
        print(f"   Dermoscopy black pixels: {black_pixels}/{total_pixels} ({100*black_pixels/total_pixels:.1f}%)")

        print("\n✅ Dataset test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Dataset test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test model forward pass"""
    print("\n" + "="*60)
    print("Testing Dual-Modal Fusion Model")
    print("="*60)

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🖥️  Using device: {device}")

        # Create model
        print("\n🔨 Building model...")
        model = DualModalFusionModel(
            model_name="swin_tiny_patch4_window7_224",
            pretrained=True
        ).to(device)

        print(f"✅ Model created successfully!")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")

        # Test forward pass
        print("\n🚀 Testing forward pass...")
        batch_size = 4
        clinical_dummy = torch.randn(batch_size, 3, 224, 224).to(device)
        dermoscopy_dummy = torch.randn(batch_size, 3, 224, 224).to(device)

        model.eval()
        with torch.no_grad():
            output = model(clinical_dummy, dermoscopy_dummy)

        print(f"   Input shapes:")
        print(f"     Clinical: {clinical_dummy.shape}")
        print(f"     Dermoscopy: {dermoscopy_dummy.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Output range: [{output.min():.4f}, {output.max():.4f}]")

        print("\n✅ Model test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Model test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test dataset + model integration"""
    print("\n" + "="*60)
    print("Testing Dataset + Model Integration")
    print("="*60)

    DATA_ROOT = r"I:\Final Year Projects\Skin-Pigmentation-Detection - Copy\backend\data"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        # Load dataset
        dataset = DualModalDataset(DATA_ROOT, transform)

        # Create model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = DualModalFusionModel(pretrained=True).to(device)

        # Load batch
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=4, shuffle=True)

        print("\n🔄 Running one training iteration...")
        batch = next(iter(loader))

        clinical = batch['clinical'].to(device)
        dermoscopy = batch['dermoscopy'].to(device)
        severity = batch['severity'].to(device).unsqueeze(1)

        # Forward pass
        model.train()
        preds = model(clinical, dermoscopy)

        # Loss
        criterion = torch.nn.MSELoss()
        loss = criterion(preds, severity)

        print(f"\n   Batch size: {clinical.shape[0]}")
        print(f"   Predictions: {preds.squeeze().tolist()}")
        print(f"   Ground truth: {severity.squeeze().tolist()}")
        print(f"   Loss: {loss.item():.6f}")

        # Backward pass
        print("\n🔙 Testing backward pass...")
        loss.backward()

        print("✅ Integration test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print(" DUAL-MODAL FUSION MODEL TEST SUITE ".center(60))
    print("="*60)

    results = {
        "Dataset": test_dataset(),
        "Model": test_model(),
        "Integration": test_integration()
    }

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<20} {status}")

    all_passed = all(results.values())

    print("\n" + "="*60)
    if all_passed:
        print("🎉 All tests passed! Ready to start training.")
        print("\nRun: python train_dual_modal.py")
    else:
        print("⚠️  Some tests failed. Please fix errors before training.")

    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
