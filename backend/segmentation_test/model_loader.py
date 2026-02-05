import torch
from huggingface_hub import hf_hub_download

def load_model():
    model_path = hf_hub_download(
        repo_id="DevBhuyan/Skin-Lesion-Segmentation",
        filename="model.pth"
    )

    model = torch.load(model_path, map_location="cpu")
    model.eval()

    return model
