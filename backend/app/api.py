from fastapi import APIRouter, UploadFile, File, HTTPException
from .inference import run_inference
from .model_loader import load_model
import io

router = APIRouter()
model = load_model()

@router.post("/predict")
async def predict(
    clinical_image: UploadFile = File(...)
):
    try:
        clinical_bytes = io.BytesIO(await clinical_image.read())
        
        result = run_inference(model, clinical_bytes)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")