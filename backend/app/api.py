from fastapi import APIRouter, UploadFile, File, HTTPException
from .inference import run_inference
from .model_loader import load_model
import io

router = APIRouter()
model = load_model()

@router.post("/predict")
async def predict(
    clinical_image: UploadFile = File(...),
    dermoscopy_image: UploadFile = File(None),
    multispectral_image: UploadFile = File(None)
):
    try:
        clinical_bytes = io.BytesIO(await clinical_image.read())
        
        dermoscopy_bytes = None
        if dermoscopy_image:
            dermoscopy_bytes = io.BytesIO(await dermoscopy_image.read())
            
        multispectral_bytes = None
        if multispectral_image:
            multispectral_bytes = io.BytesIO(await multispectral_image.read())
        
        result = run_inference(model, clinical_bytes, dermoscopy_bytes, multispectral_bytes)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")