from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from .inference import run_inference
from .model_loader import load_model
from typing import Optional
import io

router = APIRouter()
model = load_model()

@router.post("/predict")
async def predict(
    clinical_image: Optional[UploadFile] = File(None),
    affected_area: Optional[str] = Form(None),
    pigmentation_intensity: Optional[str] = Form(None),
    duration: Optional[str] = Form(None),
    progression: Optional[str] = Form(None),
    itching: Optional[str] = Form(None),
    burning: Optional[str] = Form(None),
    pain: Optional[str] = Form(None),
    sun_exposure: Optional[str] = Form(None),
    sunscreen_use: Optional[str] = Form(None),
    user_concern: Optional[str] = Form(None)
):
    try:
        # Check if we have either image or manual inputs
        has_image = clinical_image is not None
        manual_fields = [affected_area, pigmentation_intensity, duration, progression, 
                        itching, burning, pain, sun_exposure, sunscreen_use, user_concern]
        has_manual_inputs = any(field is not None and field != '' for field in manual_fields)
        
        if not has_image and not has_manual_inputs:
            raise HTTPException(status_code=400, detail="Either image or assessment fields are required")
        
        clinical_bytes = None
        if has_image:
            clinical_bytes = io.BytesIO(await clinical_image.read())
        
        manual_data = {
            'affected_area': affected_area,
            'pigmentation_intensity': pigmentation_intensity,
            'duration': duration,
            'progression': progression,
            'itching': itching,
            'burning': burning,
            'pain': pain,
            'sun_exposure': sun_exposure,
            'sunscreen_use': sunscreen_use,
            'user_concern': user_concern
        } if has_manual_inputs else None
        
        result = run_inference(model, clinical_bytes, manual_data)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")