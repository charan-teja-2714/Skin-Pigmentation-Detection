# Backend - Skin Pigmentation Detection API

## Overview
FastAPI backend with PyTorch-based multi-modal skin pigmentation detection using Swin Transformers and cross-attention fusion.

## Architecture
- **Encoders**: Three Swin Transformer encoders for clinical, dermoscopy, and multispectral images
- **Fusion**: Cross-attention mechanism with clinical features as query
- **Output**: Regression score (0-1) converted to severity labels (Mild/Moderate/Severe)

## Setup

### 1. Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 2. Run Server
```bash
cd backend
python -m app.main
```

Server runs on: http://localhost:8000

## API Usage

### POST /predict
Upload images for pigmentation analysis.

**Required:**
- `clinical_image`: Clinical image file

**Optional:**
- `dermoscopy_image`: Dermoscopy image file
- `multispectral_image`: Multispectral image file

**Response:**
```json
{
  "score": 0.63,
  "severity": "Moderate"
}
```

### Severity Mapping
- 0.0-0.25: Mild
- 0.26-0.6: Moderate  
- 0.61-1.0: Severe

## Model Details
- Uses Swin-Tiny transformers (CPU compatible)
- Input size: 224x224 RGB images
- No training required - inference only