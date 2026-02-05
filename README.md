# Dual-Modal Skin Pigmentation Detection System

A complete end-to-end system for analyzing skin pigmentation using clinical and dermoscopy images with deep learning.

## 🏗️ Architecture

### Backend (Python + FastAPI + PyTorch)
- **Dual-Modal Fusion**: Combines clinical and dermoscopy images
- **Swin Transformers**: Two separate encoders for each image type
- **Cross-Attention**: Fuses features with clinical images as query
- **Regression Output**: Produces pigmentation score (0-1) and severity label

### Frontend (React + Vite)
- Simple file upload interface
- Real-time prediction results
- Clean, responsive design
- Error handling and loading states

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 16+
- npm or yarn

### 1. Setup Backend
```bash
cd backend
pip install -r requirements.txt
python -m app.main
```
Backend runs on: http://localhost:8000

### 2. Setup Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on: http://localhost:3000

### 3. Use the System
1. Open http://localhost:3000 in your browser
2. Upload a clinical image (required)
3. Optionally upload dermoscopy image
4. Click "Analyze" to get pigmentation score and severity

## 📁 Project Structure
```
skin-pigmentation-app/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI application
│   │   ├── api.py           # API routes
│   │   ├── inference.py     # Model inference
│   │   ├── model_loader.py  # Model initialization
│   │   └── utils.py         # Image preprocessing
│   ├── models/
│   │   ├── swin_encoder.py     # Swin Transformer encoder
│   │   ├── cross_attention.py  # Cross-attention module
│   │   ├── fusion_model.py     # Main fusion model
│   │   └── prediction_head.py  # Regression head
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx             # Main React component
│   │   ├── api.js              # Backend API calls
│   │   ├── components/
│   │   │   └── UploadForm.jsx  # File upload component
│   │   └── styles.css          # Application styles
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## 🔬 Model Details

### Input Requirements
- **Clinical Image**: Required, any skin image
- **Dermoscopy Image**: Optional, dermatoscope image
- **Format**: JPG, PNG, or other common image formats
- **Processing**: Auto-resized to 224x224, normalized

### Output
- **Score**: Float between 0.0 and 1.0
- **Severity**: 
  - Mild (0.0-0.25)
  - Moderate (0.26-0.6)
  - Severe (0.61-1.0)

### Technical Specifications
- **Framework**: PyTorch with timm library
- **Architecture**: Swin Transformer + Cross-Attention
- **Compute**: CPU-only compatible
- **Memory**: ~2GB RAM recommended

## 🛠️ Development

### Backend Development
```bash
cd backend
# Install in development mode
pip install -e .
# Run with auto-reload
uvicorn app.main:app --reload --port 8000
```

### Frontend Development
```bash
cd frontend
# Install dependencies
npm install
# Run development server
npm run dev
# Build for production
npm run build
```

## 🔧 Troubleshooting

### Common Issues

**Backend won't start:**
- Check Python version (3.10+ required)
- Verify all dependencies installed: `pip install -r requirements.txt`
- Ensure port 8000 is available

**Frontend won't connect:**
- Verify backend is running on port 8000
- Check CORS settings in main.py
- Ensure frontend runs on port 3000

**Model errors:**
- Verify PyTorch installation
- Check available memory (2GB+ recommended)
- Ensure image files are valid formats

### Performance Tips
- Use smaller images for faster processing
- Close other applications to free memory
- Consider GPU setup for production use

## 📊 Expected Results

The system provides:
- **Quantitative Score**: Numerical assessment of pigmentation severity
- **Qualitative Label**: Human-readable severity classification
- **Dual-Modal Analysis**: Enhanced accuracy through image fusion
- **Real-Time Processing**: Results in seconds

## 🔒 Security Notes
- Images are processed locally only
- No data is stored or transmitted externally
- All processing happens on your machine
- CORS is configured for local development only