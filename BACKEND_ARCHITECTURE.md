# Backend Architecture Documentation
## Multi-Modal Skin Pigmentation Detection System

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Design](#architecture-design)
3. [Model Components](#model-components)
4. [Dataset Pipeline](#dataset-pipeline)
5. [Training Pipeline](#training-pipeline)
6. [Inference Pipeline](#inference-pipeline)
7. [API Layer](#api-layer)
8. [Configuration Management](#configuration-management)
9. [Technical Specifications](#technical-specifications)
10. [Implementation Details](#implementation-details)

---

## 1. System Overview

### Purpose
A deep learning system that analyzes skin pigmentation severity using multiple image modalities (clinical, dermoscopy, multispectral) through a fusion-based architecture.

### Core Capabilities
- **Multi-modal fusion**: Combines information from 3 different imaging types
- **Regression-based scoring**: Outputs continuous pigmentation score (0-1)
- **Severity classification**: Maps scores to interpretable labels (Mild/Moderate/Severe)
- **Flexible input**: Works with 1, 2, or all 3 modalities

### Technology Stack
- **Framework**: PyTorch
- **Model Architecture**: Swin Transformer + Cross-Attention
- **API**: FastAPI
- **Image Processing**: PIL, OpenCV, torchvision
- **Configuration**: YAML

---

## 2. Architecture Design

### High-Level Flow
```
Input Images → Preprocessing → Encoders → Cross-Attention Fusion → Prediction Head → Score + Severity
```

### Component Hierarchy
```
FusionModel
├── SwinEncoder (Clinical)
├── SwinEncoder (Dermoscopy)
├── SwinEncoder (Multispectral)
├── CrossAttention (Fusion Module)
└── PredictionHead (Regression)
```

### Design Philosophy
1. **Query-Key-Value Paradigm**: Clinical image acts as query, auxiliary modalities as key-value pairs
2. **Modular Encoders**: Each modality has independent feature extraction
3. **Attention-Based Fusion**: Cross-attention learns optimal feature combination
4. **End-to-End Training**: All components trained jointly

---

## 3. Model Components

### 3.1 Swin Encoder (`models/swin_encoder.py`)

**Purpose**: Extract hierarchical visual features from input images

**Architecture**:
- Base model: `swin_tiny_patch4_window7_224` from timm library
- Pretrained on ImageNet
- Classification head removed (`num_classes=0`)
- Global average pooling applied

**Key Concepts**:
- **Shifted Windows**: Computes self-attention within local windows, then shifts for cross-window connections
- **Hierarchical Features**: Multi-scale representation through patch merging
- **Efficiency**: Linear complexity relative to image size (vs quadratic in ViT)

**Input/Output**:
```
Input:  (B, 3, 224, 224) - Batch of RGB images
Output: (B, 768) - Feature embeddings (embed_dim=768 for swin_tiny)
```

**Implementation Details**:
```python
- model_name: "swin_tiny_patch4_window7_224"
- pretrained: True (ImageNet weights)
- embed_dim: 768 (automatically extracted from model.num_features)
- global_pool: "avg" (spatial dimensions collapsed)
```

**Why Swin Transformer?**
- Better than CNNs for capturing long-range dependencies
- More efficient than standard Vision Transformers
- Strong transfer learning from ImageNet
- Proven effectiveness on medical imaging tasks

---

### 3.2 Cross-Attention Module (`models/cross_attention.py`)

**Purpose**: Fuse features from multiple modalities using attention mechanism

**Architecture**:
- Multi-head attention with 8 heads
- Separate linear projections for Q, K, V
- Residual connection + Layer normalization
- Output projection

**Key Concepts**:
- **Query**: Clinical image features (what we want to enhance)
- **Key-Value**: Auxiliary modality features (dermoscopy, multispectral)
- **Attention Weights**: Learned importance of each auxiliary feature
- **Residual Connection**: Preserves original clinical information

**Mathematical Flow**:
```
1. Project query:     Q = W_q × clinical_features
2. Project key:       K = W_k × auxiliary_features
3. Project value:     V = W_v × auxiliary_features
4. Compute attention: A = softmax(Q × K^T / √d_k)
5. Apply attention:   O = A × V
6. Output projection: out = W_o × O
7. Residual + Norm:   final = LayerNorm(out + query)
```

**Input/Output**:
```
query:      (B, 1, 768) - Clinical features as single token
key_value:  (B, N, 768) - N auxiliary modality tokens
output:     (B, 1, 768) - Fused features
```

**Why Cross-Attention?**
- Allows clinical image to "attend" to relevant auxiliary information
- Learns which modality provides useful information for each case
- More flexible than simple concatenation or averaging
- Residual connection ensures clinical features aren't lost

**Critical Implementation Detail**:
The residual connection (`output + query`) is ESSENTIAL. Without it, the model may ignore clinical features entirely.

---

### 3.3 Fusion Model (`models/fusion_model.py`)

**Purpose**: Orchestrate the complete multi-modal pipeline

**Architecture Flow**:
```
1. Encode clinical image → (B, 768)
2. Encode dermoscopy image (if provided) → (B, 768)
3. Encode multispectral image (if provided) → (B, 768)
4. Stack auxiliary features → (B, N, 768) where N ∈ {0,1,2}
5. Prepare clinical as query → (B, 1, 768)
6. Cross-attention fusion → (B, 1, 768)
7. Squeeze to (B, 768)
8. Prediction head → (B, 1) score
```

**Handling Missing Modalities**:
- **Clinical only**: Zero tensor used as key-value (fallback mode)
- **Clinical + 1 auxiliary**: N=1 token
- **Clinical + 2 auxiliary**: N=2 tokens

**Design Rationale**:
- Clinical image is ALWAYS required (query)
- Auxiliary modalities are optional (key-value)
- Model gracefully degrades with fewer modalities
- All encoders share same architecture but have independent weights

**Key Implementation**:
```python
# Fallback when no auxiliary modality provided
if len(key_value_features) == 0:
    key_value_features.append(torch.zeros_like(clinical_features))
```

This ensures the model always has something to attend to, even in clinical-only mode.

---

### 3.4 Prediction Head (`models/prediction_head.py`)

**Purpose**: Map fused features to pigmentation severity score

**Architecture**:
```
Input (768) → Linear(256) → ReLU → Dropout(0.3)
            → Linear(64)  → ReLU → Dropout(0.3)
            → Linear(1)   → Sigmoid → Score [0,1]
```

**Design Choices**:
- **3-layer MLP**: Sufficient capacity without overfitting
- **Decreasing dimensions**: 768 → 256 → 64 → 1 (progressive compression)
- **ReLU activation**: Standard non-linearity
- **Dropout (0.3)**: Regularization to prevent overfitting
- **Sigmoid output**: Constrains score to [0, 1] range

**Output Interpretation**:
```
Score Range    Severity Label
[0.00, 0.25]   Mild
(0.25, 0.60]   Moderate
(0.60, 1.00]   Severe
```

**Why Regression (not Classification)?**
- Pigmentation severity is continuous, not discrete
- Regression captures subtle gradations
- Can derive classification from regression (not vice versa)
- More informative for clinical assessment

---

## 4. Dataset Pipeline

### 4.1 Individual Datasets

#### Clinical Dataset (`datasets/clinical_dataset.py`)
**Purpose**: Load standard clinical photographs

**Implementation**:
- Scans directory for image files (.jpg, .png, .jpeg)
- Loads images as RGB
- Applies transforms (resize, normalize)

**Expected Data Structure**:
```
data/clinical/images/
├── image_001.jpg
├── image_002.jpg
└── ...
```

#### Dermoscopy Dataset (`datasets/dermoscopy_dataset.py`)
**Purpose**: Load dermatoscope images

**Implementation**: Identical to ClinicalDataset
- Same loading logic
- Different source directory

**Expected Data Structure**:
```
data/dermoscopy/images/
├── image_001.jpg
├── image_002.jpg
└── ...
```

#### Multispectral Dataset (`datasets/multispectral_dataset.py`)
**Purpose**: Simulate multispectral imaging from RGB images

**Key Concept**: True multispectral cameras are expensive/rare. This simulates multispectral data using color space transformations.

**Simulation Process**:
```
1. Load RGB image
2. Convert to HSV color space → extract H, S channels
3. Convert to LAB color space → extract L channel
4. Stack [H, S, L] as pseudo-multispectral channels
5. Apply transforms
```

**Why This Works**:
- HSV separates color (H, S) from intensity (V)
- LAB separates luminance (L) from color (A, B)
- These channels capture different spectral properties
- Provides complementary information to RGB

**Limitation**: This is a SIMULATION, not real multispectral data. Real multispectral imaging would use different wavelengths (UV, IR, etc.).

---

### 4.2 Multi-Modal Dataset (`datasets/multimodal_dataset.py`)

**Purpose**: Combine all modalities into unified training samples

**Current Implementation**:
```python
def __getitem__(self, idx):
    dermo_img = self.dermo[random.randint(0, len(self.dermo) - 1)]
    clinical_img = self.clinical[random.randint(0, len(self.clinical) - 1)]
    multi_img = self.multi[random.randint(0, len(self.multi) - 1)]
    label = torch.rand(1)  # PSEUDO LABEL
    return clinical_img, dermo_img, multi_img, label
```

**Critical Limitation**: 
- **Random pairing**: Images from different modalities are randomly matched
- **Pseudo labels**: Labels are random values, not real annotations
- **No correspondence**: Clinical, dermoscopy, and multispectral images don't correspond to same patient/lesion

**Why This Exists**:
This is a PROOF-OF-CONCEPT implementation demonstrating the architecture. In production:
- Images should be matched (same lesion across modalities)
- Labels should be real clinical annotations
- Dataset should have proper train/val/test splits with patient-level separation

**What This Means**:
- The model learns general feature extraction and fusion
- It does NOT learn actual pigmentation assessment
- Training will converge but predictions are meaningless
- This is a FRAMEWORK, not a trained diagnostic system

---

### 4.3 Data Preprocessing

**Transform Pipeline**:
```python
transforms.Compose([
    transforms.Resize((224, 224)),      # Swin Transformer input size
    transforms.ToTensor(),               # Convert to [0,1] tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],     # ImageNet statistics
        std=[0.229, 0.224, 0.225]
    )
])
```

**Why ImageNet Normalization?**
- Swin encoders are pretrained on ImageNet
- Using same normalization ensures feature distributions match
- Critical for transfer learning effectiveness

**Image Size**:
- 224×224 is standard for Swin Transformer
- Smaller than original images but sufficient for feature extraction
- Balances computational cost and information retention

---

## 5. Training Pipeline

### 5.1 Training Script (`training/train.py`)

**Configuration Loading**:
```yaml
dataset:
  dermoscopy_path: "../backend/data/dermoscopy/images"
  clinical_path: "../backend/data/clinical/images"

training:
  batch_size: 8
  epochs: 10
  learning_rate: 0.0001
  num_workers: 4
  save_path: "best_model.pth"

device:
  use_gpu: true
```

**Training Loop Structure**:
```
For each epoch:
  1. Training Phase:
     - Set model to train mode
     - Iterate through batches
     - Forward pass
     - Compute loss (MSE)
     - Backward pass
     - Update weights
     - Track training loss
  
  2. Validation Phase:
     - Set model to eval mode
     - Disable gradients
     - Iterate through validation batches
     - Compute validation loss
     - Track best model
  
  3. Model Checkpointing:
     - Save if validation loss improves
```

**Loss Function**: Mean Squared Error (MSE)
```python
criterion = nn.MSELoss()
loss = criterion(predictions, labels)
```

**Why MSE?**
- Standard for regression tasks
- Penalizes large errors more than small ones
- Differentiable and well-behaved
- Simple and effective

**Optimizer**: AdamW
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0001,
    weight_decay=1e-4
)
```

**Why AdamW?**
- Adaptive learning rates per parameter
- Weight decay (L2 regularization) prevents overfitting
- Proven effective for transformer models
- Better than standard Adam for deep learning

**Data Split**: 80% train, 20% validation
```python
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
```

**Progress Tracking**: Uses tqdm for real-time progress bars
```python
train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
```

---

### 5.2 Evaluation Script (`training/evaluate.py`)

**Purpose**: Comprehensive model assessment on validation set

**Metrics Computed**:
1. **MSE (Mean Squared Error)**: Average squared difference
2. **MAE (Mean Absolute Error)**: Average absolute difference
3. **RMSE (Root Mean Squared Error)**: Square root of MSE

**Mathematical Definitions**:
```
MSE  = (1/N) Σ(y_pred - y_true)²
MAE  = (1/N) Σ|y_pred - y_true|
RMSE = √MSE
```

**Output Artifacts**:
1. **metrics.csv**: Detailed predictions with severity labels
2. **score_distribution.png**: Histogram of predicted scores
3. **severity_distribution.png**: Bar chart of severity categories

**Severity Mapping**:
```python
def score_to_severity(score):
    if score < 0.25:
        return "Mild"
    elif score < 0.6:
        return "Moderate"
    else:
        return "Severe"
```

**Usage**:
```bash
cd backend
python training/evaluate.py
```

**Output Location**: `outputs/` directory

---

### 5.3 Single Image Inference (`training/infer_single.py`)

**Purpose**: Test model on individual images

**Workflow**:
```
1. Load trained model weights
2. Load and preprocess input images
3. Run inference (no gradients)
4. Extract score and severity
5. Display results
```

**Example Usage**:
```python
clinical_image_path = "../backend/data/clinical/images/example.jpg"
dermoscopy_image_path = "../backend/data/dermoscopy/images/example.jpg"

clinical_img = load_image(clinical_image_path).to(DEVICE)
dermo_img = load_image(dermoscopy_image_path).to(DEVICE)

with torch.no_grad():
    score = model(
        clinical_img=clinical_img,
        dermoscopy_img=dermo_img,
        multispectral_img=None
    )
```

**Output Format**:
```
========== RESULT ==========
Pigmentation Score  : 0.4523
Severity Level     : Moderate
============================
```

---

## 6. Inference Pipeline

### 6.1 Model Loading (`app/model_loader.py`)

**Purpose**: Initialize model for API serving

**Implementation**:
```python
def load_model():
    model = FusionModel()
    model.eval()  # Set to evaluation mode
    return model
```

**Key Points**:
- No weights loaded (untrained model for demo)
- `model.eval()` disables dropout and batch normalization updates
- In production, would load trained weights: `model.load_state_dict(torch.load(path))`

**Severity Mapping Function**:
```python
def get_severity_label(score):
    if score <= 0.25:
        return "Mild"
    elif score <= 0.6:
        return "Moderate"
    else:
        return "Severe"
```

---

### 6.2 Image Preprocessing (`app/utils.py`)

**Purpose**: Convert uploaded images to model-ready tensors

**Implementation**:
```python
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_bytes).convert('RGB')
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor
```

**Input**: BytesIO object (from file upload)
**Output**: Tensor of shape (1, 3, 224, 224)

**Critical Steps**:
1. Open image from bytes
2. Convert to RGB (handles grayscale, RGBA, etc.)
3. Apply transforms
4. Add batch dimension (model expects batches)

---

### 6.3 Inference Execution (`app/inference.py`)

**Purpose**: Run model prediction on preprocessed images

**Implementation**:
```python
def run_inference(model, clinical_image, dermoscopy_image=None, multispectral_image=None):
    with torch.no_grad():
        clinical_tensor = preprocess_image(clinical_image)
        
        dermoscopy_tensor = None
        if dermoscopy_image:
            dermoscopy_tensor = preprocess_image(dermoscopy_image)
            
        multispectral_tensor = None
        if multispectral_image:
            multispectral_tensor = preprocess_image(multispectral_image)
        
        score = model(clinical_tensor, dermoscopy_tensor, multispectral_tensor)
        score_value = float(score.squeeze().item())
        severity = get_severity_label(score_value)
        
        return {
            "score": round(score_value, 3),
            "severity": severity
        }
```

**Key Features**:
- `torch.no_grad()`: Disables gradient computation (faster, less memory)
- Handles optional modalities gracefully
- Extracts scalar value from tensor
- Returns JSON-serializable dictionary

---

## 7. API Layer

### 7.1 FastAPI Application (`app/main.py`)

**Purpose**: HTTP server for model serving

**Configuration**:
```python
app = FastAPI(title="Skin Pigmentation Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**CORS Middleware**:
- Allows frontend (port 3000) to call backend (port 8000)
- Required for browser security
- In production, restrict to specific domains

**Root Endpoint**:
```python
@app.get("/")
def read_root():
    return {"message": "Skin Pigmentation Detection API is running"}
```

**Server Launch**:
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### 7.2 API Routes (`app/api.py`)

**Prediction Endpoint**:
```python
@router.post("/predict")
async def predict(
    clinical_image: UploadFile = File(...),      # Required
    dermoscopy_image: UploadFile = File(None),   # Optional
    multispectral_image: UploadFile = File(None) # Optional
)
```

**Request Flow**:
```
1. Receive multipart/form-data with image files
2. Read file bytes into BytesIO objects
3. Pass to inference function
4. Return JSON response
```

**Response Format**:
```json
{
  "score": 0.452,
  "severity": "Moderate"
}
```

**Error Handling**:
```python
try:
    # ... inference logic ...
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
```

**API Testing**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "clinical_image=@path/to/image.jpg"
```

---

## 8. Configuration Management

### 8.1 Config File (`config.yaml`)

**Structure**:
```yaml
dataset:
  dermoscopy_path: "../backend/data/dermoscopy/images"
  clinical_path: "../backend/data/clinical/images"

training:
  batch_size: 8
  epochs: 10
  learning_rate: 0.0001
  num_workers: 4
  save_path: "best_model.pth"

model:
  attention_heads: 8

device:
  use_gpu: true
```

**Purpose**:
- Centralized configuration
- Easy hyperparameter tuning
- Separate code from configuration
- Version control friendly

**Loading**:
```python
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

BATCH_SIZE = config["training"]["batch_size"]
```

---

## 9. Technical Specifications

### 9.1 Model Architecture Summary

| Component | Type | Input Shape | Output Shape | Parameters |
|-----------|------|-------------|--------------|------------|
| SwinEncoder | Transformer | (B,3,224,224) | (B,768) | ~28M |
| CrossAttention | Multi-head Attn | (B,1,768), (B,N,768) | (B,1,768) | ~2.4M |
| PredictionHead | MLP | (B,768) | (B,1) | ~200K |
| **Total** | - | - | - | **~90M** |

### 9.2 Computational Requirements

**Training**:
- GPU: 4-8GB VRAM (NVIDIA GTX 1060 or better)
- RAM: 8GB minimum, 16GB recommended
- Storage: 1GB for model + dataset size
- Time: ~1-2 hours for 10 epochs (depends on dataset size)

**Inference**:
- CPU: Modern multi-core processor
- RAM: 2GB minimum
- Latency: ~500ms per image (CPU), ~50ms (GPU)

### 9.3 Supported Image Formats

**Input**: JPG, JPEG, PNG, BMP
**Resolution**: Any (resized to 224×224)
**Color**: RGB (grayscale converted automatically)

---

## 10. Implementation Details

### 10.1 Key Design Decisions

**1. Why Swin Transformer over ResNet/EfficientNet?**
- Better long-range dependency modeling
- Hierarchical features (like CNNs) + global attention (like ViT)
- State-of-the-art on medical imaging benchmarks
- Efficient: O(n) complexity vs O(n²) for standard ViT

**2. Why Cross-Attention over Concatenation?**
- Learns which modality is informative (adaptive fusion)
- Handles missing modalities naturally
- More parameter-efficient than late fusion
- Attention weights provide interpretability

**3. Why Regression over Classification?**
- Pigmentation severity is continuous
- Preserves fine-grained information
- Can derive classification from regression
- More clinically meaningful

**4. Why MSE Loss?**
- Standard for regression
- Smooth gradients
- Penalizes outliers
- Simple and effective

### 10.2 Critical Implementation Notes

**1. Residual Connection in Cross-Attention**:
```python
return self.norm(output + query)  # CRITICAL: Don't remove "+ query"
```
Without this, clinical features may be ignored entirely.

**2. Feature Dimension Consistency**:
All encoders must output same dimension (768). This is automatically handled by using same Swin model.

**3. Batch Dimension Handling**:
```python
tensor = transform(image).unsqueeze(0)  # Add batch dimension for single image
```
Model expects (B, C, H, W), not (C, H, W).

**4. Gradient Management**:
```python
with torch.no_grad():  # Disable gradients during inference
    predictions = model(inputs)
```
Saves memory and speeds up inference.

### 10.3 Common Pitfalls

**1. Forgetting model.eval()**:
```python
model.eval()  # MUST call before inference
```
Without this, dropout and batch norm behave incorrectly.

**2. Incorrect Normalization**:
Must use ImageNet stats for pretrained Swin models.

**3. Device Mismatch**:
```python
model = model.to(DEVICE)
inputs = inputs.to(DEVICE)  # Both must be on same device
```

**4. Missing Batch Dimension**:
Single images need `.unsqueeze(0)` to add batch dimension.

### 10.4 Extension Points

**To Add New Modality**:
1. Create new dataset class (e.g., `InfraredDataset`)
2. Add encoder in `FusionModel.__init__()`
3. Update `forward()` to encode new modality
4. Add to key-value features stack

**To Change Backbone**:
1. Modify `SwinEncoder` to use different model
2. Update `feature_dim` accordingly
3. Adjust `CrossAttention` if dimension changes

**To Add Classification Head**:
1. Create new head in `models/`
2. Add to `FusionModel`
3. Use CrossEntropyLoss instead of MSE
4. Update severity mapping logic

---

## Summary

This system demonstrates a complete multi-modal deep learning pipeline:

**Strengths**:
- Modular, extensible architecture
- Handles variable number of input modalities
- Attention-based fusion learns optimal combination
- Complete training and inference pipelines
- Production-ready API structure

**Limitations**:
- Pseudo labels (not real clinical data)
- Random image pairing (no correspondence)
- Untrained model in API (demo only)
- Simulated multispectral data

**Production Requirements**:
- Real annotated dataset with matched modalities
- Proper train/val/test splits
- Model training on real data
- Clinical validation
- Regulatory compliance (if medical use)

This is a FRAMEWORK and PROOF-OF-CONCEPT, not a diagnostic tool.
