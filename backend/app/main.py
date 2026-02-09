from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
import uvicorn
import os

# ============================================
# FIX: Load model ONCE at startup (global scope)
# ============================================
from app.model_loader import load_model

print("[STARTUP] Loading model...")
model = load_model()
print("[STARTUP] Model loaded successfully")

app = FastAPI(title="Skin Pigmentation Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://skin-pigmentation-detection.vercel.app",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Skin Pigmentation Detection API is running"}

# ============================================
# FIX: Lightweight health endpoint for warm-up
# ============================================
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # FIX: Increase timeout for cold starts
    uvicorn.run(
        "app.main:app", 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=120  # 2 minutes for cold starts
    )