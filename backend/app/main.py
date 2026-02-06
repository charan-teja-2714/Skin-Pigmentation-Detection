from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
import uvicorn
import os

app = FastAPI(title="Skin Pigmentation Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Skin Pigmentation Detection API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)