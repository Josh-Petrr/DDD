"""
backend/main.py — FastAPI Application for Drowsiness Detection API.
"""

import io
import os
import sys
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to sys.path
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_pipeline import config
from backend.schemas import (
    DrowsinessPredictionResponse,
    FramePredictionRequest,
    HealthResponse
)
from backend.services.inference_service import inference_service

app = FastAPI(
    title="Drowsiness Detection API",
    description="Real-time Drowsiness Detection REST API powered by Feature Fusion (EfficientNet-B0 + MediaPipe Facial Landmarks).",
    version="1.0.0"
)

# Enable CORS for frontend clients (React/Vite/HTML5)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust origin domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="Root Endpoint")
def read_root():
    return {
        "message": "Drowsiness Detection REST API is online.",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    return HealthResponse(
        status="ok" if inference_service.checkpoint_found else "degraded",
        model_loaded=(inference_service.predictor is not None),
        device=str(config.DEVICE),
        active_model=inference_service.model_name,
        checkpoint_found=inference_service.checkpoint_found
    )


@app.post("/api/predict/image", response_model=DrowsinessPredictionResponse, summary="Predict Drowsiness from Image File")
async def predict_image(file: UploadFile = File(...)):
    """
    Upload an image file (JPEG/PNG) to analyze driver drowsiness and extract facial landmarks.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not an image."
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = inference_service.predict_pil_image(image)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image prediction failed: {str(e)}"
        )


@app.post("/api/predict/frame", response_model=DrowsinessPredictionResponse, summary="Predict Drowsiness from Base64 Frame")
def predict_frame(request: FramePredictionRequest):
    """
    Submit a base64 encoded image string (e.g. from webcam streaming) to analyze drowsiness.
    """
    try:
        result = inference_service.predict_base64(request.image_base64)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Frame prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
