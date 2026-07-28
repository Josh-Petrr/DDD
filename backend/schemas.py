"""
backend/schemas.py — Pydantic schemas for the Drowsiness Detection API.
"""

from typing import Optional, Dict
from pydantic import BaseModel, Field


class GeometricFeaturesResponse(BaseModel):
    ear: Optional[float] = Field(None, description="Eye Aspect Ratio")
    mar: Optional[float] = Field(None, description="Mouth Aspect Ratio")
    eyebrow_dist: Optional[float] = Field(None, description="Normalized eyebrow distance")
    head_tilt: Optional[float] = Field(None, description="Head tilt angle in degrees")


class DrowsinessPredictionResponse(BaseModel):
    success: bool = Field(..., description="Whether the inference succeeded")
    label: str = Field(..., description="Predicted class ('Drowsy' or 'Non Drowsy')")
    confidence: float = Field(..., description="Confidence score for predicted class")
    is_drowsy: bool = Field(..., description="True if driver is detected as drowsy")
    drowsy_probability: float = Field(..., description="Probability of being Drowsy (0-1)")
    non_drowsy_probability: float = Field(..., description="Probability of being Non Drowsy (0-1)")
    geometric_features: Optional[GeometricFeaturesResponse] = Field(None, description="Extracted landmark features")
    model_used: str = Field(..., description="Name of the model used for prediction")
    error: Optional[str] = Field(None, description="Error message if inference failed")


class FramePredictionRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string (e.g. data:image/jpeg;base64,...)")
    model_name: Optional[str] = Field("fusion", description="Model architecture to use ('fusion', 'baseline', 'geometric')")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    active_model: str
    checkpoint_found: bool
