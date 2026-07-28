"""
backend/services/inference_service.py — Inference engine service for backend API.
"""

import io
import os
import sys
import base64
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

# Ensure project root is in sys.path
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SERVICE_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ml_pipeline import config
from ml_pipeline.models import get_model
from ml_pipeline.landmark_features import LandmarkExtractor
from ml_pipeline.inference import DrowsinessPredictor


class InferenceService:
    """
    Singleton service managing loaded drowsiness detection model(s).
    Supports single image files, OpenCV numpy frames, and base64 strings.
    """

    def __init__(self, model_name: str = "fusion"):
        self.model_name = model_name
        self.predictor = None
        self.load_error = None
        self.checkpoint_found = False
        
        self.checkpoint_path = os.path.join(
            config.CHECKPOINTS_DIR, f"{model_name}_best.pth"
        )
        
        self._initialize_predictor()

    def _initialize_predictor(self):
        try:
            if os.path.exists(self.checkpoint_path):
                self.predictor = DrowsinessPredictor(
                    model_name=self.model_name,
                    checkpoint_path=self.checkpoint_path,
                    device=config.DEVICE
                )
                self.checkpoint_found = True
                print(f"[Backend] Loaded model '{self.model_name}' successfully from {self.checkpoint_path}")
            else:
                self.checkpoint_found = False
                self.load_error = f"Checkpoint file not found at {self.checkpoint_path}"
                print(f"[Backend Warning] {self.load_error}. Will initialize untrained model for testing.")
                # Fallback to model instance without weights for dev testing
                self.predictor = DrowsinessPredictor.__new__(DrowsinessPredictor)
                self.predictor.device = config.DEVICE
                self.predictor.model_name = self.model_name
                self.predictor.model = get_model(self.model_name, pretrained=False).to(config.DEVICE)
                self.predictor.model.eval()
                self.predictor.transform = transforms.Compose([
                    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
                ])
                self.predictor.landmark_extractor = LandmarkExtractor()
                self.predictor.geo_stats = {
                    "ear": [0.25, 0.05], "mar": [0.3, 0.1],
                    "eyebrow_dist": [0.15, 0.03], "head_tilt": [0.0, 15.0]
                }
        except Exception as e:
            self.load_error = str(e)
            print(f"[Backend Error] Failed to initialize predictor: {e}")

    def predict_pil_image(self, image: Image.Image) -> dict:
        """Process PIL Image and return structured prediction dictionary."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save temp image for predict_image or convert directly
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        return self._predict_bgr_frame(img_bgr, image)

    def predict_base64(self, base64_str: str) -> dict:
        """Decode base64 image data and run prediction."""
        if "," in base64_str:
            base64_str = base64_str.split(",", 1)[1]
            
        image_bytes = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_bytes))
        return self.predict_pil_image(image)

    def _predict_bgr_frame(self, img_bgr: np.ndarray, image_pil: Image.Image) -> dict:
        """Core inference logic on BGR frame and PIL image."""
        try:
            geo_features = self.predictor.landmark_extractor.extract(img_bgr)
            
            # Prepare image tensor
            img_tensor = self.predictor.transform(image_pil).unsqueeze(0).to(self.predictor.device)
            
            # Prepare geometric tensor
            if geo_features["success"]:
                norm_geo = []
                for k in ["ear", "mar", "eyebrow_dist", "head_tilt"]:
                    val = geo_features[k]
                    mean, std = self.predictor.geo_stats.get(k, [0.0, 1.0])
                    norm_geo.append((val - mean) / std)
                geo_tensor = torch.tensor([norm_geo], dtype=torch.float32).to(self.predictor.device)
            else:
                geo_tensor = torch.zeros((1, config.NUM_GEOMETRIC_FEATURES), dtype=torch.float32).to(self.predictor.device)
            
            # Run inference
            with torch.no_grad():
                if self.model_name == "fusion":
                    logits = self.predictor.model(img_tensor, geo_tensor)
                elif self.model_name == "baseline":
                    logits = self.predictor.model(img_tensor)
                elif self.model_name == "geometric":
                    logits = self.predictor.model(geo_tensor)
                else:
                    logits = self.predictor.model(img_tensor, geo_tensor)
                    
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
            pred_class = int(np.argmax(probs))
            label = config.CLASS_NAMES[pred_class]
            confidence = float(probs[pred_class])
            drowsy_prob = float(probs[0])
            non_drowsy_prob = float(probs[1])
            
            geo_dict = None
            if geo_features["success"]:
                geo_dict = {
                    "ear": float(geo_features["ear"]),
                    "mar": float(geo_features["mar"]),
                    "eyebrow_dist": float(geo_features["eyebrow_dist"]),
                    "head_tilt": float(geo_features["head_tilt"]),
                }
            
            return {
                "success": True,
                "label": label,
                "confidence": confidence,
                "is_drowsy": (label == "Drowsy"),
                "drowsy_probability": drowsy_prob,
                "non_drowsy_probability": non_drowsy_prob,
                "geometric_features": geo_dict,
                "model_used": self.model_name,
                "error": None,
            }
        except Exception as e:
            return {
                "success": False,
                "label": "Unknown",
                "confidence": 0.0,
                "is_drowsy": False,
                "drowsy_probability": 0.0,
                "non_drowsy_probability": 0.0,
                "geometric_features": None,
                "model_used": self.model_name,
                "error": str(e),
            }


# Singleton instance
inference_service = InferenceService()
