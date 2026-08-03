"""
app.py - FastAPI Backend for Live Drowsiness Detection.

Serves a premium web UI and runs real-time CNN-LSTM inference
on webcam frames streamed via WebSocket.
"""

import os
import sys
import json
import base64
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# Add ml_pipeline to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ML_PIPELINE_DIR = os.path.join(PROJECT_ROOT, "ml_pipeline")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if ML_PIPELINE_DIR not in sys.path:
    sys.path.insert(0, ML_PIPELINE_DIR)

from ml_pipeline import config
from ml_pipeline.models import get_model
from ml_pipeline.landmark_features import LandmarkExtractor
from ml_pipeline.dataset import get_subject_prefix

import pandas as pd


class LiveInferenceSession:
    """
    Manages the 30-frame rolling buffer and orchestrates
    the CNN -> Geometric -> LSTM pipeline for live inference.
    """
    def __init__(self, seq_len: int = 30):
        self.seq_len = seq_len
        self.buffer = []  # Rolling buffer of 1284-d feature vectors
        self.device = config.DEVICE
        
        print("Loading models...")
        
        # 1. Load CNN Feature Extractor (FUSION_GRL_V4)
        num_domains = 22  # From training
        self.cnn_model = get_model("fusion_grl_v4", pretrained=False, num_domains=num_domains)
        
        checkpoint = os.path.join(config.CHECKPOINTS_DIR, "fusion_grl_4_best.pth")
        checkpoint_v4 = os.path.join(config.CHECKPOINTS_DIR, "V4", "fusion_grl_4_best.pth")
        
        if os.path.exists(checkpoint):
            target_ckpt = checkpoint
        elif os.path.exists(checkpoint_v4):
            target_ckpt = checkpoint_v4
        else:
            raise FileNotFoundError(f"CNN checkpoint not found at {checkpoint} or {checkpoint_v4}")
        
        self.cnn_model.load_state_dict(
            torch.load(target_ckpt, map_location=self.device, weights_only=True)
        )
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
        print(f"  CNN loaded from {target_ckpt}")
        
        # 2. Load LSTM Sequence Model
        self.lstm_model = get_model("lstm")
        lstm_ckpt = os.path.join(config.CHECKPOINTS_DIR, "lstm_best.pth")
        if not os.path.exists(lstm_ckpt):
            raise FileNotFoundError(f"LSTM checkpoint not found at {lstm_ckpt}")
        
        self.lstm_model.load_state_dict(
            torch.load(lstm_ckpt, map_location=self.device, weights_only=True)
        )
        self.lstm_model = self.lstm_model.to(self.device)
        self.lstm_model.eval()
        print(f"  LSTM loaded from {lstm_ckpt}")
        
        # 3. Load MediaPipe Landmark Extractor
        self.landmark_extractor = LandmarkExtractor()
        print("  MediaPipe LandmarkExtractor loaded")
        
        # 4. Compute global baseline stats for geometric feature normalization
        self.global_baselines = self._compute_global_baselines()
        print(f"  Global baselines computed: {self.global_baselines}")
        
        # 5. Image transform (same as validation/test — no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ])
        
        print("All models loaded! Ready for live inference.\n")
    
    def _compute_global_baselines(self) -> dict:
        """
        Compute global 95th-percentile baselines from the training set's
        geometric features CSV. This matches how dataset.py normalizes
        for unseen subjects.
        """
        if not os.path.exists(config.FEATURES_FILE):
            return {"ear_max": 0.3, "mar_max": 0.5, "eyebrow_max": 0.2, "head_tilt_max": 0.15}
        
        df = pd.read_csv(config.FEATURES_FILE)
        success_df = df[df["success"] == True]
        
        return {
            "ear_max": float(np.percentile(success_df["ear"], 95)) if len(success_df) > 0 else 0.3,
            "mar_max": float(np.percentile(success_df["mar"], 95)) if len(success_df) > 0 else 0.5,
            "eyebrow_max": float(np.percentile(success_df["eyebrow_dist"], 95)) if len(success_df) > 0 else 0.2,
            "head_tilt_max": float(np.percentile(success_df["head_tilt"], 95)) if len(success_df) > 0 else 0.15,
        }
    
    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Process a single BGR frame through the full pipeline:
        1. Extract geometric features via MediaPipe
        2. Extract CNN embedding via EfficientNet
        3. Concatenate to 1284-d vector
        4. Append to rolling buffer
        5. If buffer full (30 frames), run LSTM prediction
        
        Returns a dict with prediction info.
        """
        # 1. Extract geometric features
        geo = self.landmark_extractor.extract(frame_bgr)
        
        if geo["success"]:
            raw_features = [geo["ear"], geo["mar"], geo["eyebrow_dist"], geo["head_tilt"]]
        else:
            # Fallback to global baselines if no face detected
            raw_features = [
                self.global_baselines["ear_max"],
                self.global_baselines["mar_max"],
                self.global_baselines["eyebrow_max"],
                self.global_baselines["head_tilt_max"],
            ]
        
        # Normalize using global baselines (same as dataset.py for unseen subjects)
        norm_features = [
            raw_features[0] / max(self.global_baselines["ear_max"], 1e-6),
            raw_features[1] / max(self.global_baselines["mar_max"], 1e-6),
            raw_features[2] / max(self.global_baselines["eyebrow_max"], 1e-6),
            raw_features[3] / max(self.global_baselines["head_tilt_max"], 1e-6),
        ]
        norm_features = [min(max(f, 0.0), 2.0) for f in norm_features]
        geo_tensor = torch.tensor([norm_features], dtype=torch.float32).to(self.device)
        
        # 2. Extract CNN embedding
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            cnn_emb = self.cnn_model.get_embedding(img_tensor)  # (1, 1280)
        
        # 3. Concatenate to 1284-d
        fused = torch.cat([cnn_emb, geo_tensor], dim=1).cpu().numpy()[0]
        
        # 4. Append to buffer
        self.buffer.append(fused)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)  # Remove oldest frame
        
        buffer_fill = len(self.buffer) / self.seq_len
        
        # 5. Run LSTM if buffer is full
        if len(self.buffer) == self.seq_len:
            sequence = np.stack(self.buffer)
            seq_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.lstm_model(seq_tensor)
                probs = F.softmax(logits, dim=1)[0]
                pred = probs.argmax().item()
                confidence = probs[pred].item()
            
            return {
                "status": "predicting",
                "prediction": config.CLASS_NAMES[pred],
                "is_drowsy": pred == 0,
                "confidence": round(float(confidence), 4),
                "drowsy_prob": round(float(probs[0]), 4),
                "awake_prob": round(float(probs[1]), 4),
                "buffer_fill": round(buffer_fill, 2),
                "face_detected": geo["success"],
                "geo_features": {
                    "ear": round(raw_features[0], 4),
                    "mar": round(raw_features[1], 4),
                    "eyebrow_dist": round(raw_features[2], 4),
                    "head_tilt": round(raw_features[3], 4),
                },
            }
        else:
            return {
                "status": "buffering",
                "prediction": "Waiting...",
                "is_drowsy": False,
                "confidence": 0.0,
                "drowsy_prob": 0.0,
                "awake_prob": 0.0,
                "buffer_fill": round(buffer_fill, 2),
                "face_detected": geo["success"],
                "geo_features": {
                    "ear": round(raw_features[0], 4),
                    "mar": round(raw_features[1], 4),
                    "eyebrow_dist": round(raw_features[2], 4),
                    "head_tilt": round(raw_features[3], 4),
                },
            }
    
    def reset(self):
        """Clear the rolling buffer."""
        self.buffer = []


# ─── FastAPI App ───
app = FastAPI(title="DrowsiGuard - Live Drowsiness Detection")

# Mount static files
STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global session (initialized on first connection)
session = None


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global session
    
    await websocket.accept()
    print("WebSocket connected!")
    
    # Lazy-load models on first connection
    if session is None:
        session = LiveInferenceSession(seq_len=30)
    
    session.reset()  # Fresh buffer for new session
    
    try:
        while True:
            # Receive base64-encoded JPEG frame from browser
            data = await websocket.receive_text()
            
            # Decode base64 -> bytes -> numpy array
            if "," in data:
                data = data.split(",")[1]  # Remove "data:image/jpeg;base64," prefix
            
            img_bytes = base64.b64decode(data)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame_bgr is None:
                await websocket.send_json({"error": "Could not decode frame"})
                continue
            
            # Run inference
            result = session.process_frame(frame_bgr)
            
            await websocket.send_json(result)
    
    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
