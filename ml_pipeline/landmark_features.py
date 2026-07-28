"""
landmark_features.py — Geometric facial feature extraction using MediaPipe.

Uses the MediaPipe Tasks API (FaceLandmarker) to extract 4 biologically 
meaningful features from face images:
  1. EAR  (Eye Aspect Ratio)     — drops when eyes close
  2. MAR  (Mouth Aspect Ratio)   — increases during yawning  
  3. Eyebrow-to-eye distance     — drops with drooping/fatigue
  4. Head tilt angle             — captures head nodding

Usage:
    from landmark_features import LandmarkExtractor
    extractor = LandmarkExtractor()
    features = extractor.extract(image_bgr)
    # Returns: {"ear": float, "mar": float, "eyebrow_dist": float, 
    #           "head_tilt": float, "success": bool}
"""

import os
import math
import numpy as np
import cv2
import mediapipe as mp

import config


# ──────────────────────────────────────────────
# MediaPipe FaceMesh landmark indices
# (Same 478-landmark topology as legacy FaceMesh)
# ──────────────────────────────────────────────

# Left eye landmarks (6 points for EAR)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye landmarks
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Mouth landmarks (inner lips, 4 pairs for MAR)
UPPER_LIP = [13, 312, 311, 310]
LOWER_LIP = [14, 317, 402, 318]
MOUTH_LEFT = 78
MOUTH_RIGHT = 308

# Eyebrow landmarks
LEFT_EYEBROW_BOTTOM = [276, 283, 282, 295]
RIGHT_EYEBROW_BOTTOM = [46, 53, 52, 65]
LEFT_EYE_TOP = [386, 374]
RIGHT_EYE_TOP = [159, 145]

# Head pose landmarks
NOSE_TIP = 1
CHIN = 152

# Path to the FaceLandmarker model
MODEL_PATH = config.LANDMARK_TASK_FILE


class LandmarkExtractor:
    """
    Reusable extractor using MediaPipe Tasks API.
    Initialise once, call extract() many times.
    """
    
    def __init__(self, model_path: str = MODEL_PATH,
                 num_faces: int = 1,
                 min_detection_confidence: float = 0.5):
        
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    
    def __del__(self):
        if hasattr(self, 'landmarker'):
            self.landmarker.close()
    
    def extract(self, image_bgr: np.ndarray) -> dict:
        """
        Extract geometric features from a BGR image.
        
        Returns dict with keys: ear, mar, eyebrow_dist, head_tilt, success
        """
        default_result = {
            "ear": 0.0, "mar": 0.0, "eyebrow_dist": 0.0,
            "head_tilt": 0.0, "success": False
        }
        
        if image_bgr is None or image_bgr.size == 0:
            return default_result
        
        h, w = image_bgr.shape[:2]
        
        try:
            # Convert BGR → RGB and create MediaPipe Image
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Detect landmarks
            result = self.landmarker.detect(mp_image)
            
            if not result.face_landmarks or len(result.face_landmarks) == 0:
                return default_result
            
            landmarks = result.face_landmarks[0]  # First face
            
            # Compute features
            left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0
            
            mar = _mouth_aspect_ratio(landmarks, w, h)
            eyebrow_dist = _eyebrow_eye_distance(landmarks, w, h)
            head_tilt = _head_tilt_angle(landmarks, w, h)
            
            return {
                "ear": round(ear, 6),
                "mar": round(mar, 6),
                "eyebrow_dist": round(eyebrow_dist, 6),
                "head_tilt": round(head_tilt, 6),
                "success": True
            }
        
        except Exception:
            return default_result


# ──────────────────────────────────────────────
# Feature computation helpers
# ──────────────────────────────────────────────

def _lm_to_np(landmark, w: int, h: int) -> np.ndarray:
    """Convert a MediaPipe NormalizedLandmark to pixel coordinates."""
    return np.array([landmark.x * w, landmark.y * h])


def _eye_aspect_ratio(landmarks, eye_indices, w, h) -> float:
    """
    Compute Eye Aspect Ratio (EAR).
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Where p1-p6 are the 6 eye landmark points:
        p1 = left corner, p4 = right corner (horizontal)
        p2, p3 = upper lid, p5, p6 = lower lid (vertical)
    """
    pts = [_lm_to_np(landmarks[i], w, h) for i in eye_indices]
    
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    
    # Horizontal distance
    h_dist = np.linalg.norm(pts[0] - pts[3])
    
    if h_dist < 1e-6:
        return 0.0
    
    return (v1 + v2) / (2.0 * h_dist)


def _mouth_aspect_ratio(landmarks, w, h) -> float:
    """
    Compute Mouth Aspect Ratio (MAR).
    
    MAR = avg(vertical distances) / horizontal distance
    Increases during yawning.
    """
    vertical_dists = []
    for u, l in zip(UPPER_LIP, LOWER_LIP):
        p_upper = _lm_to_np(landmarks[u], w, h)
        p_lower = _lm_to_np(landmarks[l], w, h)
        vertical_dists.append(np.linalg.norm(p_upper - p_lower))
    
    p_left = _lm_to_np(landmarks[MOUTH_LEFT], w, h)
    p_right = _lm_to_np(landmarks[MOUTH_RIGHT], w, h)
    h_dist = np.linalg.norm(p_left - p_right)
    
    if h_dist < 1e-6:
        return 0.0
    
    return np.mean(vertical_dists) / h_dist


def _eyebrow_eye_distance(landmarks, w, h) -> float:
    """
    Compute normalized eyebrow-to-eye distance.
    
    Average distance from bottom of eyebrow to top of eye,
    normalized by face height. Drops when fatigued.
    """
    nose = _lm_to_np(landmarks[NOSE_TIP], w, h)
    chin = _lm_to_np(landmarks[CHIN], w, h)
    face_h = np.linalg.norm(nose - chin)
    
    if face_h < 1e-6:
        return 0.0
    
    distances = []
    
    for eb, ey in zip(LEFT_EYEBROW_BOTTOM, LEFT_EYE_TOP):
        p_eb = _lm_to_np(landmarks[eb], w, h)
        p_ey = _lm_to_np(landmarks[ey], w, h)
        distances.append(np.linalg.norm(p_eb - p_ey))
    
    for eb, ey in zip(RIGHT_EYEBROW_BOTTOM, RIGHT_EYE_TOP):
        p_eb = _lm_to_np(landmarks[eb], w, h)
        p_ey = _lm_to_np(landmarks[ey], w, h)
        distances.append(np.linalg.norm(p_eb - p_ey))
    
    return np.mean(distances) / face_h


def _head_tilt_angle(landmarks, w, h) -> float:
    """
    Compute head tilt angle (normalized to [0, 1]).
    
    Uses the angle of the nose-to-chin vector relative to vertical.
    A forward head nod (drowsiness) produces a larger angle.
    """
    nose = _lm_to_np(landmarks[NOSE_TIP], w, h)
    chin = _lm_to_np(landmarks[CHIN], w, h)
    
    vec = chin - nose
    angle = abs(math.atan2(vec[0], vec[1]))
    
    return min(angle / (math.pi / 2), 1.0)


# ──────────────────────────────────────────────
# Standalone function for one-off use
# ──────────────────────────────────────────────

def extract_geometric_features(image_bgr: np.ndarray) -> dict:
    """
    Extract features using a temporary LandmarkExtractor.
    For batch use, prefer creating a LandmarkExtractor instance once.
    """
    extractor = LandmarkExtractor()
    result = extractor.extract(image_bgr)
    del extractor
    return result
