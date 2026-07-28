"""
inference.py — Run drowsiness detection on unseen images.

Takes a folder of completely new images (or a single image path) and
outputs per-image predictions with confidence scores.

Usage:
    python inference.py --input "path/to/folder"
    python inference.py --input "path/to/single_image.jpg"
    python inference.py --input "path/to/folder" --model fusion
"""

import os
import sys
import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config
from models import get_model
from landmark_features import LandmarkExtractor


class DrowsinessPredictor:
    """
    End-to-end predictor: image → face landmarks + CNN → drowsiness prediction.
    
    Loads a trained model checkpoint and runs inference on new images.
    """
    
    def __init__(self, model_name: str = "fusion",
                 checkpoint_path: str = None,
                 device: torch.device = config.DEVICE):
        
        self.device = device
        self.model_name = model_name
        
        # Load model
        self.model = get_model(model_name, pretrained=False)
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(config.CHECKPOINTS_DIR, 
                                           f"{model_name}_best.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        # Image transform (no augmentation)
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN,
                                 std=config.IMAGENET_STD),
        ])
        
        # Landmark extractor
        self.landmark_extractor = LandmarkExtractor()
        
        # Load geometric feature stats from training (for normalization)
        self.geo_stats = self._load_geo_stats()
        
        print(f"✅ Loaded {model_name} model from {checkpoint_path}")
    
    def _load_geo_stats(self) -> dict:
        """Load geometric feature normalization stats from the training set."""
        import pandas as pd
        
        stats_path = os.path.join(config.RESULTS_DIR, "geo_stats.json")
        
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                return json.load(f)
        
        # Fallback: compute from features file
        if os.path.exists(config.FEATURES_FILE):
            df = pd.read_csv(config.FEATURES_FILE)
            success_df = df[df["success"] == True]
            
            stats = {
                "ear": [float(success_df["ear"].mean()), 
                        float(success_df["ear"].std()) + 1e-8],
                "mar": [float(success_df["mar"].mean()),
                        float(success_df["mar"].std()) + 1e-8],
                "eyebrow_dist": [float(success_df["eyebrow_dist"].mean()),
                                 float(success_df["eyebrow_dist"].std()) + 1e-8],
                "head_tilt": [float(success_df["head_tilt"].mean()),
                              float(success_df["head_tilt"].std()) + 1e-8],
            }
            
            # Cache for future use
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            return stats
        
        # Ultimate fallback: no normalization
        return {
            "ear": [0.25, 0.05], "mar": [0.3, 0.1],
            "eyebrow_dist": [0.15, 0.05], "head_tilt": [0.1, 0.1],
        }
    
    def predict_single(self, image_path: str) -> dict:
        """
        Predict drowsiness for a single image.
        
        Returns:
            dict with keys: prediction, confidence, class_name,
                           probabilities, geometric_features
        """
        # Load image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return {
                "image": image_path, "error": "Could not load image",
                "prediction": -1, "confidence": 0.0,
                "class_name": "ERROR"
            }
        
        # Extract geometric features
        geo = self.landmark_extractor.extract(image_bgr)
        
        if geo["success"]:
            raw = [geo["ear"], geo["mar"], geo["eyebrow_dist"], geo["head_tilt"]]
        else:
            # Use mean values
            raw = [self.geo_stats["ear"][0], self.geo_stats["mar"][0],
                   self.geo_stats["eyebrow_dist"][0], self.geo_stats["head_tilt"][0]]
        
        # Normalize
        normalized = [
            (raw[0] - self.geo_stats["ear"][0]) / self.geo_stats["ear"][1],
            (raw[1] - self.geo_stats["mar"][0]) / self.geo_stats["mar"][1],
            (raw[2] - self.geo_stats["eyebrow_dist"][0]) / self.geo_stats["eyebrow_dist"][1],
            (raw[3] - self.geo_stats["head_tilt"][0]) / self.geo_stats["head_tilt"][1],
        ]
        
        geo_tensor = torch.tensor([normalized], dtype=torch.float32).to(self.device)
        
        # Load and transform image
        image_pil = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor, geo_tensor)
            probs = F.softmax(output, dim=1)[0]
            pred = probs.argmax().item()
            confidence = probs[pred].item()
        
        return {
            "image": os.path.basename(image_path),
            "prediction": pred,
            "class_name": config.CLASS_NAMES[pred],
            "confidence": confidence,
            "probabilities": {
                "Drowsy": float(probs[0]),
                "Non Drowsy": float(probs[1])
            },
            "geometric_features": {
                "ear": raw[0], "mar": raw[1],
                "eyebrow_dist": raw[2], "head_tilt": raw[3]
            },
            "face_detected": geo["success"]
        }
    
    def predict_folder(self, folder_path: str) -> list:
        """
        Predict drowsiness for all images in a folder.
        
        Returns list of prediction dicts.
        """
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
        
        image_files = sorted([
            f for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
        
        if not image_files:
            print(f"No images found in {folder_path}")
            return []
        
        print(f"\nProcessing {len(image_files)} images from: {folder_path}")
        print("-" * 50)
        
        results = []
        for fname in image_files:
            filepath = os.path.join(folder_path, fname)
            result = self.predict_single(filepath)
            results.append(result)
            
            # Print individual result
            status = "🔴 DROWSY" if result["class_name"] == "Drowsy" else "🟢 ALERT"
            print(f"  {fname:>30s}  →  {status}  "
                  f"({result['confidence']:.1%} confidence)")
        
        return results


def print_summary(results: list):
    """Print a summary of predictions."""
    total = len(results)
    drowsy = sum(1 for r in results if r["prediction"] == 0)
    non_drowsy = sum(1 for r in results if r["prediction"] == 1)
    errors = sum(1 for r in results if r["prediction"] == -1)
    
    avg_confidence = np.mean([r["confidence"] for r in results 
                              if r["prediction"] != -1])
    
    print(f"\n{'='*50}")
    print(f"PREDICTION SUMMARY")
    print(f"{'='*50}")
    print(f"  Total images:   {total}")
    print(f"  🔴 Drowsy:      {drowsy} ({100*drowsy/total:.1f}%)")
    print(f"  🟢 Non-Drowsy:  {non_drowsy} ({100*non_drowsy/total:.1f}%)")
    if errors > 0:
        print(f"  ⚠ Errors:       {errors}")
    print(f"  Avg confidence: {avg_confidence:.1%}")
    print(f"{'='*50}")
    
    if drowsy > 0:
        print(f"\n⚠ WARNING: {drowsy}/{total} images classified as DROWSY!")
    else:
        print(f"\n✅ All images classified as Non-Drowsy.")


def main():
    parser = argparse.ArgumentParser(
        description="Drowsiness Detection — Inference on unseen data"
    )
    parser.add_argument("--input", required=True,
                        help="Path to an image file or a folder of images")
    parser.add_argument("--model", default="fusion", 
                        choices=["baseline", "geometric", "fusion"],
                        help="Which model to use (default: fusion)")
    parser.add_argument("--output", default=None,
                        help="Optional: save results to JSON file")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DrowsinessPredictor(model_name=args.model)
    
    # Run inference
    if os.path.isfile(args.input):
        results = [predictor.predict_single(args.input)]
        print(f"\n  {os.path.basename(args.input)}  →  "
              f"{results[0]['class_name']}  "
              f"({results[0]['confidence']:.1%} confidence)")
    elif os.path.isdir(args.input):
        results = predictor.predict_folder(args.input)
    else:
        print(f"ERROR: Path not found: {args.input}")
        sys.exit(1)
    
    # Print summary
    if results:
        print_summary(results)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
