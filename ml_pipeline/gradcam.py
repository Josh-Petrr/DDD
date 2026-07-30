"""
gradcam.py — Grad-CAM visualization for the CNN models.

Generates attention heatmaps showing where the EfficientNet-B0 focuses
when making drowsiness predictions.
"""

import os
import numpy as np

import torch
import torch.nn.functional as F
from torch.amp import autocast
import cv2
from PIL import Image
from torchvision import transforms

import config


class GradCAM:
    """
    Grad-CAM implementation for EfficientNet-B0 based models.
    """
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        if target_layer is None:
            if hasattr(model, 'backbone'):
                target_layer = model.backbone.features[-1]
            elif hasattr(model, 'features'):
                target_layer = model.features[-1]
            else:
                raise ValueError("Cannot auto-detect target layer")
        
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, image_tensor: torch.Tensor, 
                 geo_features: torch.Tensor,
                 target_class: int = None,
                 device: torch.device = config.DEVICE) -> np.ndarray:
        image_tensor = image_tensor.to(device)
        geo_features = geo_features.to(device)
        
        self.model.zero_grad()
        if hasattr(self.model, 'domain_head'):
            output, _ = self.model(image_tensor, geo_features)
        else:
            output = self.model(image_tensor, geo_features)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        target_score = output[0, target_class]
        target_score.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2))
        
        cam = torch.zeros(activations.shape[1:], device=device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (config.IMG_SIZE, config.IMG_SIZE))
        
        return cam


def _denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
    
    img = image_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def create_overlay(image: np.ndarray, heatmap: np.ndarray, 
                   alpha: float = 0.4) -> np.ndarray:
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
    return overlay


def generate_gradcam_grid(model, test_loader, model_name: str,
                          num_samples: int = 8,
                          device: torch.device = config.DEVICE):
    import matplotlib.pyplot as plt
    
    model = model.to(device)
    grad_cam = GradCAM(model)
    
    samples = {"Drowsy": [], "Non Drowsy": []}
    target_per_class = num_samples // 2
    
    model.eval()
    for images, geo_features, labels, domain_labels in test_loader:
        for i in range(images.size(0)):
            img = images[i].unsqueeze(0)
            geo = geo_features[i].unsqueeze(0)
            label = labels[i].item()
            
            with torch.no_grad():
                if hasattr(model, 'domain_head'):
                    output, _ = model(img.to(device), geo.to(device))
                else:
                    output = model(img.to(device), geo.to(device))
                pred = output.argmax(dim=1).item()
            
            class_name = config.CLASS_NAMES[label]
            
            if pred == label and len(samples[class_name]) < target_per_class:
                heatmap = grad_cam.generate(img, geo, device=device)
                orig_img = _denormalize_image(img[0])
                overlay = create_overlay(orig_img, heatmap)
                
                prob = F.softmax(output, dim=1)[0, pred].item()
                samples[class_name].append({
                    "original": orig_img,
                    "heatmap": heatmap,
                    "overlay": overlay,
                    "label": class_name,
                    "confidence": prob,
                })
            
            if all(len(v) >= target_per_class for v in samples.values()):
                break
        
        if all(len(v) >= target_per_class for v in samples.values()):
            break
    
    all_samples = samples["Drowsy"] + samples["Non Drowsy"]
    n = len(all_samples)
    
    if n == 0:
        print("No samples collected for Grad-CAM visualization.")
        return
    
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f"Grad-CAM Visualization — {model_name.upper()}", 
                 fontsize=16, fontweight="bold", y=1.02)
    
    for i, sample in enumerate(all_samples):
        axes[i, 0].imshow(sample["original"])
        axes[i, 0].set_title(f"{sample['label']}\n({sample['confidence']:.1%} conf.)",
                             fontsize=10)
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(sample["heatmap"], cmap="jet")
        axes[i, 1].set_title("Attention Map", fontsize=10)
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(sample["overlay"])
        axes[i, 2].set_title("Overlay", fontsize=10)
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    
    save_path = os.path.join(config.RESULTS_DIR, f"gradcam_{model_name}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM grid saved: {save_path}")
