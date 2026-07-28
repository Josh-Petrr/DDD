"""
models.py — Three model architectures for drowsiness detection.

1. BaselineCNN         — EfficientNet-B0 + classification head (no geometry)
2. GeometricOnlyMLP    — Just the 4 landmark features through an MLP
3. FusionModel         — EfficientNet-B0 embedding + geometric features → fusion MLP

All models output 2-class logits [Drowsy, Non-Drowsy].
"""

import torch
import torch.nn as nn
from torchvision import models

import config


class BaselineCNN(nn.Module):
    """
    EfficientNet-B0 with a simple classification head.
    
    This is the "before" model — pure CNN, no geometric features.
    Used as the comparison baseline.
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=config.DROPOUT_RATE),
            nn.Linear(in_features, num_classes)
        )
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 1280-d embedding (before classification head)."""
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, images: torch.Tensor, geo_features: torch.Tensor = None):
        """Forward pass. geo_features is ignored (kept for API compatibility)."""
        return self.backbone(images)
    
    def freeze_backbone(self):
        """Freeze EfficientNet feature extractor layers."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True


class GeometricOnlyMLP(nn.Module):
    """
    Small MLP that uses only the 4 geometric features.
    
    This is the ablation model — shows what landmarks alone can do
    without any visual features.
    """
    
    def __init__(self, num_features: int = config.NUM_GEOMETRIC_FEATURES,
                 num_classes: int = config.NUM_CLASSES):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(num_features, config.GEO_HIDDEN_1),
            nn.BatchNorm1d(config.GEO_HIDDEN_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            
            nn.Linear(config.GEO_HIDDEN_1, config.GEO_HIDDEN_2),
            nn.BatchNorm1d(config.GEO_HIDDEN_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            
            nn.Linear(config.GEO_HIDDEN_2, num_classes),
        )
    
    def forward(self, images: torch.Tensor = None, 
                geo_features: torch.Tensor = None):
        """Forward pass. images is ignored."""
        return self.mlp(geo_features)
    
    def freeze_backbone(self):
        pass  # No backbone to freeze
    
    def unfreeze_backbone(self):
        pass


class FusionModel(nn.Module):
    """
    Main model: EfficientNet-B0 embedding + geometric features → fusion MLP.
    
    Architecture:
        CNN embedding (1280-d) + geometric features (4-d)
        → Concatenate (1284-d)
        → Linear(1284, 256) → BN → ReLU → Dropout
        → Linear(256, 64)   → BN → ReLU → Dropout
        → Linear(64, 2)
    """
    
    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        efficientnet = models.efficientnet_b0(weights=weights)
        
        # Use features + avgpool as backbone (remove classifier)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        
        # Fusion MLP
        fusion_input_dim = config.EMBEDDING_DIM + config.NUM_GEOMETRIC_FEATURES
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_1),
            nn.BatchNorm1d(config.FUSION_HIDDEN_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            
            nn.Linear(config.FUSION_HIDDEN_1, config.FUSION_HIDDEN_2),
            nn.BatchNorm1d(config.FUSION_HIDDEN_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            
            nn.Linear(config.FUSION_HIDDEN_2, num_classes),
        )
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 1280-d CNN embedding."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    def forward(self, images: torch.Tensor, geo_features: torch.Tensor):
        """
        Forward pass.
        
        Args:
            images: (B, 3, 224, 224) image tensors
            geo_features: (B, 4) geometric feature tensors
        
        Returns:
            (B, 2) logits
        """
        # Extract CNN embedding
        cnn_emb = self.get_embedding(images)  # (B, 1280)
        
        # Concatenate with geometric features
        fused = torch.cat([cnn_emb, geo_features], dim=1)  # (B, 1284)
        
        # Fusion MLP
        return self.fusion_head(fused)
    
    def freeze_backbone(self):
        """Freeze EfficientNet feature extractor layers."""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True


def get_model(model_name: str, pretrained: bool = True) -> nn.Module:
    """
    Factory function to create a model by name.
    
    Args:
        model_name: one of 'baseline', 'geometric', 'fusion'
        pretrained: whether to use ImageNet pretrained weights
    
    Returns:
        nn.Module
    """
    model_name = model_name.lower()
    
    if model_name == "baseline":
        model = BaselineCNN(pretrained=pretrained)
    elif model_name == "geometric":
        model = GeometricOnlyMLP()
    elif model_name == "fusion":
        model = FusionModel(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: baseline, geometric, fusion")
    
    return model


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
