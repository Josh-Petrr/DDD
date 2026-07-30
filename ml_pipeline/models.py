"""
models.py — Model architectures for drowsiness detection.

1. BaselineCNN         — EfficientNet-B0 + classification head (no geometry)
2. GeometricOnlyMLP    — Just the 4 landmark features through an MLP
3. FusionModel         — EfficientNet-B0 embedding + geometric features → fusion MLP
4. FusionGRLModel      — FusionModel + Gradient Reversal Layer for Domain Adaptation

All models output 2-class logits [Drowsy, Non-Drowsy].
FusionGRLModel outputs (drowsy_logits, domain_logits).
"""

import torch
import torch.nn as nn
from torchvision import models

import config


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL).
    Forward pass is the identity function.
    Backward pass multiplies the gradient by -alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class BaselineCNN(nn.Module):
    """
    EfficientNet-B0 with a simple classification head.
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
    
    def forward(self, images: torch.Tensor, geo_features: torch.Tensor = None, alpha: float = 1.0):
        """Forward pass. geo_features and alpha are ignored."""
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
                geo_features: torch.Tensor = None, alpha: float = 1.0):
        """Forward pass. images and alpha are ignored."""
        return self.mlp(geo_features)
    
    def freeze_backbone(self):
        pass
    
    def unfreeze_backbone(self):
        pass


class FusionModel(nn.Module):
    """
    Main model: EfficientNet-B0 embedding + geometric features → fusion MLP.
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
    
    def forward(self, images: torch.Tensor, geo_features: torch.Tensor, alpha: float = 1.0):
        """Forward pass."""
        cnn_emb = self.get_embedding(images)
        fused = torch.cat([cnn_emb, geo_features], dim=1)
        return self.fusion_head(fused)
    
    def freeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        for param in self.features.parameters():
            param.requires_grad = True


class FusionGRLModel(FusionModel):
    """
    FusionModel with an adversarial Gradient Reversal Layer for Domain Adaptation.
    Forces the CNN to learn identity-invariant (domain-invariant) features.
    """
    def __init__(self, num_classes: int = config.NUM_CLASSES, num_domains: int = 10, pretrained: bool = True):
        super().__init__(num_classes=num_classes, pretrained=pretrained)
        
        fusion_input_dim = config.EMBEDDING_DIM + config.NUM_GEOMETRIC_FEATURES
        self.domain_head = nn.Sequential(
            nn.Linear(fusion_input_dim, config.FUSION_HIDDEN_1),
            nn.BatchNorm1d(config.FUSION_HIDDEN_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            
            nn.Linear(config.FUSION_HIDDEN_1, config.FUSION_HIDDEN_2),
            nn.BatchNorm1d(config.FUSION_HIDDEN_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.DROPOUT_RATE),
            
            nn.Linear(config.FUSION_HIDDEN_2, num_domains),
        )
        
    def forward(self, images: torch.Tensor, geo_features: torch.Tensor, alpha: float = 1.0):
        """
        Returns (drowsy_logits, domain_logits).
        """
        cnn_emb = self.get_embedding(images)
        fused = torch.cat([cnn_emb, geo_features], dim=1)
        
        # Primary Task: Drowsiness prediction
        drowsy_logits = self.fusion_head(fused)
        
        # Adversarial Task: Domain (Subject) prediction via GRL
        reversed_fused = GradientReversalLayer.apply(fused, alpha)
        domain_logits = self.domain_head(reversed_fused)
        
        return drowsy_logits, domain_logits


def get_model(model_name: str, pretrained: bool = True, num_domains: int = 0) -> nn.Module:
    """
    Factory function to create a model by name.
    """
    model_name = model_name.lower()
    
    if model_name == "baseline":
        model = BaselineCNN(pretrained=pretrained)
    elif model_name == "geometric":
        model = GeometricOnlyMLP()
    elif model_name == "fusion":
        model = FusionModel(pretrained=pretrained)
    elif model_name == "fusion_grl":
        model = FusionGRLModel(pretrained=pretrained, num_domains=num_domains)
    else:
        raise ValueError(f"Unknown model: {model_name}. ")
    
    return model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
