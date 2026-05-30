import torch
import torch.nn as nn
import torchvision.models as models

def create_model(backbone='resnet50', num_classes=6, pretrained=True, freeze_backbone=True):
    """Create classifier with pretrained ResNet50 backbone"""
    
    model = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    
    return model

def unfreeze_layers(model, layers_to_unfreeze, backbone='resnet50'):
    """Unfreeze specific layers for fine-tuning"""
    
    for param in model.parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        for layer in layers_to_unfreeze:
            if layer in name:
                param.requires_grad = True
    
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
    
    return model

def create_two_layer_head(model, num_classes=6, hidden_size=512, dropout=0.3):
    """Replace classifier head with two-layer MLP"""
    
    if hasattr(model, 'fc'):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
    
    return model