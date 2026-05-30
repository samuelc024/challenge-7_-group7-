# src/models/dann.py
import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalFunction(Function):
    """Gradient Reversal Layer for DANN"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.saved_tensors[0]
        return -alpha * grad_output, None

class GradientReversal(nn.Module):
    """Gradient Reversal Layer wrapper"""
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

class DANNClassifier(nn.Module):
    """Domain Adversarial Neural Network for domain adaptation"""
    
    def __init__(self, backbone, num_classes=6, hidden_size=256):
        super().__init__()
        
        self.backbone = backbone
        
        # Remove original classification head
        if hasattr(backbone, 'fc'):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, '_fc'):
            in_features = backbone._fc.in_features
            backbone._fc = nn.Identity()
        
        # Class classifier
        self.class_classifier = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Domain classifier with gradient reversal
        self.domain_classifier = nn.Sequential(
            GradientReversal(alpha=1.0),
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # Source/Target
        )
    
    def forward(self, x, alpha=1.0):
        features = self.backbone(x)
        class_output = self.class_classifier(features)
        
        # Set gradient reversal alpha
        if hasattr(self.domain_classifier[0], 'alpha'):
            self.domain_classifier[0].alpha = alpha
        
        domain_output = self.domain_classifier(features)
        
        return class_output, domain_output
    
    def get_features(self, x):
        """Extract features without classification heads"""
        return self.backbone(x)