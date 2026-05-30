# src/utils/settings.py
from dataclasses import dataclass
from typing import List

@dataclass
class Settings:
    # Dataset settings
    source_domain: str = "clipart"
    target_domain: str = "sketch"
    categories: List[str] = None
    
    # Data settings
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    shots_per_class: int = 50
    val_split: float = 0.2
    
    # Model settings
    backbone: str = "resnet50"
    num_classes: int = 6
    
    # Training settings - Feature Extraction
    fe_lr: float = 1e-3
    fe_epochs: int = 30
    
    # Training settings - Fine-tuning
    ft_lr: float = 1e-4
    ft_epochs: int = 30
    ft_unfreeze_layers: List[str] = None
    
    # Style Transfer settings
    content_layers: List[str] = None
    style_layers: List[str] = None
    alpha = 10
    beta =  100
    style_steps: int =10000
    style_lr: float = 1.0
    num_style_images_per_class: int = 30
    
    # Domain Adaptation settings
    target_finetune_epochs: int = 20
    target_finetune_lr: float = 1e-4
    
    # Paths
    data_root: str = "./data"
    checkpoint_dir: str = "./models/checkpoints"
    synthetic_dir: str = "./data/synthetic_target"
    figures_dir: str = "./figures"
    tensorboard_dir: str = "./runs"
    
    # Random seed
    seed: int = 42
    num_seeds: int = 3
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = ["jacket", "pants", "shoe", "sock", "t-shirt", "umbrella"]
        
        if self.ft_unfreeze_layers is None:
            self.ft_unfreeze_layers = ['layer3', 'layer4']
        
        if self.content_layers is None:
            self.content_layers = ['relu4_2']
        
        if self.style_layers is None:
            self.style_layers = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']