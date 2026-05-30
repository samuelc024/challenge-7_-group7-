import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_num_trainable_params(model):
    """Get number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def create_few_shot_loaders(dataset, shots_per_class, batch_size, val_split=0.2, num_workers=4):
    """Create few-shot train/val/test loaders"""
    from collections import defaultdict
    
    # Get indices per class
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    # Sample shots_per_class per class
    train_indices = []
    val_indices = []
    
    for cls, indices in class_indices.items():
        num_shots = min(shots_per_class, len(indices))
        sampled = random.sample(indices, num_shots)
        
        train_size = int(num_shots * (1 - val_split))
        train_indices.extend(sampled[:train_size])
        val_indices.extend(sampled[train_size:])
    
    # Remaining indices for test
    all_indices = set(range(len(dataset)))
    used_indices = set(train_indices + val_indices)
    test_indices = list(all_indices - used_indices)
    
    # Create loaders
    train_loader = DataLoader(dataset, batch_size=batch_size, 
                            sampler=SubsetRandomSampler(train_indices),
                            num_workers=num_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size,
                          sampler=SubsetRandomSampler(val_indices),
                          num_workers=num_workers)
    test_loader = DataLoader(dataset, batch_size=batch_size,
                           sampler=SubsetRandomSampler(test_indices),
                           num_workers=num_workers)
    
    return train_loader, val_loader, test_loader