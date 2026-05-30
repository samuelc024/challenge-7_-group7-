import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DomainNetDataset(Dataset):
    def __init__(self, root_dir, domain, categories, transform=None):
        self.root_dir = root_dir
        self.domain = domain
        self.categories = categories
        self.transform = transform
        
        self.images = []
        self.labels = []
        
        for label, category in enumerate(categories):
            category_path = os.path.join(root_dir, domain, category)
            if os.path.exists(category_path):
                for img_file in os.listdir(category_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.images.append(os.path.join(category_path, img_file))
                        self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images from {domain}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, 224, 224), label

def get_transforms(img_size=224, is_train=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    return transform

def load_datasets(settings):
    source_transform = get_transforms(settings.img_size, is_train=True)
    target_transform = get_transforms(settings.img_size, is_train=False)
    
    source_dataset = DomainNetDataset(
        settings.data_root, settings.source_domain, 
        settings.categories, source_transform
    )
    target_dataset = DomainNetDataset(
        settings.data_root, settings.target_domain,
        settings.categories, target_transform
    )
    
    return source_dataset, target_dataset