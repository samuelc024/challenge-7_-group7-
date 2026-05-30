# src/training/metrics.py
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch

class DomainShiftMetrics:
    """Metrics for measuring domain shift and adaptation effectiveness"""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    def compute_domain_shift(self, source_accuracy, target_accuracy):
        """Compute domain shift penalty Δ_shift"""
        return source_accuracy - target_accuracy
    
    def compute_all_metrics(self, model, source_loader, target_loader, device):
        """Compute comprehensive metrics for model evaluation"""
        
        model.eval()
        model.to(device)
        
        # Source domain metrics
        source_acc, source_report = self.evaluate_with_per_class(model, source_loader, device)
        
        # Target domain metrics
        target_acc, target_report = self.evaluate_with_per_class(model, target_loader, device)
        
        # Domain shift
        domain_shift = self.compute_domain_shift(source_acc, target_acc)
        
        # Feature statistics
        source_features, source_labels = self.extract_features(model, source_loader, device)
        target_features, target_labels = self.extract_features(model, target_loader, device)
        
        # Maximum Mean Discrepancy (MMD) between domains
        mmd_score = self.compute_mmd(source_features, target_features)
        
        return {
            'source_accuracy': source_acc,
            'target_accuracy': target_acc,
            'domain_shift': domain_shift,
            'mmd_score': mmd_score,
            'source_class_report': source_report,
            'target_class_report': target_report
        }
    
    def evaluate_with_per_class(self, model, loader, device):
        """Evaluate model and return per-class metrics"""
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        report = classification_report(all_labels, all_preds, output_dict=True)
        
        return accuracy * 100, report
    
    def extract_features(self, model, loader, device):
        """Extract penultimate layer features for t-SNE visualization"""
        features = []
        labels = []
        
        # Register hook to get intermediate features
        def hook_fn(module, input, output):
            features.extend(output.detach().cpu().numpy())
        
        # For ResNet, hook to the layer before fc
        if hasattr(model, 'fc'):
            handle = model.fc.register_forward_hook(hook_fn)
        
        model.eval()
        with torch.no_grad():
            for images, lbls in loader:
                images = images.to(device)
                _ = model(images)
                labels.extend(lbls.numpy())
        
        if hasattr(model, 'fc'):
            handle.remove()
        
        return np.array(features), np.array(labels)
    
    def compute_mmd(self, source_features, target_features, kernel='rbf'):
        """Compute Maximum Mean Discrepancy between domains"""
        # Simplified MMD with RBF kernel
        def rbf_kernel(x, y, sigma=1.0):
            pairwise_dist = np.sum(x**2, axis=1).reshape(-1, 1) + np.sum(y**2, axis=1) - 2 * np.dot(x, y.T)
            return np.exp(-pairwise_dist / (2 * sigma**2))
        
        mmd = np.mean(rbf_kernel(source_features, source_features)) + \
              np.mean(rbf_kernel(target_features, target_features)) - \
              2 * np.mean(rbf_kernel(source_features, target_features))
        
        return mmd