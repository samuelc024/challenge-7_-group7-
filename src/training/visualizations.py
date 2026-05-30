import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from sklearn.manifold import TSNE
from torchvision import transforms
import cv2

class Visualizer:
    """Generate visualizations for the paper"""
    
    @staticmethod
    def plot_training_curves(history, title, save_path):
        """Plot training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True)
        
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_style_transfer_gallery(content_images, style_images, stylized_images, 
                                    class_names, save_path):
        """Create gallery of style transfer results"""
        num_classes = len(class_names)
        fig, axes = plt.subplots(num_classes, 3, figsize=(9, 3*num_classes))
        
        for i in range(num_classes):
            # Content image
            ax = axes[i, 0] if num_classes > 1 else axes[0]
            img = content_images[i].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f'Content: {class_names[i]}')
            ax.axis('off')
            
            # Style image
            ax = axes[i, 1] if num_classes > 1 else axes[1]
            img = style_images[i].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f'Style: {class_names[i]}')
            ax.axis('off')
            
            # Stylized image
            ax = axes[i, 2] if num_classes > 1 else axes[2]
            img = stylized_images[i].cpu().numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            ax.set_title(f'Stylized')
            ax.axis('off')
        
        plt.suptitle('Neural Style Transfer Results', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_gradcam(model, image, true_label, class_names, save_path):
        """Generate Grad-CAM visualization"""
        # Simplified Grad-CAM implementation
        model.eval()
        
        # Register hooks to get gradients and activations
        gradients = []
        activations = []
        
        def save_gradient(grad):
            gradients.append(grad)
        
        def save_activation(module, input, output):
            activations.append(output)
            output.register_hook(save_gradient)
        
        # Hook to last convolutional layer (for ResNet: layer4)
        if hasattr(model, 'layer4'):
            handle = model.layer4.register_forward_hook(save_activation)
        
        # Forward pass
        image_tensor = image.unsqueeze(0)
        output = model(image_tensor)
        _, pred = output.max(1)
        
        # Backward pass for predicted class
        model.zero_grad()
        output[0, pred].backward()
        
        # Compute Grad-CAM
        if gradients and activations:
            grad = gradients[0][0]
            act = activations[0][0]
            weights = torch.mean(grad, dim=(1, 2), keepdim=True)
            cam = torch.sum(weights * act, dim=0)
            cam = torch.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cam = cam.detach().cpu().numpy()
            
            # Resize CAM to image size
            cam_resized = cv2.resize(cam, (224, 224))
            
            # Overlay on image
            img_np = image.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay = 0.6 * img_np + 0.4 * heatmap
            
            # Plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            
            ax1.imshow(img_np)
            ax1.set_title(f'Original Image\nTrue: {class_names[true_label]}')
            ax1.axis('off')
            
            ax2.imshow(cam_resized, cmap='jet')
            ax2.set_title('Grad-CAM Heatmap')
            ax2.axis('off')
            
            ax3.imshow(overlay)
            ax3.set_title(f'Overlay\nPred: {class_names[pred.item()]}')
            ax3.axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        handle.remove()
    
    @staticmethod
    def plot_tsne(source_features, target_features, source_labels, target_labels,
                 class_names, save_path):
        """Plot t-SNE visualization of features"""
        # Combine features
        all_features = np.vstack([source_features, target_features])
        all_labels = np.concatenate([source_labels, target_labels])
        domain_labels = np.concatenate([np.zeros(len(source_features)), 
                                       np.ones(len(target_features))])
        
        # Run t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(all_features)
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # By class
        scatter1 = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=all_labels, cmap='tab10', alpha=0.6)
        ax1.set_title('t-SNE by Class')
        ax1.set_xlabel('t-SNE 1')
        ax1.set_ylabel('t-SNE 2')
        legend1 = ax1.legend(*scatter1.legend_elements(), title='Classes')
        ax1.add_artist(legend1)
        
        # By domain
        scatter2 = ax2.scatter(features_2d[:, 0], features_2d[:, 1],
                              c=domain_labels, cmap='coolwarm', alpha=0.6)
        ax2.set_title('t-SNE by Domain')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.legend(['Source (Clipart)', 'Target (Sketch)'])
        
        plt.suptitle('Feature Space Visualization with t-SNE')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_comparison_bar(results, save_path):
        """Plot comparison bar chart of adaptation strategies"""
        strategies = list(results.keys())
        source_acc = [results[s]['source_accuracy'] for s in strategies]
        target_acc = [results[s]['target_accuracy'] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, source_acc, width, label='Source Domain', color='blue')
        bars2 = ax.bar(x + width/2, target_acc, width, label='Target Domain', color='orange')
        
        ax.set_xlabel('Adaptation Strategy')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()