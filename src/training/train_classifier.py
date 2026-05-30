# src/training/train_classifier.py
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter

class ClassifierTrainer:
    def __init__(self, model, device, settings):
        self.model = model.to(device)
        self.device = device
        self.settings = settings
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, train_loader, val_loader, lr, epochs, experiment_name="experiment"):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=lr
        )
        
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0
        
        # Crear directorios
        os.makedirs(self.settings.checkpoint_dir, exist_ok=True)
        
        # Inicializar TensorBoard
        log_dir = os.path.join(self.settings.tensorboard_dir, experiment_name)
        writer = SummaryWriter(log_dir=log_dir)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # ========== TRAINING ==========
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc="Training"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            train_loss_avg = train_loss / len(train_loader)
            
            # ========== VALIDATION ==========
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation"):
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            val_acc = 100. * val_correct / val_total
            val_loss_avg = val_loss / len(val_loader)
            
            # Guardar historial
            history['train_loss'].append(train_loss_avg)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss_avg)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")
            
            # ========== TENSORBOARD LOGS ==========
            writer.add_scalar('Loss/Train', train_loss_avg, epoch)
            writer.add_scalar('Loss/Validation', val_loss_avg, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # ========== GUARDAR MEJOR MODELO ==========
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = os.path.join(self.settings.checkpoint_dir, f"{experiment_name}_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'history': history
                }, checkpoint_path)
                print(f"   Modelo guardado en {checkpoint_path} (val_acc: {val_acc:.2f}%)")
        
        # ========== GUARDAR MODELO FINAL ==========
        final_path = os.path.join(self.settings.checkpoint_dir, f"{experiment_name}_final.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': history,
            'best_val_acc': best_val_acc
        }, final_path)
        print(f"   Modelo final guardado en {final_path}")
        
        # ========== CERRAR TENSORBOARD ==========
        writer.close()
        print(f"\n TensorBoard logs guardados en: {log_dir}")
        print(f"   Para verlos: tensorboard --logdir={self.settings.tensorboard_dir}")
        
        return history, best_val_acc


def evaluate_on_domain(model, loader, device):
    """Evaluate model on a domain and return accuracy"""
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total