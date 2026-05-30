# src/models/style_transfer.py - Versión completa para Sketch (blanco y negro)
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torch.nn.functional as F

class StyleTransfer:
    def __init__(self, device='cuda'):
        self.device = device
        
        # Capas para style transfer
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        print(f"Inicializando Style Transfer en {device}")
        
        # Cargar VGG19
        self.cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
        
        # Congelar parámetros
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        # Normalización
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        # Crear modelo secuencial con nombres
        self.model = nn.Sequential()
        self.layer_names = []
        
        conv_idx = 0
        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d):
                conv_idx += 1
                name = f'conv_{conv_idx}'
            elif isinstance(layer, nn.ReLU):
                name = f'relu_{conv_idx}'
            elif isinstance(layer, nn.MaxPool2d):
                name = f'pool_{conv_idx}'
            else:
                continue
            self.model.add_module(name, layer)
            self.layer_names.append(name)
        
        print(f"VGG19 cargado: {len(self.layer_names)} capas")
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def get_features(self, x):
        """Extrae características"""
        features = {}
        x = self.normalize(x)
        for name, layer in self.model.named_children():
            x = layer(x)
            if name in self.content_layers or name in self.style_layers:
                features[name] = x
        return features
    
    def gram_matrix(self, x):
        b, c, h, w = x.size()
        x = x.view(b * c, h * w)
        gram = torch.mm(x, x.t())
        return gram.div(b * c * h * w)
    
    
    
    def transfer_style(self, content_img, style_img, num_steps=400, 
                       style_weight=1e10, content_weight=0.0001):
        """
        Transferencia de estilo - Pesos extremos para Sketch (blanco y negro)
        """
        print(f"      Style weight: {style_weight}, Content weight: {content_weight}")
        
        # Asegurar que requieren gradiente
        content_img = content_img.clone().detach().to(self.device)
        style_img = style_img.clone().detach().to(self.device)
        
        # Extraer features
        with torch.no_grad():
            content_features = self.get_features(content_img)
            style_features = self.get_features(style_img)
        
        # Gram matrices del estilo
        style_grams = {}
        for layer in self.style_layers:
            if layer in style_features:
                style_grams[layer] = self.gram_matrix(style_features[layer])
        
        # Imagen objetivo
        target = content_img.clone().requires_grad_(True)
        
        # Optimizador Adam
        optimizer = optim.Adam([target], lr=0.1)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            target_features = self.get_features(target)
            
            # Pérdida de contenido
            content_loss = torch.tensor(0.0, device=self.device)
            for layer in self.content_layers:
                if layer in target_features and layer in content_features:
                    content_loss = content_loss + F.mse_loss(target_features[layer], content_features[layer])
            
            # Pérdida de estilo
            style_loss = torch.tensor(0.0, device=self.device)
            for layer in self.style_layers:
                if layer in target_features and layer in style_grams:
                    target_gram = self.gram_matrix(target_features[layer])
                    style_loss = style_loss + F.mse_loss(target_gram, style_grams[layer])
            
            # Pérdida total
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            # Backward
            total_loss.backward()
            optimizer.step()
            
            # Clamp
            with torch.no_grad():
                target.data.clamp_(0, 1)
            
            # Progreso
            if (step + 1) % 100 == 0:
                print(f"        Paso {step+1}/{num_steps} - "
                      f"Content: {content_loss.item():.4f}, "
                      f"Style: {style_loss.item():.4f}")
        
        # Convertir a blanco y negro al final
        target = self.to_grayscale(target)
        
        return target.detach()
    
    def generate_training_data(self, source_dataset, target_dataset, 
                               num_per_class=5, alpha=0.0001, beta=1e10):
        """
        Genera imágenes con style transfer REAL para Sketch
        """
        synthetic_images = []
        synthetic_labels = []
        
        style_weight = beta / alpha if alpha > 0 else 1e10
        content_weight = 1.0
        
        print(f"\n{'='*60}")
        print(f" GENERANDO IMÁGENES CON STYLE TRANSFER PARA SKETCH")
        print(f"{'='*60}")
        print(f"   Por clase: {num_per_class} imágenes")
        print(f"   Total esperado: {num_per_class * 6} imágenes")
        print(f"   alpha (contenido): {alpha}")
        print(f"   beta (estilo): {beta}")
        print(f"   style_weight: {style_weight}")
        print(f"   content_weight: {content_weight}")
        print(f"   Esto tomará varios minutos...\n")
        
        # Obtener imagen de estilo por clase
        style_images = {}
        print("Obteniendo imágenes de estilo del dominio Sketch...")
        for label in range(6):
            for i in range(len(target_dataset)):
                img, lbl = target_dataset[i]
                if lbl == label:
                    style_images[label] = img
                    print(f"   Clase {label}: imagen de estilo obtenida")
                    break
        
        # Generar para cada clase
        for label in range(6):
            print(f"\n{'='*50}")
            print(f" Procesando clase {label}")
            print(f"{'='*50}")
            
            # Obtener imágenes de contenido
            content_list = []
            for i in range(len(source_dataset)):
                img, lbl = source_dataset[i]
                if lbl == label:
                    content_list.append(img)
                    if len(content_list) >= num_per_class:
                        break
            
            print(f"Imágenes de contenido encontradas: {len(content_list)}")
            
            if label not in style_images:
                print(f"   No hay estilo para clase {label}")
                continue
            
            style_img = style_images[label]
            style_batch = style_img.unsqueeze(0)
            
            for idx, content_img in enumerate(content_list):
                print(f"\n   Generando {idx+1}/{len(content_list)}...")
                
                content_batch = content_img.unsqueeze(0)
                
                try:
                    # Style transfer REAL con pesos para Sketch
                    stylized = self.transfer_style(
                        content_batch, style_batch,
                        num_steps=400,
                        style_weight=style_weight,
                        content_weight=content_weight
                    )
                    synthetic_images.append(stylized.squeeze(0).cpu())
                    synthetic_labels.append(label)
                    print(f"    Completada - Imagen en blanco y negro")
                    
                    # Limpiar caché GPU
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"    Error: {e}")
                    # Fallback: imagen original convertida a gris
                    gray_img = self.to_grayscale(content_batch)
                    synthetic_images.append(gray_img.squeeze(0).cpu())
                    synthetic_labels.append(label)
                    print(f"    Usando imagen original en gris")
        
        print(f"\n{'='*60}")
        print(f" GENERACIÓN COMPLETADA")
        print(f"{'='*60}")
        print(f"   Total imágenes generadas: {len(synthetic_images)}")
        print(f"   (Deberían ser {num_per_class * 6} si todo funcionó)")
        print(f"   Las imágenes deberían estar en BLANCO Y NEGRO (estilo Sketch)")
        print(f"{'='*60}\n")
        
        return synthetic_images, synthetic_labels
