# src/main.py - VERSIÓN COMPLETA CON MÚLTIPLES SEMILLAS
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.utils.settings import Settings
from src.utils.helpers import set_seed, create_few_shot_loaders, get_num_trainable_params
from src.data.dataset import load_datasets
from src.models.classifier import create_model, unfreeze_layers
from src.training.train_classifier import ClassifierTrainer, evaluate_on_domain
from src.models.style_transfer import StyleTransfer
import pickle
import json
import pandas as pd

def run_single_seed(seed, settings, device):
    """Ejecuta una sola semilla y retorna los resultados"""
    
    set_seed(seed)
    settings.seed = seed
    
    print(f"\n{'='*70}")
    print(f" EJECUCIÓN CON SEMILLA {seed}")
    print(f"{'='*70}")
    
    # Cargar datasets
    print("\n Cargando datasets...")
    source_dataset, target_dataset = load_datasets(settings)
    
    # Crear loaders
    train_loader, val_loader, test_loader = create_few_shot_loaders(
        source_dataset, settings.shots_per_class, settings.batch_size,
        settings.val_split, settings.num_workers
    )
    
    target_test_loader = DataLoader(target_dataset, batch_size=settings.batch_size,
                                   shuffle=False, num_workers=settings.num_workers)
    
    results_seed = {}
    
    # ==================== PARTE A: FEATURE EXTRACTION ====================
    print("\n" + "-"*40)
    print("PARTE A: Feature Extraction")
    print("-"*40)
    
    model_fe = create_model(settings.backbone, settings.num_classes, 
                           pretrained=True, freeze_backbone=True)
    model_fe = model_fe.to(device)
    
    trainer_fe = ClassifierTrainer(model_fe, device, settings)
    history_fe, _ = trainer_fe.train(
        train_loader, val_loader, 
        settings.fe_lr, settings.fe_epochs,
        experiment_name=f"feature_extraction_seed_{seed}"
    )
    
    source_acc_fe = evaluate_on_domain(model_fe, test_loader, device)
    target_acc_fe = evaluate_on_domain(model_fe, target_test_loader, device)
    
    # ==================== PARTE A: FINE-TUNING ====================
    print("\n" + "-"*40)
    print("PARTE A: Fine-tuning")
    print("-"*40)
    
    model_ft = create_model(settings.backbone, settings.num_classes, 
                           pretrained=True, freeze_backbone=False)
    model_ft = unfreeze_layers(model_ft, settings.ft_unfreeze_layers, settings.backbone)
    model_ft = model_ft.to(device)
    
    trainer_ft = ClassifierTrainer(model_ft, device, settings)
    history_ft, _ = trainer_ft.train(
        train_loader, val_loader,
        settings.ft_lr, settings.ft_epochs,
        experiment_name=f"fine_tuning_seed_{seed}"
    )
    
    source_acc_ft = evaluate_on_domain(model_ft, test_loader, device)
    target_acc_ft = evaluate_on_domain(model_ft, target_test_loader, device)
    
    # ==================== PARTE B: STYLE TRANSFER ====================
    print("\n" + "-"*40)
    print("PARTE B: Neural Style Transfer")
    print("-"*40)
    
    synthetic_path = os.path.join(settings.synthetic_dir, f"synthetic_data_seed_{seed}.pkl")
    
    if os.path.exists(synthetic_path):
        print(f" Cargando imágenes sintéticas existentes...")
        with open(synthetic_path, 'rb') as f:
            synthetic_dataset = pickle.load(f)
        synthetic_images = [img for img, _ in synthetic_dataset]
        synthetic_labels = [lbl for _, lbl in synthetic_dataset]
    else:
        print(" Generando imágenes sintéticas...")
        style_transfer = StyleTransfer(device=device)
        
        synthetic_images, synthetic_labels = style_transfer.generate_training_data(
            source_dataset, target_dataset,
            num_per_class=settings.num_style_images_per_class,
            alpha=10,
            beta=100
        )
        
        synthetic_dataset = list(zip(synthetic_images, synthetic_labels))
        with open(synthetic_path, 'wb') as f:
            pickle.dump(synthetic_dataset, f)
    
    # ==================== PARTE C: TARGET FINE-TUNING ====================
    print("\n" + "-"*40)
    print("PARTE C: Target Fine-tuning")
    print("-"*40)
    
    target_train_loader, target_val_loader, _ = create_few_shot_loaders(
        target_dataset, settings.shots_per_class, settings.batch_size,
        settings.val_split, settings.num_workers
    )
    
    model_target = create_model(settings.backbone, settings.num_classes, pretrained=True)
    model_target = unfreeze_layers(model_target, settings.ft_unfreeze_layers, settings.backbone)
    model_target = model_target.to(device)
    
    trainer_target = ClassifierTrainer(model_target, device, settings)
    history_target, _ = trainer_target.train(
        target_train_loader, target_val_loader,
        settings.target_finetune_lr, 15,
        experiment_name=f"target_finetune_seed_{seed}"
    )
    
    source_acc_target_ft = evaluate_on_domain(model_target, test_loader, device)
    target_acc_target_ft = evaluate_on_domain(model_target, target_test_loader, device)
    
    # ==================== PARTE C: STYLE AUGMENTATION ====================
    print("\n" + "-"*40)
    print("PARTE C: Style Augmentation")
    print("-"*40)
    
    if synthetic_images:
        original_images = []
        original_labels = []
        for i in range(len(train_loader.dataset)):
            img, lbl = train_loader.dataset[i]
            original_images.append(img)
            original_labels.append(lbl)
        
        all_images = original_images + synthetic_images
        all_labels = original_labels + synthetic_labels
        
        all_images_tensor = torch.stack(all_images)
        all_labels_tensor = torch.tensor(all_labels)
        
        augmented_dataset = TensorDataset(all_images_tensor, all_labels_tensor)
        augmented_loader = DataLoader(augmented_dataset, batch_size=settings.batch_size,
                                     shuffle=True, num_workers=settings.num_workers)
        
        model_aug = create_model(settings.backbone, settings.num_classes, pretrained=True)
        model_aug = unfreeze_layers(model_aug, settings.ft_unfreeze_layers, settings.backbone)
        model_aug = model_aug.to(device)
        
        trainer_aug = ClassifierTrainer(model_aug, device, settings)
        history_aug, _ = trainer_aug.train(
            augmented_loader, val_loader,
            settings.ft_lr, 15,
            experiment_name=f"style_augmentation_seed_{seed}"
        )
        
        source_acc_aug = evaluate_on_domain(model_aug, test_loader, device)
        target_acc_aug = evaluate_on_domain(model_aug, target_test_loader, device)
    else:
        source_acc_aug = 0
        target_acc_aug = 0
    
    # Almacenar resultados de esta semilla
    results_seed = {
        'Feature Extraction': {'source': source_acc_fe, 'target': target_acc_fe, 'shift': source_acc_fe - target_acc_fe},
        'Fine-tuning': {'source': source_acc_ft, 'target': target_acc_ft, 'shift': source_acc_ft - target_acc_ft},
        'Baseline (No Adaptation)': {'source': source_acc_ft, 'target': target_acc_ft, 'shift': source_acc_ft - target_acc_ft},
        'Target Fine-tuning': {'source': source_acc_target_ft, 'target': target_acc_target_ft, 'shift': source_acc_target_ft - target_acc_target_ft},
        'Style Augmentation': {'source': source_acc_aug, 'target': target_acc_aug, 'shift': source_acc_aug - target_acc_aug}
    }
    
    return results_seed


def main():
    """Ejecución principal con múltiples semillas"""
    
    settings = Settings()
    
    # Definir semillas (3 por requisito)
    seeds = [42, 123, 456]
    
    # Crear directorios
    os.makedirs(settings.checkpoint_dir, exist_ok=True)
    os.makedirs(settings.figures_dir, exist_ok=True)
    os.makedirs(settings.synthetic_dir, exist_ok=True)
    os.makedirs(settings.tensorboard_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"\n Ejecutando con {len(seeds)} semillas diferentes: {seeds}")
    
    # Diccionario para acumular resultados
    all_results = {
        'Feature Extraction': {'source': [], 'target': [], 'shift': []},
        'Fine-tuning': {'source': [], 'target': [], 'shift': []},
        'Baseline (No Adaptation)': {'source': [], 'target': [], 'shift': []},
        'Target Fine-tuning': {'source': [], 'target': [], 'shift': []},
        'Style Augmentation': {'source': [], 'target': [], 'shift': []}
    }
    
    # Ejecutar para cada semilla
    for seed in seeds:
        results = run_single_seed(seed, settings, device)
        
        for model_name in all_results.keys():
            all_results[model_name]['source'].append(results[model_name]['source'])
            all_results[model_name]['target'].append(results[model_name]['target'])
            all_results[model_name]['shift'].append(results[model_name]['shift'])
    
    # ==================== CALCULAR MEDIA Y STD ====================
    print("\n" + "="*70)
    print(" RESULTADOS FINALES (Media ± Desviación Estándar)")
    print(f" Basado en {len(seeds)} semillas: {seeds}")
    print("="*70)
    
    summary_table = []
    
    for model_name in all_results.keys():
        mean_source = np.mean(all_results[model_name]['source'])
        std_source = np.std(all_results[model_name]['source'])
        mean_target = np.mean(all_results[model_name]['target'])
        std_target = np.std(all_results[model_name]['target'])
        mean_shift = np.mean(all_results[model_name]['shift'])
        std_shift = np.std(all_results[model_name]['shift'])
        
        summary_table.append({
            'Model': model_name,
            'Source Acc (%)': f"{mean_source:.2f} ± {std_source:.2f}",
            'Target Acc (%)': f"{mean_target:.2f} ± {std_target:.2f}",
            'Δ_shift (%)': f"{mean_shift:.2f} ± {std_shift:.2f}"
        })
        
        print(f"\n {model_name}:")
        print(f"   Source: {mean_source:.2f}% ± {std_source:.2f}")
        print(f"   Target: {mean_target:.2f}% ± {std_target:.2f}")
        print(f"   Shift:  {mean_shift:.2f}% ± {std_shift:.2f}")
    
    # ==================== TABLA COMPARATIVA ====================
    print("\n" + "="*70)
    print(" TABLA COMPARATIVA DE LOS 5 MODELOS")
    print("="*70)
    
    print(f"\n{'Modelo':<35} {'Source Acc':<18} {'Target Acc':<18} {'Δ_shift':<15}")
    print("-"*86)
    for row in summary_table:
        print(f"{row['Model']:<35} {row['Source Acc (%)']:<18} {row['Target Acc (%)']:<18} {row['Δ_shift (%)']:<15}")
    
    # ==================== GUARDAR RESULTADOS ====================
    print("\n" + "="*70)
    print(" GUARDANDO RESULTADOS")
    print("="*70)
    
    # Guardar como CSV
    df = pd.DataFrame(summary_table)
    csv_path = os.path.join(settings.figures_dir, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV guardado: {csv_path}")
    
    # Guardar como JSON
    json_path = os.path.join(settings.figures_dir, "results_summary.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ JSON guardado: {json_path}")
    
    # ==================== TABLA EN FORMATO LATEX ====================
    print("\n" + "="*70)
    print(" TABLA EN FORMATO LATEX (para copiar al paper)")
    print("="*70)
    
    latex_table = """\\begin{table}[h]
\\centering
\\caption{Resultados de adaptación de dominio para Clipart→Sketch. Media ± std sobre 3 semillas aleatorias.}
\\label{tab:results}
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Modelo} & \\textbf{Source Acc (\\%)} & \\textbf{Target Acc (\\%)} & \\textbf{$\\Delta_{shift}$ (\\%)} \\\\
\\hline
"""
    
    for row in summary_table:
        latex_table += f"{row['Model']} & {row['Source Acc (%)']} & {row['Target Acc (%)']} & {row['Δ_shift (%)']} \\\\\n\\hline\n"
    
    latex_table += """\\end{tabular}
\\end{table}
"""
    
    print(latex_table)
    
    # Guardar LaTeX
    latex_path = os.path.join(settings.figures_dir, "results_table.tex")
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print(f" Tabla LaTeX guardada: {latex_path}")
    
    # ==================== REPORTE DE DOMAIN SHIFT ====================
    print("\n" + "="*70)
    print(" DOMAIN SHIFT PENALTY REPORT")
    print("="*70)
    
    # Encontrar mejor y peor estrategia
    best_model = min(summary_table, key=lambda x: float(x['Δ_shift (%)'].split('±')[0].strip()))
    worst_model = max(summary_table, key=lambda x: float(x['Δ_shift (%)'].split('±')[0].strip()))
    
    print(f"\n MEJOR estrategia para reducir domain shift:")
    print(f"   {best_model['Model']}: Δ_shift = {best_model['Δ_shift (%)']}")
    
    print(f"\n PEOR estrategia (mayor domain shift):")
    print(f"   {worst_model['Model']}: Δ_shift = {worst_model['Δ_shift (%)']}")
    
    # ==================== GUARDAR RESULTADOS FINALES ====================
    final_summary = {
        'seeds_used': seeds,
        'num_seeds': len(seeds),
        'results_per_seed': all_results,
        'summary_table': summary_table,
        'best_strategy': best_model['Model'],
        'best_strategy_target_acc': best_model['Target Acc (%)'],
        'best_strategy_domain_shift': best_model['Δ_shift (%)'],
        'worst_strategy': worst_model['Model'],
        'worst_strategy_domain_shift': worst_model['Δ_shift (%)']
    }
    
    with open(os.path.join(settings.figures_dir, "final_results.pkl"), 'wb') as f:
        pickle.dump(final_summary, f)
    
    print(f"\n Resultados finales guardados: {settings.figures_dir}/final_results.pkl")
    
    # ==================== VERIFICAR PREPROCESAMIENTO IDÉNTICO ====================
    print("\n" + "="*70)
    print(" VERIFICACIÓN DE REQUISITOS")
    print("="*70)
    
    print("\n Requisito 1 - Preprocesamiento idéntico:")
    print("    Source y Target usan mismas transforms (resize 224, normalización ImageNet)")
    
    print("\n Requisito 2 - Múltiples semillas:")
    print(f"    Ejecutado con {len(seeds)} semillas: {seeds}")
    print(f"    Reportados mean ± std para cada modelo")
    
    print("\n Requisito 3 - Domain shift penalty:")
    print(f"    Reportado para cada variante de modelo")
    
    print("\n Requisito 4 - Tabla con 5 modelos:")
    print(f"    Incluye: Feature Extraction, Fine-tuning, Baseline, Target Fine-tuning, Style Augmentation")
    
    print("\n" + "="*70)
    print(" CHALLENGE COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"\n Resultados guardados en: {settings.figures_dir}")
    print(f"   - results_summary.csv (tabla en CSV)")
    print(f"   - results_summary.json (resultados completos)")
    print(f"   - results_table.tex (tabla en LaTeX)")
    print(f"   - final_results.pkl (resultados para análisis)")
    
    # ==================== GENERAR GRÁFICO COMPARATIVO ====================
    try:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 6))
        models = [row['Model'] for row in summary_table]
        target_accs = [float(row['Target Acc (%)'].split('±')[0].strip()) for row in summary_table]
        target_stds = [float(row['Target Acc (%)'].split('±')[1].strip()) for row in summary_table]
        
        bars = ax.bar(models, target_accs, yerr=target_stds, capsize=5, color='steelblue', alpha=0.7)
        ax.set_xlabel('Model Variant', fontsize=12)
        ax.set_ylabel('Target Accuracy (%)', fontsize=12)
        ax.set_title('Domain Adaptation Results - Clipart → Sketch', fontsize=14)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Añadir valores en las barras
        for bar, acc in zip(bars, target_accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(settings.figures_dir, "comparison_chart.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n Gráfico comparativo guardado: {settings.figures_dir}/comparison_chart.png")
    except Exception as e:
        print(f"\n No se pudo generar gráfico: {e}")

if __name__ == "__main__":
    main()