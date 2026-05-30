# Challenge 7 - Transfer Learning: Few-Shot Classification, Neural Style Transfer, and Domain Shift Adaptation

## Group 6: Clipart -> Sketch | Clothing Categories

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains the complete implementation for Challenge 7 of the Machine Learning course at Universidad Distrital Francisco Jose de Caldas. The project demonstrates transfer learning techniques for domain adaptation between **Clipart** (source domain) and **Sketch** (target domain) across six clothing categories.

### Dataset Assignment

| Property | Value |
|----------|-------|
| Group | 6 |
| Source Domain | Clipart (flat colours, simplified contours) |
| Target Domain | Sketch (edge-only, monochromatic) |
| Categories | jacket, pants, shoe, sock, t-shirt, umbrella |
| Dataset | DomainNet (Peng et al., 2019) |

### Challenge Structure

```
Part A: Few-Shot Classification
    - Feature Extraction (frozen backbone)
    - Fine-tuning (unfrozen layer3 and layer4)

Part B: Neural Style Transfer
    - Gatys-style style transfer (VGG-19)
    - Generate 180 synthetic images (30 per class)

Part C: Domain Shift Adaptation
    - Baseline (no adaptation)
    - Target Fine-tuning (50 shots per class)
    - Style Transfer Augmentation
```

---

## Key Results

### Part A: Few-Shot Classification

| Strategy | Source Acc (%) | Target Acc (%) | Delta_shift (%) |
|----------|----------------|----------------|-----------------|
| Feature Extraction | 79.77 +- 1.20 | 53.30 +- 1.50 | 26.47 +- 1.30 |
| Fine-tuning | 89.74 +- 0.80 | 61.90 +- 1.10 | 27.84 +- 0.90 |

### Part C: Domain Adaptation

| Model Variant | Source Acc (%) | Target Acc (%) | Delta_shift (%) |
|---------------|----------------|----------------|-----------------|
| Baseline (No Adaptation) | 89.74 +- 0.80 | 61.90 +- 1.10 | 27.84 +- 0.90 |
| **Target Fine-tuning** | **76.20 +- 1.00** | **71.40 +- 1.20** | **4.80 +- 0.80** |
| Style Augmentation | 81.30 +- 0.90 | 65.70 +- 1.30 | 15.60 +- 1.10 |

> **Best Strategy**: Target Fine-tuning reduces domain shift from 27.84% to 4.80%.

---

## Repository Structure

```
challenge-7-group6/
├── src/
│   ├── __init__.py
│   ├── main.py                     # Main execution script (Parts A, B, C)
│   ├── run_part_b_c.py             # Run only Parts B and C
│   ├── generate_figures_fixed.py   # Generate all required figures
│   ├── debug_gradcam.py            # Debug Grad-CAM layer detection
│   ├── save_synthetic_images.py    # Export synthetic images as PNG
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── settings.py             # Configuration and hyperparameters
│   │   └── helpers.py              # Utility functions (seeds, loaders)
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py              # DomainNet dataset loader
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py           # ResNet-50 with feature extraction/fine-tuning
│   │   └── style_transfer.py       # Neural Style Transfer (VGG-19)
│   └── training/
│       ├── __init__.py
│       └── train_classifier.py     # Training loops and evaluation
├── data/
│   ├── clipart/                    # Source domain images
│   │   ├── jacket/
│   │   ├── pants/
│   │   ├── shoe/
│   │   ├── sock/
│   │   ├── t-shirt/
│   │   └── umbrella/
│   ├── sketch/                     # Target domain images
│   │   └── (same category structure)
│   └── synthetic_target/           # Generated stylized images (180 total)
│       ├── synthetic_data.pkl
│       ├── jacket/
│       ├── pants/
│       └── ...
├── models/
│   └── checkpoints/                # Saved model weights (.pt files)
│       ├── fine_tuning_seed_42_best.pt
│       ├── fine_tuning_seed_123_best.pt
│       └── fine_tuning_seed_456_best.pt
├── figures/                        # Generated visualizations
│   ├── fig_a_training_curves.png
│   ├── fig_b_style_transfer_gallery.png
│   ├── fig_c_gradcam_source.png
│   ├── fig_c_gradcam_target.png
│   ├── fig_d_tsne.png
│   ├── resultados_completos.txt
│   └── results_summary.csv
├── runs/                           # TensorBoard logs
│   ├── feature_extraction/
│   ├── fine_tuning/
│   ├── target_fine_tuning/
│   └── style_augmentation/
├── notebooks/                      # Jupyter notebooks (optional)
├── pyproject.toml                  # Poetry dependencies
├── README.md
├── CHECKLIST.md                    # Grading checklist
└── paper/                          # IEEE paper (LaTeX)
    └── paper.pdf
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for fast training)
- 20 GB free disk space (for DomainNet dataset)

### Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/challenge-7-group6.git
cd challenge-7-group6
```

### Step 2: Install dependencies with Poetry

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Step 3: Install PyTorch with CUDA (for GPU)

```bash
# For CUDA 11.8
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only
poetry run pip install torch torchvision torchaudio
```

### Step 4: Download DomainNet Dataset

Download the Clipart and Sketch domains from [DomainNet](http://ai.bu.edu/M3SDA/). Expected structure after download:

```
data/
├── clipart/
│   ├── jacket/
│   ├── pants/
│   ├── shoe/
│   ├── sock/
│   ├── t-shirt/
│   └── umbrella/
└── sketch/
    ├── jacket/
    ├── pants/
    ├── shoe/
    ├── sock/
    ├── t-shirt/
    └── umbrella/
```

---

## Usage

### Quick Start: Run Everything

```bash
# Complete pipeline (Parts A, B, and C)
poetry run python -m src.main
```

This will:
1. Train feature extraction and fine-tuning models (Part A)
2. Generate 180 synthetic stylized images (Part B)
3. Evaluate three adaptation strategies (Part C)
4. Save models, results, and TensorBoard logs

### Run Only Parts B and C (skip Part A)

```bash
poetry run python src/run_part_b_c.py
```

### Generate All Figures

```bash
poetry run python src/generate_figures_fixed.py
```

This generates:
- `fig_a_training_curves.png` - Training and validation curves
- `fig_b_style_transfer_gallery.png` - Style transfer gallery (6 examples)
- `fig_c_gradcam_source.png` - Grad-CAM for Clipart
- `fig_c_gradcam_target.png` - Grad-CAM for Sketch
- `fig_d_tsne.png` - t-SNE visualization

### Export Synthetic Images as PNG

```bash
poetry run python src/save_synthetic_images.py
```

### View TensorBoard

```bash
tensorboard --logdir=./runs --port=6006
# Open http://localhost:6006
```

### Run with Different Random Seeds

The code automatically runs with seeds `[42, 123, 456]` as required. To modify, edit `src/main.py`:

```python
seeds = [42, 123, 456]  # Replace with your seeds
```

---

## Configuration

Edit `src/utils/settings.py` to adjust hyperparameters:

```python
@dataclass
class Settings:
    # Data settings
    shots_per_class: int = 50       # Few-shot budget
    batch_size: int = 32

    # Training settings
    fe_lr: float = 1e-3             # Feature extraction learning rate
    fe_epochs: int = 30
    ft_lr: float = 1e-4             # Fine-tuning learning rate
    ft_epochs: int = 30

    # Style Transfer
    alpha: float = 0.0001           # Content weight (lower = more style)
    beta: float = 1e10              # Style weight (higher = more style)
    num_style_images_per_class: int = 30

    # Domain Adaptation
    target_finetune_lr: float = 1e-4
    target_finetune_epochs: int = 15
```

---

## Results

**Training Curves**: The fine-tuning model achieves 94.4% validation accuracy on Clipart at epoch 15, with training accuracy reaching 100% thereafter.

**Style Transfer Gallery**: Generated stylized images preserve clipart content while adopting sketch-style edge-based monochromatic appearance.

**Grad-CAM Attention**:
- Correct predictions: attention focuses on semantically meaningful regions (shoe silhouette, jacket collar)
- Incorrect predictions: attention is dispersed or focused on irrelevant background features

**t-SNE Visualization**: Feature distributions show clear separation before adaptation and substantial overlap after target fine-tuning, confirming domain-invariant feature learning.

---

## Reproducibility

All experiments use fixed random seeds:

| Library | Seeds |
|---------|-------|
| Python random | 42, 123, 456 |
| NumPy | 42, 123, 456 |
| PyTorch | 42, 123, 456 |
| CUDA | 42, 123, 456 |

**Environment**:
```
Python:      3.9+
PyTorch:     2.1.0
TorchVision: 0.16.0
CUDA:        11.8 (optional)
```

---

## Paper

The IEEE-formatted paper is available in the `paper/` directory:

```
paper/
├── paper.tex   # LaTeX source
└── paper.pdf   # Compiled PDF
```

Sections: Abstract, Introduction, Related Work, Methodology, Experimental Results, Discussion, Conclusions, References.

---

## Grading Checklist

See `CHECKLIST.md` for the full checklist. Key items:

- Dataset pair: Clipart -> Sketch, Clothing categories
- Backbone: ResNet-50 with ImageNet weights
- Trainable parameters: 23.5M (fine-tuning), 25K (feature extraction)
- Source/Target accuracy with mean +- std over 3 seeds
- Domain shift penalty for best Part A and Part C models
- alpha/beta ratio: 1e-4, visual quality assessment
- Best adaptation strategy explanation (200 words max)

---

## Troubleshooting

**CUDA not available**
```bash
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Out of memory** — reduce batch size in `settings.py`:
```python
batch_size: int = 16  # instead of 32
```

**Style transfer too slow** — reduce images or steps in `settings.py`:
```python
num_style_images_per_class: int = 10  # instead of 30
style_steps: int = 150                # instead of 300
```

**Missing synthetic images** — run Part B first or check the path:
```bash
ls data/synthetic_target/synthetic_data.pkl
```

---

## References

1. Peng, X., et al. (2019). "Moment matching for multi-source domain adaptation." ICCV.
2. He, K., et al. (2016). "Deep residual learning for image recognition." CVPR.
3. Gatys, L. A., et al. (2016). "Image style transfer using convolutional neural networks." CVPR.
4. Tan, C., et al. (2018). "A survey on deep transfer learning." ICANN.
5. Ganin, Y., et al. (2016). "Domain-adversarial training of neural networks." JMLR.

---

## Authors

| Name | Role |
|------|------|
| Giovanni Vargas | Lead Developer |
| Samuel Casas | Developer |
| Nahin Penaranda | Developer |

**Supervisor**: Prof. Carlos Andres Sierra, M.Sc.  
**Institution**: Universidad Distrital Francisco Jose de Caldas  
**Program**: Computer Engineering  
**Course**: Machine Learning

---

## License

This project is for academic purposes as part of the Machine Learning course requirement.

## Acknowledgments

- Prof. Carlos Andres Sierra for guidance and dataset assignment
- DomainNet dataset authors for the benchmark
- PyTorch and TorchVision teams for the frameworks

---

Repository: https://github.com/yourusername/challenge-7-group6  
Paper: IEEE Format — see `paper/paper.pdf`  
Video: Demo Video (7-10 minute demonstration)
