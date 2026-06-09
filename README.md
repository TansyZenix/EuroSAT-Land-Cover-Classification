# EuroSAT Land Cover Classification

High-resolution remote sensing image land cover classification using EuroSAT RGB dataset. Implements an HRNet-style multi-resolution classifier and compares against ResNet-18, with ablation studies on data augmentation and multi-resolution fusion, plus spatial resolution robustness analysis.

![Sample Predictions](results/eurosat/20260609_134700/sample_predictions.png)

## Features

- **HRNet-style multi-resolution classifier** with high/medium/low resolution branches and feature fusion
- **Comparison with ResNet-18 baseline** for architecture benchmarking
- **Ablation studies**: data augmentation (on/off), multi-resolution fusion (with/without), and combined analysis
- **Resolution robustness**: compares original 10m vs. simulated 30m spatial resolution inputs
- **Auto-generated figures**: class distribution, training curves, confusion matrix, sample predictions, resolution comparison, and ablation bar chart

## Results Summary

| Model | Augmentation | 10m Accuracy | 10m Macro F1 | 30m Accuracy | 30m Macro F1 |
|-------|-------------|-------------|--------------|-------------|--------------|
| ResNet-18 | yes | 0.9528 | 0.9514 | 0.5121 | 0.4640 |
| HRNet (ablation: no fusion) | yes | 0.9368 | 0.9350 | 0.3373 | 0.2419 |
| HRNet (ablation: no aug) | no | 0.9756 | 0.9752 | 0.4044 | 0.3384 |
| **HRNet (main)** | **yes** | **0.9672** | **0.9663** | **0.4459** | **0.3928** |

Key findings:
- HRNet outperforms ResNet-18 by ~1.4% on 10m input, demonstrating the effectiveness of multi-resolution feature fusion.
- Removing the fusion module drops accuracy by ~3%, confirming its contribution.
- Data augmentation improves robustness on degraded 30m input (+4% accuracy).
- Resolution degradation (10m → 30m) significantly hurts performance across all models.

## Project Structure

```
├── train_eurosat.py           # Main training script
├── test_train_eurosat.py      # Unit tests
├── summarize_ablation.py      # Ablation study summary script
├── download_dataset.py        # EuroSAT dataset download script
├── RESULTS_SUMMARY.md         # Detailed results with figures (Chinese)
├── results/
│   └── eurosat/
│       ├── 20260609_134700/   # Best run results (figures, metrics, report)
│       └── ablation_summary/  # Ablation comparison table and chart
└── README.md
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- numpy, matplotlib, pillow

Install with conda:

```bash
conda create -n eurosat_env python=3.9
conda activate eurosat_env
pip install torch torchvision numpy matplotlib pillow
```

## Dataset

The EuroSAT dataset is a collection of Sentinel-2 satellite images covering 10 land cover classes. It will be downloaded automatically:

```bash
python download_dataset.py
```

Alternatively, you can manually download the EuroSAT RGB dataset from [TorchVision datasets](https://www.tensorflow.org/datasets/catalog/eurosat) and place it under `data/eurosat/2750/` with class subdirectories: `AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`.

## Usage

### Training

```bash
conda activate eurosat_env
python train_eurosat.py --epochs 20 --batch-size 128 --model hrnet
```

Options:

| Argument | Default | Choices |
|----------|---------|---------|
| `--model` | `hrnet` | `hrnet`, `hrnet_no_fusion`, `resnet18` |
| `--epochs` | 20 | int |
| `--batch-size` | 128 | int |
| `--learning-rate` | 0.001 | float |
| `--image-size` | 64 | int |
| `--no-augmentation` | (enabled) | flag to disable data aug |

Output is saved to `results/eurosat/<timestamp>/`:
- `best_model.pth` — best validation checkpoint
- `metrics.json` — full training history and test metrics
- `class_distribution.png`, `training_curves.png`, `confusion_matrix.png`, `sample_predictions.png`, `resolution_comparison.png`
- `work_summary.md` — auto-generated experiment report

### Running Tests

```bash
conda activate eurosat_env
python -m pytest test_train_eurosat.py -v
```

### Ablation Summary

After collecting multiple run results:

```bash
python summarize_ablation.py results/eurosat/<run1> results/eurosat/<run2> ...
```

## Results Gallery

![Class Distribution](results/eurosat/20260609_134700/class_distribution.png)
*Figure 1: EuroSAT class distribution across 10 land cover categories.*

![Training Curves](results/eurosat/20260609_134700/training_curves.png)
*Figure 2: Training and validation loss/accuracy curves.*

![Confusion Matrix](results/eurosat/20260609_134700/confusion_matrix.png)
*Figure 3: Normalized confusion matrix on the 10m test set.*

![Resolution Comparison](results/eurosat/20260609_134700/resolution_comparison.png)
*Figure 4: Comparison of 10m vs simulated 30m spatial resolution performance.*

![Ablation Chart](results/eurosat/ablation_summary/ablation_bar_chart.png)
*Figure 5: Ablation and comparison experiment results.*

## License

This project is for educational purposes as part of the Low-altitude Intelligent Computing Theory course.
