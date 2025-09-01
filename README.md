# CIFAR-100 Image Classification

This project implements CIFAR-100 image classification with TensorFlow/Keras.  
It compares a small CNN with **EfficientNetV2B0** transfer learning, including the proper resize→preprocess pipeline, staged fine-tuning, and evaluation.
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kiko1992-creator/cifar100-image-classification/blob/main/notebooks/image%20classification%20final.ipynb)

---
Get the latest trained weights from the **[Releases](https://github.com/kiko1992-creator/cifar100-image-classification/releases/latest)** page:

- **Transfer model (EfficientNetV2B0)** — [final_efficientnetv2b0.keras](https://github.com/kiko1992-creator/cifar100-image-classification/releases/latest/download/final_efficientnetv2b0.keras)
- **Baseline CNN** — [final_cifar100_cnn.keras](https://github.com/kiko1992-creator/cifar100-image-classification/releases/latest/download/final_cifar100_cnn.keras)

**Open in Colab**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kiko1992-creator/cifar100-image-classification/blob/main/notebooks/image%20classification%20final.ipynb)

---
### Inference (CLI)

Download a weight file from **[Releases](https://github.com/kiko1992-creator/cifar100-image-classification/releases/latest)** (see links above), then run:
```bash
pip install -r requirements.txt

# EfficientNetV2B0 (transfer model)
python predict.py --image path/to/image.jpg --weights final_efficientnetv2b0.keras --topk 5

# Or the small CNN baseline
python predict.py --image path/to/image.jpg --weights final_cifar100_cnn.keras --topk 5

---
## Repository layout

notebooks/            # Training notebooks
results/              # Metrics, plots, confusion matrix, reports
├─ metrics_transfer.json
├─ metrics_cnn.json
├─ classification_report.txt
└─ confusion_matrix.png
releases/             # Model weights (via GitHub Releases)
requirements.txt      # Dependencies
predict.py            # Inference script (CLI)
LICENSE               # MIT License

---
## Quickstart

```bash
pip install -r requirements.txt

# EfficientNetV2B0 (transfer) model
python predict.py path/to/image.jpg --weights final_efficientnetv2b0.keras --topk 5

# Or the small CNN baseline
python predict.py path/to/image.jpg --weights final_cifar100_cnn.keras --topk 5

## Load a model in Python

import tensorflow as tf
m = tf.keras.models.load_model("final_efficientnetv2b0.keras")  # or "final_cifar100_cnn.keras"

## Results
EfficientNetV2B0 (transfer learning) significantly outperformed the small CNN baseline.

Top-1 accuracy: ~78.2%

Top-5 accuracy: ~95.9%

Artifacts (see the results/ folder):

results/metrics_transfer.json

results/metrics_cnn.json

results/classification_report.txt

Confusion Matrix (normalized)

Notes
Inputs are resized to 224×224, then preprocessed with the EfficientNetV2 preprocess layer before the backbone.

Fine-tuning uses staged unfreezing with a lower learning rate.

Labels are in labels.txt.

License
Released under the MIT License.
