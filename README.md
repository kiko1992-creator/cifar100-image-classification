# CIFAR-100 Image Classification
## Weights & Releases

Get the latest trained weights from the **[Releases](../../releases/latest)** page:
- [`final_efficientnetv2b0.keras`](../../releases/latest/download/final_efficientnetv2b0.keras) — transfer model (EfficientNetV2B0)
- [`final_cifar100_cnn.keras`](../../releases/latest/download/final_cifar100_cnn.keras) — baseline CNN

CNN baseline + EfficientNetV2B0 transfer learning in Keras/TensorFlow.
- Training notebook lives in `notebooks/`
- Metrics & plots are in `results/`
- Large model weights will be attached in **Releases** (GitHub limits uploads >25MB in the repo).

## Quickstart
```bash
pip install -r requirements.txt

# EfficientNetV2B0 transfer model
python predict.py path/to/image.jpg --weights final_efficientnetv2b0.keras

# Or the small CNN baseline
python predict.py path/to/image.jpg --weights final_cifar100_cnn.keras
