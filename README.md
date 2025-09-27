# A1-CS452: CNN Baselines for Facial Affect Recognition

This repository contains my submission for **Deep Learning (CS452) Assignment 1**.  
The task was to implement and compare convolutional neural network (CNN) baselines for **multi-task facial affect recognition**:

1. **Categorical facial expression classification** (8 classes)  
2. **Continuous valence & arousal regression** (values in [-1, 1])  

I trained four CNN backbones — **ResNet18, EfficientNet-B0, VGG16, MobileNetV2** — in a multi-task setup with shared backbone and dual heads.

---

## Requirements

Experiments were run in **Google Colab GPU** environment (Tesla T4).  
Main dependencies:

- **Python** 3.12  
- **PyTorch** >= 2.0  
- **torchvision**  
- **scikit-learn**  
- **numpy, pandas**  
- **matplotlib / seaborn**  
- **tqdm**

Install locally:

```bash
pip install torch torchvision scikit-learn matplotlib seaborn tqdm
```

---

## Dataset

- **Samples:** ~3999 matched samples (image + expression + valence + arousal)  
- **Image preprocessing:** resized to 224×224 and normalized  
- **Splits:** 70% train / 15% validation / 15% test  
- **Each item:**  

```python
(image_tensor, expression_label, valence, arousal)
```

> **Note:** Dataset files (`.npy` and images) are **not included** in this repository.

---

## Clone and Install

```bash
git clone https://github.com/<your-username>/A1-CS452.git
cd A1-CS452
```

---

## Run in Colab

Open `notebooks/assignment.ipynb` in Google Colab and enable GPU.  
The notebook contains **modular cells** for:

- Dataset loading  
- Model definitions  
- Training loop  
- Evaluation with full metrics  
- Visualization & analysis  

---

## Results

- **VGG16** → Best classification metrics (Accuracy / F1 / AUC)  
- **MobileNetV2** → Best regression performance (RMSE, CORR, CCC)  
- **ResNet18** & **EfficientNet-B0** → Similar performance across both tasks  

---

## Notes

- Training limited to **10 epochs** due to Colab GPU constraints. Longer runs and stronger augmentation would improve results.  
- Loss weighting between classification and regression fixed at **α = 0.5**; tuning this is future work.  
- **Krippendorff’s Alpha** approximated with a nominal variant.
