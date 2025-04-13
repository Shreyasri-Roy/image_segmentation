# 🐾 Oxford-IIIT Pet Image Segmentation (UNet, PyTorch)

This project prepares the Oxford-IIIT Pet Dataset and trains a UNet-based segmentation model to segment pet regions from images. It includes full preprocessing, model training, evaluation, and metric logging using PyTorch and Weights & Biases (WandB).

---

## 📂 Project Structure
 ├── dataset/ │ ├── train/images/ │ ├── train/masks/ │ ├── val/images/ │ ├── val/masks/ │ ├── test/images/ │ └── test/masks/ ├── scripts/ │ └── prepare_dataset.py ├── training/ │ └── train_unet.py (or .ipynb) ├── report.md ├── requirements.txt └── README.md

---

## ⚙️ Task 1: Dataset Preparation

- Downloads the Oxford-IIIT Pet Dataset.
- Converts `trimap` masks (original labels: 1=border, 2=pet, 3=background) to standard 0-based format:
  - 0 = Background
  - 1 = Pet
  - 2 = Border
- Resizes all images and masks to 128×128.
- Splits data into **train / validation / test** sets.
- Output masks are saved in `.png` format for fast loading.

Run:
```bash
python scripts/prepare_dataset.py
```
---

### ⚠️ Edge Case Handling

- Trimap remapping: `(1, 2, 3) → (2, 1, 0)`
- Labels outside `[1, 3]` clipped to background (`0`)
- Skips macOS system files like `._*.png`
- Masks resized using `Image.NEAREST` to avoid label interpolation
- Ensures one-to-one correspondence of image-mask pairs

---
## 🤖 Task 2: UNet Model Training

Trains a **UNet** segmentation model using PyTorch. Logs training progress and performance metrics to **Weights & Biases (wandb.ai)**. Visualizes model predictions at each epoch.

### 🔧 Training Specs

- **Optimizer:** Adam  
- **Loss Function:** CrossEntropy  
- **Input Size:** 128×128  
- **Epochs:** 5  
- **Hardware:** GPU supported (e.g. Google Colab)

### ✅ Run

```bash
python training/train_unet.py
```

---
## 📈 Evaluation Metrics
---

Computed on validation and test sets:

- ✅ Pixel Accuracy  
- 📊 Intersection over Union (IoU)  
- 🧪 Dice Score  

All metrics are automatically logged and visualized using **WandB**.

---

## 🚀 Reproducibility
---

To install dependencies:

```bash
pip install -r requirements.txt
```
---
### 🙌 Acknowledgements 
* **Dataset:** Oxford-IIIT Pet Dataset
* * **Model Architecture:** UNet
* * **Experiment Logging:** [Weights & Biases](https://wandb.ai)
---
