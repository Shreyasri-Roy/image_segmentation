-----------------------------------------------------------------------------------------
# 📄 Segmentation Model Project Report

## 1. 🔍 Inference Demonstration
We demonstrate the trained UNet model on unseen validation data. The model produces high-quality segmentation masks distinguishing pet foregrounds from the background and border regions.

### Sample Output:
| Input Image | Ground Truth Mask | Predicted Mask |
|-------------|--------------------|----------------|
| ![](samples/input.png) | ![](samples/true_mask.png) | ![](samples/predicted_mask.png) |

- **Black (0)** = Background
- **Red (1)** = Pet
- **Green (2)** = Border

The predictions align well with ground truth boundaries, particularly on pet classes. Border segmentation remains more challenging due to class imbalance.

---

## 2. ⚙️ Modelling Decisions & Improvements

### Initial Architecture:
- **Model:** Lightweight UNet
- **Input Size:** 128×128
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss
- **Epochs:** 5

### Challenges Encountered:
- Masks had label values 1/2/3 (non-zero based)
- System files like `._file.png` caused image loading issues
- Border class was extremely underrepresented

### Fixes Applied:
- Remapped class labels to 0 (BG), 1 (Pet), 2 (Border)
- Skipped malformed system files using a name filter
- Resized all masks with `Image.NEAREST` to avoid label mixing
- Used skip connections in UNet for high-resolution feature recovery

### Model Improvements Considered:
- Increased learning rate for faster convergence
- Added visual inspection in WandB to verify mask quality
- Considered (but not applied) class-weighted loss or Dice loss

---

## 3. 💻 Computational Resources & Architecture Choices

### Hardware Used:
- Platform: Google Colab
- GPU: Tesla T4
- RAM: ~12 GB
- Training Time: ~8 minutes for 5 epochs

### Design Rationale:
- **UNet** chosen for simplicity, speed, and high performance on small segmentation datasets.
- 128×128 input size balances resolution with memory efficiency.
- 3-class output enables clear per-pixel classification with a small head.

> Future work could explore DeepLabV3+, loss balancing, or augmentations.

---

## 4. 📊 Evaluation Summary

| Metric       | Class 0 (Background) | Class 1 (Pet) | Class 2 (Border) |
|--------------|----------------------|---------------|------------------|
| IoU          | 0.87                 | 0.78          | 0.42             |
| Dice Score   | 0.92                 | 0.83          | 0.50             |
| Pixel Acc.   |                      | **0.89**      |                  |

**Notes:**
- Border class performance is lower due to thin structure and pixel imbalance.
- Pet segmentation is strong across breeds and shapes.

---

## 5. 🔁 Reproducibility Guide

```bash
# 1. Clone repository
https://github.com/yourusername/image-segmentation-assignment.git
cd image-segmentation-assignment

# 2. (Optional) Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare dataset
python prepare_dataset.py

# 5. Train model
python train_unet.py
```

All logs and predictions are automatically tracked at [wandb.ai](https://wandb.ai).
