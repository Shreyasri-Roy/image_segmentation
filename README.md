# 🐾 Oxford-IIIT Pet Image Segmentation (UNet, PyTorch)

This project prepares the Oxford-IIIT Pet Dataset and trains a UNet-based segmentation model to segment pet regions from images. It includes full preprocessing, model training, evaluation, and metric logging using PyTorch and Weights & Biases (WandB).
## Processed Data

- Dataset: Oxford-IIIT Pet Dataset
- Total Images: 3,686
- Each image has a corresponding processed mask with 3 classes:
  - 0 = Background
  - 1 = Pet
  - 2 = Border

This satisfies the assignment’s requirement to process 3,000 to 8,000 images.

---
### FULL CODE PIPELINE IS THERE IN `full_code_pipeline.pynb`: YOU CAN JUST RUN IT IN COLAB/ JUPYTER NOTEBOOK AND DO NOTHING ELSE.
### Otherwise follow below instructions:
## 📂 Project Structure

    📦 project-root/
    ├── dataset/
    │   ├── annotations
    │   │   ├── trimaps
    │   │   └── ....
    │   ├── images
    │   │   ├── Abyssinian_1.jpg
    │   │   └── ....
    │   ├── masks
    │   │   ├── Abyssinian_1.png
    │   │   └── ...
    ├── prepare_dataset.py
    ├── train_unet.py (or .ipynb)
    ├── report.md
    ├── requirements.txt
    ├── run.sh
    ├── wanb report.csv
    ├── full_code_pipeline.pynb
    └── README.md


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
python prepare_dataset.py
```
Prepares dataset folder of the form:
## 📂 Dataset Folder Structure

    📦 project-root/
    ├── dataset/
    │   ├── annotations
    │   │   ├── trimaps
    │   │   └── ....
    │   ├── images
    │   │   ├── Abyssinian_1.jpg
    │   │   └── ....
    │   ├── masks
    │   │   ├── Abyssinian_1.png
    │   │   └── ...

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
python train_unet.py
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
 run.sh  -> Includes instructions to reproduce the results on a Linux environment.
---
### 🙌 Acknowledgements 
* **Dataset:** Oxford-IIIT Pet Dataset
* * **Model Architecture:** UNet
* *  **Experiment Logging:** Weights & Biases Report is present in **`wandb report.csv`**
* * **Experiment Logging:** [Weights & Biases]([https://wandb.ai](https://wandb.ai/shreyasriroy-indian-institute-of-science/oxford-pet-segmentation/runs/imp7h51g?nw=nwusershreyasriroysrmr))
---
