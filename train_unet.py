``` NOTE: First run "prepare_dataset.py" and create the dataset in the form of folder shown below and then run this code
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
```
"""##**Task 2**"""

!pip install torch torchvision matplotlib opencv-python

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn

ROOT_DIR = Path("dataset")
IMAGE_DIR = ROOT_DIR / "images"
MASKS_DIR = ROOT_DIR / "masks"
# Creating Dataset Class
class Pet_Dataset(Dataset):
  def __init__(self, image_dir, mask_dir, size=(128,128)):
      self.image_dir = image_dir
      self.mask_dir = mask_dir
      self.image_transforms = T.Compose([
          T.Resize(size),
          T.ToTensor()
      ])
      self.size = size
      self.images = sorted([f for f in os.listdir(self.image_dir) if f.endswith(".jpg")])
      self.mask_transforms = T.Compose([
          T.Resize(size, interpolation=Image.NEAREST),
          T.Lambda(lambda x: torch.from_numpy(np.array(x)).long())
      ])
  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
      image_name = self.images[index]
      image_path = os.path.join(self.image_dir, image_name)
      mask_path = os.path.join(self.mask_dir, image_name.replace(".jpg",".png"))
      image = Image.open(image_path).convert("RGB")
      mask = Image.open(mask_path)

      image = self.image_transforms(image)
      mask = self.mask_transforms(mask)

      return image,mask

dataset = Pet_Dataset(IMAGE_DIR, MASKS_DIR)

import torch
from torch.utils.data import random_split

# Creating Train, Validation, and Test Datasets and DataLoader

# Define the split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Calculate the sizes of each split
dataset_size = len(dataset)
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size

# Split the dataset
train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
)

# Print the sizes of each split
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=8,
    shuffle=True
)

# Defining the Model: UNet
class UNet(nn.Module):
  def __init__(self,n_classes):
    super(UNet,self).__init__()
    def CBR(in_channels,out_channels):
      return nn.Sequential(
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=3,padding=1),
          nn.ReLU(),
          nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
          nn.ReLU()
      )
    self.enc1 = CBR(3,64)
    self.enc2 = CBR(64,128)
    self.pool = nn.MaxPool2d(kernel_size=(2,2))
    self.up = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
    self.dec1 = nn.Conv2d(128,64,kernel_size=3,padding=1)
    self.final = nn.Conv2d(64,n_classes,1)

  def forward(self, x):
    e1 = self.enc1(x)
    # print(f"Shape e1:{e1.shape}")
    e2 = self.enc2(self.pool(e1))
    # print(f"Shape e2:{e2.shape}")

    d1 = self.up(e2)
    # print(f"Shape after passing up layer:{d1.shape}")
    d1 = torch.cat([d1,e1],dim=1)
    # print(f"Shape after concatenation:{d1.shape}")
    d1 = self.dec1(d1)
    # print(f"Shape passing decoder layer:{d1.shape}")
    return self.final(d1)
# Setting up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = UNet(n_classes=3).to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Metrics to be computed
def compute_iou(pred, target, n_classes=3):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  for cls in range(n_classes):
    pred_indices = (pred == cls)
    target_indices = (target == cls)
    intersection = (pred_indices & target_indices).sum().item()
    union = (pred_indices | target_indices).sum().item()
    if union == 0:
      ious.append(float('nan'))
    else:
      ious.append(intersection/union)

  return ious
def pixel_accuracy(pred, target):
  correct = (pred == target).sum().item()
  total = target.numel()
  return correct/total

def dice_score(pred, target, n_classes=3):
  dice = []

  for cls in range(n_classes):
    pred_cls = (pred == cls).float()
    target_cls = (target == cls).float()
    intersection = (pred_cls * target_cls).sum()
    union = pred_cls.sum() + target_cls.sum()
    if union == 0:
      dice.append(float('nan'))
    else:
      dice.append((2. * intersection)/union)

  return dice
# Function to evaluate the trained model
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())

    preds_all = torch.cat(all_preds, dim=0)
    targets_all = torch.cat(all_targets, dim=0)

    ious = compute_iou(preds_all, targets_all)
    dices = dice_score(preds_all, targets_all)
    acc = pixel_accuracy(preds_all, targets_all)

    return ious, dices, acc, preds_all, targets_all

!pip install wandb

import wandb
wandb.login()
wandb.init(
    project="oxford-pet-segmentation",
    name="unet-128x128-split",
    config={
        "epochs": 50,
        "batch_size": 8,
        "image_size": "128x128",
        "architecture": "UNet",
        "loss": "CrossEntropy",
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "num_classes": 3,
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_classes=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(wandb.config.epochs):
    model.train()
    train_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_loss/=len(train_loader)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")

    # 🎯 Validation Step
    ious, dices, acc, val_preds, val_targets = evaluate_model(model, val_loader, device)

    print(f"Val Pixel Accuracy: {acc:.4f}")
    print(f"Val IoU: {ious}")
    print(f"Val Dice: {dices}")

    # 🟩 WandB Logging
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_pixel_accuracy": acc,
        "val_iou_class_0": ious[0],
        "val_iou_class_1": ious[1],
        "val_iou_class_2": ious[2],
        "val_dice_class_0": dices[0],
        "val_dice_class_1": dices[1],
        "val_dice_class_2": dices[2],
    })

    # 🖼️ Log one sample from validation
    def mask_to_rgb(mask_tensor):
        mask = mask_tensor.numpy()
        colors = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0]])
        return colors[mask]

    val_sample_image, val_sample_mask = next(iter(val_loader))
    val_sample_image = val_sample_image[0].to(device).unsqueeze(0)
    val_pred = torch.argmax(model(val_sample_image), dim=1).squeeze().cpu()
    val_image = val_sample_image.squeeze().permute(1, 2, 0).cpu().numpy()
    val_true = val_sample_mask[0]

    wandb.log({
        "Val/Input": wandb.Image(val_image, caption="Input"),
        "Val/Prediction": wandb.Image(mask_to_rgb(val_pred), caption="Prediction"),
        "Val/True Mask": wandb.Image(mask_to_rgb(val_true), caption="Ground Truth")
    })

print("🔍 Running final evaluation on test set...")
test_ious, test_dices, test_acc, _, _ = evaluate_model(model, test_loader, device)

print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test IoU: {test_ious}")
print(f"Test Dice: {test_dices}")

wandb.log({
    "test_pixel_accuracy": test_acc,
    "test_iou_class_0": test_ious[0],
    "test_iou_class_1": test_ious[1],
    "test_iou_class_2": test_ious[2],
    "test_dice_class_0": test_dices[0],
    "test_dice_class_1": test_dices[1],
    "test_dice_class_2": test_dices[2],
})

#7. Predictions that demonstrates model inference with example predictions
model.eval()
with torch.inference_mode():
  sample_img, sample_mask = dataset[0]
  pred1 = model(sample_img.unsqueeze(0).to(device))
  pred = torch.argmax(pred1.squeeze(0),dim=0).cpu().numpy()
plt.subplot(1,3,1)
plt.imshow(sample_img.permute(1,2,0))
plt.title("Original Image")
plt.subplot(1,3,2)
plt.imshow(sample_mask)
plt.title("Original Mask")
plt.subplot(1,3,3)
plt.imshow(pred)
plt.title("Predicted Mask")
