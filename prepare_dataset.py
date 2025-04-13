
"""Name : Shreyasri Roy

Prepares dataset, saves it in the following form
├── dataset/
│   ├── annotations
│   │   ├── trimaps,etc
│   ├── images
│   │   ├── Abyssinian_1.jpg,etc
│   ├── masks/
│   │   └── Abyssinian_1.png,etc
"""

!pip install opencv-python matplotlib

import requests
import tarfile
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/masks", exist_ok=True)
os.makedirs("scripts")

DATASET_PATH = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ANNOTATIONS_PATH ="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
ROOT_DIR = Path("dataset")
IMAGE_DIR = ROOT_DIR / "images"
MASKS_DIR = ROOT_DIR / "masks"

ROOT_DIR.mkdir(exist_ok=True)

def download_and_extract(url, dest):
    """
    Downloads and extracts a .tar.gz file from a URL into the given destination folder.

    Parameters:
        url (str): The URL of the .tar.gz file to download.
        dest (Path or str): The destination directory to extract the contents into.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    # Get filename from URL (e.g., images.tar.gz)
    fname = url.split("/")[-1]
    target_folder_name = fname.replace(".tar.gz", "")
    target_path = dest / target_folder_name

    # If folder already extracted, skip
    if target_path.exists() and any(target_path.iterdir()):
        print(f"{target_folder_name}/ already exists — skipping download.")
        return

    print(f"Downloading {fname} from {url}...")
    response = requests.get(url, stream=True)
    with open(fname, 'wb') as f:
        f.write(response.content)

    print(f"Extracting {fname} into {dest}/ ...")
    with tarfile.open(fname, "r:gz") as tar:
        tar.extractall(path=dest)

    os.remove(fname)
    print(f"{fname} download and extraction complete.")

def convert_masks():
  #1. Set the path where  the original masks lie
  seg_dir = ROOT_DIR / "annotations" / "trimaps"
  #2. Make sure that the directory where the images will be saved exists
  MASKS_DIR.mkdir(exist_ok=True)
  #3. Iterate through all the images in the .png masks files
  for images in seg_dir.glob("*.png"):
    #4.1 Check if the file is a hidden system file starting with "._"
    #    If it is, skip processing this file.
    if images.name.startswith("._"):
        continue
    #4.2 Opening the Images using the PIL(Python Imaging Library)
    img = Image.open(images)
    #5. Converting the Image to numpy array so that we can manipulate the values easily
    img = np.array(img)
    #6. Original masks has labels 1:pet , 2:border, 3: background
    #   We subtract 1 from everything and make it 0-based
    #   Now its 0:pet, 1:border, 2:background
    img = img-1
    #7. Re-checking that there are no negative pixel values, if any pixel<0, we make
    #   it 0 i.e. background
    img[img<0] = 0
    #8. We convert the dtype to uint8 so that pixel values are 0-255, then convert it back
    #   to Image format and save it in the required path
    Image.fromarray(img.astype(np.uint8)).save(MASKS_DIR / images.name)

def main():
  if not IMAGE_DIR.exists() or len(os.listdir(IMAGE_DIR)) == 0:
        download_and_extract(DATASET_PATH, ROOT_DIR)
  if not (ROOT_DIR / "annotations").exists() or len(os.listdir((ROOT_DIR / "annotations"))) == 0:
        download_and_extract(ANNOTATIONS_PATH, ROOT_DIR)
  convert_masks()

if __name__ == "__main__":
    main()

