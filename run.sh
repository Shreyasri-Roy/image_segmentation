#!/bin/bash

# 1. Exit on any error
set -e

echo "📦 Setting up virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo "📁 Preparing the dataset..."
python prepare_dataset.py

echo "🚀 Starting training..."
python train_unet.py

echo "✅ Done! Your model is trained and logs are available on wandb.ai"
