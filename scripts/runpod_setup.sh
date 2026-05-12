#!/bin/bash
# RunPod setup script for BirdCLEF 2026 training
set -e

echo "=== Step 1: Install dependencies ==="
pip install kaggle librosa -q

echo "=== Step 2: Kaggle credentials ==="
mkdir -p ~/.kaggle
# KAGGLE_API_TOKEN uses Bearer auth (required for KGAT_ tokens)
echo "Set KAGGLE_API_TOKEN env var before running kaggle commands"
# export KAGGLE_API_TOKEN="your_token_here"

echo "=== Step 3: Download competition data ==="
mkdir -p /workspace/data
cd /workspace/data

# Download (requires KAGGLE_API_TOKEN to be set)
kaggle competitions download -c birdclef-2026
unzip -q birdclef-2026.zip -d birdclef-2026
echo "Data downloaded: $(du -sh birdclef-2026 | cut -f1)"

echo "=== Step 4: Set up directory structure ==="
# Mirror Kaggle's /kaggle/input structure so notebooks work without changes
mkdir -p /kaggle/input/competitions
ln -sfn /workspace/data/birdclef-2026 /kaggle/input/competitions/birdclef-2026
mkdir -p /kaggle/working

echo "=== Step 5: Clone repo ==="
cd /workspace
git clone https://github.com/goromaru5600/birdclef-2026.git
cd birdclef-2026

echo "=== Done! ==="
echo "Competition data: /kaggle/input/competitions/birdclef-2026"
echo "Working dir:      /kaggle/working"
echo "Notebooks:        /workspace/birdclef-2026/notebooks"
