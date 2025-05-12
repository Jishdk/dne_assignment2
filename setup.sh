#!/bin/bash

# Create virtual environment
python3 -m venv dne
source dne/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Setup project directories
mkdir -p data/{train,test,processed,keypoints,visualizations}
mkdir -p models results

echo "Setup complete. Virtual environment 'dne' activated."
