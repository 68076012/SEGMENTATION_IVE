#!/bin/bash
# Setup script for IVE Segmentation on Linux + RTX 6000
# Run: chmod +x setup_linux.sh && ./setup_linux.sh

set -e  # Exit on error

echo "================================"
echo "🚀 IVE Segmentation Setup"
echo "For: Linux + NVIDIA RTX 6000"
echo "================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: This script is for Linux only${NC}"
    exit 1
fi

# Check NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. Please install NVIDIA drivers first.${NC}"
    echo "Visit: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
fi

# Update system packages
echo -e "${GREEN}Step 1/7: Updating system packages...${NC}"
sudo apt update
sudo apt install -y build-essential git wget curl software-properties-common

# Install Python 3.10 if not exists
echo -e "${GREEN}Step 2/7: Installing Python 3.10...${NC}"
if ! command -v python3.10 &> /dev/null; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils
fi

# Install system dependencies for OpenCV and other packages
echo -e "${GREEN}Step 3/7: Installing system dependencies...${NC}"
sudo apt install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-dev

# Create virtual environment
echo -e "${GREEN}Step 4/7: Creating virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Remove it first if you want to recreate.${NC}"
else
    python3.10 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip
echo -e "${GREEN}Step 5/7: Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
echo -e "${GREEN}Step 6/7: Installing PyTorch (CUDA 11.8)...${NC}"
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo -e "${GREEN}Step 7/7: Installing project dependencies...${NC}"
pip install Cython
pip install -r requirements.txt

# Verify installation
echo ""
echo "================================"
echo "🔍 Verifying installation..."
echo "================================"
echo ""

python << 'EOF'
import sys
print(f"Python version: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Memory: {vram:.2f} GB")
except Exception as e:
    print(f"✗ PyTorch error: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers error: {e}")

try:
    import cv2
    print(f"✓ OpenCV: {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV error: {e}")

try:
    import peft
    print(f"✓ PEFT: {peft.__version__}")
except Exception as e:
    print(f"✗ PEFT error: {e}")

try:
    import jupyter
    print(f"✓ Jupyter: installed")
except Exception as e:
    print(f"✗ Jupyter error: {e}")

try:
    import pycocotools
    print(f"✓ pycocotools: installed")
except Exception as e:
    print(f"✗ pycocotools error: {e}")

print("\n🎉 Setup verification complete!")
EOF

echo ""
echo "================================"
echo "✅ Setup Complete!"
echo "================================"
echo ""
echo "Next steps:"
echo "1. Activate environment: source .venv/bin/activate"
echo "2. Login Hugging Face: huggingface-cli login"
echo "3. Run Jupyter: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo ""
echo "For SSH tunnel: ssh -L 8888:localhost:8888 user@your-server"
echo ""
