#!/bin/bash
# =============================================================================
# Setup Script for Identity-Aware Segmentation with SAM 3 & InsightFace
# =============================================================================
# ใช้สำหรับติดตั้งโปรเจคบนเครื่องใหม่ (Ubuntu/Debian)
# =============================================================================

set -e  # หยุดทันทีถ้ามี error

echo "=========================================="
echo "🚀 Setting up Identity-Aware Segmentation"
echo "=========================================="

# =============================================================================
# 1. Check OS
# =============================================================================
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  Warning: This script is designed for Linux (Ubuntu/Debian)"
    echo "   For other OS, please install dependencies manually."
    exit 1
fi

# =============================================================================
# 2. Install System Dependencies
# =============================================================================
echo ""
echo "📦 Installing system dependencies..."
echo "   (cmake, python3-dev, ffmpeg, etc.)"

sudo apt-get update
sudo apt-get install -y \
    cmake \
    python3-dev \
    python3-pip \
    python3-venv \
    python3.10-venv \
    build-essential \
    ffmpeg \
    libgl1 \
    git \
    wget

echo "✅ System dependencies installed"

# =============================================================================
# 3. Verify Python Version
# =============================================================================
echo ""
echo "🐍 Checking Python version..."

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python 3.10+ is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION is compatible"

# =============================================================================
# 4. Create Virtual Environment
# =============================================================================
echo ""
echo "🌐 Creating virtual environment..."

if [ -d ".venv" ]; then
    echo "   Found existing .venv, removing..."
    rm -rf .venv
fi

python3 -m venv .venv
echo "✅ Virtual environment created at .venv"

# =============================================================================
# 5. Activate and Upgrade pip
# =============================================================================
echo ""
echo "⬆️  Upgrading pip..."

source .venv/bin/activate
# setuptools<70: SAM 3 ใช้ pkg_resources ซึ่งถูกลบออกใน setuptools v70+
pip install --upgrade pip wheel "setuptools<70"

echo "✅ pip upgraded"

# =============================================================================
# 6. Install PyTorch with CUDA 12.6
# =============================================================================
echo ""
echo "🔥 Installing PyTorch with CUDA 12.6 support..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "✅ PyTorch installed"

# =============================================================================
# 7. Install Python Dependencies (except insightface first)
# =============================================================================
echo ""
echo "📚 Installing Python dependencies..."

# ติดตั้ง dependencies อื่นก่อน (ยกเว้น insightface เพราะต้อง compile)
echo "   Step 1: Installing base dependencies..."
pip install numpy>=1.26.0,<2.0.0
pip install onnxruntime-gpu>=1.17.0 onnx>=1.15.0
pip install opencv-python>=4.9.0 opencv-contrib-python>=4.9.0
pip install Pillow>=10.0.0 imageio>=2.34.0 imageio-ffmpeg>=0.4.9
pip install gradio>=5.0.0
pip install matplotlib>=3.8.0 seaborn>=0.13.0
pip install scikit-learn>=1.4.0 scipy>=1.12.0
pip install huggingface-hub>=0.22.0 transformers>=4.40.0 accelerate>=0.29.0
pip install tqdm>=4.66.0 decord>=0.6.0 av>=11.0.0
pip install pandas>=2.2.0
pip install jupyter>=1.0.0 ipython>=8.22.0

echo "   Step 2: Installing insightface (this may take a while)..."
pip install insightface>=0.7.3

echo "✅ Python dependencies installed"

# =============================================================================
# 8. Clone and Install SAM 3
# =============================================================================
echo ""
echo "✂️  Cloning SAM 3 repository..."

if [ -d "sam3" ]; then
    echo "   Found existing sam3 directory, removing..."
    rm -rf sam3
fi

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e ".[notebooks]"
cd ..

echo "✅ SAM 3 installed"

# =============================================================================
# 9. Create Required Directories
# =============================================================================
echo ""
echo "📁 Creating required directories..."

mkdir -p Input
mkdir -p outputs
mkdir -p insightface_models

echo "✅ Directories created"

# =============================================================================
# 10. Verify Installation
# =============================================================================
echo ""
echo "🧪 Verifying installation..."

python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✅ PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import insightface
    print(f"✅ InsightFace: {insightface.__version__}")
except ImportError as e:
    print(f"❌ InsightFace: {e}")

try:
    import cv2
    print(f"✅ OpenCV: {cv2.__version__}")
except ImportError as e:
    print(f"❌ OpenCV: {e}")

try:
    sys.path.insert(0, 'sam3')
    from sam3 import build_sam3_image_model
    print(f"✅ SAM 3: import successful")
except ImportError as e:
    print(f"❌ SAM 3: {e}")

try:
    import gradio
    print(f"✅ Gradio: {gradio.__version__}")
except ImportError as e:
    print(f"❌ Gradio: {e}")

print("\n🎉 Setup completed successfully!")
EOF

# =============================================================================
# 11. Final Instructions
# =============================================================================
echo ""
echo "=========================================="
echo "🎉 Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "   1. Activate virtual environment:"
echo "      source .venv/bin/activate"
echo ""
echo "   2. Login to HuggingFace (for SAM 3 weights):"
echo "      huggingface-cli login"
echo "      # OR set environment variable:"
echo "      export HF_TOKEN='your_token_here'"
echo ""
echo "   3. Run the notebook:"
echo "      jupyter notebook main.ipynb"
echo ""
echo "   4. Open browser at http://127.0.0.1:7861 after running Section 6"
echo ""
echo "📖 For troubleshooting, see README.md"
echo ""
