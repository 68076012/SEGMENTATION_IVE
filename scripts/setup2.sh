#!/bin/bash
# =============================================================================
# setup2.sh — Quick Setup for main2.ipynb (RTX 6000 / CUDA 12.x)
# =============================================================================
# รัน: chmod +x scripts/setup2.sh && ./scripts/setup2.sh
# เวลารวม (เครื่องใหม่): ~15–20 นาที
# เวลารวม (มี .venv แล้ว): ~2–3 นาที
# =============================================================================

set -euo pipefail

# ── สี สำหรับ output ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

ok()   { echo -e "${GREEN}✅ $*${NC}"; }
info() { echo -e "${BLUE}   $*${NC}"; }
warn() { echo -e "${YELLOW}⚠️  $*${NC}"; }
err()  { echo -e "${RED}❌ $*${NC}"; exit 1; }
step() { echo -e "\n${BOLD}━━━ $* ━━━${NC}"; }

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════╗"
echo "║  IVE Segmentation — Setup for main2.ipynb       ║"
echo "║  InsightFace + SAM 3 + Ultralytics (RTX 6000)  ║"
echo "╚══════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── ตรวจสอบ OS ───────────────────────────────────────────────────────────────
[[ "$OSTYPE" == "linux-gnu"* ]] || err "รองรับเฉพาะ Linux"

# ── ตรวจสอบ CUDA ─────────────────────────────────────────────────────────────
if ! command -v nvidia-smi &> /dev/null; then
    warn "ไม่พบ nvidia-smi — ตรวจสอบ driver ก่อน"
else
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    CUDA_VER=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
    ok "GPU: $GPU_NAME | CUDA $CUDA_VER"
fi

# =============================================================================
# STEP 1: System packages (ข้ามถ้าติดตั้งแล้ว)
# =============================================================================
step "1. System Dependencies"

MISSING_PKGS=()
for pkg in cmake python3-dev python3-venv build-essential ffmpeg git; do
    dpkg -s "$pkg" &>/dev/null || MISSING_PKGS+=("$pkg")
done

# libgl1 ตรวจแยก (ชื่อแตกต่างกันใน distro)
dpkg -s libgl1 &>/dev/null || dpkg -s libgl1-mesa-glx &>/dev/null || MISSING_PKGS+=("libgl1")

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    info "ติดตั้ง: ${MISSING_PKGS[*]}"
    # ใช้ || true เพื่อ ignore GPG/PPA errors (เช่น deadsnakes key หมดอายุ)
    # packages หลักอยู่ใน main Ubuntu repo ไม่กระทบ
    sudo apt-get update -qq 2>&1 | grep -v "NO_PUBKEY\|GPG error\|not signed\|signatures couldn't" || true
    sudo apt-get install -y "${MISSING_PKGS[@]}"
    ok "System packages installed"
else
    ok "System packages already installed (ข้าม)"
fi

# =============================================================================
# STEP 2: Python version check
# =============================================================================
step "2. Python Version"

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

[[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -ge 10 ]] || err "ต้องการ Python 3.10+ (พบ $PY_VER)"
ok "Python $PY_VER"

# =============================================================================
# STEP 3: Virtual environment
# =============================================================================
step "3. Virtual Environment"

VENV_DIR=".venv"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    ok ".venv มีอยู่แล้ว — ข้ามการสร้างใหม่"
else
    info "สร้าง .venv ..."
    python3 -m venv "$VENV_DIR"
    ok ".venv created"
fi

source "$VENV_DIR/bin/activate"
info "Active: $(which python3)"

# =============================================================================
# STEP 4: pip + setuptools (critical: <70 สำหรับ SAM 3)
# =============================================================================
step "4. pip / setuptools"

pip install --quiet --upgrade pip wheel
pip install --quiet "setuptools<70"
ok "pip $(pip --version | awk '{print $2}') | setuptools<70"

# =============================================================================
# STEP 5: PyTorch with CUDA
# =============================================================================
step "5. PyTorch + CUDA"

# ตรวจ PyTorch ที่ติดตั้งอยู่แล้ว
if python3 -c "import torch; assert torch.cuda.is_available()" &>/dev/null; then
    PT_VER=$(python3 -c "import torch; print(torch.__version__)")
    CUDA_PT=$(python3 -c "import torch; print(torch.version.cuda)")
    ok "PyTorch $PT_VER + CUDA $CUDA_PT (ข้าม)"
else
    # ตรวจ CUDA version จาก driver เพื่อเลือก index url ที่เหมาะสม
    CUDA_MAJOR=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' | cut -d. -f1 || echo "12")
    CUDA_MINOR=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}' | cut -d. -f2 || echo "1")

    if [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 6 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu126"
        info "ใช้ cu126 index (CUDA $CUDA_MAJOR.$CUDA_MINOR)"
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        info "ใช้ cu121 index (CUDA $CUDA_MAJOR.$CUDA_MINOR)"
    fi

    info "กำลัง install PyTorch (ใช้เวลา 2–5 นาที)..."
    pip install --quiet torch torchvision torchaudio --index-url "$TORCH_INDEX"
    ok "PyTorch installed"
fi

# =============================================================================
# STEP 6: Python dependencies
# =============================================================================
step "6. Python Dependencies"

# แบ่งเป็น group เพื่อง่ายต่อการ debug ถ้า fail
info "Core packages..."
pip install --quiet \
    "numpy>=1.26.0,<2.0.0" \
    "scipy>=1.12.0" \
    "scikit-learn>=1.4.0"

info "Vision packages..."
pip install --quiet \
    "opencv-python>=4.9.0" \
    "Pillow>=10.0.0"

info "HuggingFace + IO..."
pip install --quiet \
    "huggingface-hub>=0.22.0" \
    "transformers>=4.40.0" \
    "accelerate>=0.29.0"

info "Jupyter + UI..."
pip install --quiet \
    "jupyter>=1.0.0" \
    "ipywidgets>=8.0.0" \
    "tqdm>=4.66.0" \
    "gradio>=5.0.0"

info "Video + misc..."
pip install --quiet \
    "imageio>=2.34.0" \
    "imageio-ffmpeg>=0.4.9" \
    "matplotlib>=3.8.0" \
    "pandas>=2.2.0" \
    "onnx>=1.15.0"

info "ONNX Runtime GPU (สำหรับ InsightFace)..."
pip install --quiet "onnxruntime-gpu>=1.17.0"

info "ONNX Runtime CPU (fallback สำหรับ insightface บน CPU-only setup)..."
# ติดตั้ง onnxruntime ธรรมดาด้วยเพื่อป้องกัน conflict
pip install --quiet "onnxruntime>=1.17.0" || true

info "InsightFace (compile ~5–10 นาที ถ้าไม่มี wheel)..."
pip install --quiet "insightface>=0.7.3"

# ── Ultralytics SAM + CLIP ────────────────────────────────────────────────────
info "Ultralytics (SAM segmentation + YOLO)..."
pip install --quiet -U ultralytics

info "Ultralytics CLIP (text-to-segment)..."
# uninstall ก่อนเพราะ ultralytics/CLIP conflict กับ openai/clip
pip uninstall -y clip 2>/dev/null || true
pip install --quiet "git+https://github.com/ultralytics/CLIP.git"

ok "Python dependencies installed"

# =============================================================================
# STEP 7: SAM 3
# =============================================================================
step "7. SAM 3"

if [ -d "sam3" ]; then
    ok "sam3/ มีอยู่แล้ว"
    info "ตรวจสอบ installation..."
    if ! python3 -c "from sam3 import build_sam3_image_model" &>/dev/null; then
        info "reinstall SAM 3..."
        cd sam3 && pip install --quiet -e ".[notebooks]" && cd ..
    fi
else
    info "Clone SAM 3..."
    git clone --depth 1 https://github.com/facebookresearch/sam3.git
    info "Install SAM 3 (pip install -e)..."
    cd sam3 && pip install --quiet -e ".[notebooks]" && cd ..
fi

ok "SAM 3 ready"

# =============================================================================
# STEP 7b: Pre-download Ultralytics SAM weights
# =============================================================================
step "7b. Ultralytics SAM Weights"

# sam_b.pt (~358 MB) — Ultralytics SAM base model
# ถ้ายังไม่มีให้ download ล่วงหน้าเพื่อ notebook รันได้เลย
SAM_PT_PATH="sam_b.pt"

if [ -f "$SAM_PT_PATH" ]; then
    SAM_SIZE=$(du -sh "$SAM_PT_PATH" | cut -f1)
    ok "sam_b.pt มีอยู่แล้ว ($SAM_SIZE) — ข้าม"
else
    info "Downloading sam_b.pt (~358 MB)..."
    python3 << 'PYEOF'
try:
    from ultralytics import SAM
    # การสร้าง SAM object จะ trigger auto-download
    model = SAM("sam_b.pt")
    print("   ✅ sam_b.pt downloaded!")
except Exception as e:
    print(f"   ⚠️  {e}")
    # Fallback: download ตรงๆ ด้วย wget
    import subprocess, os
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam_b.pt"
    r = subprocess.run(["wget", "-q", "--show-progress", "-O", "sam_b.pt", url])
    if r.returncode == 0:
        print("   ✅ sam_b.pt downloaded via wget!")
    else:
        print("   ❌ download ล้มเหลว — รัน: wget -O sam_b.pt " + url)
PYEOF
fi

# =============================================================================
# STEP 8: Directories
# =============================================================================
step "8. Project Directories"

mkdir -p Input outputs insightface_models Dataset

ok "Directories: Input/ outputs/ insightface_models/ Dataset/"

# =============================================================================
# STEP 9: HuggingFace Token
# =============================================================================
step "9. HuggingFace Token"

# ตรวจสอบ token จาก env หรือ cached credentials
if python3 -c "from huggingface_hub import whoami; whoami()" &>/dev/null; then
    HF_USER=$(python3 -c "from huggingface_hub import whoami; print(whoami()['name'])")
    ok "Already logged in as: $HF_USER"
elif [ -n "${HF_TOKEN:-}" ]; then
    python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
    ok "Logged in via HF_TOKEN env"
else
    warn "ยังไม่ได้ login HuggingFace"
    echo ""
    echo "   ทำอย่างใดอย่างหนึ่ง:"
    echo "   A) รัน: huggingface-cli login"
    echo "   B) export HF_TOKEN='hf_...' แล้วรัน script นี้อีกครั้ง"
    echo ""
    echo "   (จำเป็นสำหรับ download SAM 3 weights ครั้งแรก)"
fi

# =============================================================================
# STEP 10: Pre-download SAM 3 weights (optional แต่ช่วยให้ notebook รันไว)
# =============================================================================
step "10. Pre-download SAM 3 Weights"

# SAM 3 weights อยู่ที่ HuggingFace: facebook/sam3
# จะ download อัตโนมัติครั้งแรกที่ build_sam3_image_model() ถูกเรียก
# แต่เราสามารถ pre-cache ได้

if python3 -c "from huggingface_hub import whoami; whoami()" &>/dev/null; then
    info "Pre-downloading SAM 3 weights (~5 GB, ใช้เวลา 3–10 นาที)..."
    python3 << 'PYEOF'
import sys
sys.path.insert(0, 'sam3')

try:
    from huggingface_hub import snapshot_download
    import os

    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    repo_id   = "facebook/sam3"

    print(f"   Downloading {repo_id} → {cache_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=None,   # ใช้ HF cache ปกติ
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
    )
    print("   ✅ SAM 3 weights cached!")
except Exception as e:
    print(f"   ⚠️  ข้าม pre-download: {e}")
    print("   (weights จะ download อัตโนมัติเมื่อรัน notebook)")
PYEOF
else
    warn "ข้าม pre-download (ต้อง login HuggingFace ก่อน)"
fi

# =============================================================================
# STEP 11: Pre-download InsightFace buffalo_l
# =============================================================================
step "11. Pre-download InsightFace buffalo_l"

BUFFALO_DIR="insightface_models/models/buffalo_l"

if [ -d "$BUFFALO_DIR" ] && [ "$(ls -A $BUFFALO_DIR)" ]; then
    ok "buffalo_l มีอยู่แล้ว (ข้าม)"
else
    info "กำลัง download buffalo_l (~500 MB)..."
    python3 << 'PYEOF'
import insightface
from insightface.app import FaceAnalysis
import os

try:
    fa = FaceAnalysis(
        name="buffalo_l",
        root="./insightface_models",
        providers=["CPUExecutionProvider"],
    )
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    print("   ✅ buffalo_l downloaded!")
except Exception as e:
    print(f"   ⚠️  {e}")
PYEOF
fi

# =============================================================================
# STEP 12: Jupyter kernel
# =============================================================================
step "12. Jupyter Kernel"

if python3 -c "import ipykernel" &>/dev/null; then
    python3 -m ipykernel install --user --name ive-seg --display-name "IVE Segmentation (.venv)"
    ok "Kernel 'IVE Segmentation' registered"
else
    warn "ipykernel ไม่พบ — ข้ามการ register kernel"
fi

# =============================================================================
# STEP 13: Verify everything
# =============================================================================
step "13. Final Verification"

python3 << 'PYEOF'
import sys
sys.path.insert(0, 'sam3')

results = []

checks = [
    ("torch + CUDA",   "import torch; assert torch.cuda.is_available(), 'no CUDA'; "
                       "print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | '+"
                       "f'GPU: {torch.cuda.get_device_name(0)}')"),
    ("insightface",    "import insightface; print(f'v{insightface.__version__}')"),
    ("onnxruntime-gpu","import onnxruntime as rt; print(f'v{rt.__version__} | providers: {rt.get_available_providers()}')"),
    ("opencv",         "import cv2; print(f'v{cv2.__version__}')"),
    ("sam3",           "from sam3 import build_sam3_image_model; print('import OK')"),
    ("sam3_processor", "from sam3.model.sam3_image_processor import Sam3Processor; print('OK')"),
    ("ultralytics",    "import ultralytics; print(f'v{ultralytics.__version__}')"),
    ("clip",           "import clip; print('ultralytics CLIP OK')"),
    ("sam_b.pt",       "import os; sz=os.path.getsize('sam_b.pt')/1e6 if os.path.exists('sam_b.pt') else 0; "
                       "assert sz > 100, f'not found or too small ({sz:.0f}MB)'; print(f'{sz:.0f} MB ✓')"),
    ("gradio",         "import gradio; print(f'v{gradio.__version__}')"),
    ("scipy",          "import scipy; print(f'v{scipy.__version__}')"),
    ("tqdm",           "from tqdm.notebook import tqdm; print('notebook OK')"),
    ("PIL",            "from PIL import Image; print('OK')"),
    ("huggingface_hub","from huggingface_hub import whoami; u=whoami(); print(f'logged in: {u[\"name\"]}')"),
]

pad = max(len(k) for k,_ in checks)
for name, code in checks:
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(code)
        detail = buf.getvalue().strip()
        print(f"  ✅ {name:<{pad}}  {detail}")
        results.append(True)
    except Exception as e:
        print(f"  ❌ {name:<{pad}}  {e}")
        results.append(False)

print()
passed = sum(results)
total  = len(results)
if passed == total:
    print(f"  🎉 All {total}/{total} checks passed — ready to run main2.ipynb!")
else:
    print(f"  ⚠️  {passed}/{total} passed — แก้ไข ❌ ก่อนรัน notebook")
PYEOF

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗"
echo -e "║  Setup Complete!  $(date '+%H:%M:%S')                      ║"
echo -e "╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BOLD}วิธีรัน main2.ipynb:${NC}"
echo ""
echo "   # 1. Activate venv"
echo "   source .venv/bin/activate"
echo ""
echo "   # 2. เปิด Jupyter"
echo "   jupyter notebook main2.ipynb"
echo "   # หรือ Lab:"
echo "   jupyter lab main2.ipynb"
echo ""
echo "   # 3. เลือก kernel: IVE Segmentation (.venv)"
echo "   # 4. Run All Cells (ข้าม Cell 1.1–1.2 ถ้าไม่ต้องการ reinstall)"
echo ""
echo -e "${YELLOW}หมายเหตุ:${NC}"
echo "   - SAM 3 weights (~5GB) จะ download ครั้งแรกที่รัน Cell 3.1"
echo "     (ถ้า pre-download สำเร็จ จะโหลดจาก cache เร็วกว่ามาก)"
echo "   - InsightFace buffalo_l (~500MB) download ครั้งแรกที่รัน Cell 2.1"
echo "   - Cell 1.1 (pip install) และ Cell 1.2 (git clone) ข้ามได้เลย"
echo "   - sam_b.pt อยู่ที่ root โปรเจค (Ultralytics จะหาเจออัตโนมัติ)"
echo ""
echo -e "${BOLD}Packages ที่ติดตั้ง:${NC}"
echo "   Minimal : insightface onnxruntime-gpu opencv-python"
echo "   SAM 3   : facebook/sam3 (native HF, ~5GB weights)"
echo "   Ultralytics: ultralytics + ultralytics/CLIP + sam_b.pt (~358MB)"
echo ""
