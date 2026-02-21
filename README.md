# 🎯 IVE Member Segmentation (Identity & Possession Context)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![SAM 3](https://img.shields.io/badge/Meta-SAM_3-blueviolet)
![LoRA](https://img.shields.io/badge/LoRA-Fine--tuning-green)
![Hardware](https://img.shields.io/badge/Hardware-RTX_6000-orange)

This project is an intelligent **Text-to-Segment** system designed to overcome the limitations of standard segmentation models. By **fine-tuning SAM 3** with custom annotations, the system can identify specific individuals (Identity) and understand contextual ownership (Possession) within images or videos of the K-pop group IVE.

---

## 💡 The Core Concepts

This project goes beyond simply drawing bounding boxes around generic "people". It is engineered to address two advanced challenges:

1. **Identity Awareness (Fine-tuned SAM 3):** 
   Base models like SAM 3 understand the concept of a "Person," but lack knowledge of specific entities like "Wonyoung." 
   
   **Our Approach:** Fine-tune SAM 3 ด้วย **LoRA (Low-Rank Adaptation)** บน dataset ที่เราสร้างขึ้นเองโดยใช้ **X-AnyLabeling** สำหรับการ annotate รูปภาพ
   - ใช้ X-AnyLabeling + SAM ViT-H สร้าง mask คุณภาพสูง
   - Export เป็น COCO Format
   - Fine-tune SAM 3 ด้วย LoRA เฉพาะ Prompt Encoder และ Mask Decoder

2. **Possession & Association (Contextual Understanding):**
   The system is capable of handling complex prompts such as *"Wonyoung's shirt"*. This is achieved through a two-step pipeline:
   - **Step 1:** Locate the target identity (e.g., Wonyoung) and generate the identity mask.
   - **Step 2:** Utilize this mask as a Region of Interest (ROI) alongside the object text prompt (e.g., "shirt") to run SAM 3 again. This ensures the output segments the shirt worn *specifically* by Wonyoung, ignoring similar clothing worn by others in the frame.

---

## 🛠️ Architecture & Tech Stack

- **Segmentation Engine:** [Meta SAM 3](https://github.com/facebookresearch/segment-anything) (via Hugging Face) + LoRA Fine-tuning
- **Data Annotation:** [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) with SAM ViT-H
- **Training Framework:** PyTorch + PEFT (LoRA)
- **UI & Visualization:** Gradio
- **Environment:** Entire pipeline executed within `main_finetune.ipynb` notebook
- **Hardware Optimization:** Fully optimized for **NVIDIA RTX 6000 (48GB VRAM)**

---

## 📁 Project Structure

```text
SEGMENTATION_IVE/
├── main_finetune.ipynb          # Main Jupyter Notebook (Fine-tuning + Inference + UI)
├── Dataset/                     # Solo images of IVE members (180 total)
│   ├── An_Yujin/               
│   ├── Jang_Wonyoung/          
│   ├── Kim_Gaeul/              
│   ├── Kim_Jiwon/              
│   ├── Lee_Hyunseo/            
│   └── Naoi_Rei/               
├── data/                        # Generated during setup
│   ├── images/                  # Annotated images
│   ├── masks/                   # Generated masks
│   └── annotations.json         # COCO format annotations
├── models/                      # Saved fine-tuned models
├── outputs/                     # Output videos/images
├── requirements.txt             # Python dependencies
├── ANNOTATION_GUIDE.md          # Guide for X-AnyLabeling
├── PLAN.md                      # Detailed logic & roadmap
└── README.md                    # This file
```

---

## 🚀 Installation & Setup (Using .venv)

### 1. Create and activate the virtual environment:

```bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate

# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies:

```bash
pip install -r requirements.txt
```

### 3. Hugging Face Authentication (Required for SAM):

```bash
huggingface-cli login
```

---

## 📝 Data Annotation with X-AnyLabeling

### Step 1: Install X-AnyLabeling

```bash
git clone https://github.com/CVHub520/X-AnyLabeling.git
cd X-AnyLabeling
pip install -r requirements.txt
python anylabeling/app.py
```

### Step 2: Annotate Images

1. **Open Directory**: เลือกโฟลเดอร์รูปภาพ (e.g., `Dataset/Jang_Wonyoung/`)
2. **Enable SAM**: Click `AI-Powered` → `Segment Anything (SAM)` → เลือก `sam_vit_h`
3. **Create Masks**: Click บนวัตถุเพื่อสร้าง mask อัตโนมัติ
4. **Label**: กำหนด label name:
   - `Wonyoung` - คลุมทั้งตัว
   - `Wonyoung_shirt` - เฉพาะเสื้อ
5. **Export**: Click `Export` → `COCO` → บันทึกเป็น `data/annotations.json`

ดูรายละเอียดเพิ่มเติมใน [ANNOTATION_GUIDE.md](ANNOTATION_GUIDE.md)

---

## 💻 Usage

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook main_finetune.ipynb
```

Run cells sequentially:
1. **Environment Setup**: ติดตั้ง dependencies และ verify GPU
2. **Data Preparation**: โหลด COCO annotations
3. **Model Fine-tuning**: Fine-tune SAM 3 with LoRA
4. **Inference**: ทดสอบกับรูปภาพหรือวิดีโอ
5. **Gradio UI**: Launch web interface

### Option 2: Command Line (ถ้ามีเพิ่มเติม)

```bash
python train.py  # ถ้าจะเทรนผ่าน command line
python inference.py --image path/to/image.jpg --prompt "Wonyoung"
```

---

## 🎓 Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| LoRA Rank (r) | 8 | Low-rank dimension |
| LoRA Alpha | 16 | Scaling factor |
| Learning Rate | 1e-4 | AdamW optimizer |
| Batch Size | 2-8 | ขึ้นอยู่กับ VRAM |
| Epochs | 5-10 | ขึ้นอยู่กับ dataset size |
| Loss Function | Dice + Focal | จัดการ class imbalance |

---

## 🏷️ Supported Labels

| Category ID | Label | Type |
|-------------|-------|------|
| 1 | `Wonyoung` | Identity |
| 2 | `Wonyoung_shirt` | Possession |
| 3 | `Yujin` | Identity |
| 4 | `Yujin_shirt` | Possession |
| 5 | `Gaeul` | Identity |
| 6 | `Rei` | Identity |
| 7 | `Liz` | Identity |
| 8 | `Leeseo` | Identity |

---

## ⚠️ Notes & Limitations

- **Annotation Required**: ต้องมี COCO annotations ที่สมบูรณ์ก่อนการ fine-tune
- **GPU Memory**: Fine-tuning SAM ต้องการ VRAM อย่างน้อย 16GB (แนะนำ 24GB+)
- **Possession Segmentation**: ความแม่นยำขึ้นอยู่กับความชัดเจนของภาพและคุณภาพของ annotation

---

## 📚 References

- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)
- [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)
- [PEFT - Hugging Face](https://huggingface.co/docs/peft)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
