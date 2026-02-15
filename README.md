# 🎯 IVE Member Segmentation (Identity & Possession Context)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![SAM 3](https://img.shields.io/badge/Meta-SAM_3-blueviolet)
![InsightFace](https://img.shields.io/badge/InsightFace-Buffalo__L-success)
![Hardware](https://img.shields.io/badge/Hardware-RTX_6000-orange)

This project is an intelligent **Text-to-Segment** system designed to overcome the limitations of standard segmentation models. By bridging **Computer Vision** and **Language Models**, the system can identify specific individuals (Identity) and understand contextual ownership (Possession) within images or videos of the K-pop group IVE.

---

## 💡 The Core Concepts & "Hidden Agenda"

This project goes beyond simply drawing bounding boxes around generic "people". It is engineered to address two advanced challenges:

1. **Identity Awareness (Beyond Generic Objects):** Base models like SAM 3 understand the concept of a "Person," but lack knowledge of specific entities like "Wonyoung." Instead of performing a resource-heavy full model fine-tuning, this project implements a **Vision Retrieval-Augmented Generation (RAG)** approach. We utilize **InsightFace** to extract Face Embeddings (512-dimensional vectors) of IVE members to serve as a knowledge base. This maps specific names to facial features before feeding the precise bounding box to SAM 3 for pixel-level segmentation.
2. **Possession & Association (Contextual Understanding):**
   The system is capable of handling complex prompts such as *"Wonyoung's shirt"*. This is achieved through a two-step pipeline:
   - **Step 1:** Locate the target identity (e.g., Wonyoung) and generate the identity mask.
   - **Step 2:** Utilize this mask as a Region of Interest (ROI) alongside the object text prompt (e.g., "shirt") to run SAM 3 again. This ensures the output segments the shirt worn *specifically* by Wonyoung, ignoring similar clothing worn by others in the frame.

---

## 🛠️ Architecture & Tech Stack

- **Segmentation Engine:** [Meta SAM 3](https://github.com/facebookresearch/sam3) (via Hugging Face)
- **Identity Knowledge Base:** [InsightFace](https://github.com/deepinsight/insightface) (buffalo_l model)
- **UI & Visualization:** Gradio
- **Environment:** Entire pipeline executed within a single `main.ipynb` notebook.
- **Hardware Optimization:** Fully optimized for **NVIDIA RTX 6000 (48GB VRAM)** using CUDA execution providers.

---

## 📁 Project Structure

```text
SEGMENTATION_IVE/
├── main.ipynb                   # Main Jupyter Notebook (Core Logic & UI)
├── Dataset/                     # Solo images of IVE members for Vector DB (180 total)
│   ├── An_Yujin/               
│   ├── Jang_Wonyoung/          
│   ├── Kim_Gaeul/              
│   ├── Kim_Jiwon/              
│   ├── Lee_Hyunseo/            
│   └── Naoi_Rei/               
├── requirements.txt             # Python dependencies
├── PROJECT_PLAN.md              # Development roadmap & architecture
├── PLAN.md                      # Detailed logic for Identity & Possession tasks
└── README.md                    # This file
🚀 Installation & Setup (Using .venv)
It is highly recommended to isolate the project dependencies using a Python virtual environment (.venv).

Clone the repository and navigate to the project directory:

Bash
git clone <your-repo-url>
cd SEGMENTATION_IVE
Create and activate the virtual environment:

Bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
Install Dependencies:
With your .venv activated, install the required packages:

Bash
pip install -r requirements.txt
Hugging Face Authentication (Required for SAM 3):
You must request access to the facebook/sam3 repository on Hugging Face. Once approved, log in via the CLI using your access token:

Bash
huggingface-cli login
Prepare the Dataset:
Ensure the Dataset/ directory is fully populated with member images so the system can accurately build the Face Embedding Database.

💻 Usage
Open main.ipynb within your activated .venv environment.

Run the cells sequentially:

Environment Setup: Load libraries and verify GPU (RTX 6000) availability.

Build Identity Database: The system will process images in Dataset/ to extract and store Face Embeddings.

Initialize Models: Load SAM 3 and InsightFace into VRAM.

Launch Gradio UI: Run the final cell to start the web interface.

Interact with the UI:

Upload an image or a video frame.

Enter a target name or a possession-based prompt (e.g., "Wonyoung" or "Wonyoung's shirt").

View the generated mask, annotated bounding boxes, and similarity scores.

⚠️ Notes & Limitations
Possession segmentation relies on the overlap between the Identity Bounding Box and the Object Text Prompt. Accuracy depends heavily on the clarity and angle of the source image.

If InsightFace cannot detect a face (e.g., subject is facing away or heavily blurred), the system will default to "Unknown" and cannot accurately segment based on specific identities.