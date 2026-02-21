# PLAN.md — SAM 3 Custom Fine-Tuning for Identity & Possession (IVE Segmentation)

## 🎯 Objective (เป้าหมายหลัก)
พัฒนาระบบ **Text-to-Segment** ในรูปแบบวิดีโอผ่านการ **Fine-tune โมเดล SAM 3** แบบ End-to-End เพื่อแก้โจทย์ 2 ระดับ:
1.  **Identity Awareness:** สอนให้โมเดลรู้จักชื่อเฉพาะ "Wonyoung"
2.  **Possession Understanding:** สอนให้เข้าใจบริบทความเป็นเจ้าของ "Wonyoung's shirt"

## 🛠 Tech Stack & Constraints (เงื่อนไขการทำงาน)
* **Model:** SAM 3 (Meta) ผ่าน Hugging Face (`facebook/sam3`).
* **Data Annotation:** X-AnyLabeling (ใช้ SAM ViT-H ช่วยทำ Auto-segmentation แล้ว Export เป็น COCO Format).
* **Training Method:** LoRA (Low-Rank Adaptation) เพื่อความรวดเร็วและประหยัด Parameter.
* **Hardware:** NVIDIA RTX 6000 (VRAM 48GB) - *ใช้ Batch Size ใหญ่ได้สบาย.*
* **Delivery Format:** เขียนทุกอย่างลงในไฟล์ Jupyter Notebook (`main_finetune.ipynb`) ไฟล์เดียว.

---

## 📅 Execution Roadmap (สำหรับ AI Agent)

### Phase 1: Data Annotation Preparation (Manual Work by Developer)
*เป้าหมาย: สร้าง Ground Truth Dataset คุณภาพสูง*
* **Action:** Developer ทำการสุ่มภาพกลุ่มและภาพเดี่ยวของ IVE (ประมาณ 50-100 ภาพ).
* **Tool:** ใช้ `X-AnyLabeling` รันโมเดล SAM (ViT-H) เพื่อไดคัทด้วยการคลิก.
* **Labels Needed:** ต้องมี Label อย่างน้อย 2 ประเภทคือ `Wonyoung` (คลุมทั้งตัว) และ `Wonyoung_shirt` (คลุมเฉพาะเสื้อ).
* **Output:** โฟลเดอร์ `images/` และไฟล์ `annotations.json` (COCO Format).

### Phase 2: Data Preprocessing & PyTorch Dataset
*เป้าหมาย: แปลงไฟล์ COCO JSON ให้เป็น Dataset ที่ SAM 3 อ่านเข้าใจ*
* **Step 2.1:** เขียนฟังก์ชันอ่านไฟล์ `annotations.json` เพื่อดึงพิกัด Polygon.
* **Step 2.2:** ใช้ `cv2.fillPoly` แปลง Polygon เป็น Binary Mask (0 และ 1).
* **Step 2.3:** สร้าง Custom PyTorch `Dataset` Class ที่ Return ค่าประกอบด้วย:
    * `pixel_values` (รูปภาพที่ผ่าน `Sam3Processor`)
    * `input_ids` (Text Prompt ที่ผ่าน Tokenizer เช่น "Wonyoung")
    * `labels` (Ground Truth Binary Mask)

### Phase 3: Model Fine-Tuning (Parameter-Efficient with LoRA)
*เป้าหมาย: เทรน SAM 3 เฉพาะส่วนที่จำเป็นด้วย RTX 6000*
* **Step 3.1:** โหลด `Sam3Model` จาก `facebook/sam3`.
* **Step 3.2:** Freeze น้ำหนัก (Weights) ของ Image Encoder (ViT backbone) ทั้งหมด.
* **Step 3.3:** ใช้ไลบรารี `peft` เพื่อแทรก LoRA adapters เข้าไปที่ **Prompt Encoder** และ **Mask Decoder**.
* **Step 3.4:** Setup Training Loop:
    * **Loss Function:** Dice Loss + Focal Loss (จัดการปัญหา Background เยอะกว่า Foreground).
    * **Optimizer:** AdamW (Learning rate ~ 1e-4).
    * **Batch Size:** ปรับให้เหมาะสมกับ VRAM 48GB (อาจจะดันได้ถึง 8 หรือ 16).
* **Step 3.5:** Save โมเดล (เฉพาะ LoRA Weights).

### Phase 4: Inference & Video Processing
*เป้าหมาย: ทดสอบโมเดลกับวิดีโอโจทย์*
* **Step 4.1:** โหลด Base SAM 3 และนำ LoRA Weights ที่เทรนเสร็จแล้วมา Merge.
* **Step 4.2:** เขียนฟังก์ชัน `process_video(video_path, text_prompt)` โดยใช้ `cv2.VideoCapture` อ่านทีละเฟรม.
* **Step 4.3:** นำแต่ละเฟรมส่งเข้าโมเดลพร้อม Text Prompt ("Wonyoung" หรือ "Wonyoung's shirt").
* **Step 4.4:** นำ Prediction Mask ที่ได้มา Overlay สีทับลงบนเฟรมวิดีโอ แล้ว Write เป็นไฟล์ `.mp4` ใหม่.

### Phase 5: Gradio UI Integration
*เป้าหมาย: สร้าง Interface สำหรับ Demo ส่งอาจารย์*
* สร้าง Gradio Blocks ภายใน `.ipynb`.
* **Input:** ช่องอัปโหลดวิดีโอ/รูปภาพ + Textbox สำหรับพิมพ์ Prompt.
* **Output:** วิดีโอ/รูปภาพที่ถูก Overlay Mask เรียบร้อยแล้ว.