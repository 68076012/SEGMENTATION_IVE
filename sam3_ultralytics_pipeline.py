"""
SAM 3 (Ultralytics) + InsightFace Pipeline
===========================================

ใช้ SAM3SemanticPredictor จาก Ultralytics ร่วมกับ InsightFace
สำหรับ Identity-Aware Segmentation

Workflow:
---------
1. InsightFace: หา Wonyoung (face recognition)
2. ขยาย BBox: Face → Body
3. SAM3SemanticPredictor: segment "person" ในกรอบนั้น
   (ใช้ text prompt จริง ๆ ไม่ใช่ box-only)

ติดตั้ง:
--------
pip install -U ultralytics
# ดาวน์โหลด sam3.pt จาก Hugging Face

ข้อแตกต่าง SAM3SemanticPredictor vs SAM3InstancePredictor:
----------------------------------------------------------
- SAM3SemanticPredictor: segment ทุก instance ที่ match text prompt
  (เหมาะสำหรับ "person" → ได้ทุกคนในภาพ)
  
- SAM3InstancePredictor: segment ตาม box/point prompts
  (เหมาะสำหรับ bounding box ที่กำหนด)

ดังนั้นเราจะใช้ทั้งสองแบบ:
- InsightFace + BBox → ระบุ Wonyoung
- SAM3SemanticPredictor + BBox constraint → segment "person" เฉพาะในกรอบ
"""

import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from collections import deque, defaultdict
from typing import List, Tuple, Optional, Dict, Union
import warnings


class SAM3UltralyticsSegmenter:
    """
    Segmenter ที่ใช้ SAM 3 (Ultralytics) + InsightFace
    
    ใช้ SAM3SemanticPredictor สำหรับ text-based segmentation
    โดยจำกัด region ด้วย BBox จาก InsightFace
    """
    
    def __init__(
        self,
        face_analyzer,
        embeddings_db: Dict[str, np.ndarray],
        sam3_model_path: str = "sam3.pt",
        device: str = 'cuda',
    ):
        """
        Args:
            face_analyzer: InsightFace FaceAnalysis model
            embeddings_db: Database ของ face embeddings
            sam3_model_path: path ไปยัง sam3.pt
            device: 'cuda' หรือ 'cpu'
        """
        self.face_analyzer = face_analyzer
        self.embeddings_db = embeddings_db
        self.device = device
        
        # โหลด SAM 3 Semantic Predictor
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor
            self.sam3_predictor = SAM3SemanticPredictor(
                overrides=dict(model=sam3_model_path, device=device)
            )
            self.use_semantic = True
            print("✅ โหลด SAM3SemanticPredictor สำเร็จ")
        except ImportError:
            print("⚠️ ไม่พบ ultralytics, ใช้วิธี fallback")
            self.sam3_predictor = None
            self.use_semantic = False
        
        # Online gallery
        self.online_gallery = defaultdict(lambda: deque(maxlen=30))
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """L2 Normalization"""
        norm = np.linalg.norm(embedding)
        if norm < 1e-10:
            return embedding
        return embedding / norm
    
    def best_match(self, embedding: np.ndarray, threshold: float = 0.40) -> Tuple[str, float]:
        """หา match ที่ดีที่สุดจาก embeddings database"""
        best_name = "Unknown"
        best_score = -1.0
        
        for name, ref_emb in self.embeddings_db.items():
            if isinstance(ref_emb, np.ndarray) and ref_emb.ndim == 2:
                score = float(np.max(ref_emb @ embedding))
            else:
                score = float(np.dot(embedding, ref_emb))
            
            # Check online gallery
            if name in self.online_gallery and len(self.online_gallery[name]) > 0:
                online_mat = np.stack(list(self.online_gallery[name]), axis=0)
                online_score = float(np.max(online_mat @ embedding))
                score = max(score, online_score)
            
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_score < threshold:
            best_name = "Unknown"
        
        return best_name, best_score
    
    def face_to_body_bbox(
        self,
        face_bbox: np.ndarray,
        img_shape: Tuple[int, int],
        width_scale: float = 3.0,
        height_top_scale: float = 1.2,
        height_bottom_scale: float = 5.0
    ) -> np.ndarray:
        """ขยาย face bbox เป็น body bbox"""
        x1, y1, x2, y2 = face_bbox.astype(float)
        face_center_x = (x1 + x2) / 2.0
        face_center_y = (y1 + y2) / 2.0
        face_width = x2 - x1
        face_height = y2 - y1
        
        img_h, img_w = img_shape
        
        half_body_width = (face_width * width_scale) / 2.0
        body_x1 = face_center_x - half_body_width
        body_x2 = face_center_x + half_body_width
        
        body_y1 = face_center_y - (face_height * height_top_scale)
        body_y2 = face_center_y + (face_height * height_bottom_scale)
        
        # Clamp
        body_x1 = max(0, int(body_x1))
        body_y1 = max(0, int(body_y1))
        body_x2 = min(img_w - 1, int(body_x2))
        body_y2 = min(img_h - 1, int(body_y2))
        
        return np.array([body_x1, body_y1, body_x2, body_y2], dtype=np.int32)
    
    def segment_with_text_in_bbox(
        self,
        image_rgb: np.ndarray,
        bbox: List[int],
        text_prompt: str = "person",
        iou_threshold: float = 0.3
    ) -> np.ndarray:
        """
        🎯 Core Function: Segment ด้วย text prompt ในกรอบ BBox
        
        วิธีทำ:
        1. SAM3SemanticPredictor: segment ทุก "person" ในภาพ
        2. เลือก mask ที่มี IoU สูงสุดกับ target bbox
        
        Args:
            image_rgb: ภาพ RGB
            bbox: [x1, y1, x2, y2] กรอบที่ต้องการ
            text_prompt: noun phrase (เช่น "person", "shirt")
            iou_threshold: ขั้นต่ำของ IoU ที่ยอมรับ
            
        Returns:
            mask: binary mask
        """
        if not self.use_semantic:
            # Fallback: ใช้ box-based segmentation
            return self._fallback_segment(image_rgb, bbox)
        
        # Set image
        self.sam3_predictor.set_image(image_rgb)
        
        # Predict with text prompt
        results = self.sam3_predictor(text=[text_prompt])
        
        if len(results) == 0 or len(results[0].masks) == 0:
            return np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        
        # Get all masks
        masks = results[0].masks.data.cpu().numpy()  # [N, H, W]
        
        # คำนวณ IoU กับ target bbox สำหรับแต่ละ mask
        bx1, by1, bx2, by2 = bbox
        bbox_area = (bx2 - bx1) * (by2 - by1)
        
        best_mask = None
        best_iou = 0
        
        for mask in masks:
            # หา bbox ของ mask
            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue
            
            mx1, my1 = xs.min(), ys.min()
            mx2, my2 = xs.max(), ys.max()
            
            # คำนวณ IoU
            xA = max(bx1, mx1)
            yA = max(by1, my1)
            xB = min(bx2, mx2)
            yB = min(by2, my2)
            
            inter = max(0, xB - xA) * max(0, yB - yA)
            mask_area = (mx2 - mx1) * (my2 - my1)
            union = bbox_area + mask_area - inter
            
            iou = inter / (union + 1e-6)
            
            if iou > best_iou:
                best_iou = iou
                best_mask = mask
        
        if best_mask is None or best_iou < iou_threshold:
            # ถ้าไม่มี mask ที่ IoU สูงพอ ใช้ box-based fallback
            print(f"⚠️ Low IoU ({best_iou:.2f}), using fallback")
            return self._fallback_segment(image_rgb, bbox)
        
        return (best_mask > 0).astype(np.uint8)
    
    def _fallback_segment(self, image_rgb: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Fallback: ใช้ box-based segmentation"""
        # ใช้ cv2.grabCut หรือวิธีอื่นง่าย ๆ
        mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = bbox
        mask[y1:y2, x1:x2] = 1
        return mask
    
    def segment_member(
        self,
        image_bgr: np.ndarray,
        member_name: str,
        text_prompt: str = "person",
        similarity_threshold: float = 0.45,
        return_all_members: bool = False,
    ) -> Dict:
        """
        🔥 Main Pipeline: Segment สมาชิกด้วย SAM 3 + Text Prompt
        
        Args:
            image_bgr: Input image (BGR)
            member_name: ชื่อสมาชิก (เช่น "Wonyoung")
            text_prompt: noun phrase สำหรับ SAM 3 (default: "person")
                แนะนำ: "person", "face", "shirt", "dress", "hair"
            similarity_threshold: threshold สำหรับ face recognition
            return_all_members: คืนข้อมูลสมาชิกทั้งหมด
            
        Returns:
            Dict: {
                'overlay': ภาพต้นฉบับ + mask สี,
                'cutout': ภาพตัดออกมา (พื้นหลังโปร่งใส),
                'annotated': ภาพพร้อม bbox และ label,
                'mask': binary mask,
                'status': ข้อความสถานะ,
                'all_members': list ของสมาชิกทั้งหมด (optional)
            }
        """
        if member_name not in self.embeddings_db:
            return self._empty_result(f"❌ ไม่พบ embedding ของ '{member_name}'")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_bgr.shape[:2]
        
        # Step 1: Detect faces with InsightFace
        faces = self.face_analyzer.get(image_bgr)
        faces = [f for f in faces if f.det_score >= 0.45]
        
        if len(faces) == 0:
            return self._empty_result("❌ ไม่พบใบหน้าในภาพ")
        
        # Step 2: Identify all members
        all_members = []
        target = None
        
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            emb = face.normed_embedding if face.normed_embedding is not None else face.embedding
            emb = self.normalize_embedding(emb)
            
            name, sim = self.best_match(emb, threshold=similarity_threshold)
            
            member_info = {
                'name': name,
                'bbox': bbox,
                'similarity': float(sim),
                'det_score': float(face.det_score),
                'embedding': emb
            }
            all_members.append(member_info)
            
            if name == member_name:
                target = member_info
        
        if target is None:
            result = self._empty_result(f"❌ ไม่พบ {member_name} ในภาพ")
            if return_all_members:
                result['all_members'] = all_members
            return result
        
        # Update online gallery
        self.online_gallery[member_name].append(target['embedding'].copy())
        
        # Step 3: Expand face bbox to body bbox
        face_bbox = np.array(target['bbox'])
        body_bbox = self.face_to_body_bbox(face_bbox, (img_h, img_w))
        
        # Step 4: SAM 3 Segmentation with TEXT PROMPT! 🎯
        mask = self.segment_with_text_in_bbox(
            image_rgb,
            body_bbox.tolist(),
            text_prompt=text_prompt
        )
        
        # Step 5: Clean mask
        mask = self.clean_mask(mask, kernel_size=7)
        
        # Step 6: Create visualizations
        overlay = self.create_overlay(image_rgb, mask)
        cutout = self.create_cutout(image_rgb, mask)
        
        # Annotated image
        annotated = self.create_annotated(
            image_rgb, all_members, target, body_bbox, text_prompt
        )
        
        status = f"✅ Found {member_name} (sim={target['similarity']:.3f}) | SAM3: '{text_prompt}'"
        
        result = {
            'overlay': overlay,
            'cutout': cutout,
            'annotated': annotated,
            'mask': mask,
            'status': status,
        }
        
        if return_all_members:
            result['all_members'] = all_members
            
        return result
    
    def segment_specific_part(
        self,
        image_bgr: np.ndarray,
        member_name: str,
        part: str = "shirt",
        similarity_threshold: float = 0.45,
    ) -> Dict:
        """
        Segment เฉพาะส่วนของสมาชิก
        
        Args:
            image_bgr: Input image
            member_name: ชื่อสมาชิก
            part: ส่วนที่ต้องการ (shirt, hair, face, dress, pants, skirt, shoes)
            similarity_threshold: face recognition threshold
        """
        part_to_prompt = {
            'shirt': 'shirt',
            'dress': 'dress',
            'hair': 'hair',
            'face': 'face',
            'pants': 'pants',
            'skirt': 'skirt',
            'shoes': 'shoes',
            'jacket': 'jacket',
        }
        
        text_prompt = part_to_prompt.get(part.lower(), part)
        
        return self.segment_member(
            image_bgr=image_bgr,
            member_name=member_name,
            text_prompt=text_prompt,
            similarity_threshold=similarity_threshold,
        )
    
    def segment_multiple_prompts(
        self,
        image_bgr: np.ndarray,
        member_name: str,
        text_prompts: List[str],
        similarity_threshold: float = 0.45,
    ) -> Dict:
        """
        Segment ด้วยหลาย text prompts พร้อมกัน
        
        ตัวอย่าง: ["person", "shirt", "hair"] → ได้ mask แยกกัน
        """
        results = {}
        
        for prompt in text_prompts:
            result = self.segment_member(
                image_bgr=image_bgr,
                member_name=member_name,
                text_prompt=prompt,
                similarity_threshold=similarity_threshold,
            )
            results[prompt] = result
        
        return results
    
    def create_annotated(
        self,
        image_rgb: np.ndarray,
        all_members: List[Dict],
        target: Dict,
        body_bbox: np.ndarray,
        text_prompt: str
    ) -> np.ndarray:
        """สร้างภาพ annotated"""
        annotated = image_rgb.copy()
        
        # Draw all face bboxes
        for m in all_members:
            x1, y1, x2, y2 = m['bbox']
            is_target = (m['name'] == target['name'])
            color = (0, 255, 0) if is_target else (200, 200, 200)
            thickness = 3 if is_target else 1
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            label = f"{m['name']} ({m['similarity']:.2f})"
            cv2.putText(annotated, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw body bbox with SAM 3 info
        bx1, by1, bx2, by2 = body_bbox
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 0, 0), 2)
        prompt_label = f"SAM3: '{text_prompt}'"
        cv2.putText(annotated, prompt_label, (bx1, by1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return annotated
    
    def clean_mask(self, mask: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        """Clean mask ด้วย morphological operations"""
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255
        else:
            mask = (mask > 0).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Keep largest component
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_idx = np.argmax(areas) + 1
            mask = (labels == largest_idx).astype(np.uint8) * 255
        
        return (mask > 0).astype(np.uint8)
    
    def create_overlay(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        color: Optional[List[int]] = None,
        alpha: float = 0.5
    ) -> np.ndarray:
        """สร้างภาพ overlay"""
        if color is None:
            color = [0, 255, 128]
        
        overlay = image_rgb.copy()
        mask_bool = mask.astype(bool)
        color_layer = np.zeros_like(image_rgb)
        color_layer[mask_bool] = color
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, alpha, 0)
        return overlay
    
    def create_cutout(self, image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """สร้างภาพ cutout พื้นหลังโปร่งใส"""
        mask_bool = mask.astype(bool)
        alpha_channel = (mask_bool * 255).astype(np.uint8)
        cutout_rgba = np.dstack((image_rgb, alpha_channel))
        return cutout_rgba
    
    def _empty_result(self, status: str) -> Dict:
        """คืนค่า empty result"""
        return {
            'overlay': None,
            'cutout': None,
            'annotated': None,
            'mask': None,
            'status': status,
            'all_members': []
        }


# =============================================================================
# Video Processing with SAM3VideoSemanticPredictor
# =============================================================================

class SAM3VideoSegmenter(SAM3UltralyticsSegmenter):
    """
    สำหรับประมวลผลวิดีโอ ใช้ SAM3VideoSemanticPredictor
    """
    
    def __init__(
        self,
        face_analyzer,
        embeddings_db: Dict[str, np.ndarray],
        sam3_model_path: str = "sam3.pt",
        device: str = 'cuda',
    ):
        super().__init__(face_analyzer, embeddings_db, sam3_model_path, device)
        
        # โหลด Video Predictor
        try:
            from ultralytics.models.sam import SAM3VideoSemanticPredictor
            self.sam3_video_predictor = SAM3VideoSemanticPredictor(
                overrides=dict(model=sam3_model_path, device=device)
            )
            self.use_video = True
            print("✅ โหลด SAM3VideoSemanticPredictor สำเร็จ")
        except ImportError:
            self.sam3_video_predictor = None
            self.use_video = False
            print("⚠️ ไม่พบ SAM3VideoSemanticPredictor")
    
    def process_video(
        self,
        input_path: str,
        output_path: str,
        target_member: str,
        text_prompt: str = "person",
        frame_sampling: int = 5,
    ):
        """
        ประมวลผลวิดีโอ
        
        Args:
            input_path: path ไปยัง video input
            output_path: path สำหรับบันทึกผลลัพธ์
            target_member: ชื่อสมาชิกที่ต้องการ track
            text_prompt: noun phrase สำหรับ SAM 3
            frame_sampling: process ทุกกี่เฟรม
        """
        import os
        from tqdm import tqdm
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps / frame_sampling, (width, height))
        
        frame_idx = 0
        processed_count = 0
        
        with tqdm(total=total_frames, desc="Processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_sampling == 0:
                    # Process frame
                    result = self.segment_member(
                        image_bgr=frame,
                        member_name=target_member,
                        text_prompt=text_prompt
                    )
                    
                    if result['overlay'] is not None:
                        output_frame = cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR)
                    else:
                        output_frame = frame
                    
                    out.write(output_frame)
                    processed_count += 1
                
                frame_idx += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        print(f"\n✅ เสร็จสิ้น! ประมวลผล {processed_count} frames")
        print(f"   บันทึกที่: {output_path}")
        
        return output_path


# =============================================================================
# Helper Functions
# =============================================================================

def create_sam3_segmenter(face_analyzer, embeddings_db, model_path="sam3.pt"):
    """
    สร้าง SAM3UltralyticsSegmenter
    
    Usage:
        from sam3_ultralytics_pipeline import create_sam3_segmenter
        
        segmenter = create_sam3_segmenter(
            face_analyzer=face_analyzer,
            embeddings_db=embeddings_db,
            model_path="sam3.pt"
        )
        
        # Segment Wonyoung
        result = segmenter.segment_member(
            image_bgr=cv2.imread("input.jpg"),
            member_name="Wonyoung",
            text_prompt="person"
        )
    """
    return SAM3UltralyticsSegmenter(
        face_analyzer=face_analyzer,
        embeddings_db=embeddings_db,
        sam3_model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


def download_sam3_model():
    """ดาวน์โหลด sam3.pt จาก Hugging Face (ถ้ายังไม่มี)"""
    from huggingface_hub import hf_hub_download
    
    try:
        model_path = hf_hub_download(
            repo_id="facebook/sam3",
            filename="sam3.pt",
            cache_dir="./.sam3_cache"
        )
        print(f"✅ ดาวน์โหลด SAM 3 model: {model_path}")
        return model_path
    except Exception as e:
        print(f"❌ ไม่สามารถดาวน์โหลด SAM 3: {e}")
        print("   กรุณาดาวน์โหลดเองจาก https://huggingface.co/facebook/sam3")
        return "sam3.pt"


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("SAM 3 (Ultralytics) + InsightFace Pipeline")
    print("=" * 60)
    print("")
    print("ติดตั้ง:")
    print("  pip install -U ultralytics")
    print("")
    print("ดาวน์โหลดโมเดล:")
    print("  huggingface-cli download facebook/sam3 sam3.pt")
    print("")
    print("Usage ใน main.ipynb:")
    print("-" * 60)
    print("""
from sam3_ultralytics_pipeline import create_sam3_segmenter, download_sam3_model

# ดาวน์โหลดโมเดล (ถ้ายังไม่มี)
model_path = download_sam3_model()

# สร้าง segmenter
segmenter = create_sam3_segmenter(
    face_analyzer=face_analyzer,
    embeddings_db=embeddings_db,
    model_path=model_path
)

# 1. Segment Wonyoung ด้วย prompt "person"
result = segmenter.segment_member(
    image_bgr=cv2.imread("input.jpg"),
    member_name="Wonyoung",
    text_prompt="person"
)

# 2. Segment เฉพาะเสื้อ
result = segmenter.segment_specific_part(
    image_bgr=cv2.imread("input.jpg"),
    member_name="Wonyoung",
    part="shirt"
)

# 3. Segment หลายอย่างพร้อมกัน
results = segmenter.segment_multiple_prompts(
    image_bgr=cv2.imread("input.jpg"),
    member_name="Wonyoung",
    text_prompts=["person", "shirt", "hair"]
)

# แสดงผล
print(result['status'])
cv2.imwrite("output.png", cv2.cvtColor(result['overlay'], cv2.COLOR_RGB2BGR))
    """)
