# =============================================================================
# Ultralytics SAM + InsightFace Text-to-Segment Pipeline
# =============================================================================
# ใช้ InsightFace หาคน → ได้ BBox → ใช้ SAM (Ultralytics) Segment
# =============================================================================

import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque

# InsightFace
from insightface.app import FaceAnalysis

class IVEInsightFaceUltralyticsSAM3:
    """
    Pipeline ที่ผสม InsightFace (Face Recognition) + SAM (Segment)
    
    การทำงาน:
    1. InsightFace detect & recognize หน้า → ได้ BBox + ชื่อสมาชิก
    2. ขยาย BBox จากหน้าไปเป็นร่างกาย
    3. SAM segment ในบริเวณ BBox นั้น
    """
    
    def __init__(self, device='cuda', sam_model='sam_b.pt'):
        self.device = device
        
        # 1. โหลด InsightFace
        print("🚀 Loading InsightFace (buffalo_l)...")
        self.face_analyzer = FaceAnalysis(
            name='buffalo_l',
            root='./insightface_models',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        print("✅ InsightFace ready!")
        
        # 2. โหลด SAM จาก Ultralytics (ใช้ SAM ธรรมดา ไม่ใช่ SAM3)
        print(f"🚀 Loading SAM ({sam_model})...")
        try:
            from ultralytics import SAM
            self.sam_model = SAM(sam_model)
            print(f"✅ SAM model ({sam_model}) ready!")
        except Exception as e:
            print(f"⚠️ Could not load SAM: {e}")
            self.sam_model = None
        
        # Colors for visualization
        self.colors = {
            'Wonyoung': (255, 105, 180),  # Hot pink
            'Yujin': (0, 191, 255),       # Deep sky blue
            'Gaeul': (255, 165, 0),       # Orange
            'Liz': (148, 0, 211),         # Violet
            'Leeseo': (50, 205, 50),      # Lime green
            'Rei': (255, 20, 147)         # Deep pink
        }
    
    def face_to_body_bbox(self, face_bbox, img_shape, scale=3.0):
        """ขยาย face bbox ให้เป็น body bbox"""
        x1, y1, x2, y2 = face_bbox.astype(float)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        
        img_h, img_w = img_shape
        
        new_w = w * scale
        new_h = h * scale * 1.5
        
        new_x1 = max(0, cx - new_w/2)
        new_y1 = max(0, cy - new_h/3)
        new_x2 = min(img_w, cx + new_w/2)
        new_y2 = min(img_h, cy + new_h)
        
        return np.array([new_x1, new_y1, new_x2, new_y2], dtype=int)
    
    def detect_members(self, image_bgr, embeddings_db, threshold=0.40):
        """
        Detect และ recognize สมาชิก IVE ด้วย InsightFace
        """
        faces = self.face_analyzer.get(image_bgr)
        results = []
        
        for face in faces:
            if face.det_score < 0.5:
                continue
            
            emb = face.normed_embedding
            if emb is None:
                continue
            
            best_name = "Unknown"
            best_score = -1
            
            for name, ref_emb in embeddings_db.items():
                if isinstance(ref_emb, np.ndarray) and ref_emb.ndim == 2:
                    score = float(np.max(ref_emb @ emb))
                else:
                    score = float(np.dot(emb, ref_emb))
                
                if score > best_score:
                    best_score = score
                    best_name = name
            
            if best_score < threshold:
                best_name = "Unknown"
            
            results.append({
                'name': best_name,
                'bbox': face.bbox.astype(int).tolist(),
                'embedding': emb,
                'sim': best_score,
                'det_score': face.det_score
            })
        
        return results
    
    def segment_with_box(self, image_bgr, box):
        """
        Segment ด้วย box ใช้ SAM
        
        Args:
            image_bgr: numpy array (H, W, 3) BGR
            box: [x1, y1, x2, y2]
        
        Returns:
            mask: binary mask
        """
        if self.sam_model is None:
            print("❌ SAM model not available")
            return None
        
        # Predict with box
        results = self.sam_model(image_bgr, bboxes=[box])
        
        if results and len(results) > 0:
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                mask = results[0].masks.data[0].cpu().numpy()
                return mask
        
        return None
    
    def segment_member_objects(self, image_bgr, member_name, object_text, embeddings_db):
        """
        Segment วัตถุที่เกี่ยวข้องกับสมาชิก
        
        Args:
            image_bgr: input image BGR
            member_name: ชื่อสมาชิก (เช่น "Wonyoung")
            object_text: วัตถุที่ต้องการ (ใช้สำหรับแสดงผล label)
            embeddings_db: database ของ face embeddings
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_bgr.shape[:2]
        
        # 1. หาสมาชิกในภาพ
        members = self.detect_members(image_bgr, embeddings_db)
        
        target = None
        for m in members:
            if m['name'] == member_name:
                target = m
                break
        
        if target is None:
            return image_rgb, None, f"❌ ไม่พบ {member_name} ในภาพ"
        
        # 2. ขยาย bbox เป็น body bbox
        face_bbox = np.array(target['bbox'])
        body_bbox = self.face_to_body_bbox(face_bbox, (img_h, img_w), scale=3.5)
        
        # 3. Segment ด้วย box
        print(f"🎯 Segmenting {member_name}'s {object_text}...")
        mask = self.segment_with_box(image_bgr, body_bbox.tolist())
        
        if mask is None:
            return image_rgb, None, f"❌ Segmentation failed"
        
        # Resize mask ถ้าขนาดไม่ตรง
        if mask.shape != (img_h, img_w):
            mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
        
        # 4. สร้าง visualization
        color = self.colors.get(member_name, (0, 255, 0))
        
        # Draw overlay
        overlay = image_rgb.copy()
        mask_bool = mask.astype(bool)
        color_layer = np.zeros_like(image_rgb)
        color_layer[mask_bool] = color
        overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.5, 0)
        
        # Draw face box
        x1, y1, x2, y2 = target['bbox']
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{member_name} ({target['sim']:.2f})"
        cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw body box
        bx1, by1, bx2, by2 = body_bbox
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
        cv2.putText(overlay, f"Body: {object_text}", (bx1, by1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        status = f"✅ {member_name}'s {object_text}: found"
        
        return overlay, mask, status
    
    def segment_all_people(self, image_bgr, embeddings_db=None):
        """
        Segment ทุกคนในภาพ
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = image_bgr.shape[:2]
        
        if embeddings_db:
            members = self.detect_members(image_bgr, embeddings_db)
        else:
            faces = self.face_analyzer.get(image_bgr)
            members = [{'name': f'Person_{i}', 'bbox': f.bbox.astype(int).tolist(), 
                       'sim': f.det_score} for i, f in enumerate(faces)]
        
        annotated = image_rgb.copy()
        all_masks = []
        
        for m in members:
            face_bbox = np.array(m['bbox'])
            body_bbox = self.face_to_body_bbox(face_bbox, (img_h, img_w), scale=3.0)
            
            mask = self.segment_with_box(image_bgr, body_bbox.tolist())
            
            if mask is not None:
                if mask.shape != (img_h, img_w):
                    mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)
                
                all_masks.append(mask)
                
                color = self.colors.get(m['name'], tuple(np.random.randint(50, 200, 3).tolist()))
                mask_bool = mask.astype(bool)
                color_layer = np.zeros_like(image_rgb)
                color_layer[mask_bool] = color
                annotated = cv2.addWeighted(annotated, 1.0, color_layer, 0.4, 0)
            
            x1, y1, x2, y2 = m['bbox']
            color = self.colors.get(m['name'], (0, 255, 0))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            label = f"{m['name']}"
            if 'sim' in m and m['sim']:
                label += f" ({m['sim']:.2f})"
            cv2.putText(annotated, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated, all_masks


def create_text_to_segment_pipeline(device='cuda'):
    """สร้าง pipeline instance"""
    return IVEInsightFaceUltralyticsSAM3(device=device)
