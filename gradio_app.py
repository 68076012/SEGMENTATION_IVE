"""
Thai Text-to-Segment System - Enhanced Gradio UI
Master's Thesis Project

This module provides a comprehensive Gradio interface for the Thai Text-to-Segment
system with support for all 3 levels (basic, identity, possession), multiple
visualization options, and metrics display.
"""

import gradio as gr
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import cv2
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os


# =============================================================================
# Enums and Data Classes
# =============================================================================

class SegmentLevel(Enum):
    """Three levels of segmentation supported by the system."""
    BASIC = "basic"
    IDENTITY = "identity"
    POSSESSION = "possession"


class InputType(Enum):
    """Supported input types."""
    IMAGE = "image"
    VIDEO = "video"


@dataclass
class SegmentationResult:
    """Container for segmentation results."""
    mask: np.ndarray
    annotated_image: np.ndarray
    bounding_boxes: List[Dict[str, Any]]
    faces_detected: List[Dict[str, Any]]
    confidence: float
    iou_score: Optional[float] = None
    dice_score: Optional[float] = None
    pixel_accuracy: Optional[float] = None


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    iou: float
    dice: float
    pixel_accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time: float


# =============================================================================
# Mock Pipeline (Replace with actual implementation)
# =============================================================================

class ThaiSegmentationPipeline:
    """
    Mock Thai Text-to-Segment Pipeline.
    Replace this with the actual pipeline implementation.
    """
    
    def __init__(self):
        self.supported_levels = [SegmentLevel.BASIC, SegmentLevel.IDENTITY, SegmentLevel.POSSESSION]
        self.device = "cuda"  # or "cpu"
    
    def segment(
        self,
        image: np.ndarray,
        text_prompt: str,
        level: SegmentLevel = SegmentLevel.BASIC,
        show_face_detection: bool = False,
        show_bbox: bool = True,
        show_overlay: bool = True,
        overlay_alpha: float = 0.5
    ) -> SegmentationResult:
        """
        Perform segmentation based on text prompt.
        
        Args:
            image: Input image as numpy array
            text_prompt: Text description in Thai or English
            level: Segmentation level (basic/identity/possession)
            show_face_detection: Whether to detect and mark faces
            show_bbox: Whether to show bounding boxes
            show_overlay: Whether to show mask overlay
            overlay_alpha: Transparency of mask overlay
            
        Returns:
            SegmentationResult containing mask and annotations
        """
        # Mock implementation - replace with actual segmentation logic
        h, w = image.shape[:2]
        
        # Create mock mask (centered ellipse)
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (w // 4, h // 4)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        
        # Create annotated image
        annotated = image.copy()
        
        # Mock bounding boxes
        bboxes = [{
            'x1': w // 4,
            'y1': h // 4,
            'x2': 3 * w // 4,
            'y2': 3 * h // 4,
            'label': text_prompt,
            'confidence': 0.85
        }]
        
        # Mock face detection
        faces = []
        if show_face_detection:
            faces = [{
                'x': w // 2 - 50,
                'y': h // 2 - 50,
                'width': 100,
                'height': 100,
                'confidence': 0.92
            }]
        
        # Apply visualizations
        if show_overlay:
            overlay = np.zeros_like(image)
            overlay[mask > 0] = [0, 255, 0]  # Green overlay
            annotated = cv2.addWeighted(annotated, 1, overlay, overlay_alpha, 0)
        
        if show_bbox:
            for bbox in bboxes:
                cv2.rectangle(
                    annotated,
                    (bbox['x1'], bbox['y1']),
                    (bbox['x2'], bbox['y2']),
                    (255, 0, 0),  # Blue
                    2
                )
                cv2.putText(
                    annotated,
                    f"{bbox['label']}: {bbox['confidence']:.2f}",
                    (bbox['x1'], bbox['y1'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )
        
        if show_face_detection and faces:
            for face in faces:
                cv2.rectangle(
                    annotated,
                    (face['x'], face['y']),
                    (face['x'] + face['width'], face['y'] + face['height']),
                    (0, 0, 255),  # Red
                    2
                )
                cv2.putText(
                    annotated,
                    "Face",
                    (face['x'], face['y'] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )
        
        return SegmentationResult(
            mask=mask,
            annotated_image=annotated,
            bounding_boxes=bboxes,
            faces_detected=faces,
            confidence=0.85,
            iou_score=0.78,
            dice_score=0.82,
            pixel_accuracy=0.91
        )
    
    def segment_batch(
        self,
        images: List[np.ndarray],
        text_prompt: str,
        level: SegmentLevel = SegmentLevel.BASIC,
        **kwargs
    ) -> List[SegmentationResult]:
        """Process multiple images."""
        return [self.segment(img, text_prompt, level, **kwargs) for img in images]


# =============================================================================
# Utility Functions
# =============================================================================

def calculate_metrics(pred_mask: np.ndarray, gt_mask: Optional[np.ndarray] = None) -> MetricsResult:
    """
    Calculate segmentation metrics.
    
    Args:
        pred_mask: Predicted segmentation mask
        gt_mask: Ground truth mask (optional)
        
    Returns:
        MetricsResult with calculated metrics
    """
    if gt_mask is None:
        # Return mock metrics if no ground truth
        return MetricsResult(
            iou=0.78,
            dice=0.82,
            pixel_accuracy=0.91,
            precision=0.85,
            recall=0.80,
            f1_score=0.82,
            inference_time=0.15
        )
    
    # Calculate actual metrics
    pred_binary = (pred_mask > 0).astype(np.uint8)
    gt_binary = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (pred_binary.sum() + gt_binary.sum() + 1e-8)
    pixel_acc = (pred_binary == gt_binary).sum() / pred_binary.size
    
    tp = intersection
    fp = pred_binary.sum() - intersection
    fn = gt_binary.sum() - intersection
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return MetricsResult(
        iou=iou,
        dice=dice,
        pixel_accuracy=pixel_acc,
        precision=precision,
        recall=recall,
        f1_score=f1,
        inference_time=0.15
    )


def create_comparison_view(
    original: np.ndarray,
    result: SegmentationResult,
    ground_truth: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Create side-by-side comparison view.
    
    Args:
        original: Original input image
        result: Segmentation result
        ground_truth: Optional ground truth mask
        
    Returns:
        Concatenated comparison image
    """
    h, w = original.shape[:2]
    
    # Prepare images for comparison
    images = [original, result.annotated_image]
    labels = ["Original", "Segmented"]
    
    if ground_truth is not None:
        gt_vis = original.copy()
        overlay = np.zeros_like(original)
        overlay[ground_truth > 0] = [255, 0, 255]  # Magenta
        gt_vis = cv2.addWeighted(gt_vis, 1, overlay, 0.5, 0)
        images.append(gt_vis)
        labels.append("Ground Truth")
    
    # Add mask visualization
    mask_vis = np.zeros_like(original)
    mask_vis[result.mask > 0] = [0, 255, 255]  # Cyan
    images.append(mask_vis)
    labels.append("Mask Only")
    
    # Create comparison grid
    n_images = len(images)
    grid_w = w * n_images
    grid = np.zeros((h + 40, grid_w, 3), dtype=np.uint8)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        x_offset = i * w
        grid[:h, x_offset:x_offset + w] = img
        cv2.putText(
            grid,
            label,
            (x_offset + 10, h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
    
    return grid


def format_metrics_html(metrics: MetricsResult) -> str:
    """Format metrics as HTML for display."""
    return f"""
    <div style="font-family: Arial, sans-serif; padding: 10px;">
        <h4>Segmentation Metrics</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #f0f0f0;">
                <td style="padding: 8px; border: 1px solid #ddd;"><b>IoU (Jaccard)</b></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{metrics.iou:.4f}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><b>Dice Coefficient</b></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{metrics.dice:.4f}</td>
            </tr>
            <tr style="background: #f0f0f0;">
                <td style="padding: 8px; border: 1px solid #ddd;"><b>Pixel Accuracy</b></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{metrics.pixel_accuracy:.4f}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><b>Precision</b></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{metrics.precision:.4f}</td>
            </tr>
            <tr style="background: #f0f0f0;">
                <td style="padding: 8px; border: 1px solid #ddd;"><b>Recall</b></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{metrics.recall:.4f}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><b>F1 Score</b></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{metrics.f1_score:.4f}</td>
            </tr>
            <tr style="background: #f0f0f0;">
                <td style="padding: 8px; border: 1px solid #ddd;"><b>Inference Time</b></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{metrics.inference_time:.3f}s</td>
            </tr>
        </table>
    </div>
    """


# =============================================================================
# Gradio Interface Functions
# =============================================================================

# Global pipeline instance
_pipeline: Optional[ThaiSegmentationPipeline] = None


def get_pipeline() -> ThaiSegmentationPipeline:
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ThaiSegmentationPipeline()
    return _pipeline


def process_single_image(
    image: np.ndarray,
    text_prompt: str,
    level: str,
    show_face_detection: bool,
    show_bbox: bool,
    show_overlay: bool,
    overlay_alpha: float,
    ground_truth: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, str, np.ndarray]:
    """
    Process a single image through the segmentation pipeline.
    
    Args:
        image: Input image
        text_prompt: Text description
        level: Segmentation level
        show_face_detection: Enable face detection
        show_bbox: Show bounding boxes
        show_overlay: Show mask overlay
        overlay_alpha: Overlay transparency
        ground_truth: Optional ground truth mask
        
    Returns:
        Tuple of (annotated_image, mask, metrics_html, comparison_view)
    """
    if image is None:
        return None, None, "Please upload an image.", None
    
    if not text_prompt.strip():
        return image, None, "Please enter a text prompt.", None
    
    pipeline = get_pipeline()
    
    # Convert level string to enum
    level_enum = SegmentLevel(level)
    
    # Run segmentation
    result = pipeline.segment(
        image=image,
        text_prompt=text_prompt,
        level=level_enum,
        show_face_detection=show_face_detection,
        show_bbox=show_bbox,
        show_overlay=show_overlay,
        overlay_alpha=overlay_alpha
    )
    
    # Calculate metrics
    metrics = calculate_metrics(result.mask, ground_truth)
    metrics_html = format_metrics_html(metrics)
    
    # Create comparison view
    comparison = create_comparison_view(image, result, ground_truth)
    
    return result.annotated_image, result.mask, metrics_html, comparison


def process_batch(
    images: List[np.ndarray],
    text_prompt: str,
    level: str,
    show_face_detection: bool,
    show_bbox: bool,
    show_overlay: bool,
    overlay_alpha: float
) -> List[Tuple[np.ndarray, str]]:
    """
    Process multiple images in batch.
    
    Args:
        images: List of input images
        text_prompt: Text description
        level: Segmentation level
        show_face_detection: Enable face detection
        show_bbox: Show bounding boxes
        show_overlay: Show mask overlay
        overlay_alpha: Overlay transparency
        
    Returns:
        List of (annotated_image, metrics_summary) tuples
    """
    if not images:
        return []
    
    if not text_prompt.strip():
        return [(img, "No prompt provided") for img in images]
    
    pipeline = get_pipeline()
    level_enum = SegmentLevel(level)
    
    results = pipeline.segment_batch(
        images=images,
        text_prompt=text_prompt,
        level=level_enum,
        show_face_detection=show_face_detection,
        show_bbox=show_bbox,
        show_overlay=show_overlay,
        overlay_alpha=overlay_alpha
    )
    
    output = []
    for img, result in zip(images, results):
        metrics = calculate_metrics(result.mask)
        summary = f"IoU: {metrics.iou:.3f}, Dice: {metrics.dice:.3f}, Time: {metrics.inference_time:.3f}s"
        output.append((result.annotated_image, summary))
    
    return output


def compare_levels(
    image: np.ndarray,
    text_prompt: str,
    show_face_detection: bool,
    show_bbox: bool,
    show_overlay: bool,
    overlay_alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Compare all three segmentation levels side by side.
    
    Args:
        image: Input image
        text_prompt: Text description
        show_face_detection: Enable face detection
        show_bbox: Show bounding boxes
        show_overlay: Show mask overlay
        overlay_alpha: Overlay transparency
        
    Returns:
        Tuple of (basic_result, identity_result, possession_result, comparison_metrics)
    """
    if image is None or not text_prompt.strip():
        return None, None, None, "Please provide both image and text prompt."
    
    pipeline = get_pipeline()
    results = {}
    metrics_dict = {}
    
    for level in SegmentLevel:
        result = pipeline.segment(
            image=image,
            text_prompt=text_prompt,
            level=level,
            show_face_detection=show_face_detection,
            show_bbox=show_bbox,
            show_overlay=show_overlay,
            overlay_alpha=overlay_alpha
        )
        results[level] = result
        metrics = calculate_metrics(result.mask)
        metrics_dict[level.value] = metrics
    
    # Format comparison metrics
    comparison_html = """
    <div style="font-family: Arial, sans-serif; padding: 10px;">
        <h4>Level Comparison Metrics</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr style="background: #333; color: white;">
                <th style="padding: 8px; border: 1px solid #ddd;">Level</th>
                <th style="padding: 8px; border: 1px solid #ddd;">IoU</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Dice</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Accuracy</th>
                <th style="padding: 8px; border: 1px solid #ddd;">Time (s)</th>
            </tr>
    """
    
    colors = {"basic": "#e3f2fd", "identity": "#f3e5f5", "possession": "#e8f5e9"}
    
    for level_name, metrics in metrics_dict.items():
        bg_color = colors.get(level_name, "white")
        comparison_html += f"""
            <tr style="background: {bg_color};">
                <td style="padding: 8px; border: 1px solid #ddd; text-transform: capitalize;"><b>{level_name}</b></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{metrics.iou:.4f}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{metrics.dice:.4f}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{metrics.pixel_accuracy:.4f}</td>
                <td style="padding: 8px; border: 1px solid #ddd;">{metrics.inference_time:.3f}</td>
            </tr>
        """
    
    comparison_html += "</table></div>"
    
    return (
        results[SegmentLevel.BASIC].annotated_image,
        results[SegmentLevel.IDENTITY].annotated_image,
        results[SegmentLevel.POSSESSION].annotated_image,
        comparison_html
    )


# =============================================================================
# Create Gradio Interface
# =============================================================================

def create_gradio_interface(pipeline: Optional[ThaiSegmentationPipeline] = None) -> gr.Blocks:
    """
    Create enhanced Gradio interface for Thai Text-to-Segment System.
    
    Args:
        pipeline: Optional pre-configured pipeline instance
        
    Returns:
        gr.Blocks: The configured Gradio interface
    """
    global _pipeline
    if pipeline is not None:
        _pipeline = pipeline
    
    # Custom CSS for better styling
    custom_css = """
    .segmentation-output {
        border: 2px solid #4CAF50;
        border-radius: 8px;
    }
    .metrics-panel {
        background: #f5f5f5;
        border-radius: 8px;
        padding: 10px;
    }
    .level-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
    }
    .level-basic { background: #e3f2fd; color: #1565c0; }
    .level-identity { background: #f3e5f5; color: #7b1fa2; }
    .level-possession { background: #e8f5e9; color: #2e7d32; }
    """
    
    with gr.Blocks(css=custom_css, title="Thai Text-to-Segment System") as interface:
        
        # Header
        gr.Markdown("""
        # 🎯 Thai Text-to-Segment System
        ### Master's Thesis Project - Enhanced Segmentation Interface
        
        This interface supports **three levels of segmentation**:
        - **Basic**: Simple object segmentation
        - **Identity**: Person-aware segmentation with identity preservation
        - **Possession**: Segmentation of objects and their possessions/attributes
        """)
        
        # Main tabs
        with gr.Tabs() as tabs:
            
            # ==================== TAB 1: Single Image Segmentation ====================
            with gr.TabItem("🖼️ Segment", id="segment"):
                gr.Markdown("### Single Image Segmentation")
                
                with gr.Row():
                    # Left column: Inputs
                    with gr.Column(scale=1):
                        gr.Markdown("#### Input Settings")
                        
                        input_image = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            height=300
                        )
                        
                        text_prompt = gr.Textbox(
                            label="Text Prompt (Thai/English)",
                            placeholder="e.g., 'คนใส่เสื้อสีแดง' or 'person wearing red shirt'",
                            lines=2
                        )
                        
                        segmentation_level = gr.Radio(
                            choices=["basic", "identity", "possession"],
                            value="basic",
                            label="Segmentation Level"
                        )
                        
                        with gr.Accordion("⚙️ Visualization Options", open=True):
                            show_face_detection = gr.Checkbox(
                                label="Show Face Detection",
                                value=False
                            )
                            show_bbox = gr.Checkbox(
                                label="Show Bounding Boxes",
                                value=True
                            )
                            show_overlay = gr.Checkbox(
                                label="Show Mask Overlay",
                                value=True
                            )
                            overlay_alpha = gr.Slider(
                                label="Overlay Transparency",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.1
                            )
                        
                        segment_btn = gr.Button(
                            "🚀 Run Segmentation",
                            variant="primary",
                            size="lg"
                        )
                    
                    # Right column: Outputs
                    with gr.Column(scale=2):
                        gr.Markdown("#### Results")
                        
                        with gr.Row():
                            output_image = gr.Image(
                                label="Segmented Result",
                                type="numpy",
                                elem_classes=["segmentation-output"]
                            )
                            output_mask = gr.Image(
                                label="Mask",
                                type="numpy"
                            )
                        
                        metrics_display = gr.HTML(
                            label="Metrics",
                            elem_classes=["metrics-panel"]
                        )
                        
                        with gr.Accordion("📊 Comparison View", open=False):
                            comparison_view = gr.Image(
                                label="Side-by-Side Comparison",
                                type="numpy"
                            )
                
                # Example prompts
                gr.Markdown("#### 📝 Example Prompts")
                examples = gr.Examples(
                    examples=[
                        ["คนใส่เสื้อสีแดง", "basic"],
                        ["person wearing red shirt", "basic"],
                        ["ผู้หญิงถือกระเป๋า", "possession"],
                        ["woman holding a bag", "possession"],
                        ["ชายสวมแว่นตา", "identity"],
                        ["man wearing glasses", "identity"],
                    ],
                    inputs=[text_prompt, segmentation_level],
                    label="Click to use example"
                )
                
                # Event handlers
                segment_btn.click(
                    fn=process_single_image,
                    inputs=[
                        input_image, text_prompt, segmentation_level,
                        show_face_detection, show_bbox, show_overlay,
                        overlay_alpha
                    ],
                    outputs=[output_image, output_mask, metrics_display, comparison_view]
                )
            
            # ==================== TAB 2: Level Comparison ====================
            with gr.TabItem("🔍 Compare", id="compare"):
                gr.Markdown("### Compare All Three Segmentation Levels")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        compare_image = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            height=300
                        )
                        compare_prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Enter prompt to compare across all levels...",
                            lines=2
                        )
                        
                        with gr.Accordion("⚙️ Visualization Options", open=True):
                            compare_face = gr.Checkbox(
                                label="Show Face Detection",
                                value=False
                            )
                            compare_bbox = gr.Checkbox(
                                label="Show Bounding Boxes",
                                value=True
                            )
                            compare_overlay = gr.Checkbox(
                                label="Show Mask Overlay",
                                value=True
                            )
                            compare_alpha = gr.Slider(
                                label="Overlay Transparency",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.1
                            )
                        
                        compare_btn = gr.Button(
                            "🔬 Compare Levels",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Comparison Results")
                        
                        with gr.Row():
                            basic_output = gr.Image(
                                label="Basic Level",
                                type="numpy"
                            )
                            identity_output = gr.Image(
                                label="Identity Level",
                                type="numpy"
                            )
                            possession_output = gr.Image(
                                label="Possession Level",
                                type="numpy"
                            )
                        
                        compare_metrics = gr.HTML(
                            label="Level Comparison Metrics"
                        )
                
                compare_btn.click(
                    fn=compare_levels,
                    inputs=[
                        compare_image, compare_prompt,
                        compare_face, compare_bbox, compare_overlay, compare_alpha
                    ],
                    outputs=[basic_output, identity_output, possession_output, compare_metrics]
                )
            
            # ==================== TAB 3: Batch Processing ====================
            with gr.TabItem("📁 Batch", id="batch"):
                gr.Markdown("### Batch Image Processing")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_images = gr.File(
                            label="Upload Multiple Images",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        batch_prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Prompt applied to all images...",
                            lines=2
                        )
                        batch_level = gr.Radio(
                            choices=["basic", "identity", "possession"],
                            value="basic",
                            label="Segmentation Level"
                        )
                        
                        with gr.Accordion("⚙️ Visualization Options", open=True):
                            batch_face = gr.Checkbox(
                                label="Show Face Detection",
                                value=False
                            )
                            batch_bbox = gr.Checkbox(
                                label="Show Bounding Boxes",
                                value=True
                            )
                            batch_overlay = gr.Checkbox(
                                label="Show Mask Overlay",
                                value=True
                            )
                            batch_alpha = gr.Slider(
                                label="Overlay Transparency",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.1
                            )
                        
                        batch_btn = gr.Button(
                            "⚡ Process Batch",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column(scale=2):
                        gr.Markdown("#### Batch Results")
                        batch_gallery = gr.Gallery(
                            label="Segmented Images",
                            columns=3,
                            rows=2,
                            height="auto"
                        )
                        batch_status = gr.Textbox(
                            label="Processing Status",
                            interactive=False
                        )
            
            # ==================== TAB 4: Help/Documentation ====================
            with gr.TabItem("❓ Help", id="help"):
                gr.Markdown("""
                ## 📖 How to Use
                
                ### Segmentation Levels
                
                1. **Basic Level** (`basic`)
                   - Simple object segmentation based on text description
                   - Best for: General object detection and segmentation
                   - Example: "car", "dog", "tree"
                
                2. **Identity Level** (`identity`)
                   - Person-aware segmentation with identity preservation
                   - Best for: Segmenting specific people, facial features
                   - Example: "man wearing glasses", "woman with long hair"
                
                3. **Possession Level** (`possession`)
                   - Segmentation of objects and their possessions/attributes
                   - Best for: Complex scenes with object relationships
                   - Example: "person holding a phone", "child with a toy"
                
                ### Visualization Options
                
                - **Face Detection**: Detects and highlights faces in the image
                - **Bounding Boxes**: Shows detection boxes around segmented objects
                - **Mask Overlay**: Overlays the segmentation mask on the original image
                - **Overlay Transparency**: Controls the opacity of the mask overlay
                
                ### Metrics Explanation
                
                - **IoU (Intersection over Union)**: Measures overlap between predicted and ground truth masks
                - **Dice Coefficient**: Similar to IoU but gives more weight to overlap
                - **Pixel Accuracy**: Percentage of correctly classified pixels
                - **Precision**: Ratio of true positives to total predicted positives
                - **Recall**: Ratio of true positives to total actual positives
                - **F1 Score**: Harmonic mean of precision and recall
                
                ### Supported Languages
                
                - Thai: ภาษาไทย
                - English
                
                ### Tips for Better Results
                
                1. Use clear and specific prompts
                2. Choose the appropriate segmentation level for your task
                3. Adjust overlay transparency for better visibility
                4. Use face detection for person-related queries
                """)
        
        # Footer
        gr.Markdown("""
        ---
        <div style="text-align: center; color: #666;">
            Thai Text-to-Segment System | Master's Thesis Project | 2024
        </div>
        """)
    
    return interface


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Launch the Gradio interface."""
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    )


if __name__ == "__main__":
    main()
