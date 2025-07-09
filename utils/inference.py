from typing import List, Dict, Tuple, Any
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append("./")

from utils.facenet import apply_nms, load_retinaface,  detect_faces
from utils.AgeGenderRaceNet import predict_bbox, load_model

def preprocess_for_retinaface(image_bgr: np.ndarray, device: str = 'cuda') -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Preprocess image for RetinaFace detection"""
    img_raw = image_bgr.copy()
    img = np.float32(image_bgr)
    
    if len(img.shape) != 3:
        raise ValueError(f"Expected 3D image (H, W, C), got shape: {img.shape}")
    
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
    
    img -= np.array([104, 117, 123], dtype=np.float32)  # BGR mean subtraction
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
    return img_tensor, img_raw, scale

def preprocess_for_bbox(face_crop: np.ndarray, img_size: int = 224) -> torch.Tensor:
    """Preprocess face crop for gender classification"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Convert BGR to RGB and to PIL
    face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    
    # Apply transforms
    face_tensor = transform(face_pil).unsqueeze(0)
    
    return face_tensor

def predict(image_bgr: np.ndarray, retinaface_net: Any, retinaface_cfg: Any, 
                             model: Any, device: str = 'cuda', 
                             face_threshold: float = 0.6):
    """
    Complete pipeline: face detection + gender classification
    
    Args:
        image_bgr: Input image in BGR format
        retinaface_net: Loaded RetinaFace model
        retinaface_cfg: RetinaFace configuration
        gender_model: Loaded gender classification model
        gender_indices: Tuple of (start, end) indices for gender logits
        device: Device to run inference on
        face_threshold: Confidence threshold for face detection
    
    Returns:
        List of CVAT format results: [{"type": "rectangle", "label": "person_info", "confidence": "0.95", "points": [x1,y1,x2,y2], "attributes": {}}, ...]
    """
    
    h, w, _ = image_bgr.shape
    
    # Step 1: Preprocess for face detection
    img_tensor, img_raw, scale = preprocess_for_retinaface(image_bgr, device)
    
    # Step 2: Detect faces
    boxes, scores = detect_faces(
        img_tensor=img_tensor,
        net=retinaface_net,
        cfg=retinaface_cfg,
        scale=scale,
        device=device
    )
    
    # Step 3: Filter faces by confidence
    valid_faces = np.where(np.array(scores) > face_threshold)[0]
    
    if len(valid_faces) == 0:
        return []
    
    filtered_boxes = boxes[valid_faces].astype(int)
    filtered_scores = scores[valid_faces]
    
    # Step 4: Extract face crops and preprocess for gender
    input_bbox: List[torch.Tensor] = []
    valid_face_data: List[Tuple[List[int], float]] = [] 
    
    nms_indices = apply_nms(filtered_boxes, filtered_scores)
    filtered_boxes = filtered_boxes[nms_indices]
    filtered_scores = filtered_scores[nms_indices]

    for box, face_score in zip(filtered_boxes, filtered_scores):
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are within bounds
        x1 = max(0, x1)
        y1 = max(0, y1) 
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Skip if invalid bounding box
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Extract face crop
        face_crop = img_raw[y1:y2, x1:x2]
        
        # Check if face crop is valid (not empty and has reasonable size)
        if face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
            # Preprocess for gender classification
            gender_input = preprocess_for_bbox(face_crop)
            input_bbox.append(gender_input.to(device))
            valid_face_data.append(([x1, y1, x2, y2], face_score))
    
    # Step 5: Predict gender for all faces
    if not input_bbox:
        return [], []
    
    prdictions = predict_bbox(
        input_bbox,
        model=model,
        device=device
    )
    
    return valid_face_data, prdictions