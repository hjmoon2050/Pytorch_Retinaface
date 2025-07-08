from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import sys
sys.path.append("./")

from utils.face_detect import detect_faces
from utils.gender_predict import predict_gender_batch

def preprocess_for_retinaface(image_bgr: np.ndarray, device: str = 'cuda') -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Preprocess image for RetinaFace detection"""
    img_raw = image_bgr.copy()
    img = np.float32(image_bgr)
    
    # Ensure image has 3 dimensions
    if len(img.shape) != 3:
        raise ValueError(f"Expected 3D image (H, W, C), got shape: {img.shape}")
    
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
    
    # RetinaFace preprocessing - use numpy array instead of tuple
    img -= np.array([104, 117, 123], dtype=np.float32)  # BGR mean subtraction
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
    return img_tensor, img_raw, scale

def preprocess_for_gender(face_crop: np.ndarray, img_size: int = 224) -> torch.Tensor:
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

def detect_and_classify_gender(image_bgr: np.ndarray, retinaface_net, retinaface_cfg, 
                             gender_model, gender_indices: Tuple[int, int], device: str = 'cuda', 
                             face_threshold: float = 0.6) -> List[Dict]:
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
        List of CVAT format results: [{"type": "rectangle", "label": "male", "confidence": "0.95", "points": [x1,y1,x2,y2], "attributes": {}}, ...]
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
    valid_faces = np.where(scores > face_threshold)[0]
    
    if len(valid_faces) == 0:
        return []
    
    filtered_boxes = boxes[valid_faces].astype(int)
    filtered_scores = scores[valid_faces]
    
    # Step 4: Extract face crops and preprocess for gender
    gender_inputs = []
    valid_face_data = []  # Store box coordinates and scores
    
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
            gender_input = preprocess_for_gender(face_crop)
            gender_inputs.append(gender_input.to(device))
            valid_face_data.append(([x1, y1, x2, y2], face_score))
    
    # Step 5: Predict gender for all faces
    if not gender_inputs:
        return []
    
    gender_predictions = predict_gender_batch(
        gender_tensors=gender_inputs,
        model=gender_model,
        gender_indices=gender_indices,
        device=device
    )
    
    # Step 6: Format results in CVAT format
    results = []
    for i, ((x1, y1, x2, y2), face_score) in enumerate(valid_face_data):
        if i < len(gender_predictions):
            gender_pred = gender_predictions[i]
            
            results.append({
                "type": "rectangle",
                "label": gender_pred['gender'],
                "confidence": f"{gender_pred['confidence']:.4f}",
                "points": [float(x1), float(y1), float(x2), float(y2)],
                "attributes": {
                    "gender": gender_pred['gender'],
                    "gender_confidence": f"{gender_pred['confidence']:.4f}",
                    "face_confidence": f"{face_score:.4f}"
                }
            })
    
    return results


# Test function
def test_face_gender_detection():
    """Test function for face detection and gender classification"""
    import os
    
    # Test image path
    test_image_path = "./curve/test.jpg"  # Change to your test image path
    
    if not os.path.exists(test_image_path):
        print(f"❌ Test image not found: {test_image_path}")
        print("Please provide a test image path")
        return
    
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        from face_detect import load_retinaface
        from gender_predict import load_gender_model
        
        # Load RetinaFace
        retinaface_net, retinaface_cfg = load_retinaface(
            network="resnet50",
            trained_model="./weights/Resnet50_Final.pth",  # Change to your path
            device=device
        )
        print("✅ RetinaFace loaded")
        
        # Load Gender model
        gender_model, gender_indices = load_gender_model(
            "./weights/res34_fair_align_multi_4_20190809.pt",  # Change to your path
            device=device
        )
        print("✅ Gender model loaded")
        
        # Step 2: Load and process test image
        print(f"📷 Loading test image: {test_image_path}")
        img_bgr = cv2.imread(test_image_path)
        
        if img_bgr is None:
            print("❌ Failed to load image")
            return
        
        print(f"📐 Image shape: {img_bgr.shape}")
        
        # Step 3: Run detection and classification
        print("🔍 Running face detection and gender classification...")
        
        results = detect_and_classify_gender(
            image_bgr=img_bgr,
            retinaface_net=retinaface_net,
            retinaface_cfg=retinaface_cfg,
            gender_model=gender_model,
            gender_indices=gender_indices,
            device=device,
            face_threshold=0.6
        )
        
        # Step 4: Display results
        print(f"\n🎯 Results: Found {len(results)} faces")
        print("=" * 70)
        
        if results:
            for i, result in enumerate(results, 1):
                x1, y1, x2, y2 = result['points']
                gender = result['label']
                confidence = result['confidence']
                face_conf = result['attributes']['face_confidence']
                
                print(f"Face {i}:")
                print(f"  Gender: {gender} (confidence: {confidence})")
                print(f"  Face detection confidence: {face_conf}")
                print(f"  Bounding box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                print(f"  Size: {x2-x1:.0f}x{y2-y1:.0f}")
                print()
        else:
            print("No faces detected")
        
        print("=" * 70)
        print("✅ Test completed successfully!")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run test
    test_results = test_face_gender_detection()
    
    # Example with custom image path
    # custom_image = "./your_test_image.jpg"
    # if os.path.exists(custom_image):
    #     img = cv2.imread(custom_image)
    #     # Load your models here...
    #     # results = detect_and_classify_gender(img, ...)
    #     # print(results)