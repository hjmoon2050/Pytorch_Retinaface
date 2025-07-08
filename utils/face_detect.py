from data.config import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
import torch
import numpy as np
import cv2

from layers.functions.prior_box import PriorBox
from utils.box_utils import decode

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_retinaface(network="resnet50", trained_model="./weights/Resnet50_Final.pth", device='cuda'):
    """Load RetinaFace model"""
    if network == "mobile0.25":
        cfg = cfg_mnet
    elif network == "resnet50":
        cfg = cfg_re50
    else:
        raise ValueError("Unsupported network type")

    net = RetinaFace(cfg=cfg, phase='test')
    pretrained_dict = torch.load(trained_model, map_location=lambda storage, loc: storage)
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    net.load_state_dict(pretrained_dict, strict=False)
    net.eval()
    net.to(device)
    return net, cfg

def detect_faces(img_tensor, net, cfg, scale, device='cuda'):
    """
    Detect faces from preprocessed image tensor
    
    Args:
        img_tensor: Preprocessed image tensor (1, C, H, W)
        net: RetinaFace model
        cfg: RetinaFace configuration
        scale: Scale tensor for box coordinates
        device: Device to run on
        
    Returns:
        boxes: numpy array of bounding boxes (N, 4)
        scores: numpy array of confidence scores (N,)
    """
    # Forward pass
    loc, conf, landms = net(img_tensor)
    
    # Get image dimensions from tensor
    im_height, im_width = img_tensor.shape[2], img_tensor.shape[3]
    
    # Decode predictions
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward().to(device)
    boxes = decode(loc.data.squeeze(0), priors.data, cfg['variance'])
    boxes = boxes * scale
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    
    return boxes, scores

# Legacy function for backward compatibility
def get_face_crops(img, net, cfg, device='cuda', confidence_threshold=0.6):
    """
    Legacy function: Get face crops with preprocessing included
    
    Note: This function does preprocessing internally.
    For new code, use preprocess_for_retinaface() + detect_faces() instead.
    """
    # Preprocess image
    img_raw = img.copy()
    img = np.float32(img)
    
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(device)
    
    # RetinaFace preprocessing
    img -= np.array([104, 117, 123], dtype=np.float32)
    img = img.transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    
    # Detect faces
    boxes, scores = detect_faces(img_tensor, net, cfg, scale, device)
    
    # Filter and extract crops
    inds = np.where(scores > confidence_threshold)[0]
    filtered_boxes = boxes[inds]
    
    face_crops = []
    for box in filtered_boxes.astype(int):
        x1, y1, x2, y2 = box
        
        # Ensure coordinates are within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(im_width, x2)
        y2 = min(im_height, y2)
        
        # Extract face crop
        if x2 > x1 and y2 > y1:
            face = img_raw[y1:y2, x1:x2]
            face_crops.append(face)
    
    return face_crops