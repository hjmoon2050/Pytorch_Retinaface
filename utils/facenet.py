from typing import List
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

def detect_faces(img_tensor, net, cfg, scale, device='cuda', face_threshold: float = 0.6, nms_threshold: float = 0.4):
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
    
    nms_indices = apply_nms(boxes, scores, face_threshold, nms_threshold)
    
    final_boxes = boxes[nms_indices].astype(int)
    final_scores = scores[nms_indices]
    
    if len(nms_indices) == 0:
        return [], []
    
    return final_boxes, final_scores


def nms(dets, thresh):
    """Pure Python NMS baseline from Fast R-CNN."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def apply_nms(boxes: np.ndarray, scores: np.ndarray, score_threshold: float = 0.6, nms_threshold: float = 0.4) -> List[int]:
    """
    Apply Non-Maximum Suppression using Fast R-CNN's py_cpu_nms
    
    Args:
        boxes: Array of bounding boxes [N, 4] in format [x1, y1, x2, y2]
        scores: Array of confidence scores [N]
        score_threshold: Minimum score threshold
        nms_threshold: IoU threshold for NMS
    
    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []
    
    # Filter by score threshold first
    valid_indices = np.where(scores > score_threshold)[0]
    if len(valid_indices) == 0:
        return []
    
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]
    
    # Combine boxes and scores into the format expected by py_cpu_nms
    # dets format: [x1, y1, x2, y2, score]
    dets = np.column_stack((valid_boxes, valid_scores))
    
    # Apply Fast R-CNN NMS
    keep_indices = nms(dets, nms_threshold)
    
    # Map back to original indices
    final_indices = [valid_indices[i] for i in keep_indices]
    
    return final_indices