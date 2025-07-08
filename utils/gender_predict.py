import torch
import torch.nn.functional as F
import torchvision.models as models

def load_gender_model(model_path, device='cuda'):
    """Load gender classification model"""
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    keys = list(state_dict.keys())
    fc_weight_shape = state_dict['fc.weight'].shape
    num_outputs = fc_weight_shape[0]
    
    # Determine gender indices based on output size
    if num_outputs == 15:
        gender_indices = (4, 6)  # 4-race model: gender 4-5
    elif num_outputs == 18:
        gender_indices = (7, 9)  # FairFace: gender 7-8
    else:
        raise ValueError(f"Unsupported model output size: {num_outputs}")
    
    # Determine ResNet variant
    if any('layer4.2.' in key for key in keys):
        model = models.resnet34(pretrained=False)
    else:
        model = models.resnet18(pretrained=False)
    
    model.fc = torch.nn.Linear(model.fc.in_features, num_outputs)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, gender_indices

def predict_gender_batch(gender_tensors, model, gender_indices, device='cuda'):
    """
    Batch predict gender from preprocessed tensors
    Args:
        gender_tensors: List of preprocessed face tensors
        model: Gender classification model
        gender_indices: Tuple of (start, end) indices for gender logits
        device: Device to run on
    Returns:
        List of prediction dictionaries
    """
    if not gender_tensors:
        return []
    
    predictions = []
    model.eval()
    
    with torch.no_grad():
        for gender_tensor in gender_tensors:
            # Forward pass
            outputs = model(gender_tensor)
            
            # Extract gender logits
            gender_logits = outputs[:, gender_indices[0]:gender_indices[1]]
            probabilities = F.softmax(gender_logits, dim=1)
            
            # Get prediction (0: female, 1: male)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()
            gender_label = "male" if pred_class == 0 else "female"
            
            predictions.append({
                'gender': gender_label,
                'confidence': confidence
            })
    
    return predictions