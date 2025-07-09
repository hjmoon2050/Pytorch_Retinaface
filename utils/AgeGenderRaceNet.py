import torch
import torch.nn.functional as F
import torchvision.models as models

def load_model(model_path, device='cuda'):
    """Load gender classification model"""
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    keys = list(state_dict.keys())
    fc_weight_shape = state_dict['fc.weight'].shape
    num_outputs = fc_weight_shape[0]

    # Determine ResNet variant
    if any('layer4.2.' in key for key in keys):
        model = models.resnet34(pretrained=False)
    else:
        model = models.resnet18(pretrained=False)
    
    model.fc = torch.nn.Linear(model.fc.in_features, num_outputs)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model

def predict_bbox(tensors, model, device='cuda'):
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

    age_groups = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    gender_groups = ['male', 'female']
    race_groups = ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Aisan', 'Indian', 'Middle Eastern']
    
    if not tensors:
        return []
    
    predictions = []
    model.eval()
    
    with torch.no_grad():   
        for tensor in tensors:     
            outputs = model(tensor)
            
            race_logits = outputs[:, 0:6]
            gender_logits = outputs[:, 7:9] 
            age_logits = outputs[:, 10:17]      
            
            race_probs = F.softmax(race_logits, dim=1)
            gender_probs = F.softmax(gender_logits, dim=1)
            age_probs = F.softmax(age_logits, dim=1)
            
            race_class = int(torch.argmax(race_probs, dim=1).item())
            gender_class = int(torch.argmax(gender_probs, dim=1).item())
            age_class = int(torch.argmax(age_probs, dim=1).item())
            
            race_confidence = float(race_probs[0, race_class].item())
            gender_confidence = float(gender_probs[0, gender_class].item())
            age_confidence = float(age_probs[0, age_class].item())
       
            race_label = race_groups[race_class] if race_class < len(race_groups) else f"race_{race_class}"
            gender_label = gender_groups[gender_class] if gender_class < len(gender_groups) else f"gender_{gender_class}"
            age_label = age_groups[age_class] if age_class < len(age_groups) else f"age_{age_class}"
            
            predictions.append({
                'race': race_label,
                'race_confidence': race_confidence,
                'gender': gender_label,
                'gender_confidence': gender_confidence,
                'age': age_label,
                'age_confidence': age_confidence
            })
            
    return predictions