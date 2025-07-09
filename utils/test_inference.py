import cv2
import torch
import os
import sys

sys.path.append("./")
from utils.inference import predict

# Test function with enhanced output
def test_inference():
    """Test function for face detection and gender classification with NMS"""
    
    # Test image path
    test_image_path = "/home/vv-team/hajin/Pytorch_Retinaface/results/image (26).png"  # Change to your test image path
    
    if not os.path.exists(test_image_path):
        print(f" Test image not found: {test_image_path}")
        print("Please provide a test image path")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        from utils.facenet import load_retinaface
        from utils.AgeGenderRaceNet import load_model
        
        # Load RetinaFace
        retinaface_net, retinaface_cfg = load_retinaface(
            network="resnet50",
            trained_model="./weights/Resnet50_Final.pth",  # Change to your path
            device=device
        )
        print(" RetinaFace loaded")
        
        # Load Gender model
        AgeGenderRaceNet = load_model(
            "./weights/res34_fair_align_multi_4_20190809.pt",  # Change to your path
            device=device
        )
        print("model loaded")
        
        # Step 2: Load and process test image
        print(f" Loading test image: {test_image_path}")
        img_bgr = cv2.imread(test_image_path)
        
        if img_bgr is None:
            print("Failed to load image")
            return
        
        print(f" Image shape: {img_bgr.shape}")
        

        face_data, predictions = predict(
            image_bgr=img_bgr,
            retinaface_net=retinaface_net,
            retinaface_cfg=retinaface_cfg,
            model=AgeGenderRaceNet,
            device=device,
            face_threshold=0.6
        )
        
        # Step 4: Display results and draw bounding boxes
        results = []
        vis_image = img_bgr.copy()  # Copy image for visualization
        
        for i, ((x1, y1, x2, y2), face_confidence) in enumerate(face_data):
            if i < len(predictions):
                                
                results.append({
                    "type": "rectangle",
                    "label": "person_info",
                    "points": [float(x1), float(y1), float(x2), float(y2)],
                    "attributes": {
                        "race": predictions[i]['race'],
                        "race_confidence": predictions[i]['race_confidence'],
                        "gender": predictions[i]['gender'],
                        "gender_confidence": f"{predictions[i]['gender_confidence']:.4f}",
                        "age": predictions[i]['age'],
                        "age_confidence": predictions[i]['age_confidence']                        
                    }
                })
                
        if results:
            print(f" Found {len(results)} unique faces:")
            
            # Draw bounding boxes and labels
            for i, result in enumerate(results, 1):
                x1, y1, x2, y2 = result['points']
                # Ensure coordinates are integers
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                race = result['attributes']['race']
                race_conf = float(result['attributes']['race_confidence'])
                
                gender = result['attributes']['gender']
                gender_conf = float(result['attributes']['gender_confidence'])
                
                age = result['attributes']['age']
                age_conf = float(result['attributes']['age_confidence'])
                
                # Choose color based on gender
                if gender.lower() == 'male':
                    color = (255, 0, 0)  # Blue for male
                else:
                    color = (255, 0, 255)  # Magenta for female
                
                # Draw bounding box
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Prepare text
                label_text = [
                    f"{race} {race_conf:.2f}",
                    f"{gender} {gender_conf:.2f}", 
                    f"{age} {age_conf:.2f}"
                ]
                
                # Calculate text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                line_spacing = 3
                
                # Get text size
                text_widths = []
                text_heights = []
                
                for line in label_text:
                    (label_w, label_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
                    text_widths.append(label_w)
                    text_heights.append(label_h)
            
                max_width = max(text_widths)
                total_height = sum(text_heights) + line_spacing * (len(label_text) - 1)

                padding = 3
                bg_x1 = x1
                bg_y1 = y1 - total_height - padding * 2
                bg_x2 = x1 + max_width + padding * 2
                bg_y2 = y1

                # Draw text background rectangles
                cv2.rectangle(vis_image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

                
                for i, line in enumerate(label_text):
                    y1 -= text_heights[i]
                    cv2.putText(vis_image, line, (x1 + padding, y1 + padding), 
                                font, font_scale, (255, 255, 255), thickness)
                    if i < len(label_text) - 1:  
                        y1 -= line_spacing
                
                # Console output
                print(f"  Face {i}:")
                print(f"  Gender: {gender} (conf: {gender_conf:.4f})")
                print(f"  Bounding box: [{x1}, {y1}, {x2}, {y2}]")
                print(f"  Size: {x2-x1}x{y2-y1}")
            
            # Save visualization
            name = (test_image_path.split("/")[-1]).split(".")[0]
            output_path = f"{name}_result.jpg"
            cv2.imwrite(output_path, vis_image)
            print(f" Visualization saved: {output_path}")
                

        print("\n" + "=" * 70)
        print(" Test completed successfully!")
        
    except Exception as e:
        print(f" Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run test
    test_results = test_inference()
    
    # if test_results:
    #     print(f"\n📋 Final results: {len(test_results)} faces detected")
    #     for i, result in enumerate(test_results, 1):
    #         print(f"Face {i}: {result['label']} at {result['points']}")