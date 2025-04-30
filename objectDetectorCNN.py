import numpy as np
import os
import cv2
from pathlib import Path
import torch
from ultralytics import YOLO
from torchvision import transforms

class ObjectDetectorCNN:
    def __init__(self, model_path, seed=21):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        self.model_path = Path(__file__).resolve().parent / model_path
        self.model = self.load_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    
    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        model = YOLO(self.model_path)
        return model

    def preprocess_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(img_rgb).unsqueeze(0).to(self.device)
        return img_tensor

    def preprocess_image_array(self, img_array):
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(img_rgb).unsqueeze(0).to(self.device)
        return img_tensor

    def detect_objects(self, image_tensor):
        results = self.model(image_tensor, verbose=False)  # Run inference on the image tensor

        if len(results) == 0:
            print("No detections found.")
            return []
        
        obb_data = results[0].obb.data  # Get the OBB tensor
        class_names = results[0].names
        detections = []

        
        classes_found = set()
        if obb_data is not None and isinstance(obb_data, torch.Tensor):
            for obb in obb_data.cpu().numpy():
                x_center, y_center, width, height, angle, conf, cls = obb
                cls_name = class_names[int(cls)]
                angle_deg = np.degrees(angle)
                
                if cls_name not in classes_found:
                    classes_found.add(cls_name)
                else:
                    # keep highest confidence 
                    if cls_name in [det["class_name"] for det in detections]:
                        existing_det = next(det for det in detections if det["class_name"] == cls_name)
                        if existing_det["confidence"] < conf:
                            detections.remove(existing_det)
                        else:
                            continue                   
                
                detection = {
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "angle": angle_deg,
                    "confidence": conf,
                    "class_name": cls_name
                }

                detections.append(detection)

        return detections

    def draw_detections(self, img, detections):

        for det in detections:
            x_center = int(det["x_center"])
            y_center = int(det["y_center"])
            width = int(det["width"])
            height = int(det["height"])
            angle = det["angle"]
            class_name = det["class_name"]
            confidence = det["confidence"]

            # Draw the rotated rectangle
            rect = ((x_center, y_center), (width, height), angle)
            box_points = cv2.boxPoints(rect)
            box_points = np.int0(box_points)

            cv2.polylines(img, [box_points], isClosed=True, color=(0, 255, 0), thickness=1)
            cv2.putText(img, f"{class_name} {confidence:.2f}", (int(x_center-width/2), int(y_center) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
         
        return img

