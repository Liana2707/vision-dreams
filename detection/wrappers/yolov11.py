import os

import torch
from ultralytics import YOLO
from .base_detector import BaseDetector

class YOLODetector(BaseDetector):
    def __init__(self, model_name):
        self.model_type = "yolo"
        super().__init__(model_name)
        model_path = "models/yolo11n.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        self.model = YOLO(model_path)  

    def train(self, data_path, epochs=10):
        pass

    def evaluate(self, data_path):
        pass

    def predict(self, image_path, **params):
        results = self.model(image_path, **params)
        classes = [result.boxes.cls for result in results] 
        scores = [result.boxes.conf for result in results] 
        bboxes = [result.boxes.xywhn for result in results] 

        classes = torch.cat(classes) 
        scores = torch.cat(scores)
        bboxes = torch.cat(bboxes)
        return {
            "class":  classes,
            "score": scores,
            "bbox": bboxes,
        }
