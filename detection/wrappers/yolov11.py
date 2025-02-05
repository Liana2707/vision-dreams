import os
from ultralytics import YOLO
from .base_detector import BaseDetector

class YOLODetector(BaseDetector):
    def __init__(self):
        super().__init__("yolov11")
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
        return results
