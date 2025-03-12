import os

import torch
from ultralytics import YOLO
import PIL.Image as Image

from schemas.output_detect_schema import ModelDescription, RootModel
from schemas.output_detect_schema import TLBR, XYWHN, OutputItem
from .base_detector import BaseDetector

class YOLODetector(BaseDetector):
    def __init__(self, model_name):
        self.model_type = "yolo"
        super().__init__(model_name)
        model_path = "models/yolo11n.pt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        self.model = YOLO(model_path)  

    def load(self):
        pass


    def train(self, data_path, epochs=10):
        return self.model.train(data=data_path, epochs=epochs, project="models", name=self.model_name)

    def evaluate(self, data_path):
        pass

    def predict(self, image, **params):
        results = self.model(image, **params)
        output_items = []
        for result in results:  
            boxes = result.boxes  
            classes = boxes.cls.cpu().tolist() 
            scores = boxes.conf.cpu().tolist()  
            xywhn = boxes.xywhn.cpu().tolist() 
            tlbr = boxes.xyxy.cpu().tolist()  

            for i in range(len(classes)):
                class_id = int(classes[i])
                class_name = self.model.names[class_id] 
                score = float(scores[i])
                x_center, y_center, width, height = map(float, xywhn[i])
                xmin, ymin, xmax, ymax = map(float, tlbr[i])

                xywhn_obj = XYWHN(xn=x_center, yn=y_center, wn=width, hn=height)
                tlbr_obj = TLBR(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

                output_item = OutputItem(
                    class_id=class_id,
                    class_name=class_name,
                    score=score,
                    xywhn=xywhn_obj,
                    tlbr=tlbr_obj
                )
                output_items.append(output_item)

        model_description = ModelDescription(id=str(self.uuid), name=self.model_name,type=self.model_type, problem_type='detection')

        return RootModel(model_description=model_description, model_output=[output_items])
    

    def plot_image(self, image, **params):
        results = self.model(image, **params)
        im_array = results[0].plot()
        im = Image.fromarray(im_array[..., ::-1])
        return im
        
    def save(self, dir):
        self.model.save(f"{dir}/{self.model_name}.pt")
