import numpy as np
import torch
import cv2

import torch

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from .base_detector import BaseDetector

class SSDLiteDetector(BaseDetector):
    def __init__(self):
        super().__init__("ssdlite")
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights)
      
    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self, image_path, **params):
        self.model.eval()

        target_size=(300, 300)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0
        
        processed_image = self.__process_images(image)
        return processed_image          

    def __process_images(self, img, threshold = 0.24):
        preprocess = self.weights.transforms()
        batch = preprocess(img).unsqueeze(0)
        with torch.no_grad():
            output = self.model(batch)
        
        scores = output[0]['scores']
        labels = output[0]['labels']
        boxes = output[0]['boxes']
        image_np = img.permute(1, 2, 0).numpy() 
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        for i in range(len(scores)):
            if scores[i] >= threshold:
                box = boxes[i]
                class_id = labels[i].item()
                score = scores[i].item()
                category_name = self.weights.meta["categories"][class_id - 1] 
                cv2.rectangle(image_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(image_np, f"{category_name}: {100 * score:.1f}%", 
                            (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (255, 255, 255), 2)
        return image_np
