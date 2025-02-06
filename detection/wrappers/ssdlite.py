import numpy as np
import torch
import cv2

import torch

from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights
from .base_detector import BaseDetector

class SSDLiteDetector(BaseDetector):
    def __init__(self, model_name):
        self.model_type = "ssdlite"
        super().__init__(model_name)
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

        filtered_indices = scores > threshold
        filtered_scores = scores[filtered_indices]
        filtered_labels = labels[filtered_indices]
        filtered_boxes = boxes[filtered_indices]

        img_height, img_width = img.shape[1], img.shape[2]
        filtered_boxes_normalized = filtered_boxes / torch.tensor([img_width, img_height, img_width, img_height])

        results = {
            "class": filtered_labels,
            "conf": filtered_scores,
            "xywhn": filtered_boxes_normalized 
        }
        return results
