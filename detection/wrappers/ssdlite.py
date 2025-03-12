import os
import numpy as np
import torch
import cv2

import torch
import PIL.Image as Image
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights

from schemas.output_detect_schema import TLBR, XYWHN, ModelDescription, OutputItem, RootModel
from .base_detector import BaseDetector

class SSDLiteDetector(BaseDetector):
    def __init__(self, model_name):
        self.model_type = "ssdlite"
        super().__init__(model_name)
        self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=self.weights)
    
    def load(self):
        pass
      
    def train(self):
        pass

    def evaluate(self):
        pass

    def predict(self, image, **params):
        self.model.eval()

        target_size=(300, 300)
        original_image = np.array(image) 
        img_height, img_width = original_image.shape[0], original_image.shape[1]
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1) / 255.0

        conf = params['conf']

        preprocess = self.weights.transforms()
        batch = preprocess(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(batch)
        
        scores = output[0]['scores']
        labels = output[0]['labels']
        boxes = output[0]['boxes']

        filtered_indices = scores > conf
        filtered_scores = scores[filtered_indices]
        filtered_labels = labels[filtered_indices]
        filtered_boxes = boxes[filtered_indices]

        scale_x = img_width / target_size[0]
        scale_y = img_height / target_size[1]
        filtered_boxes_scaled = filtered_boxes * torch.tensor([scale_x, scale_y, scale_x, scale_y])

        filtered_boxes_normalized = filtered_boxes / torch.tensor([target_size[0], target_size[1], target_size[0], target_size[1]])

        output_items = []
        for i in range(len(filtered_scores)):
            class_id = int(filtered_labels[i])  
            score = float(filtered_scores[i])  

            xmin_norm = float(filtered_boxes_normalized[i][0])
            ymin_norm = float(filtered_boxes_normalized[i][1])
            xmax_norm = float(filtered_boxes_normalized[i][2])
            ymax_norm = float(filtered_boxes_normalized[i][3])

            widthn = xmax_norm - xmin_norm
            heightn = ymax_norm - ymin_norm
            xcn = xmin_norm + widthn / 2  # Центр x
            ycn = ymin_norm + heightn / 2  # Центр y

            xywhn = XYWHN(xn=xcn, yn=ycn, wn=widthn, hn=heightn)

            tlbr = TLBR(xmin=float(filtered_boxes_scaled[i][0]), ymin=float(filtered_boxes_scaled[i][1]), xmax=float(filtered_boxes_scaled[i][2]), ymax=float(filtered_boxes_scaled[i][3]))

            output_item = OutputItem(
                class_id=class_id,
                class_name=self.weights.meta["categories"][class_id],
                score=score,
                xywhn=xywhn,
                tlbr=tlbr,
            )
            output_items.append(output_item)

        model_description = ModelDescription(id=str(self.uuid), name=self.model_name,type=self.model_type, problem_type='detection')

        return RootModel(
            model_description=model_description, 
            model_output=[output_items]
        )

    def plot_image(self, image_np, scores, boxes, labels):
        image_np = image_np.copy() 

        for i in range(len(scores)):
            box = boxes[i]
            class_id = labels[i].item()
            score = scores[i].item()
            category_name = self.weights.meta["categories"][class_id]

            x1 = int(box[0] * image_np.shape[1])
            y1 = int(box[1] * image_np.shape[0])
            x2 = int(box[2] * image_np.shape[1])
            y2 = int(box[3] * image_np.shape[0])
            
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"{category_name}: {100 * score:.1f}%",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
        pil_image = Image.fromarray(image_np)
        return pil_image
    
    def save(self, dir):
        model_path = os.path.join(dir, f"{self.model_name}.pt") 
        torch.save(self.model.state_dict(), model_path)

    
        
