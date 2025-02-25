import os
from PIL import Image
from matplotlib import pyplot as plt
import torch
from transformers import AutoFeatureExtractor
from transformers import YolosForObjectDetection
from PIL import Image
import PIL.Image as Image
import io

from .base_detector import BaseDetector

class YOLOTransformerDetector(BaseDetector):
    def __init__(self, model_name):
        self.model_type = "yolos"
        super().__init__(model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small")
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")

    def load(self):
        pass

    def train(self, data_path, epochs=10):
        pass

    def evaluate(self, data_path):
        pass

    def predict(self, image_path, **params):
        image = image_path
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values

        with torch.no_grad():
            outputs = self.model(pixel_values, output_attentions=True)

        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > params['conf']

        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
        results = self.feature_extractor.post_process(outputs, target_sizes)[0]

        boxes = results["boxes"][keep]
        labels = results["labels"][keep]
        scores = results["scores"][keep]

        boxes_xywh = self.xyxy_to_xywh(boxes)
        boxes_xywhn = self.normalize_boxes(boxes_xywh, image.size)

        img_with_boxes = self.plot_results(image, scores, boxes, labels)
        return img_with_boxes

    def plot_results(self, pil_img: Image.Image, prob, boxes, classes):
        plt.figure(figsize=(16,10))
        plt.imshow(pil_img)
        ax = plt.gca()
        for p, (xmin, ymin, xmax, ymax), cl in zip(prob, boxes.tolist(),classes):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color='red', linewidth=3))
            text = f'{self.model.config.id2label[cl.item()]}: {p:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')

        # Convert the matplotlib plot to a PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)

        img_with_boxes = Image.open(buf)
        return img_with_boxes
    

    def xyxy_to_xywh(self, boxes):
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        width = xmax - xmin
        height = ymax - ymin
        x_center = xmin + width / 2
        y_center = ymin + height / 2
        return torch.stack((x_center, y_center, width, height), dim=1)

    def normalize_boxes(self, boxes, image_size):
        width, height = image_size
        boxes[:, 0] /= width  # x_center
        boxes[:, 1] /= height # y_center
        boxes[:, 2] /= width  # width
        boxes[:, 3] /= height # height
        return boxes
    
    def save(self, dir):
        model_path = os.path.join(dir, f"{self.model_name}.pth")
        torch.save(self.model.state_dict(), model_path)
