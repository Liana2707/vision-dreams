from PIL import Image
import torch
from transformers import AutoFeatureExtractor
from transformers import YolosForObjectDetection

from .base_detector import BaseDetector

class YOLOTransformerDetector(BaseDetector):
    def __init__(self):
        super().__init__("yolos")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("hustvl/yolos-small")
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-small")

    def train(self, data_path, epochs=10):
        pass

    def evaluate(self, data_path):
        pass

    def predict(self, image_path, **params):
        image = Image.open(image_path)
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values

        with torch.no_grad():
            outputs = self.model(pixel_values, output_attentions=True)

        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.9

        target_sizes = torch.tensor(image.size[::-1]).unsqueeze(0)
        postprocessed_outputs = self.feature_extractor.post_process(outputs, target_sizes)
        bboxes_scaled = postprocessed_outputs[0]['boxes']

        return bboxes_scaled
