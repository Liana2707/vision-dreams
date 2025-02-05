
from wrappers.ssdlite import SSDLiteDetector
from wrappers.yolo_transformer import YOLOTransformerDetector
from wrappers.yolov11 import YOLODetector


models_config = {
    "yolo": {
        "class": YOLODetector,
    },
    "yolos": {
        "class": YOLOTransformerDetector, 
    },
    "ssdlite": {
        "class": SSDLiteDetector,
    },

}