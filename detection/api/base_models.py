from wrappers.ssdlite import SSDLiteDetector
from wrappers.yolo_transformer import YOLOTransformerDetector
from wrappers.yolov11 import YOLODetector


created_models = {"554d3bd3-b0ef-4025-b95a-3f0e1a127d40": 
                                SSDLiteDetector('ssdlite'),
                  "554d3bd3-b0ef-4025-b95a-3f0e1a127d41": 
                                YOLODetector('yolo'),
                  "554d3bd3-b0ef-4025-b95a-3f0e1a127d42":       
                                YOLOTransformerDetector('transformer')
                }