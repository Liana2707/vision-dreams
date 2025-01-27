from enum import Enum


class ModelType(str, Enum):
    yolo11n = 'YOLOv11'
    ssdlite = 'SSDLite'