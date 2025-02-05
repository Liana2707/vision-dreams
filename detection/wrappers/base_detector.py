from abc import ABC, abstractmethod

class BaseDetector(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
    
    @abstractmethod
    def train(self, **params):
        pass
    
    @abstractmethod
    def evaluate(self, **params):
        pass
    
    @abstractmethod
    def predict(self, image_path, **params):
        pass
