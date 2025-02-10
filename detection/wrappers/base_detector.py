from abc import ABC, abstractmethod
import uuid

class BaseDetector(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.uuid = str(uuid.uuid4())
    
    @abstractmethod
    def train(self, **params):
        pass
    
    @abstractmethod
    def evaluate(self, **params):
        pass
    
    @abstractmethod
    def predict(self, image_path, **params):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass
