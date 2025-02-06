from models_config import models_config

class GeneralBuilder:
    
    def __init__(self, model_type="yolo", model_name=""):
        self.model_type = model_type
        self.model_name = model_name


    def build(self):
        if self.model_type in models_config.keys():
            return models_config[self.model_type]["class"](self.model_name)
        raise KeyError(f"Модель '{self.model_type}' не найдена.")

    def __str__(self):
       return f"GeneralBuilder({self.model_type})"
    