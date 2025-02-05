from models_config import models_config


class GeneralBuilder:
    
    def __init__(self, model_type="yolo"):
        self.model_type = model_type


    def build(self):
        if self.model_type in models_config.keys():
            return models_config[self.model_type]["class"]()
        raise KeyError(f"Модель '{self.model_type}' не найдена.")

    def __str__(self):
       return f"GeneralBuilder({self.model_type})"
    