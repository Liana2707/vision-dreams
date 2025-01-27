


from builders.config_builder import ConfigBuilder
from builders.mode_builder import ModeBuilder
from wrappers.model_wrapper import ModelWrapper


class GeneralBuilder:
    __models = {
        "YOLOv11": "YOLO",
    }
    
    def __init__(self, mode_builder=None, config_builder=None):
        """
        Инициализирует ModelBuilder с экземплярами ModeBuilder и ConfigBuilder.

        Args:
            mode_builder (ModeBuilder): Экземпляр ModeBuilder (по умолчанию создается новый).
            config_builder (ConfigBuilder): Экземпляр ConfigBuilder (по умолчанию создается новый).
        """
        self.mode_builder = mode_builder if mode_builder else ModeBuilder()
        self.config_builder = config_builder if config_builder else ConfigBuilder()

    
    def get_models(self):
        """Возвращает список всех моделей, которые можно создать"""
        return list(self.__models.keys())


    def build(self):
        """
        Выполняет сборку, сначала применяя ModeBuilder, а затем ConfigBuilder.

        Returns:
            ModelWrapper: оболочка для выбранной модели.
        """
        mode = self.mode_builder.build()
        config = self.config_builder.build()

        if config['model_type'] in self.get_models():
            return ModelWrapper(mode, config)
        
        return {"message": f"Такой модели нет в списке, выберите что-то из {self.get_models()}"}
    
    def save_model(self):
        """
        Вызывает метод save_model.
        """
        pass

    def __str__(self):
       return f"GeneralBuilder(mode_builder={self.mode_builder}, config_builder={self.config_builder})"


    