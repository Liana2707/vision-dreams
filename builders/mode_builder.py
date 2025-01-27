class ModeBuilder:
    def __init__(self, inference_mode=None, train_mode=None):
        """
        Инициализирует ModeBuilder с режимами инференса, обучения.

        Args:
            inference_mode : Режим инференса.
            train_mode : Режим обучения.
        """
        self.inference_mode = inference_mode
        self.train_mode = train_mode
        


    def get_config(self):
        """
        Возвращает текущую конфигурацию режимов и сохранения в виде словаря.

        Returns:
            dict: Словарь с режимами и настройками сохранения.
        """
        return {
            "inference_mode": self.inference_mode,
            "train_mode": self.train_mode,
            "save_settings": self.save_settings
        }

    def build(self):
        """
        Возвращает текущую конфигурацию режимов и сохранения в виде словаря (синоним get_config).

        Returns:
            dict: Словарь с режимами и настройками сохранения.
        """
        return self.get_config()

    def __str__(self):
      return f"ModeBuilder(inference_mode={self.inference_mode}, train_mode={self.train_mode}, save_settings={self.save_settings})"