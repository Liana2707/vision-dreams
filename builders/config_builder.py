

class ConfigBuilder:
    def __init__(self, model_type="YOLOv11", params=None):
        """
        Инициализирует ConfigBuilder с типом модели и параметрами.

        Args:
            model_type (str): Тип модели (по умолчанию "YOLOv11").
            params (dict): Словарь параметров модели (по умолчанию None).
        """
        self.model_type = model_type
        self.params = params if params is not None else {}

    def set_model_type(self, model_type):
        """
        Устанавливает новый тип модели.

        Args:
            model_type (str): Новый тип модели.

        Returns:
            ConfigBuilder: Возвращает экземпляр ConfigBuilder.
        """
        self.model_type = model_type
        return self

    def set_param(self, key, value):
        """
        Устанавливает или обновляет параметр модели.

        Args:
            key (str): Ключ параметра.
            value: Значение параметра.

        Returns:
            ConfigBuilder: Возвращает экземпляр ConfigBuilder.
        """
        self.params[key] = value
        return self

    def set_params(self, params):
      """
      Устанавливает несколько параметров модели сразу.
      
      Args:
          params (dict): Словарь новых параметров.
      
      Returns:
          ConfigBuilder: Возвращает экземпляр ConfigBuilder.
      """
      self.params.update(params)
      return self


    def get_config(self):
        """
        Возвращает текущую конфигурацию модели в виде словаря.

        Returns:
            dict: Словарь с типом модели и параметрами.
        """
        return {
            "model_type": self.model_type,
            "params": self.params
        }

    def build(self):
        """
        Возвращает текущую конфигурацию модели в виде словаря (синоним get_config).

        Returns:
            dict: Словарь с типом модели и параметрами.
        """
        return self.get_config()

    def __str__(self):
         return f"ConfigBuilder(model_type='{self.model_type}', params={self.params})"
