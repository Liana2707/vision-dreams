from abc import ABC, abstractmethod
from typing import Any, Callable

# Publisher (Abstract)
class Publisher(ABC):
    @abstractmethod
    def publish(self, topic: str, message: Any) -> None:
        pass