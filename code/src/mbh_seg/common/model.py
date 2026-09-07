from abc import ABC, abstractmethod
from typing import Any


class InferenceModel(ABC):
    @abstractmethod
    def preprocess(self, image: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def predict(self, image: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def postprocess(self, prediction: Any) -> Any:
        raise NotImplementedError
