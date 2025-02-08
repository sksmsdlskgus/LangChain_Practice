from abc import ABC, abstractmethod


class BaseChain(ABC):
    def __init__(self, model: str = "llama3.1-instruct-8b:latest", temperature: float = 0, **kwargs):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs

    @abstractmethod
    def setup(self):
        """체인 설정을 위한 추상 메서드"""
        pass

    def create(self):
        """체인을 생성하고 반환합니다."""
        return self.setup()