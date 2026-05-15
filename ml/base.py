from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToxicityScore:
    score: float           # агрегированная оценка [0..1]
    categories: dict[str, float]  # детализация по категориям
    model_version: str


class TextClassifier(ABC):
    @abstractmethod
    async def classify(self, text: str) -> ToxicityScore:
        ...

    @abstractmethod
    async def classify_batch(self, texts: list[str]) -> list[ToxicityScore]:
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        ...


@dataclass(frozen=True)
class OCRResult:
    text: str
    # Достоверность от 0 до 1 — пригодится, чтобы фильтровать
    # очень неуверенные распознавания (часто это шум, а не текст).
    confidence: float
    # Опционально — отдельные фрагменты с их confidence.
    fragments: tuple[tuple[str, float], ...] = field(default_factory=tuple)


class OCREngine(ABC):
    @abstractmethod
    async def extract(self, image_bytes: bytes) -> OCRResult:
        ...

    @property
    @abstractmethod
    def engine_version(self) -> str:
        ...