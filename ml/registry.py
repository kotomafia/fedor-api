from api.ml.base import OCREngine, TextClassifier
from api.ml.ocr import EasyOCREngine
from api.ml.toxicity import RubertTinyToxicity, SNlpToxicityClassifier

from api.config import api_settings


def build_classifier(model_name: str | None = None) -> TextClassifier:
    name = model_name or api_settings.ml_model_name
    return RubertTinyToxicity(model_name=name)


def build_ocr_classifier(model_name: str | None = None) -> TextClassifier:
    name = model_name or api_settings.ocr_classifier_name
    return SNlpToxicityClassifier(model_name=name)


def build_ocr_engine() -> OCREngine:
    return EasyOCREngine()
