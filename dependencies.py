from fastapi import Request

from api.ml.base import TextClassifier


def get_classifier(request: Request) -> TextClassifier:
    return request.app.state.classifier