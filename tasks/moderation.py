import asyncio
from typing import Any

from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from loguru import logger

from api.celery_app import celery_app
from api.config import api_settings
from api.ml.registry import build_classifier, build_ocr_classifier, build_ocr_engine
from api.ml.base import TextClassifier, OCREngine
from api.ml.text_normalize import normalize_ocr_text


class MLTask(Task):
    """Базовый класс для тасок с ML-моделью.

    Модель загружается лениво при первом вызове и кэшируется на уровне воркер-процесса.
    Это критично: иначе модель будет грузиться в каждой задаче (~10 секунд).
    """
    _classifier: TextClassifier | None = None

    @property
    def classifier(self) -> TextClassifier:
        if self._classifier is None:
            logger.info("Worker loading model: {m}", m=api_settings.ml_model_name)
            self._classifier = build_classifier(api_settings.ml_model_name)
            logger.info("Worker model ready, version={v}", v=self._classifier.model_version)
        return self._classifier


@celery_app.task(name="tasks.moderation.classify_text", base=MLTask, bind=True)
def classify_text(self: MLTask, content: str, message_id: str,
                  guild_id: str, channel_id: str, author_id: str) -> dict:
    async def _run():
        from api.db.engine import async_session_factory
        from api.services.moderation_service import ModerationService

        async with async_session_factory() as session:
            try:
                service = ModerationService(session, self.classifier)
                result = await service.moderate_text(
                    message_id=message_id, guild_id=guild_id,
                    channel_id=channel_id, author_id=author_id,
                    content=content,
                )
                await session.commit()
                return result
            except Exception:
                await session.rollback()
                raise

    return asyncio.run(_run())


class ImageTask(Task):
    """Тяжёлая таска: OCR + два классификатора (rubert-tiny и s-nlp для OCR-текста)."""
    _ocr: OCREngine | None = None
    _classifier: TextClassifier | None = None
    _ocr_classifier: TextClassifier | None = None

    @property
    def ocr(self) -> OCREngine:
        if self._ocr is None:
            self._ocr = build_ocr_engine()
        return self._ocr

    @property
    def classifier(self) -> TextClassifier:
        if self._classifier is None:
            self._classifier = build_classifier(api_settings.ml_model_name)
        return self._classifier

    @property
    def ocr_classifier(self) -> TextClassifier:
        if self._ocr_classifier is None:
            logger.info(
                "Worker loading OCR classifier: {m}",
                m=api_settings.ocr_classifier_name,
            )
            self._ocr_classifier = build_ocr_classifier(api_settings.ocr_classifier_name)
            logger.info(
                "OCR classifier ready, version={v}",
                v=self._ocr_classifier.model_version,
            )
        return self._ocr_classifier


MAX_IMAGE_BYTES = 10 * 1024 * 1024


@celery_app.task(
    name="tasks.moderation.classify_image",
    base=ImageTask,
    bind=True,
    autoretry_for=(ConnectionError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 2},
)
def classify_image(self: ImageTask, image_bytes_b64: str, message_id: str) -> dict[str, Any]:
    import base64

    image_bytes = base64.b64decode(image_bytes_b64)

    if len(image_bytes) > MAX_IMAGE_BYTES:
        return {
            "message_id": message_id,
            "score": 0.0,
            "categories": {},
            "model_version": "skipped",
            "extracted_text": "",
            "ocr_confidence": 0.0,
            "skipped_reason": "image_too_large",
        }

    try:
        ocr_result = asyncio.run(self.ocr.extract(image_bytes))
    except Exception as e:
        logger.exception("OCR failed | message_id={mid}", mid=message_id)
        return {
            "message_id": message_id,
            "score": 0.0,
            "categories": {},
            "model_version": "ocr_error",
            "extracted_text": "",
            "ocr_confidence": 0.0,
            "skipped_reason": f"ocr_error: {type(e).__name__}",
        }

    if (
        ocr_result.text.strip()
        and ocr_result.confidence < api_settings.min_ocr_overall_confidence
    ):
        logger.info(
            "OCR overall confidence too low ({conf:.3f}) | text={t!r}",
            conf=ocr_result.confidence,
            t=ocr_result.text,
        )
        return {
            "message_id": message_id,
            "score": 0.0,
            "categories": {},
            "model_version": "no_text",
            "extracted_text": ocr_result.text,
            "ocr_confidence": ocr_result.confidence,
            "skipped_reason": "ocr_confidence_too_low",
        }

    min_conf = api_settings.min_ocr_fragment_confidence
    confident_text = " ".join(
        t for t, c in ocr_result.fragments if c >= min_conf
    ).strip()
    used_low_confidence_fallback = False

    if not confident_text and ocr_result.text.strip():
        confident_text = ocr_result.text.strip()
        used_low_confidence_fallback = True
        logger.info(
            "OCR low fragment confidence ({conf:.3f}), using full text fallback | text={t!r}",
            conf=ocr_result.confidence,
            t=confident_text,
        )

    if not confident_text:
        return {
            "message_id": message_id,
            "score": 0.0,
            "categories": {},
            "model_version": "no_text",
            "extracted_text": ocr_result.text,
            "ocr_confidence": ocr_result.confidence,
            "skipped_reason": "no_confident_text",
        }

    normalized = normalize_ocr_text(confident_text)
    if not normalized:
        return {
            "message_id": message_id,
            "score": 0.0,
            "categories": {},
            "model_version": "log_text_filtered",
            "extracted_text": confident_text,
            "ocr_confidence": ocr_result.confidence,
            "skipped_reason": "log_text_filtered",
        }

    tiny_result = asyncio.run(self.classifier.classify(normalized))
    snlp_result = asyncio.run(self.ocr_classifier.classify(normalized))
    score = max(tiny_result.score, snlp_result.score)

    categories = {**tiny_result.categories, **snlp_result.categories}

    logger.info(
        "image classify | text={t!r} | tiny={ts:.3f} | snlp={ss:.3f} | final={fs:.3f}",
        t=normalized,
        ts=tiny_result.score,
        ss=snlp_result.score,
        fs=score,
    )

    result: dict[str, Any] = {
        "message_id": message_id,
        "score": score,
        "categories": categories,
        "model_version": tiny_result.model_version,
        "ocr_classifier_version": snlp_result.model_version,
        "extracted_text": normalized,
        "ocr_confidence": ocr_result.confidence,
        "ocr_engine": self.ocr.engine_version,
    }
    if used_low_confidence_fallback:
        result["ocr_low_confidence_fallback"] = True
    return result
