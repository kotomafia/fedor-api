from datetime import datetime, timezone

from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.celery_app import celery_app
from api.config import api_settings
from api.schemas import (
    ImageModerationRequest,
    ModerationAction,
    ModerationLabel,
    ModerationResponse,
    TaskResultResponse,
    TaskStatus,
    TaskSubmittedResponse,
    TextModerationRequest,
)
from api.tasks.moderation import classify_text, classify_image

router = APIRouter(prefix="/api/v1/moderate", tags=["moderation"])


def classify_score(score: float) -> tuple[ModerationLabel, ModerationAction]:
    if score < api_settings.threshold_uncertain:
        return ModerationLabel.NEUTRAL, ModerationAction.ALLOW
    if score < api_settings.threshold_toxic:
        return ModerationLabel.UNCERTAIN, ModerationAction.FLAG
    return ModerationLabel.TOXIC, ModerationAction.BLOCK


def classify_score_ocr(score: float) -> tuple[ModerationLabel, ModerationAction]:
    if score < api_settings.threshold_uncertain:
        return ModerationLabel.NEUTRAL, ModerationAction.ALLOW
    if score < api_settings.threshold_toxic_ocr:
        return ModerationLabel.UNCERTAIN, ModerationAction.FLAG
    return ModerationLabel.TOXIC, ModerationAction.BLOCK


@router.post("/text", response_model=TaskSubmittedResponse, status_code=202)
async def submit_text_moderation(payload: TextModerationRequest) -> TaskSubmittedResponse:
    task = classify_text.delay(
        payload.content, payload.message_id, payload.guild_id,
        payload.channel_id, payload.author_id,
    )
    return TaskSubmittedResponse(task_id=task.id)


@router.post(
    "/image",
    response_model=TaskSubmittedResponse,
    status_code=202,
)
async def submit_image_moderation(payload: ImageModerationRequest) -> TaskSubmittedResponse:
    task = classify_image.delay(
        payload.image_b64,
        payload.message_id,
        payload.guild_id,
        payload.channel_id,
        payload.author_id,
    )
    logger.info(
        "image task submitted | task_id={tid} | author={a} | source={src}",
        tid=task.id, a=payload.author_id, src=payload.source_hint,
    )
    return TaskSubmittedResponse(task_id=task.id)

@router.get(
    "/result/{task_id}",
    response_model=TaskResultResponse,
)
async def get_moderation_result(task_id: str) -> TaskResultResponse:
    async_result = AsyncResult(task_id, app=celery_app)

    if not async_result.ready():
        return TaskResultResponse(task_id=task_id, status=TaskStatus.PENDING)

    if async_result.failed():
        error = str(async_result.result)
        logger.warning("task failed | task_id={tid} | err={err}", tid=task_id, err=error)
        return TaskResultResponse(task_id=task_id, status=TaskStatus.FAILURE, error=error)

    raw = async_result.result
    score = raw["score"]
    classify_fn = classify_score_ocr if raw.get("ocr_engine") else classify_score
    label, action = classify_fn(score)

    moderation = ModerationResponse(
        verdict_id=raw.get("verdict_id"),
        message_id=raw["message_id"],
        score=score,
        label=label,
        action=action,
        model_version=raw.get("model_version"),
        processed_at=datetime.now(timezone.utc),
        categories=raw.get("categories", {}),
        extracted_text=raw.get("extracted_text"),
        ocr_confidence=raw.get("ocr_confidence"),
        skipped_reason=raw.get("skipped_reason"),
    )
    return TaskResultResponse(
        task_id=task_id,
        status=TaskStatus.SUCCESS,
        result=moderation,
    )