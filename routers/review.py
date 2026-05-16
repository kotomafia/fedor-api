from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.engine import get_session
from api.db.models import Verdict
from api.db.repositories import ReviewQueueRepository, VerdictRepository
from api.schemas import (
    ReviewCreateRequest,
    ReviewDecisionRequest,
    ReviewEntryResponse,
    ReviewStatusEnum,
    VerdictResponse,
)

router = APIRouter(prefix="/api/v1", tags=["review"])


def _entry_to_response(entry) -> ReviewEntryResponse:
    return ReviewEntryResponse(
        id=entry.id,
        verdict_id=entry.verdict_id,
        status=ReviewStatusEnum(entry.status),
        review_message_id=entry.review_message_id,
        reviewer_id=entry.reviewer_id,
        reviewer_decision_at=entry.reviewer_decision_at,
        corrected_label=entry.corrected_label,
        corrected_score=entry.corrected_score,
        notes=entry.notes,
        created_at=entry.created_at,
    )


@router.post(
    "/review",
    response_model=ReviewEntryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_review_entry(
    payload: ReviewCreateRequest,
    session: AsyncSession = Depends(get_session),
) -> ReviewEntryResponse:
    """Бот вызывает после отправки flag-сообщения в review channel."""
    if await session.get(Verdict, payload.verdict_id) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"verdict {payload.verdict_id} not found",
        )

    repo = ReviewQueueRepository(session)
    entry = await repo.upsert_pending(
        verdict_id=payload.verdict_id,
        review_message_id=payload.review_message_id,
    )
    await session.commit()
    return _entry_to_response(entry)


@router.patch(
    "/review/{verdict_id}",
    response_model=ReviewEntryResponse,
)
async def update_review_entry(
    verdict_id: int,
    payload: ReviewDecisionRequest,
    session: AsyncSession = Depends(get_session),
) -> ReviewEntryResponse:
    """Записывает решение модератора.

    Может прийти как из кнопок в review channel, так и из /override-verdict —
    одинаковая логика.
    """
    repo = ReviewQueueRepository(session)
    entry = await repo.record_decision(
        verdict_id=verdict_id,
        status=payload.status.value,
        corrected_label=payload.corrected_label.value if payload.corrected_label else None,
        reviewer_id=payload.reviewer_id,
        notes=payload.notes,
    )
    await session.commit()
    return _entry_to_response(entry)


@router.get(
    "/verdicts/by-message/{guild_id}/{message_id}",
    response_model=VerdictResponse,
)
async def get_verdict_by_message(
    guild_id: str,
    message_id: str,
    session: AsyncSession = Depends(get_session),
) -> VerdictResponse:
    """Резолв verdict_id по Discord message_id для /override-verdict.

    Возвращает свежайший вердикт по (guild_id, message_id). 404 если сообщение
    не модерировалось (например, оно было в игнор-канале или от админа).
    """
    repo = VerdictRepository(session)
    verdict = await repo.find_by_message(guild_id, message_id)
    if verdict is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="verdict for this message not found",
        )
    return VerdictResponse(
        id=verdict.id,
        message_id=verdict.message_id,
        guild_id=verdict.guild_id,
        channel_id=verdict.channel_id,
        author_id=verdict.author_id,
        source_kind=verdict.source_kind,
        score=verdict.score,
        label=verdict.label,
        action=verdict.action,
        content=verdict.content,
        created_at=verdict.created_at,
    )
