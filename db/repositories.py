from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import ContentCache, ModelVersion, Verdict


class ModelVersionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_or_create(self, name: str, base_model: str) -> ModelVersion:
        stmt = select(ModelVersion).where(ModelVersion.name == name)
        existing = (await self._session.execute(stmt)).scalar_one_or_none()
        if existing:
            return existing
        mv = ModelVersion(name=name, base_model=base_model, is_active=True)
        self._session.add(mv)
        await self._session.flush()  # получаем id, но без commit
        return mv


class CacheRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, content_hash: str, source_kind: str) -> ContentCache | None:
        stmt = select(ContentCache).where(
            ContentCache.content_hash == content_hash,
            ContentCache.source_kind == source_kind,
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def upsert(self, *, content_hash: str, source_kind: str,
                     score: float, label: str, action: str,
                     categories: dict, model_version_id: int,
                     extracted_text: str | None = None) -> None:
        # ON CONFLICT для атомарной вставки/обновления счётчика.
        # Postgres-специфичный insert с .on_conflict_do_update.
        stmt = insert(ContentCache).values(
            content_hash=content_hash,
            source_kind=source_kind,
            score=score, label=label, action=action,
            categories=categories,
            model_version_id=model_version_id,
            extracted_text=extracted_text,
        ).on_conflict_do_update(
            index_elements=["content_hash", "source_kind"],
            set_=dict(
                hit_count=ContentCache.hit_count + 1,
                last_hit_at=func.now(),  # type: ignore
            ),
        )
        await self._session.execute(stmt)


class VerdictRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, **kwargs) -> Verdict:
        verdict = Verdict(**kwargs)
        self._session.add(verdict)
        await self._session.flush()
        return verdict

    async def find_recent_for_message(
        self, message_id: str, source_kind: str, content_hash: str,
    ) -> Verdict | None:
        """Для идемпотентности: уже есть свежий вердикт для этого сообщения?"""
        stmt = (
            select(Verdict)
            .where(
                Verdict.message_id == message_id,
                Verdict.source_kind == source_kind,
                Verdict.content_hash == content_hash,
            )
            .order_by(Verdict.created_at.desc())
            .limit(1)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()