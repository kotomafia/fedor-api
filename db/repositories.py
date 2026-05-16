from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import (
    ContentCache,
    GuildIgnoredChannel,
    GuildIgnoredRole,
    GuildSettings,
    GuildWhitelist,
    LogLevel,
    ModelVersion,
    ReviewQueue,
    ReviewStatus,
    Verdict,
)


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

    async def find_by_message(
        self, guild_id: str, message_id: str,
    ) -> Verdict | None:
        """Свежайший вердикт по (guild_id, message_id) для override-команды."""
        stmt = (
            select(Verdict)
            .where(
                Verdict.guild_id == guild_id,
                Verdict.message_id == message_id,
            )
            .order_by(Verdict.created_at.desc())
            .limit(1)
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()


class GuildSettingsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, guild_id: str) -> GuildSettings | None:
        stmt = select(GuildSettings).where(GuildSettings.guild_id == guild_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_or_create(self, guild_id: str) -> GuildSettings:
        existing = await self.get(guild_id)
        if existing:
            return existing
        gs = GuildSettings(guild_id=guild_id)
        self._session.add(gs)
        await self._session.flush()
        return gs

    async def patch(
        self,
        guild_id: str,
        *,
        review_channel_id: str | None | object = ...,
        log_channel_id: str | None | object = ...,
        log_level: str | object = ...,
    ) -> GuildSettings:
        """Частичное обновление. Передавайте только те поля, что нужно изменить.

        None — допустимое значение (сбросить канал). Сентинел Ellipsis отличает
        "не передано" от "явно установить None".
        """
        gs = await self.get_or_create(guild_id)
        if review_channel_id is not ...:
            gs.review_channel_id = review_channel_id  # type: ignore[assignment]
        if log_channel_id is not ...:
            gs.log_channel_id = log_channel_id  # type: ignore[assignment]
        if log_level is not ...:
            gs.log_level = log_level  # type: ignore[assignment]
        await self._session.flush()
        return gs


class GuildWhitelistRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self, guild_id: str) -> list[GuildWhitelist]:
        stmt = (
            select(GuildWhitelist)
            .where(GuildWhitelist.guild_id == guild_id)
            .order_by(GuildWhitelist.added_at.desc())
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def get_phrases(self, guild_id: str) -> list[str]:
        stmt = select(GuildWhitelist.phrase).where(
            GuildWhitelist.guild_id == guild_id,
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def add(
        self, guild_id: str, phrase: str, added_by: str,
    ) -> GuildWhitelist | None:
        """Добавляет фразу. Возвращает None если фраза уже есть в списке гильдии."""
        entry = GuildWhitelist(
            guild_id=guild_id, phrase=phrase, added_by=added_by,
        )
        self._session.add(entry)
        try:
            await self._session.flush()
        except IntegrityError:
            await self._session.rollback()
            return None
        return entry

    async def delete(self, guild_id: str, phrase_id: int) -> bool:
        stmt = (
            delete(GuildWhitelist)
            .where(
                GuildWhitelist.id == phrase_id,
                GuildWhitelist.guild_id == guild_id,
            )
        )
        result = await self._session.execute(stmt)
        return result.rowcount > 0


class GuildIgnoredChannelRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self, guild_id: str) -> list[GuildIgnoredChannel]:
        stmt = (
            select(GuildIgnoredChannel)
            .where(GuildIgnoredChannel.guild_id == guild_id)
            .order_by(GuildIgnoredChannel.added_at.desc())
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def get_channel_ids(self, guild_id: str) -> list[str]:
        stmt = select(GuildIgnoredChannel.channel_id).where(
            GuildIgnoredChannel.guild_id == guild_id,
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def add(
        self, guild_id: str, channel_id: str, added_by: str,
    ) -> GuildIgnoredChannel | None:
        entry = GuildIgnoredChannel(
            guild_id=guild_id, channel_id=channel_id, added_by=added_by,
        )
        self._session.add(entry)
        try:
            await self._session.flush()
        except IntegrityError:
            await self._session.rollback()
            return None
        return entry

    async def delete(self, guild_id: str, channel_id: str) -> bool:
        stmt = (
            delete(GuildIgnoredChannel)
            .where(
                GuildIgnoredChannel.guild_id == guild_id,
                GuildIgnoredChannel.channel_id == channel_id,
            )
        )
        result = await self._session.execute(stmt)
        return result.rowcount > 0


class GuildIgnoredRoleRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list(self, guild_id: str) -> list[GuildIgnoredRole]:
        stmt = (
            select(GuildIgnoredRole)
            .where(GuildIgnoredRole.guild_id == guild_id)
            .order_by(GuildIgnoredRole.added_at.desc())
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def get_role_ids(self, guild_id: str) -> list[str]:
        stmt = select(GuildIgnoredRole.role_id).where(
            GuildIgnoredRole.guild_id == guild_id,
        )
        return list((await self._session.execute(stmt)).scalars().all())

    async def add(
        self, guild_id: str, role_id: str, added_by: str,
    ) -> GuildIgnoredRole | None:
        entry = GuildIgnoredRole(
            guild_id=guild_id, role_id=role_id, added_by=added_by,
        )
        self._session.add(entry)
        try:
            await self._session.flush()
        except IntegrityError:
            await self._session.rollback()
            return None
        return entry

    async def delete(self, guild_id: str, role_id: str) -> bool:
        stmt = (
            delete(GuildIgnoredRole)
            .where(
                GuildIgnoredRole.guild_id == guild_id,
                GuildIgnoredRole.role_id == role_id,
            )
        )
        result = await self._session.execute(stmt)
        return result.rowcount > 0


# Маппинг для автоматического выставления corrected_score по corrected_label.
# Модератор не вводит число вручную — всё, что от него нужно, это выбор метки.
LABEL_TO_SCORE: dict[str, float] = {
    "neutral": 0.0,
    "uncertain": 0.5,
    "toxic": 1.0,
}


class ReviewQueueRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, verdict_id: int) -> ReviewQueue | None:
        stmt = select(ReviewQueue).where(ReviewQueue.verdict_id == verdict_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def upsert_pending(
        self,
        *,
        verdict_id: int,
        review_message_id: str | None,
    ) -> ReviewQueue:
        """Создаёт запись pending, либо обновляет review_message_id у существующей."""
        existing = await self.get(verdict_id)
        if existing:
            if review_message_id is not None:
                existing.review_message_id = review_message_id
            await self._session.flush()
            return existing
        entry = ReviewQueue(
            verdict_id=verdict_id,
            status=ReviewStatus.PENDING.value,
            review_message_id=review_message_id,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def record_decision(
        self,
        *,
        verdict_id: int,
        status: str,
        corrected_label: str | None,
        reviewer_id: str,
        notes: str | None = None,
    ) -> ReviewQueue:
        """Записывает решение модератора. corrected_score выводится из corrected_label."""
        entry = await self.get(verdict_id)
        if entry is None:
            # override может прийти на верект, которого ещё нет в очереди —
            # создаём запись на лету
            entry = ReviewQueue(verdict_id=verdict_id, status=ReviewStatus.PENDING.value)
            self._session.add(entry)
            await self._session.flush()
        entry.status = status
        entry.corrected_label = corrected_label
        entry.corrected_score = (
            LABEL_TO_SCORE.get(corrected_label) if corrected_label else None
        )
        entry.reviewer_id = reviewer_id
        entry.reviewer_decision_at = datetime.now(timezone.utc)
        if notes is not None:
            entry.notes = notes
        await self._session.flush()
        return entry