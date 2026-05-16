import time
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from api.config import api_settings
from api.db.repositories import (
    CacheRepository,
    GuildWhitelistRepository,
    ModelVersionRepository,
    VerdictRepository,
)
from api.ml.base import TextClassifier
from api.services.hashing import hash_image, hash_text


def _content_matches_whitelist(content: str, phrases: list[str]) -> bool:
    """Подстрочная проверка без регистра."""
    if not phrases:
        return False
    haystack = content.lower()
    return any(phrase.lower() in haystack for phrase in phrases)


class ModerationService:
    def __init__(self, session: AsyncSession, classifier: TextClassifier) -> None:
        self._session = session
        self._classifier = classifier
        self._verdicts = VerdictRepository(session)
        self._cache = CacheRepository(session)
        self._versions = ModelVersionRepository(session)
        self._whitelist = GuildWhitelistRepository(session)

    def _classify_score(self, score: float) -> tuple[str, str]:
        if score < api_settings.threshold_uncertain:
            return "neutral", "allow"
        if score < api_settings.threshold_toxic:
            return "uncertain", "flag"
        return "toxic", "block"

    async def moderate_text(
        self, *, message_id: str, guild_id: str, channel_id: str,
        author_id: str, content: str,
    ) -> dict:
        content_hash = hash_text(content)

        # 1. Идемпотентность: тот же message_id с тем же контентом уже обрабатывался?
        existing = await self._verdicts.find_recent_for_message(
            message_id, "text", content_hash,
        )
        if existing:
            return self._verdict_to_dict(existing, cache_hit=False, repeat=True)

        # 2. Белый список гильдии — обходит кеш и инференс целиком.
        #    Контент НЕ кешируется, чтобы добавление/удаление слова из списка
        #    сразу влияло на дальнейшие сообщения с тем же текстом.
        whitelist = await self._whitelist.get_phrases(guild_id)
        if _content_matches_whitelist(content, whitelist):
            model_version = await self._versions.get_or_create(
                name=self._classifier.model_version,
                base_model=self._classifier.model_version,
            )
            verdict = await self._verdicts.create(
                message_id=message_id, guild_id=guild_id,
                channel_id=channel_id, author_id=author_id,
                source_kind="text",
                content_hash=content_hash, content=content,
                score=0.0, label="neutral", action="allow",
                categories={"whitelist": 1.0},
                model_version_id=model_version.id,
                cache_hit=False, inference_ms=0,
            )
            return self._verdict_to_dict(verdict, cache_hit=False)

        # 3. Кеш по хешу контента
        cached = await self._cache.get(content_hash, "text")
        model_version = await self._versions.get_or_create(
            name=self._classifier.model_version,
            base_model=self._classifier.model_version,
        )

        if cached and cached.model_version_id == model_version.id:
            verdict = await self._verdicts.create(
                message_id=message_id, guild_id=guild_id,
                channel_id=channel_id, author_id=author_id,
                source_kind="text",
                content_hash=content_hash, content=content,
                score=cached.score, label=cached.label, action=cached.action,
                categories=cached.categories,
                model_version_id=model_version.id,
                cache_hit=True, inference_ms=0,
            )
            await self._cache.upsert(
                content_hash=content_hash, source_kind="text",
                score=cached.score, label=cached.label, action=cached.action,
                categories=cached.categories, model_version_id=model_version.id,
            )
            return self._verdict_to_dict(verdict, cache_hit=True)

        # 3. Реальный инференс
        start = time.perf_counter()
        result = await self._classifier.classify(content)
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        label, action = self._classify_score(result.score)

        verdict = await self._verdicts.create(
            message_id=message_id, guild_id=guild_id,
            channel_id=channel_id, author_id=author_id,
            source_kind="text",
            content_hash=content_hash, content=content,
            score=result.score, label=label, action=action,
            categories=result.categories,
            model_version_id=model_version.id,
            cache_hit=False, inference_ms=elapsed_ms,
        )
        await self._cache.upsert(
            content_hash=content_hash, source_kind="text",
            score=result.score, label=label, action=action,
            categories=result.categories,
            model_version_id=model_version.id,
        )
        return self._verdict_to_dict(verdict, cache_hit=False)

    def _verdict_to_dict(
        self, verdict, *, cache_hit: bool, repeat: bool = False,
    ) -> dict:
        return {
            "verdict_id": verdict.id,
            "message_id": verdict.message_id,
            "score": verdict.score,
            "label": verdict.label,
            "action": verdict.action,
            "categories": verdict.categories,
            "model_version": str(verdict.model_version_id),
            "cache_hit": cache_hit,
            "repeat": repeat,
            "extracted_text": None,
            "ocr_confidence": None,
        }