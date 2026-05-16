from datetime import datetime
from enum import StrEnum

from sqlalchemy import (
    BigInteger, CheckConstraint, DateTime, Float, ForeignKey, Index,
    Integer, String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class SourceKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"   # модератор согласен с моделью
    REJECTED = "rejected"   # модератор не согласен; для обучения это и ценно
    SKIPPED = "skipped"     # модератор пропустил, не разметил


class LogLevel(StrEnum):
    OFF = "off"                 # ничего не логировать в Discord log-channel
    BLOCK = "block"             # только блокировки
    BLOCK_FLAG = "block_flag"   # блокировки + спорные (рекомендуемый дефолт)
    ALL = "all"                 # включая allow


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    base_model: Mapped[str] = mapped_column(String(256), nullable=False)
    # для будущих fine-tuned версий: hash весов, путь к чекпоинту в S3, и т.д.
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    is_active: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )


class Verdict(Base):
    __tablename__ = "verdicts"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)

    # Discord-идентификаторы как строки: snowflakes больше int32,
    # хранить как BigInt тоже можно, но строки удобнее в JSON-API.
    message_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    guild_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    channel_id: Mapped[str] = mapped_column(String(32), nullable=False)
    author_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)

    source_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    # SHA-256 от исходного контента (текста или image bytes)
    # 64 символа hex; используется для кеша и для идемпотентности
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Сам контент: для text — оригинал; для image — extracted_text от OCR
    content: Mapped[str] = mapped_column(Text, nullable=False)

    score: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    categories: Mapped[dict] = mapped_column(JSONB, default=dict)

    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), nullable=False,
    )
    # Был ли результат взят из кеша (для аналитики экономии)
    cache_hit: Mapped[bool] = mapped_column(default=False, nullable=False)

    # OCR-метаданные, NULL для текстовых
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ocr_engine: Mapped[str | None] = mapped_column(String(64), nullable=True)

    inference_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        nullable=False, index=True,
    )

    __table_args__ = (
        CheckConstraint("score >= 0 AND score <= 1", name="score_range"),
        Index("ix_verdicts_guild_created", "guild_id", "created_at"),
    )


class ContentCache(Base):
    __tablename__ = "content_cache"

    # Натуральный PK — сам хеш контента (плюс kind, чтобы текст и картинку
    # с одинаковым хешем не перепутать; практически невозможно, но дисциплинарно правильно)
    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_kind: Mapped[str] = mapped_column(String(16), primary_key=True)

    # Денормализованная копия результата — чтобы не джойнить с verdicts на горячем пути
    score: Mapped[float] = mapped_column(Float, nullable=False)
    label: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    categories: Mapped[dict] = mapped_column(JSONB, default=dict)
    model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), nullable=False,
    )
    extracted_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    hit_count: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    last_hit_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )


class ReviewQueue(Base):
    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    verdict_id: Mapped[int] = mapped_column(
        ForeignKey("verdicts.id", ondelete="CASCADE"), nullable=False, unique=True,
    )
    verdict: Mapped[Verdict] = relationship("Verdict")

    status: Mapped[str] = mapped_column(String(16), nullable=False,
                                        default=ReviewStatus.PENDING.value, index=True)
    # ID сообщения модерационного бота в канале модерации, чтобы потом обновить эмбед
    review_message_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reviewer_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reviewer_decision_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    # Скорректированная метка модератором (neutral / uncertain / toxic)
    corrected_label: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Числовое целевое значение для дообучения. Заполняется автоматически
    # по corrected_label: neutral -> 0.0, toxic -> 1.0, uncertain -> 0.5.
    # Модератор не вводит число руками.
    corrected_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )


class GuildSettings(Base):
    __tablename__ = "guild_settings"

    guild_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    review_channel_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    log_channel_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    log_level: Mapped[str] = mapped_column(
        String(16), nullable=False, default=LogLevel.BLOCK_FLAG.value,
        server_default=LogLevel.BLOCK_FLAG.value,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(),
        onupdate=func.now(), nullable=False,
    )


class GuildWhitelist(Base):
    __tablename__ = "guild_whitelist"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    guild_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    phrase: Mapped[str] = mapped_column(Text, nullable=False)
    added_by: Mapped[str] = mapped_column(String(32), nullable=False)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("guild_id", "phrase", name="uq_guild_whitelist_phrase"),
    )


class GuildIgnoredChannel(Base):
    __tablename__ = "guild_ignored_channels"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    guild_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    channel_id: Mapped[str] = mapped_column(String(32), nullable=False)
    added_by: Mapped[str] = mapped_column(String(32), nullable=False)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("guild_id", "channel_id", name="uq_guild_ignored_channel"),
    )


class GuildIgnoredRole(Base):
    __tablename__ = "guild_ignored_roles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    guild_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    role_id: Mapped[str] = mapped_column(String(32), nullable=False)
    added_by: Mapped[str] = mapped_column(String(32), nullable=False)
    added_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("guild_id", "role_id", name="uq_guild_ignored_role"),
    )