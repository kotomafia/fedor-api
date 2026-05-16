from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ModerationLabel(StrEnum):
    NEUTRAL = "neutral"
    UNCERTAIN = "uncertain"
    TOXIC = "toxic"


class ModerationAction(StrEnum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


class TextModerationRequest(BaseModel):
    message_id: str
    guild_id: str
    channel_id: str
    author_id: str
    content: str = Field(min_length=1, max_length=4000)


class ImageModerationRequest(BaseModel):
    message_id: str
    guild_id: str
    channel_id: str
    author_id: str
    # base64-encoded image bytes
    image_b64: str = Field(min_length=1)
    # Имя файла или URL — для логов и аудита, не используется в инференсе
    source_hint: str | None = None


class ModerationResponse(BaseModel):
    # ID записи в таблице verdicts. Используется ботом для записи
    # ReviewQueue и override-команды. None — если вердикт по каким-то
    # причинам не был сохранён в БД (например, skipped).
    verdict_id: int | None = None
    message_id: str
    score: float = Field(ge=0.0, le=1.0)
    label: ModerationLabel
    action: ModerationAction
    model_version: str
    processed_at: datetime
    categories: dict[str, float] = Field(default_factory=dict)
    # для изображений
    extracted_text: str | None = None
    ocr_confidence: float | None = None
    skipped_reason: str | None = None

class TaskStatus(StrEnum):
    PENDING = "pending"      # ещё в очереди или выполняется
    SUCCESS = "success"
    FAILURE = "failure"


class TaskSubmittedResponse(BaseModel):
    task_id: str
    status: TaskStatus = TaskStatus.PENDING


class TaskResultResponse(BaseModel):
    task_id: str
    status: TaskStatus
    result: ModerationResponse | None = None
    error: str | None = None


class LogLevelEnum(StrEnum):
    OFF = "off"
    BLOCK = "block"
    BLOCK_FLAG = "block_flag"
    ALL = "all"


class GuildSettingsResponse(BaseModel):
    """Агрегированный ответ для бота: настройки + игнор-списки одним запросом."""
    guild_id: str
    review_channel_id: str | None
    log_channel_id: str | None
    log_level: LogLevelEnum
    ignored_channel_ids: list[str] = Field(default_factory=list)
    ignored_role_ids: list[str] = Field(default_factory=list)
    whitelist_count: int = 0


class GuildSettingsPatchRequest(BaseModel):
    """Все поля опциональны. None — допустимое значение (сброс канала)."""
    review_channel_id: str | None = Field(default=None)
    log_channel_id: str | None = Field(default=None)
    log_level: LogLevelEnum | None = Field(default=None)

    # Чтобы отличить "не передано" от "передан null", используем set-fields
    # на стороне роутера через model_fields_set.


class WhitelistEntry(BaseModel):
    id: int
    guild_id: str
    phrase: str
    added_by: str
    added_at: datetime


class WhitelistAddRequest(BaseModel):
    phrase: str = Field(min_length=1, max_length=200)
    added_by: str = Field(min_length=1, max_length=32)


class IgnoredChannelEntry(BaseModel):
    id: int
    guild_id: str
    channel_id: str
    added_by: str
    added_at: datetime


class IgnoredChannelAddRequest(BaseModel):
    channel_id: str = Field(min_length=1, max_length=32)
    added_by: str = Field(min_length=1, max_length=32)


class IgnoredRoleEntry(BaseModel):
    id: int
    guild_id: str
    role_id: str
    added_by: str
    added_at: datetime


class IgnoredRoleAddRequest(BaseModel):
    role_id: str = Field(min_length=1, max_length=32)
    added_by: str = Field(min_length=1, max_length=32)


class ReviewStatusEnum(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SKIPPED = "skipped"


class ReviewCreateRequest(BaseModel):
    """Бот создаёт pending-запись после отправки embed'а в review-канал."""
    verdict_id: int
    review_message_id: str | None = None


class ReviewDecisionRequest(BaseModel):
    """Решение модератора. corrected_score вычисляется на сервере по corrected_label."""
    status: ReviewStatusEnum
    corrected_label: ModerationLabel | None = None
    reviewer_id: str = Field(min_length=1, max_length=32)
    notes: str | None = Field(default=None, max_length=500)


class ReviewEntryResponse(BaseModel):
    id: int
    verdict_id: int
    status: ReviewStatusEnum
    review_message_id: str | None
    reviewer_id: str | None
    reviewer_decision_at: datetime | None
    corrected_label: ModerationLabel | None
    corrected_score: float | None
    notes: str | None
    created_at: datetime


class VerdictResponse(BaseModel):
    """Лёгкий ответ для override-команды: бот находит verdict_id по message_id."""
    id: int
    message_id: str
    guild_id: str
    channel_id: str
    author_id: str
    source_kind: str
    score: float
    label: str
    action: str
    content: str
    created_at: datetime