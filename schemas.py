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