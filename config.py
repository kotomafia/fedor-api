from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # пороги
    threshold_uncertain: float = 0.3
    threshold_toxic: float = 0.7
    threshold_toxic_ocr: float = 0.5  # для image/OCR-пути — ниже, чем для текста

    # ML
    ml_model_name: str = "cointegrated/rubert-tiny-toxicity"
    ocr_classifier_name: str = "s-nlp/russian_toxicity_classifier"
    ml_max_length: int = 256
    min_ocr_fragment_confidence: float = 0.15
    min_ocr_overall_confidence: float = 0.2  # ниже — текст слишком ненадёжен

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Celery / Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None     # если не задан, берём redis_url
    celery_result_backend: str | None = None # то же
    celery_task_time_limit: int = 60         # hard limit, секунды
    celery_task_soft_time_limit: int = 50    # soft limit, секунды
    celery_result_expires: int = 3600        # TTL результатов в backend

    # Database
    # Задаётся через DATABASE_URL в .env; значение ниже — только fallback для локальной разработки.
    database_url: str = (
        "postgresql+asyncpg://moderator:CHANGE_ME@localhost:5432/moderator"
    )

api_settings = APISettings()