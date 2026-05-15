from celery import Celery

from api.config import api_settings


def build_celery() -> Celery:
    broker = api_settings.celery_broker_url or api_settings.redis_url
    backend = api_settings.celery_result_backend or api_settings.redis_url

    app = Celery(
        "moderator",
        broker=broker,
        backend=backend,
        include=["api.tasks.moderation"],
    )

    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],

        task_time_limit=api_settings.celery_task_time_limit,
        task_soft_time_limit=api_settings.celery_task_soft_time_limit,
        result_expires=api_settings.celery_result_expires,

        # Один таск на воркер за раз. Для ML это правильно:
        # параллелить инференс лучше через несколько воркеров, чем prefetch.
        worker_prefetch_multiplier=1,
        task_acks_late=True,

        # Раздельные очереди — на будущее.
        task_routes={
            "tasks.moderation.classify_text": {"queue": "inference_text"},
            "tasks.moderation.classify_image": {"queue": "inference_image"},
        },

        timezone="UTC",
        enable_utc=True,
    )
    return app


celery_app = build_celery()