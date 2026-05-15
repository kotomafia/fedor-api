# fedor-api

FastAPI-шлюз, Celery-воркеры и слой БД для [fedor](https://github.com/kotomafia/fedor).

Subtree в мета-репозитории: `api/` ← этот репозиторий.  
ML-код — subtree [fedor-ml](https://github.com/kotomafia/fedor-ml) в каталоге `ml/` (в мета-репо путь `api/ml`).

## Клонирование

Самодостаточный репозиторий, `ml/` уже включён как обычная папка:

```bash
git clone https://github.com/kotomafia/fedor-api.git
```

Или весь стек через мета-репозиторий:

```bash
git clone https://github.com/kotomafia/fedor.git
```

## Зависимости

```powershell
pip install -r requirements.txt
pip install -r ml/requirements-ml.txt
```

## Запуск

Из **корня** [fedor](https://github.com/kotomafia/fedor) (Redis, Postgres, `.env`):

```powershell
docker compose up -d redis postgres
python -m alembic upgrade head
python -m api
python -m celery -A api.celery_app:celery_app worker -Q inference_text -l info --pool=solo
python -m celery -A api.celery_app:celery_app worker -Q inference_image -l info --pool=solo
```

Очереди: `inference_text`, `inference_image`.

## Переменные окружения

Основные (см. `.env.example` в `fedor`):

| Переменная | Описание |
|------------|----------|
| `DATABASE_URL` | Postgres (asyncpg) |
| `REDIS_URL` | Брокер Celery |
| `THRESHOLD_*` | Пороги модерации |
| `ML_MODEL_NAME`, `OCR_CLASSIFIER_NAME` | Модели Hugging Face |

API **без аутентификации** — не открывайте порт 8000 в интернет без защиты.

## Структура

```
app.py, routers/, services/   HTTP API
tasks/, celery_app.py         Celery
db/                           SQLAlchemy, репозитории
ml/                           ← subtree fedor-ml
```

## Синхронизация с fedor-ml

```powershell
git remote add fedor-ml https://github.com/kotomafia/fedor-ml.git

.\scripts\sync-subtrees.ps1 -Direction pull
.\scripts\sync-subtrees.ps1 -Direction push
```

Источник правды для `ml/` — мета-репо [fedor](https://github.com/kotomafia/fedor). Если правите ml прямо здесь, после `subtree push` сделайте такой же push в `fedor/api/ml` (или просто работайте из `fedor`).

## Связанные репозитории

- [fedor](https://github.com/kotomafia/fedor) — инфраструктура и миграции
- [fedor-bot](https://github.com/kotomafia/fedor-bot) — Discord-клиент
- [fedor-ml](https://github.com/kotomafia/fedor-ml) — токсичность и OCR
