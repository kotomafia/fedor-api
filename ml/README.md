# fedor-ml

ML-модели для [fedor](https://github.com/kotomafia/fedor): классификация токсичности текста и OCR на изображениях.

Subtree в [fedor-api](https://github.com/kotomafia/fedor-api) (`ml/`) и в [fedor](https://github.com/kotomafia/fedor) (`api/ml/`). Этот репозиторий — листовой источник правды; правки распространяются через `git subtree push`.

## Модели

| Назначение | Модель |
|------------|--------|
| Текст | [cointegrated/rubert-tiny-toxicity](https://huggingface.co/cointegrated/rubert-tiny-toxicity) |
| OCR + токсичность | [s-nlp/russian_toxicity_classifier](https://huggingface.co/s-nlp/russian_toxicity_classifier), EasyOCR |

Веса скачиваются с Hugging Face при первом запуске воркера.

## Зависимости

```powershell
pip install -r requirements-ml.txt
```

Вместе с API (из корня `fedor`):

```powershell
pip install -r requirements.txt
pip install -r api/ml/requirements-ml.txt
```

## Использование в коде

Импорты: `from api.ml...` — пакет `api` должен быть на `PYTHONPATH`.  
Запускайте Celery-воркеры из **корня** мета-репозитория [fedor](https://github.com/kotomafia/fedor), не из этого каталога.

```powershell
python -m celery -A api.celery_app:celery_app worker -Q inference_text -l info --pool=solo
python -m celery -A api.celery_app:celery_app worker -Q inference_image -l info --pool=solo
```

## Клонирование

```bash
git clone https://github.com/kotomafia/fedor-ml.git
```

В составе API (subtree уже внутри):

```bash
git clone https://github.com/kotomafia/fedor-api.git
```

Весь проект:

```bash
git clone https://github.com/kotomafia/fedor.git
```

## Связанные репозитории

- [fedor-api](https://github.com/kotomafia/fedor-api) — оркестрация и очереди
- [fedor](https://github.com/kotomafia/fedor) — полный dev-стек
