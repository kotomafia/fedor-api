import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from api.ml.base import TextClassifier, ToxicityScore


# Категории модели cointegrated/rubert-tiny-toxicity:
#   non-toxic, insult, obscenity, threat, dangerous
# Агрегируем как 1 - P(non-toxic), считая всё кроме non-toxic токсичным.
NON_TOXIC_LABEL = "non-toxic"


class RubertTinyToxicity(TextClassifier):
    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny-toxicity",
        device: str | None = None,
        max_length: int = 256,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading tokenizer: {model}", model=model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("Loading model: {model} on {device}", model=model_name, device=self._device)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

        self._id2label = self._model.config.id2label
        logger.info("Model labels: {labels}", labels=self._id2label)

        # Один поток-исполнитель для инференса.
        # Несколько потоков на одну модель без блокировки не дадут выигрыша
        # (GIL + сам torch удерживает ресурс), но один тред уводит работу с event loop.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-inference")

    @property
    def model_version(self) -> str:
        # На третьем шаге версия = имя модели. Позже добавим хеш чекпоинта и тег.
        return self._model_name

    def _infer_sync(self, texts: list[str]) -> list[dict[str, float]]:
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length,
        ).to(self._device)

        with torch.inference_mode():
            logits = self._model(**inputs).logits
            # У этой модели multi-label, выход — независимые сигмоиды на каждый класс.
            probs = torch.sigmoid(logits).cpu().numpy()

        results = []
        for row in probs:
            categories = {
                self._id2label[i]: float(p) for i, p in enumerate(row)
            }
            results.append(categories)
        return results

    async def classify(self, text: str) -> ToxicityScore:
        results = await self.classify_batch([text])
        return results[0]

    async def classify_batch(self, texts: list[str]) -> list[ToxicityScore]:
        if not texts:
            return []

        loop = asyncio.get_running_loop()
        categories_list = await loop.run_in_executor(
            self._executor, partial(self._infer_sync, texts)
        )

        scores = []
        for categories in categories_list:
            # Агрегирование: вероятность хоть какой-то токсичности = 1 - P(non-toxic)
            non_toxic = categories.get(NON_TOXIC_LABEL, 0.0)
            score = 1.0 - non_toxic
            scores.append(
                ToxicityScore(
                    score=score,
                    categories=categories,
                    model_version=self.model_version,
                )
            )
        return scores

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)


class SNlpToxicityClassifier(TextClassifier):
    """Бинарный классификатор toxic/non-toxic — лучше на коротких фрагментах и OCR-тексте."""

    def __init__(
        self,
        model_name: str = "s-nlp/russian_toxicity_classifier",
        device: str | None = None,
        max_length: int = 256,
    ) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading OCR tokenizer: {model}", model=model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info(
            "Loading OCR classifier: {model} on {device}",
            model=model_name,
            device=self._device,
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

        self._id2label = self._model.config.id2label
        self._toxic_idx = self._resolve_toxic_index()
        logger.info("OCR classifier labels: {labels} | toxic_idx={idx}", labels=self._id2label, idx=self._toxic_idx)

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr-cls")

    def _resolve_toxic_index(self) -> int:
        for idx, label in self._id2label.items():
            s = str(label).lower()
            if s in ("toxic", "label_1", "1", "positive") or ("toxic" in s and "non" not in s):
                return int(idx)
        return 1

    @property
    def model_version(self) -> str:
        return self._model_name

    def _infer_sync(self, texts: list[str]) -> list[tuple[float, dict[str, float]]]:
        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length,
        ).to(self._device)

        with torch.inference_mode():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results: list[tuple[float, dict[str, float]]] = []
        for row in probs:
            categories = {
                str(self._id2label[i]): float(p) for i, p in enumerate(row)
            }
            toxic_score = float(row[self._toxic_idx])
            results.append((toxic_score, categories))
        return results

    async def classify(self, text: str) -> ToxicityScore:
        results = await self.classify_batch([text])
        return results[0]

    async def classify_batch(self, texts: list[str]) -> list[ToxicityScore]:
        if not texts:
            return []

        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            self._executor, partial(self._infer_sync, texts)
        )

        return [
            ToxicityScore(
                score=score,
                categories=categories,
                model_version=self.model_version,
            )
            for score, categories in raw
        ]

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)