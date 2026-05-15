import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import easyocr
from loguru import logger

from api.ml.base import OCREngine, OCRResult
from api.ml.image_preprocess import extract_frames, prepare_frame_for_ocr


def _aggregate_fragments(seen: dict[str, float]) -> OCRResult:
    if not seen:
        return OCRResult(text="", confidence=0.0, fragments=())

    fragments = tuple(seen.items())
    total_chars = sum(len(t) for t in seen)
    avg_conf = (
        sum(len(t) * c for t, c in fragments) / total_chars if total_chars else 0.0
    )
    return OCRResult(
        text=" ".join(seen.keys()),
        confidence=avg_conf,
        fragments=fragments,
    )


class EasyOCREngine(OCREngine):
    def __init__(
        self,
        languages: tuple[str, ...] = ("ru", "en"),
        gpu: bool | None = None,
        max_frames: int = 5,
    ) -> None:
        self._languages = languages
        self._max_frames = max_frames
        if gpu is None:
            import torch

            gpu = torch.cuda.is_available()
        logger.info(
            "Loading EasyOCR | langs={langs} | gpu={gpu} | max_frames={mf}",
            langs=languages,
            gpu=gpu,
            mf=max_frames,
        )
        self._reader = easyocr.Reader(list(languages), gpu=gpu, verbose=False)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
        logger.info("EasyOCR ready")

    @property
    def engine_version(self) -> str:
        return f"easyocr-{'_'.join(self._languages)}"

    def _extract_sync(self, image_bytes: bytes) -> OCRResult:
        frames = extract_frames(image_bytes, max_frames=self._max_frames)
        denoise = len(frames) == 1
        seen: dict[str, float] = {}

        for frame in frames:
            prepared = prepare_frame_for_ocr(frame, denoise=denoise)
            raw = self._reader.readtext(prepared, detail=1, paragraph=False)
            for _, text, conf in raw:
                text = text.strip()
                if not text:
                    continue
                seen[text] = max(seen.get(text, 0.0), float(conf))

        return _aggregate_fragments(seen)

    async def extract(self, image_bytes: bytes) -> OCRResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, partial(self._extract_sync, image_bytes)
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
