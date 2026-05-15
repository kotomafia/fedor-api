import io

import cv2
import numpy as np
from PIL import Image


MAX_DIMENSION = 1600  # длинная сторона; больше — медленнее OCR без выигрыша в качестве
MIN_DIMENSION = 600   # очень мелкие картинки апскейлим — OCR работает лучше


def load_image_bgr(image_bytes: bytes) -> np.ndarray:
    """Грузим через Pillow (понимает WebP, HEIC и пр.), конвертируем в BGR для OpenCV."""
    pil = Image.open(io.BytesIO(image_bytes))
    pil = pil.convert("RGB")
    arr = np.array(pil)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def resize_if_needed(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)

    if longest > MAX_DIMENSION:
        scale = MAX_DIMENSION / longest
    elif longest < MIN_DIMENSION:
        scale = MIN_DIMENSION / longest
    else:
        return image

    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def denoise_colored(image: np.ndarray) -> np.ndarray:
    """Лёгкое шумоподавление; на GIF не применяем — дорого и мало пользы."""
    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=3,
        hColor=3,
        templateWindowSize=7,
        searchWindowSize=21,
    )


def prepare_frame_for_ocr(frame_bgr: np.ndarray, *, denoise: bool) -> np.ndarray:
    frame = resize_if_needed(frame_bgr)
    if denoise:
        frame = denoise_colored(frame)
    return frame


def preprocess_for_ocr(image_bytes: bytes) -> np.ndarray:
    """Один кадр: resize + denoise (для прямого вызова без extract_frames)."""
    return prepare_frame_for_ocr(load_image_bgr(image_bytes), denoise=True)


def extract_frames(image_bytes: bytes, max_frames: int = 5) -> list[np.ndarray]:
    """Сырые BGR-кадры без resize/denoise — дальше готовит OCR."""
    pil = Image.open(io.BytesIO(image_bytes))

    if not getattr(pil, "is_animated", False):
        return [load_image_bgr(image_bytes)]

    total = pil.n_frames
    if total <= max_frames:
        indices = list(range(total))
    else:
        step = (total - 1) / (max_frames - 1)
        indices = sorted({round(i * step) for i in range(max_frames)})

    frames: list[np.ndarray] = []
    for idx in indices:
        pil.seek(idx)
        arr = np.array(pil.convert("RGB"))
        frames.append(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
    return frames
