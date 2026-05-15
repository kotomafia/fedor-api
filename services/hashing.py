import hashlib


def hash_text(text: str) -> str:
    """Канонизация перед хешем: trim + lower + сжатие пробелов.
    Это даёт нечувствительность к тривиальным изменениям —
    лишним пробелам, регистру."""
    canonical = " ".join(text.lower().strip().split())
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def hash_image(image_bytes: bytes) -> str:
    """Для картинок — точный байтовый хеш.
    Тот же мем, перепосланный без перекодирования, попадёт в кеш."""
    return hashlib.sha256(image_bytes).hexdigest()