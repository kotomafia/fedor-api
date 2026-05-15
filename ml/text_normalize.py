import re

_HOMOGLYPH: dict[str, str] = {
    "a": "а",
    "e": "е",
    "o": "о",
    "p": "р",
    "c": "с",
    "x": "х",
    "y": "у",
    "k": "к",
    "m": "м",
    "t": "т",
    "A": "А",
    "E": "Е",
    "O": "О",
    "B": "В",
    "C": "С",
    "H": "Н",
    "K": "К",
    "M": "М",
    "P": "Р",
    "T": "Т",
    "X": "Х",
    "0": "о",
}

_LOG_TOKEN = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"|^\d{2}:\d{2}:\d{2}"
    r"|^[0-9a-f]{8}-[0-9a-f]{4}"
    r"|^(INFO|ERROR|WARNING|DEBUG|MainProcess|MainThread)$"
    r"|^tasks?\."
    r"|\d{10,}",
    re.IGNORECASE,
)

MAX_TOKENS_FOR_CLASSIFY = 64
_MIN_LOG_DUMP_TOKENS = 10

_CYR_VOWELS = frozenset("аеёиоуыэюяАЕЁИОУЫЭЮЯ")
_LAT_VOWELS = frozenset("aeiouAEIOU")
_VOWELS = _CYR_VOWELS | _LAT_VOWELS


def _has_cyrillic(text: str) -> bool:
    return any("\u0400" <= ch <= "\u04ff" for ch in text)


def _has_vowel(token: str) -> bool:
    return any(ch in _VOWELS for ch in token)


def replace_homoglyphs(text: str) -> str:
    return "".join(_HOMOGLYPH.get(ch, ch) for ch in text)


def _is_log_token(token: str) -> bool:
    return bool(_LOG_TOKEN.search(token))


def normalize_ocr_text(text: str) -> str:
    """Homoglyph (латиница → кириллица в смешанных токенах) + отсев лог-мусора OCR."""
    raw_tokens = text.split()
    if not raw_tokens:
        return ""

    apply_homoglyphs = _has_cyrillic(text)
    kept: list[str] = []
    for raw in raw_tokens:
        t = raw.strip(".,!?;:\"'()[]{}")
        if not t:
            continue
        # Короткий цифровой мусор без кириллицы (щ4.) — отбрасываем; «эт0» оставляем
        if len(t) <= 4 and any(c.isdigit() for c in t) and not _has_cyrillic(t):
            continue
        if len(t) == 1:
            continue
        if _is_log_token(t):
            continue
        # OCR-шум: токены длиной >= 3 с долей гласных < 20%
        # «НШШ» → 0/3 = 0%  «КРТШ» → 0%  «пизДеЦ» → 50%
        if len(t) >= 3:
            vowel_ratio = sum(1 for ch in t if ch in _VOWELS) / len(t)
            if vowel_ratio < 0.2:
                continue
        if apply_homoglyphs:
            t = replace_homoglyphs(t)
        elif _has_cyrillic(t):
            t = replace_homoglyphs(t)
        kept.append(t)

    if len(kept) > MAX_TOKENS_FOR_CLASSIFY:
        return ""

    if len(raw_tokens) >= _MIN_LOG_DUMP_TOKENS and len(kept) < 3:
        return ""

    return " ".join(kept)
