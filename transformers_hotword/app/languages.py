from __future__ import annotations


LANGUAGE_CODE_TO_NAME = {
    "ar": "Arabic",
    "yue": "Cantonese",
    "zh": "Chinese",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "fil": "Filipino",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
}


def resolve_language(value: str | None) -> str | None:
    if value is None or not value.strip() or value.strip().lower() == "auto":
        return None
    value = value.strip()
    resolved = LANGUAGE_CODE_TO_NAME.get(value.lower())
    if resolved:
        return resolved
    for name in LANGUAGE_CODE_TO_NAME.values():
        if value.lower() == name.lower():
            return name
    supported = ", ".join(sorted(LANGUAGE_CODE_TO_NAME))
    raise ValueError(f"unsupported language {value!r}; use auto or one of: {supported}")
