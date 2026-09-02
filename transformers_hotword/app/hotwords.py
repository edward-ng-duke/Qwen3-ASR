from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any


class HotwordValidationError(ValueError):
    pass


_HOTWORD_SEPARATOR = re.compile(r"[,，;；\n]+")


@dataclass(frozen=True, slots=True)
class ContextGuard:
    prompt: str | None
    hotwords: tuple[str, ...]

    @property
    def active(self) -> bool:
        return bool(self.prompt or self.hotwords)


def parse_hotwords(value: str | list[str] | None) -> list[str]:
    """Parse the shared HTTP/WS hotword contract.

    JSON string arrays are the preferred wire representation.  Plain strings
    remain supported for compatibility with the original vLLM gateway.
    """
    if value is None or value == "" or value == []:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("["):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise HotwordValidationError(
                    "hotwords beginning with '[' must be a valid JSON string array"
                ) from exc
        else:
            if raw.casefold().startswith("vocabulary:"):
                raw = raw.split(":", 1)[1]
            value = _HOTWORD_SEPARATOR.split(raw)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise HotwordValidationError("hotwords must be a string or string array")
    return normalize_hotwords(value)


def normalize_hotwords(hotwords: list[str]) -> list[str]:
    if len(hotwords) > 100:
        raise HotwordValidationError("at most 100 hotwords are allowed")
    result: list[str] = []
    seen: set[str] = set()
    for raw in hotwords:
        item = raw.strip()
        if not item:
            continue
        if len(item) > 64:
            raise HotwordValidationError(
                "each hotword must be at most 64 Unicode characters"
            )
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def merge_prompt(prompt: str | None, hotwords: list[str] | None) -> str | None:
    base = (prompt or "").strip()
    words = normalize_hotwords(hotwords or [])
    vocabulary = f"Vocabulary: {', '.join(words)}" if words else ""
    merged = "\n".join(part for part in (base, vocabulary) if part)
    if len(merged) > 4096:
        raise HotwordValidationError(
            "combined prompt and hotwords must be at most 4096 characters"
        )
    return merged or None


def make_context(
    prompt: str | None, hotwords: str | list[str] | None
) -> tuple[str | None, ContextGuard]:
    cleaned_prompt = (prompt or "").strip() or None
    cleaned_hotwords = parse_hotwords(hotwords)
    return (
        merge_prompt(cleaned_prompt, cleaned_hotwords),
        ContextGuard(cleaned_prompt, tuple(cleaned_hotwords)),
    )


def compact_context_text(value: str | None) -> str:
    return "".join(
        character.casefold() for character in (value or "") if character.isalnum()
    )


def looks_like_context_prefix(text: str, context: ContextGuard) -> bool:
    """Suppress incremental hypotheses that are only a context echo prefix."""
    output = compact_context_text(text)
    if not output or not context.active:
        return False
    prompt = compact_context_text(context.prompt)
    if prompt and prompt.startswith(output):
        return True
    vocabulary = "".join(compact_context_text(word) for word in context.hotwords)
    return bool(vocabulary and vocabulary.startswith(output))


def is_context_leak(text: str, context: ContextGuard) -> bool:
    """Detect verbatim prompt echoes and multi-hotword vocabulary echoes."""
    output = compact_context_text(text)
    if not output or not context.active:
        return False

    prompt = compact_context_text(context.prompt)
    if prompt and (output == prompt or (len(prompt) >= 8 and prompt in output)):
        return True

    words = [compact_context_text(word) for word in context.hotwords]
    words = [word for word in words if word]
    if len(words) < 2:
        return False
    matched = sum(word in output for word in words)
    # Half-list detection is useful for large vocabularies, but a two-of-four
    # match is common in legitimate speech. Ordered vocabulary-only echoes are
    # handled below, so require at least three unordered matches here.
    threshold = max(3, math.ceil(len(words) / 2))
    if matched >= threshold:
        return True

    vocabulary = "".join(words)
    return matched >= 2 and vocabulary.startswith(output)


def redact_request_fields(data: dict[str, Any]) -> dict[str, Any]:
    """Return metadata safe for logs; intentionally omit user/model text."""
    blocked = {
        "prompt",
        "hotwords",
        "text",
        "raw_text",
        "confirmed",
        "preview",
        "delta",
    }
    return {key: value for key, value in data.items() if key not in blocked}
