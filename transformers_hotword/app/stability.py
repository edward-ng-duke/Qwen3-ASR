from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher


_UNIT_RE = re.compile(
    r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\W\d_]+",
    re.UNICODE,
)


def _unit_spans(text: str) -> list[tuple[int, int]]:
    """Return semantic display units while ignoring punctuation/whitespace.

    CJK characters count individually; Latin letters and digits count as a word.
    The offsets still refer to the original string so punctuation stays attached to
    the changed suffix when a boundary is selected.
    """
    return [(match.start(), match.end()) for match in _UNIT_RE.finditer(text)]


def _unit_ends(text: str) -> list[int]:
    return [end for _start, end in _unit_spans(text)]


def _common_prefix_length(left: str, right: str) -> int:
    length = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        length += 1
    return length


def _common_prefix_units(left: str, right: str) -> int:
    left_units = [left[start:end] for start, end in _unit_spans(left)]
    right_units = [right[start:end] for start, end in _unit_spans(right)]
    count = 0
    for left_unit, right_unit in zip(left_units, right_units):
        if left_unit != right_unit:
            break
        count += 1
    return count


def _change_within_budget(previous: str, current: str, rollback_units: int) -> bool:
    """Whether a revision only replaces the previous visible tail budget."""
    if previous == current or current.startswith(previous):
        return True
    common_length = _common_prefix_length(previous, current)
    previous_tail = previous[common_length:]
    return len(_unit_spans(previous_tail)) <= rollback_units


def _compatible_revision(previous: str, current: str, rollback_units: int) -> bool:
    """Require two large-revision hypotheses to agree outside their tail budget."""
    if previous == current or previous.startswith(current) or current.startswith(previous):
        return True
    common_length = _common_prefix_length(previous, current)
    return (
        len(_unit_spans(previous[common_length:])) <= rollback_units
        and len(_unit_spans(current[common_length:])) <= rollback_units
    )


def stable_common_prefix(previous: str, current: str, rollback_units: int = 2) -> str:
    common_length = _common_prefix_length(previous, current)
    prefix = current[:common_length]
    ends = _unit_ends(prefix)
    if len(ends) <= rollback_units:
        return ""
    return prefix[: ends[-rollback_units - 1]].rstrip()


def merge_text(existing: str, addition: str, max_overlap_chars: int = 120) -> str:
    existing = existing.strip()
    addition = addition.strip()
    if not existing:
        return addition
    if not addition:
        return existing
    if addition.startswith(existing):
        return addition
    if existing.startswith(addition):
        return existing
    if addition in existing[-max_overlap_chars:]:
        return existing
    limit = min(len(existing), len(addition), max_overlap_chars)
    for size in range(limit, 0, -1):
        if existing[-size:] == addition[:size]:
            return existing + addition[size:]
    separator = (
        ""
        if (existing[-1].isascii() is False or addition[0].isascii() is False)
        else " "
    )
    return existing + separator + addition


def normalize_preview_after_confirmed(
    confirmed: str,
    preview: str,
    *,
    min_replay_units: int = 3,
    max_overlap_chars: int = 120,
) -> str:
    """Return only a display-safe tail after cumulative confirmed text.

    A new inference window may replay old audio after ``segment_final``. Exact
    suffix overlap can be removed. A revised replay of the beginning is
    ambiguous, so it is held until a later partial or final resolves it.
    """
    confirmed = confirmed.strip()
    preview = preview.strip()
    if not confirmed or not preview:
        return preview
    if preview.startswith(confirmed):
        return preview[len(confirmed) :].lstrip()

    limit = min(len(confirmed), len(preview), max_overlap_chars)
    for size in range(limit, 0, -1):
        overlap = preview[:size]
        if confirmed.endswith(overlap) and len(_unit_spans(overlap)) >= 2:
            return preview[size:].lstrip()

    if _common_prefix_units(confirmed, preview) >= min_replay_units:
        return ""
    return preview


def _merge_from_exact_suffix_anchor(
    confirmed: str,
    current: str,
    *,
    min_units: int = 2,
    max_units: int = 32,
) -> str | None:
    spans = _unit_spans(confirmed)
    for count in range(min(len(spans), max_units), min_units - 1, -1):
        suffix = confirmed[spans[-count][0] :].strip()
        if not suffix:
            continue
        anchor = current.rfind(suffix)
        if anchor < 0:
            continue
        tail = current[anchor + len(suffix) :].strip()
        return merge_text(confirmed, tail) if tail else confirmed
    return None


def _merge_from_compatible_alignment(
    confirmed: str,
    current: str,
    last_compatible: str,
    *,
    rollback_units: int = 2,
) -> str | None:
    source_spans = _unit_spans(last_compatible)
    target_spans = _unit_spans(current)
    if not source_spans or not target_spans:
        return None
    source_units = [last_compatible[start:end] for start, end in source_spans]
    target_units = [current[start:end] for start, end in target_spans]
    matcher = SequenceMatcher(None, source_units, target_units, autojunk=False)
    candidates = [
        block
        for block in matcher.get_matching_blocks()
        if block.size >= 3
        and block.a + block.size >= len(source_units) - rollback_units
    ]
    if not candidates:
        return None
    anchor = max(
        candidates,
        key=lambda block: (block.a + block.size, block.b + block.size, block.size),
    )
    current_end = target_spans[anchor.b + anchor.size - 1][1]
    tail = current[current_end:].strip()
    return merge_text(confirmed, tail) if tail else confirmed


def merge_revised_cumulative(
    confirmed: str,
    current: str,
    last_compatible: str | None = None,
) -> str:
    """Keep confirmed monotonic and append only a safely aligned new suffix.

    A revised cumulative hypothesis must never be appended wholesale: doing so
    duplicates the already-confirmed transcript. If no trustworthy terminal
    anchor can be found, retaining confirmed text is safer than inventing a
    continuation.
    """
    confirmed = confirmed.strip()
    current = current.strip()
    if not confirmed:
        return current
    if not current or confirmed.startswith(current):
        return confirmed
    if current.startswith(confirmed):
        return current
    exact = _merge_from_exact_suffix_anchor(confirmed, current)
    if exact is not None:
        return exact
    if last_compatible:
        aligned = _merge_from_compatible_alignment(
            confirmed,
            current,
            last_compatible,
        )
        if aligned is not None:
            return aligned
    return confirmed


def merge_final_segment(existing: str, current: str) -> str:
    """Merge an independent final while rejecting cumulative-prefix replay.

    VAD normally submits non-overlapping audio, so a genuinely new final must be
    appended. Very short residuals can nevertheless make the model replay the
    beginning of the cumulative transcript. A shared leading anchor of three
    display units identifies that replay shape; replay candidates use the
    conservative cumulative merger and are never appended wholesale when no
    trustworthy alignment exists.
    """
    existing = existing.strip()
    current = current.strip()
    if not existing:
        return current
    if not current:
        return existing
    if current.startswith(existing) or _common_prefix_units(existing, current) >= 3:
        return merge_revised_cumulative(existing, current)
    return merge_text(existing, current)


@dataclass(slots=True)
class StabilizedText:
    confirmed: str
    preview: str
    text: str


class PrefixStabilizer:
    def __init__(self, rollback_units: int = 2) -> None:
        self.rollback_units = rollback_units
        self.previous = ""
        self.confirmed = ""
        self.displayed_text = ""
        self.pending_revision = ""
        self.pending_count = 0
        self.last_compatible = ""

    def _clear_pending(self) -> None:
        self.pending_revision = ""
        self.pending_count = 0

    def _display(self, text: str) -> StabilizedText:
        self.displayed_text = text
        preview = (
            text[len(self.confirmed) :].lstrip()
            if text.startswith(self.confirmed)
            else ""
        )
        return StabilizedText(self.confirmed, preview, text)

    def update(self, current: str) -> StabilizedText:
        current = current.strip()
        candidate = stable_common_prefix(self.previous, current, self.rollback_units)
        if candidate.startswith(self.confirmed):
            self.confirmed = candidate
        self.previous = current

        # Rebase the last visible hypothesis onto newly grown confirmed text.
        # A two-frame consensus may legitimately replace an old preview, but it
        # must never retract the confirmed prefix.
        if not self.displayed_text or self.confirmed.startswith(self.displayed_text):
            self.displayed_text = self.confirmed
        elif not self.displayed_text.startswith(self.confirmed):
            self.displayed_text = self.confirmed

        if self.confirmed and not current.startswith(self.confirmed):
            # The model rewrote inside confirmed. Preserve the last display-ready
            # tail instead of emitting preview="" and making the UI collapse.
            self._clear_pending()
            return self._display(self.displayed_text or self.confirmed)

        self.last_compatible = current
        if _change_within_budget(
            self.displayed_text,
            current,
            self.rollback_units,
        ):
            self._clear_pending()
            return self._display(current)

        if self.pending_revision and _compatible_revision(
            self.pending_revision,
            current,
            self.rollback_units,
        ):
            self.pending_count += 1
        else:
            self.pending_revision = current
            self.pending_count = 1

        if self.pending_count >= 2:
            self._clear_pending()
            return self._display(current)
        return self._display(self.displayed_text or self.confirmed)

    def finalize(self, current: str) -> str:
        final = merge_revised_cumulative(
            self.confirmed,
            current,
            self.last_compatible,
        )
        self.previous = final
        self.confirmed = final
        self.displayed_text = final
        self.last_compatible = final
        self._clear_pending()
        return final
