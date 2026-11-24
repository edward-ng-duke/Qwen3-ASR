#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def compact(text: str) -> str:
    return "".join(character.casefold() for character in text if character.isalnum())


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, a in enumerate(left, 1):
        current = [index]
        for offset, b in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[offset] + 1,
                    previous[offset - 1] + (a != b),
                )
            )
        previous = current
    return previous[-1]


def cer(actual: str, expected: str) -> float:
    expected_clean = compact(expected)
    return edit_distance(compact(actual), expected_clean) / max(1, len(expected_clean))


def post(
    client: httpx.Client,
    base_url: str,
    sample: dict,
    *,
    hotwords: list[str] | None,
    prompt: str | None,
) -> dict:
    path = Path(sample["path"])
    data = {"language": sample.get("language", "auto")}
    if sample.get("model"):
        data["model"] = sample["model"]
    if prompt:
        data["prompt"] = prompt
    if hotwords is not None:
        data["hotwords"] = json.dumps(hotwords, ensure_ascii=False)
    started = time.perf_counter()
    with path.open("rb") as audio:
        response = client.post(
            base_url.rstrip("/") + "/v1/audio/transcriptions",
            data=data,
            files={"file": (path.name, audio, "application/octet-stream")},
        )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    payload = response.json()
    return {"text": str(payload.get("text") or ""), "elapsed_ms": elapsed * 1000}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--candidate", default="http://127.0.0.1:6001")
    parser.add_argument("--reference", default="http://127.0.0.1:6002")
    parser.add_argument("--asset-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    samples = manifest.get("samples", manifest)
    for sample in samples:
        if "path" not in sample:
            sample["path"] = str(args.asset_dir / sample["filename"])
    rows = []
    with httpx.Client(timeout=600, trust_env=False) as client:
        for sample in samples:
            reference_text = sample.get("reference", "")
            hotwords = sample.get("hotwords", [])
            unrelated = sample.get("unrelated_hotwords", ["无关术语", "irrelevant"])
            conditions = {
                "none": (None, None),
                "correct": (hotwords, sample.get("prompt")),
                "unrelated": (unrelated, None),
            }
            for condition, (words, prompt) in conditions.items():
                candidate = post(
                    client, args.candidate, sample, hotwords=words, prompt=prompt
                )
                reference = post(
                    client, args.reference, sample, hotwords=words, prompt=prompt
                )
                targets = [compact(word) for word in hotwords]
                row = {
                    "id": sample.get("id", Path(sample["path"]).stem),
                    "category": sample.get("category", "unspecified"),
                    "condition": condition,
                    "candidate": candidate,
                    "reference_service": reference,
                    "exact_match": compact(candidate["text"])
                    == compact(reference["text"]),
                    "candidate_cer": cer(candidate["text"], reference_text)
                    if reference_text
                    else None,
                    "reference_cer": cer(reference["text"], reference_text)
                    if reference_text
                    else None,
                    "candidate_hotword_hits": sum(
                        target in compact(candidate["text"]) for target in targets
                    ),
                    "reference_hotword_hits": sum(
                        target in compact(reference["text"]) for target in targets
                    ),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))

    exact = sum(row["exact_match"] for row in rows)
    report = {
        "candidate": args.candidate,
        "reference": args.reference,
        "samples": len(samples),
        "runs": len(rows),
        "exact_matches": exact,
        "exact_match_rate": exact / max(1, len(rows)),
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
