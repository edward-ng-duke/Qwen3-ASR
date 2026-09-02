#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
MAX_ATTEMPTS = 4


def verify(path: Path, expected: str) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest() == expected


def fetch(sample: dict[str, str], target: Path) -> str:
    temporary = target.with_suffix(target.suffix + ".part")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        digest = hashlib.sha256()
        request = urllib.request.Request(
            sample["url"], headers={"User-Agent": "qwen3-asr-offline-builder/1.0"}
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                temporary.open("wb") as output,
            ):
                while block := response.read(1024 * 1024):
                    digest.update(block)
                    output.write(block)
            actual = digest.hexdigest()
            if actual != sample["sha256"]:
                raise RuntimeError(
                    f"checksum mismatch for {sample['filename']}: {actual}"
                )
            temporary.replace(target)
            return actual
        except (OSError, urllib.error.URLError):
            temporary.unlink(missing_ok=True)
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(2 ** (attempt - 1))

    raise AssertionError("unreachable")


def main() -> None:
    manifest = json.loads((ASSETS / "manifest.json").read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        target = ASSETS / sample["filename"]
        if verify(target, sample["sha256"]):
            actual = sample["sha256"]
        else:
            actual = fetch(sample, target)
        print(f"verified {sample['filename']} {actual}")


if __name__ == "__main__":
    main()
