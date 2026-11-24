#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from bson import ObjectId
from gridfs import GridFS
from pymongo import MongoClient

from compare_services import cer, compact

HOTWORDS = [
    "商飞",
    "英伟达",
    "DeepSeek",
    "智谱",
    "声纹识别",
    "日产日清",
    "区块链",
    "量子纠缠",
    "李国强",
    "王建军",
    "GLEIF",
    "Kubernetes",
]


@dataclass(slots=True)
class Sample:
    id: str
    category: str
    language: str
    duration_ms: int
    reference: str
    filename: str
    audio: bytes


GROUPS = [
    ("cantonese", {"language": "粤语", "duration_ms": {"$gt": 0}}, 13, "yue"),
    (
        "zh_very_short",
        {"language": "中文", "duration_ms": {"$gt": 0, "$lt": 2000}},
        40,
        "zh",
    ),
    (
        "zh_short",
        {"language": "中文", "duration_ms": {"$gte": 2000, "$lt": 3000}},
        30,
        "zh",
    ),
    (
        "en_short",
        {"language": "English", "duration_ms": {"$gt": 0, "$lt": 3000}},
        25,
        "en",
    ),
    (
        "zh_medium",
        {"language": "中文", "duration_ms": {"$gte": 3000, "$lte": 8000}},
        20,
        "zh",
    ),
    (
        "zh_supplement",
        {"language": "中文", "duration_ms": {"$gt": 8000, "$lte": 12000}},
        22,
        "zh",
    ),
]


def load_samples(uri: str, database: str) -> list[Sample]:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    db = client[database]
    segments = db["segments"]
    fs = GridFS(db, collection="audio_segments")
    seen: set[ObjectId] = set()
    result: list[Sample] = []
    for category, query, target, language in GROUPS:
        selected = 0
        criteria = {**query, "audio_file_id": {"$exists": True, "$ne": ""}}
        for document in segments.find(criteria).sort("_id", -1).limit(target * 8):
            if document["_id"] in seen:
                continue
            try:
                stored = fs.get(ObjectId(str(document["audio_file_id"])))
                audio = stored.read()
            except Exception:
                continue
            if not audio:
                continue
            seen.add(document["_id"])
            result.append(
                Sample(
                    id=str(document["_id"]),
                    category=category,
                    language=language,
                    duration_ms=int(document.get("duration_ms") or 0),
                    reference=str(document.get("raw_text") or document.get("text") or ""),
                    filename=stored.filename or f"{document['_id']}.wav",
                    audio=audio,
                )
            )
            selected += 1
            if selected == target:
                break
        if selected != target:
            raise RuntimeError(f"category {category} selected {selected}/{target}")
    client.close()
    return result


def request(base_url: str, sample: Sample, contextual: bool) -> dict:
    data = {"language": sample.language}
    if contextual:
        data["hotwords"] = json.dumps(HOTWORDS, ensure_ascii=False)
    with httpx.Client(timeout=600, trust_env=False) as client:
        response = client.post(
            base_url.rstrip("/") + "/v1/audio/transcriptions",
            data=data,
            files={"file": (sample.filename, sample.audio, "application/octet-stream")},
        )
    text = response.json().get("text", "") if response.status_code == 200 else ""
    return {"status": response.status_code, "text": str(text)}


def leaks(text: str) -> bool:
    output = compact(text)
    matches = sum(compact(word) in output for word in HOTWORDS)
    return matches >= max(3, (len(HOTWORDS) + 1) // 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mongo-uri", default="mongodb://10.0.0.93:27017")
    parser.add_argument("--database", default="conference_ws")
    parser.add_argument("--candidate", default="http://127.0.0.1:6001")
    parser.add_argument("--reference", default="http://127.0.0.1:6002")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples = load_samples(args.mongo_uri, args.database)
    jobs = [
        (service, base, contextual, sample)
        for service, base in (
            ("candidate", args.candidate),
            ("reference", args.reference),
        )
        for contextual in (False, True)
        for sample in samples
    ]

    responses: dict[tuple[str, bool, str], dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(request, base, sample, contextual): (
                service,
                contextual,
                sample.id,
            )
            for service, base, contextual, sample in jobs
        }
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            key = futures[future]
            responses[key] = future.result()
            if index % 50 == 0:
                print(f"completed {index}/{len(futures)}", flush=True)

    rows = []
    for sample in samples:
        candidate_plain = responses[("candidate", False, sample.id)]
        candidate_context = responses[("candidate", True, sample.id)]
        reference_plain = responses[("reference", False, sample.id)]
        reference_context = responses[("reference", True, sample.id)]
        rows.append(
            {
                "id": sample.id,
                "category": sample.category,
                "duration_ms": sample.duration_ms,
                "candidate_status": [candidate_plain["status"], candidate_context["status"]],
                "reference_status": [reference_plain["status"], reference_context["status"]],
                "plain_exact": compact(candidate_plain["text"])
                == compact(reference_plain["text"]),
                "context_exact": compact(candidate_context["text"])
                == compact(reference_context["text"]),
                "candidate_false_kill": bool(candidate_plain["text"])
                and not candidate_context["text"],
                "reference_false_kill": bool(reference_plain["text"])
                and not reference_context["text"],
                "candidate_leak": leaks(candidate_context["text"]),
                "reference_leak": leaks(reference_context["text"]),
                "candidate_cer": cer(candidate_context["text"], sample.reference)
                if sample.reference
                else None,
                "reference_cer": cer(reference_context["text"], sample.reference)
                if sample.reference
                else None,
            }
        )

    summary = {
        "sample_count": len(rows),
        "category_counts": {
            category: sum(row["category"] == category for row in rows)
            for category, *_rest in GROUPS
        },
        "all_requests_ok": all(
            status == 200
            for row in rows
            for status in row["candidate_status"] + row["reference_status"]
        ),
        "plain_exact": sum(row["plain_exact"] for row in rows),
        "context_exact": sum(row["context_exact"] for row in rows),
        "candidate_false_kills": sum(row["candidate_false_kill"] for row in rows),
        "reference_false_kills": sum(row["reference_false_kill"] for row in rows),
        "candidate_leaks": sum(row["candidate_leak"] for row in rows),
        "reference_leaks": sum(row["reference_leak"] for row in rows),
        "candidate_mean_cer": sum(row["candidate_cer"] or 0 for row in rows)
        / len(rows),
        "reference_mean_cer": sum(row["reference_cer"] or 0 for row in rows)
        / len(rows),
    }
    report = {"summary": summary, "rows": rows}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if not summary["all_requests_ok"] or summary["candidate_leaks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

