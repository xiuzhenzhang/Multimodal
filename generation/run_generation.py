"""Batch entry point for the multimodal fake-news construction pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from generation.pipeline.poster_pipeline import PosterPipeline
from generation.utils.news_loader import NewsLoader


def read_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("records", "data", "articles", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Input JSON must be a list or contain records/data/articles/items.")


def normalize_record(record: dict[str, Any], index: int) -> dict[str, Any]:
    article = next(
        (
            str(record[key]).strip()
            for key in ("original_article", "content", "article", "text")
            if record.get(key)
        ),
        "",
    )
    article_id = str(
        record.get("article_id") or record.get("id") or record.get("sample_id") or f"article_{index:06d}"
    )
    images = record.get("source_images") or record.get("image_paths") or record.get("images") or []
    if not images and record.get("image_path"):
        images = [record["image_path"]]
    if isinstance(images, str):
        images = [images]
    return {
        "article_id": article_id,
        "original_article": article,
        "source_images": [str(value) for value in images if value],
    }


def records_from_dataset_dir(dataset_dir: Path) -> list[dict[str, Any]]:
    loader = NewsLoader(dataset_path=str(dataset_dir))
    records: list[dict[str, Any]] = []
    for index, article_dir in enumerate(loader.get_all_articles()):
        parsed = loader.parse_article(article_dir)
        if not parsed:
            continue
        records.append(
            {
                "article_id": parsed.get("article_id") or Path(article_dir).name,
                "original_article": parsed.get("content", ""),
                "source_images": parsed.get("image_paths") or [],
            }
        )
    return records


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_records(records: list[dict[str, Any]], offset: int, limit: int | None) -> Iterable[tuple[int, dict[str, Any]]]:
    stop = None if limit is None else offset + limit
    for index, record in enumerate(records[offset:stop], start=offset):
        yield index, normalize_record(record, index)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct paired multimodal true/fake social-media posts.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-json", type=Path)
    source.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--post-method", type=int, default=1)
    parser.add_argument("--text-max-attempts", type=int, default=None)
    parser.add_argument("--image-max-attempts", type=int, default=None)
    parser.add_argument("--no-intermediate", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = (
        read_json_records(args.input_json)
        if args.input_json
        else records_from_dataset_dir(args.dataset_dir)
    )
    pipeline = PosterPipeline(
        output_dir=str(args.output_dir),
        text_max_attempts=args.text_max_attempts,
        image_max_attempts=args.image_max_attempts,
    )
    accepted_path = args.output_dir / "accepted_manifest.jsonl"
    failed_path = args.output_dir / "failed_manifest.jsonl"
    summary = {"selected": 0, "accepted": 0, "failed": 0, "errors": 0}

    for index, record in select_records(records, args.offset, args.limit):
        summary["selected"] += 1
        article_id = record["article_id"]
        if not record["original_article"]:
            summary["errors"] += 1
            append_jsonl(failed_path, {"article_id": article_id, "error": "empty source article"})
            continue
        try:
            result = pipeline.process(
                record["original_article"],
                save_intermediate=not args.no_intermediate,
                post_method=args.post_method,
                article_id=article_id,
                source_images=record["source_images"],
            )
            accepted = result.get("status") == "success" and result.get("text_stage_passed") is True
            row = {
                "article_id": article_id,
                "accepted": accepted,
                "status": result.get("status"),
                "text_stage_passed": result.get("text_stage_passed"),
                "output_dir": result.get("output_dir"),
            }
            append_jsonl(accepted_path if accepted else failed_path, row)
            summary["accepted" if accepted else "failed"] += 1
        except Exception as exc:
            summary["errors"] += 1
            append_jsonl(failed_path, {"article_id": article_id, "error": repr(exc)})

    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
