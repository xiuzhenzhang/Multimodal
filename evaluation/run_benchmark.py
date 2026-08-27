from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.clients import (
    ModelConfig,
    build_client,
    extract_cot_text,
    extract_json_payload,
    extract_prediction,
    extract_think_blocks,
)
from evaluation.dataset import WORKSPACE_IMAGE_PREFIX, load_samples
from evaluation.metrics import (
    load_jsonl_records,
    summarize_records_by_source,
    write_summary_files,
)
from evaluation.prompts import build_direct_prompt, build_reasoning_prompt


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "fake_news_detection_dataset.json"
DEFAULT_DATASET_ROOT = Path(__file__).resolve().parents[1] / "fake_news_dataset" / "dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multimodal fake-news inference benchmarks.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the model configuration JSON.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional subset of model names to run.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["direct", "reasoning"],
        choices=["direct", "reasoning"],
        help="Inference modes to run.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    parser.add_argument("--offset", type=int, default=0, help="Optional dataset offset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where JSONL results and summary files will be written.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip samples that already exist in the output JSONL.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(base_dir: Path, value: str | None, fallback: Path) -> Path:
    if not value:
        return fallback
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def build_output_dir(requested: Path | None) -> Path:
    if requested:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(__file__).resolve().parent / "results" / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_processed_ids(result_file: Path) -> set[str]:
    processed: set[str] = set()
    for record in load_jsonl_records(result_file):
        sample_id = record.get("sample_id")
        if sample_id:
            processed.add(str(sample_id))
    return processed


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def compact_result_path(result_file: Path) -> Path:
    return result_file.with_name(f"{result_file.stem}__compact.json")


def build_compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "dataset_index": record.get("dataset_index"),
        "label": record.get("label"),
        "mode": record.get("mode"),
        "model_name": record.get("model_name"),
        "source": record.get("source"),
        "image_path": record.get("image_path"),
        "response": record.get("raw_response"),
        "error": record.get("error"),
    }


def write_compact_results(result_file: Path) -> None:
    records = load_jsonl_records(result_file)
    compact_records = [build_compact_record(record) for record in records]
    compact_result_path(result_file).write_text(
        json.dumps(compact_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def prompt_for_mode(mode: str, sample: Any) -> str:
    if mode == "direct":
        return build_direct_prompt(sample)
    if mode == "reasoning":
        return build_reasoning_prompt(sample)
    raise ValueError(f"Unsupported mode: {mode}")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    config_dir = args.config.resolve().parent

    dataset_config = config.get("dataset", {})
    dataset_path = resolve_path(
        config_dir,
        dataset_config.get("path"),
        DEFAULT_DATASET,
    )
    dataset_root = resolve_path(
        config_dir,
        dataset_config.get("dataset_root"),
        DEFAULT_DATASET_ROOT,
    )
    workspace_prefix = dataset_config.get("workspace_image_prefix", WORKSPACE_IMAGE_PREFIX)

    samples = load_samples(
        dataset_path=dataset_path,
        dataset_root=dataset_root,
        limit=args.limit,
        offset=args.offset,
        workspace_prefix=workspace_prefix,
    )
    if not samples:
        raise ValueError("No samples loaded. Check dataset path and offset / limit arguments.")

    selected_models = set(args.models or [])
    model_configs = [
        ModelConfig.from_dict(item, base_dir=config_dir)
        for item in config.get("models", [])
        if not selected_models or item["name"] in selected_models
    ]
    if not model_configs:
        raise ValueError("No models selected. Check --models or the config file.")

    output_dir = build_output_dir(args.output_dir)
    summaries: list[dict[str, Any]] = []

    print(f"Loaded {len(samples)} samples from {dataset_path}")
    print(f"Image root: {dataset_root}")
    print(f"Output directory: {output_dir}")

    for model_config in model_configs:
        print(f"\n=== Model: {model_config.name} ({model_config.backend}) ===")
        try:
            client = build_client(model_config)
        except Exception as error:  # noqa: BLE001
            print(f"Skip model {model_config.name}: {type(error).__name__}: {error}")
            continue

        for mode in args.modes:
            result_file = output_dir / f"{model_config.name}__{mode}.jsonl"
            processed_ids = read_processed_ids(result_file) if args.resume else set()

            print(
                f"Running mode={mode} | result_file={result_file.name} | "
                f"resume_skips={len(processed_ids)}"
            )
            for index, sample in enumerate(samples, start=1):
                if sample.sample_id in processed_ids:
                    continue

                prompt_text = prompt_for_mode(mode, sample)
                started_at = time.time()
                raw_response = ""
                think_blocks: list[str] = []
                cot_text = None
                parsed_output = None
                prediction = None
                error_message = None

                try:
                    raw_response = client.generate(
                        prompt_text=prompt_text,
                        image_path=sample.image_path,
                        preserve_think_blocks=(mode == "reasoning"),
                    )
                    if mode == "reasoning":
                        think_blocks = extract_think_blocks(raw_response)
                        cot_text = extract_cot_text(raw_response)
                    parsed_output = extract_json_payload(raw_response)
                    prediction = extract_prediction(parsed_output, raw_response)
                except Exception as error:  # noqa: BLE001
                    error_message = f"{type(error).__name__}: {error}"

                elapsed = time.time() - started_at
                record = {
                    "sample_id": sample.sample_id,
                    "dataset_index": sample.dataset_index,
                    "label": sample.label,
                    "prediction": prediction,
                    "mode": mode,
                    "model_name": model_config.name,
                    "model_id": model_config.model_id,
                    "backend": model_config.backend,
                    "image_path": str(sample.image_path),
                    "headline": sample.headline,
                    "body": sample.body,
                    "full_text": sample.full_text,
                    "source": sample.source,
                    "latency_sec": round(elapsed, 4),
                    "raw_response": raw_response,
                    "cot_text": cot_text,
                    "think_blocks": think_blocks,
                    "parsed_output": parsed_output,
                    "error": error_message,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                }
                append_jsonl(result_file, record)

                status = "ok" if prediction in (0, 1) else "unparsed"
                if error_message:
                    status = "error"
                print(
                    f"[{model_config.name}][{mode}] "
                    f"{index}/{len(samples)} sample={sample.sample_id} status={status} "
                    f"pred={prediction} time={elapsed:.2f}s"
                )

            records = load_jsonl_records(result_file)
            write_compact_results(result_file)
            for summary in summarize_records_by_source(records):
                summary.update(
                    {
                        "model_name": model_config.name,
                        "mode": mode,
                        "result_file": str(result_file),
                        "compact_result_file": str(compact_result_path(result_file)),
                    }
                )
                summaries.append(summary)

    write_summary_files(
        summaries=summaries,
        json_path=output_dir / "metrics_summary.json",
        csv_path=output_dir / "metrics_summary.csv",
    )

    print("\n=== Finished ===")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
