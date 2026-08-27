from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.metrics import (
    load_jsonl_records,
    summarize_records_by_source,
    write_summary_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute metrics from JSONL result files.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory that contains per-model JSONL result files.",
    )
    return parser.parse_args()


def parse_result_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" not in stem:
        return stem, "unknown"
    model_name, mode = stem.split("__", 1)
    return model_name, mode


def main() -> int:
    args = parse_args()
    result_files = sorted(args.results_dir.glob("*.jsonl"))
    if not result_files:
        raise ValueError(f"No JSONL files found in {args.results_dir}")

    summaries = []
    for result_file in result_files:
        model_name, mode = parse_result_filename(result_file)
        records = load_jsonl_records(result_file)
        for summary in summarize_records_by_source(records):
            summary.update(
                {
                    "model_name": model_name,
                    "mode": mode,
                    "result_file": str(result_file),
                }
            )
            summaries.append(summary)

    write_summary_files(
        summaries=summaries,
        json_path=args.results_dir / "metrics_summary.json",
        csv_path=args.results_dir / "metrics_summary.csv",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
