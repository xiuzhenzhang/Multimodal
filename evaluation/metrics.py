from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


def _fake_probability(record: dict[str, Any]) -> float | None:
    """Derive P(fake) from a predicted label and its self-reported confidence."""
    parsed = record.get("parsed_output")
    confidence = parsed.get("confidence") if isinstance(parsed, dict) else record.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        return None
    if confidence > 1 and confidence <= 100:
        confidence /= 100
    if not 0 <= confidence <= 1 or record.get("prediction") not in (0, 1):
        return None
    return confidence if record["prediction"] == 1 else 1 - confidence


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    valid_records = [
        record
        for record in records
        if record.get("label") in (0, 1) and record.get("prediction") in (0, 1)
    ]

    summary: dict[str, Any] = {
        "total_samples": total,
        "valid_predictions": len(valid_records),
        "coverage": (len(valid_records) / total) if total else 0.0,
    }

    if not valid_records:
        summary.update(
            {
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "precision_true": 0.0,
                "recall_true": 0.0,
                "f1_true": 0.0,
                "precision_fake": 0.0,
                "recall_fake": 0.0,
                "f1_fake": 0.0,
                "auc": None,
                "auc_samples": 0,
                "auc_coverage": 0.0,
                "confusion_matrix": [[0, 0], [0, 0]],
            }
        )
        return summary

    y_true = [record["label"] for record in valid_records]
    y_pred = [record["prediction"] for record in valid_records]
    scored_records = [
        (record["label"], score)
        for record in valid_records
        if (score := _fake_probability(record)) is not None
    ]
    auc = None
    if scored_records and len({label for label, _ in scored_records}) == 2:
        auc = roc_auc_score(
            [label for label, _ in scored_records],
            [score for _, score in scored_records],
        )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_true, recall_true, f1_true, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=0,
        zero_division=0,
    )
    precision_fake, recall_fake, f1_fake, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    summary.update(
        {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_true": precision_true,
            "recall_true": recall_true,
            "f1_true": f1_true,
            "precision_fake": precision_fake,
            "recall_fake": recall_fake,
            "f1_fake": f1_fake,
            "auc": auc,
            "auc_samples": len(scored_records),
            "auc_coverage": len(scored_records) / total if total else 0.0,
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "true_news_support": sum(1 for label in y_true if label == 0),
            "fake_news_support": sum(1 for label in y_true if label == 1),
        }
    )
    return summary


def summarize_records_by_source(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return the aggregate metrics followed by one summary per explicit source."""
    aggregate = summarize_records(records)
    aggregate["source"] = "ALL"
    summaries = [aggregate]
    sources = sorted(
        {
            str(record["source"]).strip()
            for record in records
            if isinstance(record.get("source"), str) and str(record["source"]).strip()
        }
    )
    for source in sources:
        source_records = [record for record in records if str(record.get("source", "")).strip() == source]
        summary = summarize_records(source_records)
        summary["source"] = source
        summaries.append(summary)
    return summaries


def write_summary_files(
    summaries: list[dict[str, Any]],
    json_path: Path,
    csv_path: Path,
) -> None:
    json_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = [
        "model_name",
        "mode",
        "source",
        "total_samples",
        "valid_predictions",
        "coverage",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_true",
        "recall_true",
        "f1_true",
        "precision_fake",
        "recall_fake",
        "f1_fake",
        "auc",
        "auc_samples",
        "auc_coverage",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key) for key in fieldnames})
