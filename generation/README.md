# Dataset construction

This package implements the controlled multimodal true/fake post construction pipeline used by the paper. Its functional roles are:

1. `SynthesisAgent`: grounding, faithful true-summary generation, counterfactual fake-text generation, and image-format planning.
2. `VisualProducerAgent`: FLUX-based image synthesis and poster rendering.
3. `CriticAgent`: independent text and image review with bounded revision loops.

The orchestrator records all attempts. `run_generation.py` writes accepted and failed manifests separately; an item enters `accepted_manifest.jsonl` only when both the text stage and image stage pass.

The default model identifiers reproduce the paper stack: DeepSeek-V4-pro for story generation and visual planning, FLUX.1-dev for image synthesis, and an independent GPT-4o critic. They may be overridden through process environment variables for later experiments.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r generation/requirements.txt
```

Provide credentials and model endpoints through process environment variables. Do not commit credentials. Runtime roles and quality thresholds are defined in the core module `generation/runtime.py`; it loads no configuration file and contains no credential values.

## Input

`--input-json` accepts a JSON list. Each object must provide article text under `original_article`, `content`, `article`, or `text`. Optional fields are `article_id` and `source_images`/`image_paths`/`image_path`.

```json
[
  {
    "article_id": "example-001",
    "original_article": "Source article text...",
    "source_images": ["/path/to/source-image.jpg"]
  }
]
```

The alternative `--dataset-dir` mode reads the article-directory layouts supported by `generation.utils.news_loader.NewsLoader`.

## Run

```bash
python -m generation.run_generation \
  --input-json /path/to/source_articles.json \
  --output-dir outputs/generated_posts \
  --limit 10
```

Every article receives a `final_report.json`; review traces are stored in `intermediate_results.json` unless `--no-intermediate` is used. Failed candidates remain auditable but are not listed as accepted data.
