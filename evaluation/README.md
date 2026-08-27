# Multimodal fake-news evaluation

The evaluation package supports direct and structured-reasoning prompts over local Hugging Face VLMs and OpenAI-compatible APIs. It writes one JSONL record per sample and reports coverage, accuracy, precision, recall, class-wise F1, macro F1, ROC-AUC, and confusion matrices for the full set and each explicit source.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r evaluation/requirements.txt
```

## Dataset format

The runner expects a JSON list with these fields:

```json
{
  "sample_id": "optional-stable-id",
  "source": "nature-or-nih-or-snopes",
  "post_text": "Headline and post body",
  "image_path": "/path/or/remappable/path/to/image.jpg",
  "label": 0
}
```

Labels are `0 = true news` and `1 = fake news`. Preserve `source` when possible; for legacy files it is inferred from Nature/NIH/Snopes image paths. Image paths can be remapped with `dataset_root` and `workspace_image_prefix` in the model configuration, including the legacy `/workspace/mutil-agent/...` layout.

ROC-AUC uses the model's self-reported confidence converted to `P(fake)`. Records without a valid confidence are excluded from AUC only, and `auc_coverage`/`auc_samples` are reported. AUC is never silently computed from hard class labels.

## Run local or API models

Create a local, untracked model configuration, then run a smoke test:

```bash
python -m evaluation.run_benchmark \
  --config /path/to/local-model-config.json \
  --modes direct reasoning \
  --limit 10
```

Use `--output-dir` and `--resume` for long runs. Recompute summaries with:

```bash
python -m evaluation.summarize_results --results-dir results/your_run
```

The model configuration is intentionally not included in the public repository because it is deployment-specific and may contain private endpoints. `ModelConfig` in `evaluation/clients.py` documents the accepted fields. G-Eval prompt text is provided under `prompts/geval.md`; API launch scripts and credential files are intentionally excluded.
