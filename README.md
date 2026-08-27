# Can Multimodal Large Language Models Generate and Detect Multimodal Social Media Fake News?

Official code release for multimodal news-post construction and multimodal fake-news detection.

## Repository layout

```text
generation/   Data-construction agents, critic loop, FLUX image branch, and batch entry point
evaluation/   Direct/CoT benchmark runner, local/API model adapters, and metrics
prompts/      Human-readable generation, detection, and G-Eval prompt appendices
```

Start with [generation/README.md](generation/README.md) or [evaluation/README.md](evaluation/README.md). Prompt text is collected in [prompts/](prompts/README.md).

## Paper-to-code map

| Paper component | Public implementation |
| --- | --- |
| Story Generator (DeepSeek-V4-pro) | `generation/agents/transformer.py` (`SynthesisAgent`) |
| Format-aware Image Generator (DeepSeek-V4-pro + FLUX.1-dev) | `generation/agents/visual_strategy_selector.py` and `visual_producer_sd_local.py` |
| Independent GPT-4o Critic | `generation/agents/sentinel.py` (`CriticAgent`) |
| Text/image feedback loops and pass-only retention | `generation/pipeline/poster_pipeline.py` and `run_generation.py` |
| Direct and structured-CoT detection protocols | `evaluation/prompts.py` and `run_benchmark.py` |
| Per-source F1, accuracy, AUC, coverage, and confusion metrics | `evaluation/metrics.py` |
| Paper appendix prompts | `prompts/generation.md`, `detection.md`, and `geval.md` |

The public scope intentionally excludes source-site download/curation code, raw data, deployment configurations, API launch scripts, credentials, logs, and result files. G-Eval is released as its paper prompt, not as a provider-specific launch script.

## Data

The dataset is distributed separately because it contains images and is too large for a regular Git repository:

[Download the dataset from Google Drive](https://drive.google.com/file/d/1NK2Ury8cciCEhOUQrW2rxDUlEzSuiIqp/view?usp=drive_link)

Users are responsible for complying with the license and reuse terms of every source article and image. No dataset, model weights, generated media, credentials, or experiment logs are tracked in this code repository.

## Responsible use

The generation pipeline is released for misinformation research and detector evaluation. Do not publish generated claims as real news, impersonate real outlets or people, or use the code for targeted deception.

## Reproducibility

- Use stable `sample_id` and `source` fields in evaluation data.
- Record model IDs, revisions, decoding parameters, and prompt mode for every run.
- Treat invalid model outputs as missing predictions and report coverage alongside conditional metrics.
- Keep failed generation candidates separate from accepted benchmark items.
