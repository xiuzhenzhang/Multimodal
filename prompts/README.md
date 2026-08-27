# Prompt inventory

- `generation.md`: grounding, true-summary, counterfactual fake-text, critic revision, image planning, and image review prompts.
- `detection.md`: direct and structured-reasoning fake-news detection prompts, including the four diagnostic checks.
- `geval.md`: GPT-4o G-Eval prompts for Coherence, Fluency, and Faithfulness.

The executable sources of truth for generation and detection are `generation/utils/prompt_templates.py` and `evaluation/prompts.py`. The Markdown files are provided for inspection and paper appendix use; deployment-specific API launch code is intentionally excluded.
