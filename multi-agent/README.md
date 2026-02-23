# Multi-Agent Fake News Generation System

A multi-agent system for generating mirrored fake news articles and visual posts using AI agents.

## Features

- **Transformer Agent**: Extracts facts, reverses claims, and creates mirrored articles
- **Sentinel Agent**: Quality control and content validation
- **Visual Producer Agent**: Generates visual content with strategic design approaches
- **Visual Strategy Selector**: Selects optimal visual representation strategies

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and configure your API keys:
```bash
cp .env.example .env
```

4. Edit `.env` with your API keys:
   - `DEEPSEEK_API_KEY`: Your DeepSeek API key (required)
   - `OPENAI_API_KEY`: Your OpenAI API key (optional, for image generation)

## Configuration

Key configuration options in `.env`:

- `IMAGE_GEN_PROVIDER`: Choose between `openai` or `sd_local` for image generation
- `POST_METHOD`: Choose generation method (1: modified facts, 2: modified evidence)
- `DATASET_PATH`: Path to your news dataset
- `OUTPUT_DIR`: Directory for generated outputs


## Project Structure

```
├── agents/              # AI agent implementations
├── config/              # Configuration management
├── pipeline/            # Processing pipeline
├── scripts/             # Batch processing scripts
├── utils/               # Utility functions
└── output/              # Generated outputs
```


