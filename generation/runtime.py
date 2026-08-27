"""Core runtime options.

Values are read only from the process environment.  This module never loads a
configuration or credential file.
"""
import os
from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel


@dataclass(frozen=True)
class ScoreConfig:
    """Centralized scoring configuration for each review type."""

    weights: dict[str, int]
    threshold: float
    minimum_dimensions: dict[str, float]
    veto_rules: tuple[str, ...]


@dataclass(frozen=True)
class ModelEndpointConfig:
    """Provider/model endpoint used by a runtime agent."""

    provider: str
    model_id: str
    api_key: str
    base_url: Optional[str]


@dataclass(frozen=True)
class ImageFormatConfig:
    """Configurable image format used by the fake branch image stage."""

    id: str
    label: str
    description: str
    suitable_use_cases: tuple[str, ...] = ()


TRUE_SUMMARY_SCORE_CONFIG = ScoreConfig(
    weights={
        "faithfulness": 50,
        "key_info_coverage": 20,
        "social_fit": 15,
        "fluency": 10,
        "brevity": 5,
    },
    # Relaxed defaults: only clearly unfaithful summaries should fail.
    threshold=72.0,
    minimum_dimensions={"faithfulness": 3.0},
    veto_rules=(
        "Added a core fact not present in the source article.",
        "Misstated a number, time, or location from the source article.",
        "Clearly contradicts the source article.",
    ),
)

FAKE_TEXT_SCORE_CONFIG = ScoreConfig(
    weights={
        "frame_flip_success": 20,
        "fact_support_for_new_frame": 25,
        "internal_consistency": 20,
        "plausibility": 15,
        "anchor_retention": 10,
        "social_fit": 10,
    },
    # Relaxed defaults: preserve core logic checks, but allow borderline drafts to pass.
    threshold=68.0,
    minimum_dimensions={
        "frame_flip_success": 2.5,
        "fact_support_for_new_frame": 2.5,
        "internal_consistency": 2.5,
    },
    veto_rules=(
        "Only changed emotional wording without changing supporting facts.",
        "Changed facts, but they still do not support the new frame.",
        "Contains self-contradictory numbers, timing, or subject identity.",
        "Drifted entirely away from the original news topic.",
    ),
)

IMAGE_SCORE_CONFIG = ScoreConfig(
    weights={
        "format_fit": 15,
        "visual_quality": 20,
        "readability": 15,
        "consistency_with_text_narrative": 25,
        "support_for_key_text_facts": 15,
        "artifact_control": 10,
    },
    # Relaxed defaults: unless the image is clearly broken or contradictory, prefer passing.
    threshold=72.0,
    minimum_dimensions={
        "consistency_with_text_narrative": 3.0,
        "support_for_key_text_facts": 3.0,
    },
    veto_rules=(
        "The text implies an official notice, but the image is an obviously mismatched meme or stylized promo.",
        "A key number shown in the image conflicts with the text.",
        "The main subject in the image conflicts with the main subject in the text.",
        "Text in the image is unreadable.",
        "The image has severe artifacts or is obviously broken.",
    ),
)


DEFAULT_IMAGE_FORMAT_CATALOG: tuple[ImageFormatConfig, ...] = (
    ImageFormatConfig(
        id="documentary_photo",
        label="Documentary Photo",
        description="A realistic scene or press-style photograph that depicts the fake narrative as if it were a genuine news image.",
        suitable_use_cases=("events", "people", "street scenes", "press moments"),
    ),
    ImageFormatConfig(
        id="social_media_screenshot",
        label="Social Media Screenshot",
        description="A UI-like capture of a viral post, message thread, or platform card supporting the fake narrative.",
        suitable_use_cases=("rumors", "viral reactions", "screenshots", "platform-native claims"),
    ),
    ImageFormatConfig(
        id="official_notice",
        label="Official Notice",
        description="A formal bulletin, notice, or announcement card suggesting an institutional statement.",
        suitable_use_cases=("government notices", "school notices", "company announcements"),
    ),
    ImageFormatConfig(
        id="infographic",
        label="Infographic",
        description="A compact explainer card with charts, callouts, and designed information blocks to support the fake claim.",
        suitable_use_cases=("statistics", "comparisons", "explainer visuals"),
    ),
    ImageFormatConfig(
        id="chart_card",
        label="Chart Card",
        description="A chart-first composition where a graph or metric display is the main evidence for the fake narrative.",
        suitable_use_cases=("trends", "polls", "market moves", "performance claims"),
    ),
    ImageFormatConfig(
        id="report_table_screenshot",
        label="Report Table Screenshot",
        description="A screenshot-like crop of a report, spreadsheet, or tabular result that appears to document the fake claim.",
        suitable_use_cases=("rankings", "budget figures", "tables", "reports"),
    ),
    ImageFormatConfig(
        id="map_card",
        label="Map Card",
        description="A map-centric card with highlighted regions, routes, or markers to support place-based fake narratives.",
        suitable_use_cases=("locations", "spread", "regional impact", "travel"),
    ),
    ImageFormatConfig(
        id="timeline_card",
        label="Timeline Card",
        description="A time-ordered visual card showing a sequence of events, milestones, or narrative escalation.",
        suitable_use_cases=("before/after", "chronology", "policy change", "incident buildup"),
    ),
    ImageFormatConfig(
        id="split_comparison",
        label="Split Comparison",
        description="A side-by-side comparison layout emphasizing contrast between two people, places, outcomes, or states.",
        suitable_use_cases=("comparisons", "contrasts", "before vs after", "claim rebuttals"),
    ),
    ImageFormatConfig(
        id="poster_card",
        label="Poster Card",
        description="A designed poster-like news card with headline emphasis and branded visual framing.",
        suitable_use_cases=("campaign framing", "share cards", "headline-first posts"),
    ),
    ImageFormatConfig(
        id="meme",
        label="Meme",
        description="A deliberately internet-native meme format used when the fake branch needs ironic or highly shareable framing.",
        suitable_use_cases=("satire-like virality", "internet subcultures", "joke-forward posts"),
    ),
    ImageFormatConfig(
        id="cartoon",
        label="Cartoon",
        description="An illustrated cartoon scene simplifying the fake narrative into an expressive drawn composition.",
        suitable_use_cases=("simplified narratives", "symbolic scenes", "illustrated explainers"),
    ),
    ImageFormatConfig(
        id="anime",
        label="Anime",
        description="A stylized anime-inspired illustration for narratives where an overtly illustrated treatment is intentional.",
        suitable_use_cases=("stylized fandom content", "highly illustrative narratives"),
    ),
    ImageFormatConfig(
        id="photo_edit",
        label="Photo Edit",
        description="An edited-photo composition suggesting that a real-world image was modified to support the fake narrative.",
        suitable_use_cases=("manipulated evidence", "composite scenes", "edited proof"),
    ),
)


class Settings(BaseModel):
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: Optional[str] = os.getenv("DEEPSEEK_BASE_URL") or None
    # Paper setting: DeepSeek-V4-pro for story generation and visual planning.
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")

    synthesis_provider: str = os.getenv("SYNTHESIS_PROVIDER", "deepseek")
    synthesis_api_key: str = os.getenv("SYNTHESIS_API_KEY", os.getenv("DEEPSEEK_API_KEY", ""))
    synthesis_base_url: Optional[str] = os.getenv(
        "SYNTHESIS_BASE_URL",
        os.getenv("DEEPSEEK_BASE_URL") or None,
    )
    synthesis_model: str = os.getenv(
        "SYNTHESIS_MODEL",
        os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
    )

    # Paper setting: an independent GPT-4o critic.
    critic_provider: str = os.getenv("CRITIC_PROVIDER", "openai")
    critic_api_key: str = os.getenv(
        "CRITIC_API_KEY",
        os.getenv("OPENAI_API_KEY", ""),
    )
    critic_base_url: Optional[str] = os.getenv(
        "CRITIC_BASE_URL",
        os.getenv("OPENAI_BASE_URL") or None,
    )
    critic_model: str = os.getenv(
        "CRITIC_MODEL",
        "gpt-4o",
    )

    # Vision-capable critic configuration. Defaults to the critic endpoint unless explicitly overridden.
    critic_vision_provider: str = os.getenv(
        "CRITIC_VISION_PROVIDER",
        os.getenv("CRITIC_PROVIDER", "openai"),
    )
    critic_vision_api_key: str = os.getenv(
        "CRITIC_VISION_API_KEY",
        os.getenv("CRITIC_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    )
    critic_vision_base_url: Optional[str] = os.getenv(
        "CRITIC_VISION_BASE_URL",
        os.getenv(
            "CRITIC_BASE_URL",
            os.getenv("OPENAI_BASE_URL") or None,
        ),
    )
    critic_vision_model: str = os.getenv(
        "CRITIC_VISION_MODEL",
        os.getenv(
            "CRITIC_MODEL",
            "gpt-4o",
        ),
    )

    # Paper setting: DeepSeek for visual planning and FLUX.1-dev for rendering.
    image_gen_provider: str = os.getenv("IMAGE_GEN_PROVIDER", "sd_local")
    # Keep the requested model identifier in config for consistency with local model downloads.
    image_gen_model: str = os.getenv("IMAGE_GEN_MODEL", "black-forest-labs/FLUX.1-dev")
    # Local image model path or HF repo id. Download the weights first; runtime loads with local_files_only by default.
    sd_model_path: str = os.getenv("SD_MODEL_PATH", os.getenv("IMAGE_GEN_MODEL", "black-forest-labs/FLUX.1-dev"))
    sd_device: str = os.getenv("SD_DEVICE", "cuda")
    sd_dtype: str = os.getenv("SD_DTYPE", "bfloat16")
    # Optional deployment optimization; disabled by default to preserve the paper stack.
    sd_enable_quantization: bool = os.getenv("SD_ENABLE_QUANTIZATION", "false").lower() in {"1", "true", "yes", "on"}
    sd_quantization_mode: str = os.getenv("SD_QUANTIZATION_MODE", "bnb_4bit")
    sd_quantized_model_path: str = os.getenv("SD_QUANTIZED_MODEL_PATH", "")
    sd_enable_cpu_offload: bool = os.getenv("SD_ENABLE_CPU_OFFLOAD", "true").lower() in {"1", "true", "yes", "on"}
    # Favor throughput for large-batch dataset generation; override in .env if a slower,
    # higher-fidelity run is needed for a smaller subset.
    sd_num_inference_steps: int = int(os.getenv("SD_NUM_INFERENCE_STEPS", "8"))
    sd_guidance_scale: float = float(os.getenv("SD_GUIDANCE_SCALE", "7.0"))
    sd_local_files_only: bool = os.getenv("SD_LOCAL_FILES_ONLY", "true").lower() in {"1", "true", "yes", "on"}
    sd_use_reference_images: bool = os.getenv("SD_USE_REFERENCE_IMAGES", "true").lower() in {"1", "true", "yes", "on"}
    sd_reference_max_images: int = int(os.getenv("SD_REFERENCE_MAX_IMAGES", "4"))

    output_dir: str = os.getenv("OUTPUT_DIR", "./sample_dataset_news")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    # Runtime dataset root. The current preferred structure is fake_news_dataset/created_news/<source>/<sample_id>/news_data.json
    dataset_path: Optional[str] = os.getenv("DATASET_PATH", "fake_news_dataset/created_news")
    # Root directory used to resolve original source-image paths stored inside created_news JSON records.
    source_image_root: Optional[str] = os.getenv("SOURCE_IMAGE_ROOT", "fake_news_dataset")

    poster_width: int = int(os.getenv("POSTER_WIDTH", "1024"))
    poster_height: int = int(os.getenv("POSTER_HEIGHT", "1024"))
    max_poster_text_length: int = int(os.getenv("MAX_POSTER_TEXT_LENGTH", "300"))

    text_max_attempts: int = int(os.getenv("TEXT_MAX_ATTEMPTS", "3"))
    image_max_attempts: int = int(os.getenv("IMAGE_MAX_ATTEMPTS", "3"))
    # Parallelize only text-side LLM calls. Keep image generation single-concurrency for single-GPU safety.
    enable_parallel_text_calls: bool = os.getenv("ENABLE_PARALLEL_TEXT_CALLS", "true").lower() in {"1", "true", "yes", "on"}
    parallel_text_workers: int = int(os.getenv("PARALLEL_TEXT_WORKERS", "2"))

    # Legacy option kept for CLI compatibility.
    post_method: int = int(os.getenv("POST_METHOD", "1"))

settings = Settings()


def get_model_config(role: str, runtime_settings: Settings = settings) -> ModelEndpointConfig:
    """Return a model endpoint config for the requested runtime role."""
    valid_roles = {"synthesis", "critic", "critic_vision"}
    if role not in valid_roles:
        raise ValueError(f"Unknown runtime role: {role}")
    prefix = role
    return ModelEndpointConfig(
        provider=getattr(runtime_settings, f"{prefix}_provider"),
        model_id=getattr(runtime_settings, f"{prefix}_model"),
        api_key=getattr(runtime_settings, f"{prefix}_api_key"),
        base_url=getattr(runtime_settings, f"{prefix}_base_url"),
    )


def infer_model_family(model_id: str) -> Optional[str]:
    """Best-effort family detection used to separate runtime models."""
    if not model_id:
        return None

    normalized = model_id.strip().lower()
    known_prefixes = (
        "gpt-4.1",
        "gpt-4o",
        "gpt-4",
        "o4-mini",
        "o3-mini",
        "o3",
        "claude-3-7",
        "claude-3-5",
        "claude-3",
        "gemini-2.5",
        "gemini-2.0",
        "gemini-1.5",
        "deepseek-v4",
        "deepseek-chat",
        "deepseek-reasoner",
        "llama-3.3",
        "llama-3.2",
        "llama-3.1",
    )
    for prefix in known_prefixes:
        if normalized.startswith(prefix):
            return prefix
    return None


def validate_runtime_model_configs(runtime_settings: Settings = settings) -> None:
    """Ensure synthesis remains separate from critic and critic_vision."""
    synthesis_cfg = get_model_config("synthesis", runtime_settings)
    critic_cfg = get_model_config("critic", runtime_settings)
    critic_vision_cfg = get_model_config("critic_vision", runtime_settings)

    for other_name, other_cfg in (("CriticAgent", critic_cfg), ("critic_vision", critic_vision_cfg)):
        if synthesis_cfg.model_id == other_cfg.model_id:
            raise ValueError(
                f"SynthesisAgent and {other_name} cannot use the same model_id. "
                "Configure synthesis and critic runtime models to different values."
            )

    synthesis_family = infer_model_family(synthesis_cfg.model_id)
    for other_name, other_cfg in (("CriticAgent", critic_cfg), ("critic_vision", critic_vision_cfg)):
        other_family = infer_model_family(other_cfg.model_id)
        if synthesis_family and other_family and synthesis_family == other_family:
            raise ValueError(
                f"SynthesisAgent and {other_name} cannot use the same model family. "
                "Choose different families when family detection is available."
            )


def get_image_format_catalog() -> tuple[ImageFormatConfig, ...]:
    """Return the configurable image format catalog."""
    return DEFAULT_IMAGE_FORMAT_CATALOG


def get_image_format_config(format_id: str) -> Optional[ImageFormatConfig]:
    """Look up a format config by ID."""
    for format_cfg in get_image_format_catalog():
        if format_cfg.id == format_id:
            return format_cfg
    return None


def render_image_format_catalog() -> str:
    """Render the image format catalog into a prompt-friendly string."""
    lines = []
    for format_cfg in get_image_format_catalog():
        use_cases = ", ".join(format_cfg.suitable_use_cases) if format_cfg.suitable_use_cases else "general use"
        lines.append(
            f"- {format_cfg.id} | {format_cfg.label}: {format_cfg.description} "
            f"Suitable use cases: {use_cases}."
        )
    return "\n".join(lines)
