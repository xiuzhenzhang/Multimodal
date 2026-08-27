"""Shared runtime models for the two-agent orchestration pipeline."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Grounding(BaseModel):
    """Grounded parse of the source article."""

    who: list[str] = Field(default_factory=list)
    what: list[str] = Field(default_factory=list)
    when: list[str] = Field(default_factory=list)
    where: list[str] = Field(default_factory=list)
    why: list[str] = Field(default_factory=list)
    how: list[str] = Field(default_factory=list)
    original_claims: str = Field(default="")

    @field_validator("who", "what", "when", "where", "why", "how", mode="before")
    @classmethod
    def _coerce_grounding_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            normalized = value.strip()
            return [normalized] if normalized else []
        if isinstance(value, (list, tuple, set)):
            normalized_items = []
            for item in value:
                if item is None:
                    continue
                item_text = str(item).strip()
                if item_text:
                    normalized_items.append(item_text)
            return normalized_items
        item_text = str(value).strip()
        return [item_text] if item_text else []


class ChangeLogEntry(BaseModel):
    """How a supporting fact was changed to sustain the new fake frame."""

    category: str = Field(default="")
    original: str = Field(default="")
    revised: str = Field(default="")
    rationale: str = Field(default="")


class TextSynthesisResult(BaseModel):
    """Output produced by the synthesis runtime before image generation."""

    grounding: Grounding
    true_summary: str
    fake_text: str
    fake_frame: str = Field(default="")
    change_log: list[ChangeLogEntry] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """Unified review schema shared by text and image critics."""

    model_config = ConfigDict(populate_by_name=True)

    pass_: bool = Field(default=False, alias="pass", serialization_alias="pass")
    total_score: float = Field(default=0.0)
    threshold: float = Field(default=0.0)
    dimension_scores: dict[str, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    revision_advice: list[str] = Field(default_factory=list)
    keep_unchanged: list[str] = Field(default_factory=list)


class ReviewPayload(BaseModel):
    """Raw critic payload before deterministic threshold normalization."""

    dimension_scores: dict[str, float] = Field(default_factory=dict)
    issues: list[str] = Field(default_factory=list)
    revision_advice: list[str] = Field(default_factory=list)
    keep_unchanged: list[str] = Field(default_factory=list)
    hard_fail_reasons: list[str] = Field(default_factory=list)


class TextReviewResult(BaseModel):
    """Combined text review covering the true and fake branches."""

    true_summary_review: ReviewResult
    fake_text_review: ReviewResult


class ImageReviewResult(BaseModel):
    """Image review wrapper including candidate metadata."""

    review: ReviewResult
    selected_format: Optional[str] = None
    candidate_id: Optional[str] = None
    recommended_format: Optional[str] = None
    should_change_format: bool = False
    format_change_reason: Optional[str] = None


class TextReviewPayload(BaseModel):
    """Optional combined critic payload for text review calls."""

    true_summary_review: ReviewPayload
    fake_text_review: ReviewPayload


class ImageReviewPayload(BaseModel):
    """Optional combined critic payload for image review calls."""

    review: ReviewPayload
    selected_format: Optional[str] = None
    candidate_id: Optional[str] = None
    recommended_format: Optional[str] = None
    should_change_format: bool = False
    format_change_reason: Optional[str] = None


class ImageCandidate(BaseModel):
    """A generated image candidate and its associated review history."""

    candidate_id: str
    selected_format: Optional[str] = None
    format_rationale: Optional[str] = None
    strategy_name: Optional[str] = None
    strategy_details: Optional[str] = None
    image_prompt: str = Field(default="")
    background_image_path: str = Field(default="")
    final_post_path: str = Field(default="")
    review: Optional[ImageReviewResult] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class FinalDatasetItem(BaseModel):
    """Final assembled dataset record saved by the orchestrator."""

    status: str
    text_stage_passed: bool = Field(default=False)
    selected_text_attempt: Optional[int] = None
    source_article: str
    source_images: list[str] = Field(default_factory=list)
    grounding: Grounding
    true_branch: dict[str, Any]
    fake_branch: dict[str, Any]
    change_log: list[ChangeLogEntry] = Field(default_factory=list)
    selected_text_review: Optional[dict[str, Any]] = None
    text_reviews: list[dict[str, Any]] = Field(default_factory=list)
    image_candidate_history: list[dict[str, Any]] = Field(default_factory=list)
    selected_image: Optional[dict[str, Any]] = None
