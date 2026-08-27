"""Synthesis runtime agent for the two-agent dataset generation pipeline."""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from typing import Optional

from pydantic import BaseModel, Field
from pydantic import ValidationError

from generation.agents.llm_utils import build_chat_model, parse_json_response
from generation.runtime import (
    get_image_format_config,
    get_model_config,
    render_image_format_catalog,
    settings,
)
from generation.pipeline.models import (
    ChangeLogEntry,
    Grounding,
    ImageCandidate,
    ImageReviewResult,
    TextReviewResult,
    TextSynthesisResult,
)
from generation.utils.prompt_templates import (
    FAKE_TEXT_PROMPT,
    GROUNDING_PROMPT,
    IMAGE_FORMAT_REVISION_PROMPT,
    IMAGE_FORMAT_SELECTION_PROMPT,
    IMAGE_PROMPT_REVISION_PROMPT,
    TEXT_REVISION_PROMPT,
    TRUE_SUMMARY_PROMPT,
)

logger = logging.getLogger(__name__)


class FakeDraftPayload(BaseModel):
    fake_frame: str = Field(default="")
    fake_text: str = Field(default="")
    change_log: list[ChangeLogEntry] = Field(default_factory=list)


class TextRevisionPayload(BaseModel):
    true_summary: str = Field(default="")
    fake_frame: str = Field(default="")
    fake_text: str = Field(default="")
    change_log: list[ChangeLogEntry] = Field(default_factory=list)


class FormatSelectionPayload(BaseModel):
    selected_format: str = Field(default="documentary_photo")
    rationale: str = Field(default="")


class FormatPlan(BaseModel):
    """Compatibility payload for visual helper prompt generation."""

    selected_strategy: int = Field(default=0)
    strategy_name: str
    strategy_details: str
    reasoning: str = Field(default="")
    probability_distribution: dict[str, float] = Field(default_factory=lambda: {"selected": 1.0})


class TransformerOutput(BaseModel):
    """Compatibility payload for legacy visual helper classes."""

    facts: Grounding
    opposite_claims: str
    mirrored_article: str
    post_text: str


class SynthesisAgent:
    """Responsible for grounded parsing, text generation/revision, and fake-branch image synthesis."""

    def __init__(self, model_name: Optional[str] = None, visual_helper=None):
        endpoint = get_model_config("synthesis")
        if model_name:
            endpoint = endpoint.__class__(
                provider=endpoint.provider,
                model_id=model_name,
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
            )

        self.endpoint = endpoint
        self.llm = build_chat_model(endpoint, temperature=0.5)
        self.model_name = endpoint.model_id
        self.visual_helper = visual_helper or self._create_visual_helper()

    def _create_visual_helper(self):
        helper_kwargs = {
            "model_name": self.endpoint.model_id,
            "api_key": self.endpoint.api_key,
            "base_url": self.endpoint.base_url,
        }
        from generation.runtime import settings

        if settings.image_gen_provider != "sd_local":
            raise ValueError("The public generation package supports only the local sd_local image backend.")
        from generation.agents.visual_producer_sd_local import VisualProducerAgent
        return VisualProducerAgent(**helper_kwargs)

    def _call_json_prompt(self, template: str, model_cls, **kwargs):
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(**kwargs)
        response = self.llm.invoke(messages)
        return parse_json_response(response.content, model_cls)

    def _call_json_prompt_with_retries(self, template: str, model_cls, max_attempts: int = 2, **kwargs):
        last_error = None
        for attempt_index in range(1, max_attempts + 1):
            try:
                return self._call_json_prompt(template, model_cls, **kwargs)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = exc
                if attempt_index >= max_attempts:
                    raise
                logger.warning(
                    "Structured JSON parse failed for %s on attempt %s/%s: %s. Retrying once.",
                    getattr(model_cls, "__name__", str(model_cls)),
                    attempt_index,
                    max_attempts,
                    exc,
                )
        raise last_error

    def _call_text_prompt(self, template: str, **kwargs) -> str:
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(**kwargs)
        response = self.llm.invoke(messages)
        return response.content.strip().replace("**", "").strip()

    def extract_grounding(self, source_article: str) -> Grounding:
        logger.info("Extracting grounded article facts.")
        return self._call_json_prompt_with_retries(
            GROUNDING_PROMPT,
            Grounding,
            source_article=source_article,
        )

    def generate_true_summary(self, source_article: str, grounding: Grounding) -> str:
        logger.info("Generating true summary draft.")
        return self._call_text_prompt(
            TRUE_SUMMARY_PROMPT,
            grounding=json.dumps(grounding.model_dump(), ensure_ascii=False, indent=2),
            source_article=source_article,
        )

    def generate_fake_text(self, source_article: str, grounding: Grounding) -> FakeDraftPayload:
        logger.info("Generating fake text draft with frame flip and fact edits.")
        return self._call_json_prompt_with_retries(
            FAKE_TEXT_PROMPT,
            FakeDraftPayload,
            grounding=json.dumps(grounding.model_dump(), ensure_ascii=False, indent=2),
            source_article=source_article,
        )

    def synthesize_text(
        self,
        source_article: str,
        post_method: Optional[int] = None,
    ) -> TextSynthesisResult:
        _ = post_method
        grounding = self.extract_grounding(source_article)

        if settings.enable_parallel_text_calls and settings.parallel_text_workers >= 2:
            logger.info("Running true-summary and fake-text generation in parallel.")
            with ThreadPoolExecutor(max_workers=min(settings.parallel_text_workers, 2)) as executor:
                true_summary_future = executor.submit(self.generate_true_summary, source_article, grounding)
                fake_draft_future = executor.submit(self.generate_fake_text, source_article, grounding)
                true_summary = true_summary_future.result()
                fake_draft = fake_draft_future.result()
        else:
            true_summary = self.generate_true_summary(source_article, grounding)
            fake_draft = self.generate_fake_text(source_article, grounding)

        return TextSynthesisResult(
            grounding=grounding,
            true_summary=true_summary,
            fake_text=fake_draft.fake_text,
            fake_frame=fake_draft.fake_frame,
            change_log=fake_draft.change_log,
        )

    def revise_text(
        self,
        source_article: str,
        current_result: TextSynthesisResult,
        text_review: TextReviewResult,
    ) -> TextSynthesisResult:
        logger.info("Revising text outputs from critic advice.")
        payload = self._call_json_prompt_with_retries(
            TEXT_REVISION_PROMPT,
            TextRevisionPayload,
            grounding=json.dumps(current_result.grounding.model_dump(), ensure_ascii=False, indent=2),
            source_article=source_article,
            true_summary=current_result.true_summary,
            fake_frame=current_result.fake_frame,
            fake_text=current_result.fake_text,
            change_log=json.dumps(
                [entry.model_dump() for entry in current_result.change_log],
                ensure_ascii=False,
                indent=2,
            ),
            true_summary_revision_advice=json.dumps(
                text_review.true_summary_review.revision_advice,
                ensure_ascii=False,
                indent=2,
            ),
            true_summary_keep_unchanged=json.dumps(
                text_review.true_summary_review.keep_unchanged,
                ensure_ascii=False,
                indent=2,
            ),
            fake_text_revision_advice=json.dumps(
                text_review.fake_text_review.revision_advice,
                ensure_ascii=False,
                indent=2,
            ),
            fake_text_keep_unchanged=json.dumps(
                text_review.fake_text_review.keep_unchanged,
                ensure_ascii=False,
                indent=2,
            ),
        )
        return TextSynthesisResult(
            grounding=current_result.grounding,
            true_summary=payload.true_summary or current_result.true_summary,
            fake_text=payload.fake_text or current_result.fake_text,
            fake_frame=payload.fake_frame or current_result.fake_frame,
            change_log=payload.change_log or current_result.change_log,
        )

    def _build_visual_context(self, source_article: str, text_result: TextSynthesisResult) -> TransformerOutput:
        visual_context = (
            f"Frozen fake text: {text_result.fake_text}\n\n"
            f"Fake frame: {text_result.fake_frame}\n\n"
            f"Grounding: {json.dumps(text_result.grounding.model_dump(), ensure_ascii=False)}\n\n"
            f"Change log: {json.dumps([entry.model_dump() for entry in text_result.change_log], ensure_ascii=False)}\n\n"
            f"Source article excerpt: {source_article[:1000]}"
        )
        return TransformerOutput(
            facts=text_result.grounding,
            opposite_claims=text_result.fake_frame,
            mirrored_article=visual_context,
            post_text=text_result.fake_text,
        )

    def _build_format_plan(self, selected_format: str, rationale: str) -> FormatPlan:
        format_cfg = get_image_format_config(selected_format)
        if format_cfg is None:
            raise ValueError(f"Unknown image format: {selected_format}")
        details = format_cfg.description
        if format_cfg.suitable_use_cases:
            details += " Suitable use cases: " + ", ".join(format_cfg.suitable_use_cases) + "."
        if rationale:
            details += f" Runtime rationale: {rationale}"
        return FormatPlan(
            strategy_name=format_cfg.label,
            strategy_details=details,
            reasoning=rationale,
        )

    def _build_format_selection_guidance(self, text_result: TextSynthesisResult) -> dict[str, object]:
        text_blob = " ".join(
            [
                text_result.fake_text or "",
                text_result.fake_frame or "",
                text_result.grounding.original_claims or "",
                " ".join(text_result.grounding.what or []),
                " ".join(text_result.grounding.where or []),
                " ".join(text_result.grounding.why or []),
            ]
        ).lower()
        change_blob = " ".join(
            f"{entry.category} {entry.original} {entry.revised} {entry.rationale}"
            for entry in text_result.change_log
        ).lower()
        combined = f"{text_blob} {change_blob}"

        def has_any(*terms: str) -> bool:
            return any(term in combined for term in terms)

        numeric_signal = sum(ch.isdigit() for ch in combined) >= 6 or has_any(
            "percent", "%", "increase", "decrease", "survey", "poll", "rate", "budget", "ranking"
        )
        timeline_signal = has_any("before", "after", "timeline", "previously", "later", "earlier", "timeline")
        map_signal = has_any("map", "region", "state", "province", "border", "route", "across", "spread")
        comparison_signal = has_any("versus", "vs.", "compared", "comparison", "before and after")
        institutional_signal = has_any(
            "official", "statement", "announced", "announcement", "notice", "memo", "guideline", "policy", "order"
        )
        screenshot_signal = has_any(
            "tweet", "twitter", "x post", "post on x", "instagram post", "facebook post", "thread",
            "screenshot", "dm", "comment section", "viral post", "account posted", "channel post"
        )
        manipulated_evidence_signal = has_any("leaked photo", "edited image", "doctored", "altered", "photoshopped")

        recommended_formats: list[str] = []
        discouraged_formats: list[str] = []
        guidance_lines = [
            "Prefer visually diverse formats across the dataset instead of defaulting to social-media UI mockups.",
            "Choose the most concrete visual evidence format for the claim, not merely the fact that the text will be posted on social media.",
        ]

        if screenshot_signal:
            recommended_formats.extend(["social_media_screenshot", "photo_edit"])
            guidance_lines.append("This narrative explicitly references platform-native evidence, so screenshot-style formats are allowed.")
        else:
            discouraged_formats.append("social_media_screenshot")
            guidance_lines.append("Do not use social_media_screenshot unless the claim itself depends on a post, thread, screenshot, or account UI.")

        if institutional_signal:
            recommended_formats.extend(["official_notice", "documentary_photo"])
            guidance_lines.append("Institutional or policy language makes official_notice plausible.")
        else:
            discouraged_formats.append("official_notice")

        if numeric_signal:
            recommended_formats.extend(["infographic", "chart_card", "report_table_screenshot"])
            guidance_lines.append("The narrative contains strong numeric evidence, so prefer evidence-heavy visual formats.")

        if timeline_signal:
            recommended_formats.append("timeline_card")

        if map_signal:
            recommended_formats.append("map_card")

        if comparison_signal:
            recommended_formats.append("split_comparison")

        if manipulated_evidence_signal:
            recommended_formats.append("photo_edit")

        # Always keep a realistic fallback available for person/event-centric stories.
        recommended_formats.extend(["documentary_photo", "poster_card"])

        # Preserve order while removing duplicates.
        recommended_formats = list(dict.fromkeys(recommended_formats))
        discouraged_formats = [fmt for fmt in dict.fromkeys(discouraged_formats) if fmt not in recommended_formats]

        return {
            "recommended_formats": recommended_formats[:5],
            "discouraged_formats": discouraged_formats[:4],
            "selection_guidance": "\n".join(f"- {line}" for line in guidance_lines),
            "allow_social_media_screenshot": screenshot_signal,
            "allow_official_notice": institutional_signal,
        }

    def _normalize_selected_format(
        self,
        format_payload: FormatSelectionPayload,
        guidance: dict[str, object],
    ) -> FormatSelectionPayload:
        recommended_formats = guidance.get("recommended_formats", []) or []
        discouraged_formats = guidance.get("discouraged_formats", []) or []
        selected_format = format_payload.selected_format

        if selected_format == "social_media_screenshot" and not guidance.get("allow_social_media_screenshot", False):
            fallback = recommended_formats[0] if recommended_formats else "documentary_photo"
            return FormatSelectionPayload(
                selected_format=fallback,
                rationale=(
                    f"{format_payload.rationale} Replaced social_media_screenshot with {fallback} "
                    "because the narrative does not rely on platform UI evidence."
                ).strip(),
            )

        if selected_format == "official_notice" and not guidance.get("allow_official_notice", False):
            fallback = recommended_formats[0] if recommended_formats else "documentary_photo"
            return FormatSelectionPayload(
                selected_format=fallback,
                rationale=(
                    f"{format_payload.rationale} Replaced official_notice with {fallback} "
                    "because the narrative is not primarily a formal bulletin or memo."
                ).strip(),
            )

        if selected_format in discouraged_formats and recommended_formats:
            fallback = recommended_formats[0]
            return FormatSelectionPayload(
                selected_format=fallback,
                rationale=(
                    f"{format_payload.rationale} Switched to {fallback} to avoid an overused or weakly grounded format."
                ).strip(),
            )

        return format_payload

    def select_image_format(self, text_result: TextSynthesisResult) -> FormatSelectionPayload:
        logger.info("Selecting image format for frozen fake text.")
        guidance = self._build_format_selection_guidance(text_result)
        payload = self._call_json_prompt_with_retries(
            IMAGE_FORMAT_SELECTION_PROMPT,
            FormatSelectionPayload,
            fake_text=text_result.fake_text,
            fake_frame=text_result.fake_frame,
            grounding=json.dumps(text_result.grounding.model_dump(), ensure_ascii=False, indent=2),
            format_catalog=render_image_format_catalog(),
            selection_guidance=guidance["selection_guidance"],
            recommended_formats=", ".join(guidance["recommended_formats"]),
            discouraged_formats=", ".join(guidance["discouraged_formats"]) or "none",
        )
        return self._normalize_selected_format(payload, guidance)

    def select_or_revise_image_format(
        self,
        frozen_text_result: TextSynthesisResult,
        previous_candidate: Optional[ImageCandidate] = None,
        previous_review: Optional[ImageReviewResult] = None,
    ) -> FormatSelectionPayload:
        """Allow failed image attempts to keep or switch formats in a structured way."""
        if previous_candidate is None:
            return self.select_image_format(frozen_text_result)

        guidance = self._build_format_selection_guidance(frozen_text_result)

        if previous_review and previous_review.should_change_format and previous_review.recommended_format:
            if get_image_format_config(previous_review.recommended_format):
                payload = FormatSelectionPayload(
                    selected_format=previous_review.recommended_format,
                    rationale=previous_review.format_change_reason or "Switched per critic recommendation.",
                )
                return self._normalize_selected_format(payload, guidance)

        if previous_review is None:
            payload = FormatSelectionPayload(
                selected_format=previous_candidate.selected_format or "documentary_photo",
                rationale=previous_candidate.format_rationale or "Preserved previous format.",
            )
            return self._normalize_selected_format(payload, guidance)

        payload = self._call_json_prompt_with_retries(
            IMAGE_FORMAT_REVISION_PROMPT,
            FormatSelectionPayload,
            fake_text=frozen_text_result.fake_text,
            fake_frame=frozen_text_result.fake_frame,
            grounding=json.dumps(frozen_text_result.grounding.model_dump(), ensure_ascii=False, indent=2),
            current_format=previous_candidate.selected_format or "documentary_photo",
            current_format_rationale=previous_candidate.format_rationale or "",
            issues=json.dumps(previous_review.review.issues, ensure_ascii=False, indent=2),
            revision_advice=json.dumps(previous_review.review.revision_advice, ensure_ascii=False, indent=2),
            recommended_format=previous_review.recommended_format or "",
            should_change_format=str(previous_review.should_change_format),
            format_change_reason=previous_review.format_change_reason or "",
            format_catalog=render_image_format_catalog(),
            selection_guidance=guidance["selection_guidance"],
            recommended_formats=", ".join(guidance["recommended_formats"]),
            discouraged_formats=", ".join(guidance["discouraged_formats"]) or "none",
        )
        return self._normalize_selected_format(payload, guidance)

    def _revise_image_prompt(
        self,
        fake_text: str,
        selected_format: str,
        current_image_prompt: str,
        revision_advice: list[str],
        keep_unchanged: list[str],
    ) -> str:
        logger.info("Revising image prompt from critic advice.")
        return self._call_text_prompt(
            IMAGE_PROMPT_REVISION_PROMPT,
            fake_text=fake_text,
            selected_format=selected_format,
            current_image_prompt=current_image_prompt,
            revision_advice=json.dumps(revision_advice, ensure_ascii=False, indent=2),
            keep_unchanged=json.dumps(keep_unchanged, ensure_ascii=False, indent=2),
        )

    def generate_or_revise_image(
        self,
        source_article: str,
        frozen_text_result: TextSynthesisResult,
        output_dir: str,
        attempt_index: int,
        previous_candidate: Optional[ImageCandidate] = None,
        previous_review: Optional[ImageReviewResult] = None,
        source_images: Optional[list[str]] = None,
    ) -> ImageCandidate:
        """Generate or revise a fake-branch image candidate without changing the frozen fake text."""
        logger.info("Generating image candidate %s.", attempt_index)
        os.makedirs(output_dir, exist_ok=True)

        format_payload = self.select_or_revise_image_format(
            frozen_text_result=frozen_text_result,
            previous_candidate=previous_candidate,
            previous_review=previous_review,
        )
        format_plan = self._build_format_plan(format_payload.selected_format, format_payload.rationale)

        visual_context = self._build_visual_context(source_article, frozen_text_result)
        semantic_extraction = self.visual_helper.extract_semantics(visual_context, format_plan)
        base_prompt = self.visual_helper.generate_image_prompt(
            semantic_extraction,
            frozen_text_result.fake_text,
            frozen_text_result.fake_frame,
            format_plan,
            visual_context.mirrored_article,
        )

        image_prompt = base_prompt
        if previous_review is not None:
            image_prompt = self._revise_image_prompt(
                fake_text=frozen_text_result.fake_text,
                selected_format=format_payload.selected_format,
                current_image_prompt=base_prompt,
                revision_advice=previous_review.review.revision_advice,
                keep_unchanged=previous_review.review.keep_unchanged,
            )

        background_image_path = os.path.join(output_dir, f"candidate_{attempt_index}_background.png")
        final_post_path = os.path.join(output_dir, f"candidate_{attempt_index}_post.png")
        background_image_path = self.visual_helper.generate_image(
            image_prompt,
            background_image_path,
            reference_images=source_images,
        )
        final_post_path = self.visual_helper.create_final_post(
            background_image_path,
            frozen_text_result.fake_text,
            final_post_path,
        )

        return ImageCandidate(
            candidate_id=str(attempt_index),
            selected_format=format_payload.selected_format,
            format_rationale=format_payload.rationale,
            strategy_name=format_plan.strategy_name,
            strategy_details=format_plan.strategy_details,
            image_prompt=image_prompt,
            background_image_path=background_image_path,
            final_post_path=final_post_path,
            metadata={
                "source_images": source_images or [],
                "used_reference_images": bool(source_images),
                "format_rationale": format_payload.rationale,
                "format_id": format_payload.selected_format,
            },
        )

    def process(self, news_article: str, post_method: Optional[int] = None) -> TextSynthesisResult:
        return self.synthesize_text(news_article, post_method=post_method)


Facts = Grounding
TransformerAgent = SynthesisAgent
