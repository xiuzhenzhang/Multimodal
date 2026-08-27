"""Critic runtime agent for structured text and image review."""
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from dataclasses import asdict
from typing import Optional

from PIL import Image, ImageStat

from generation.agents.llm_utils import build_chat_model, invoke_json_with_image, parse_json_response
from generation.runtime import (
    FAKE_TEXT_SCORE_CONFIG,
    IMAGE_SCORE_CONFIG,
    TRUE_SUMMARY_SCORE_CONFIG,
    get_image_format_config,
    get_model_config,
    settings,
)
from generation.pipeline.models import (
    ImageCandidate,
    ImageReviewPayload,
    ImageReviewResult,
    ReviewPayload,
    ReviewResult,
    TextReviewResult,
    TextSynthesisResult,
)
from generation.utils.prompt_templates import (
    FAKE_TEXT_REVIEW_PROMPT,
    IMAGE_REVIEW_PROMPT,
    TRUE_SUMMARY_REVIEW_PROMPT,
)

logger = logging.getLogger(__name__)


class CriticAgent:
    """Reviews text and image outputs with deterministic threshold normalization."""

    def __init__(self, model_name: Optional[str] = None):
        endpoint = get_model_config("critic")
        if model_name:
            endpoint = endpoint.__class__(
                provider=endpoint.provider,
                model_id=model_name,
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
            )
        self.endpoint = endpoint
        self.llm = build_chat_model(endpoint, temperature=0.2)
        self.model_name = endpoint.model_id

        self.vision_endpoint = get_model_config("critic_vision")
        if (
            self.vision_endpoint.provider == self.endpoint.provider
            and self.vision_endpoint.model_id == self.endpoint.model_id
            and self.vision_endpoint.api_key == self.endpoint.api_key
            and self.vision_endpoint.base_url == self.endpoint.base_url
        ):
            self.vision_llm = self.llm
        else:
            self.vision_llm = build_chat_model(self.vision_endpoint, temperature=0.1)

    def _call_json_prompt(self, template: str, model_cls, **kwargs):
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(template)
        messages = prompt.format_messages(**kwargs)
        response = self.llm.invoke(messages)
        return parse_json_response(response.content, model_cls)

    def _score_total(self, payload: ReviewPayload, score_config) -> tuple[float, dict[str, float]]:
        dimension_scores: dict[str, float] = {}
        total_score = 0.0
        for dimension, weight in score_config.weights.items():
            raw_value = float(payload.dimension_scores.get(dimension, 0.0))
            clipped = max(0.0, min(5.0, raw_value))
            dimension_scores[dimension] = clipped
            total_score += (clipped / 5.0) * weight
        return round(total_score, 2), dimension_scores

    def _normalize_review(self, payload: ReviewPayload, score_config) -> ReviewResult:
        total_score, dimension_scores = self._score_total(payload, score_config)

        issues = list(payload.issues)
        # Be conservative with hard fails: only explicit hard_fail_reasons from the critic
        # should trigger a veto, not loosely worded issues.
        hard_fail_reasons = list(payload.hard_fail_reasons)

        min_dimension_failures = []
        for dimension, minimum in score_config.minimum_dimensions.items():
            if dimension_scores.get(dimension, 0.0) < minimum:
                min_dimension_failures.append(
                    f"{dimension} below minimum: {dimension_scores.get(dimension, 0.0)}/5 < {minimum}/5"
                )

        for item in hard_fail_reasons + min_dimension_failures:
            if item not in issues:
                issues.append(item)

        passed = (
            total_score >= score_config.threshold
            and not hard_fail_reasons
            and not min_dimension_failures
        )

        revision_advice = list(payload.revision_advice)
        if not passed and not revision_advice:
            revision_advice = ["Revise the draft to address the listed issues and raise the failed dimensions."]

        return ReviewResult(
            pass_=passed,
            total_score=total_score,
            threshold=score_config.threshold,
            dimension_scores=dimension_scores,
            issues=issues,
            revision_advice=revision_advice,
            keep_unchanged=list(payload.keep_unchanged),
        )

    def review_text(self, source_article: str, text_result: TextSynthesisResult) -> TextReviewResult:
        logger.info("Reviewing text outputs.")
        grounding_json = json.dumps(text_result.grounding.model_dump(), ensure_ascii=False, indent=2)
        true_kwargs = dict(
            score_config=json.dumps(asdict(TRUE_SUMMARY_SCORE_CONFIG), ensure_ascii=False, indent=2),
            grounding=grounding_json,
            source_article=source_article,
            true_summary=text_result.true_summary,
        )
        fake_kwargs = dict(
            score_config=json.dumps(asdict(FAKE_TEXT_SCORE_CONFIG), ensure_ascii=False, indent=2),
            grounding=grounding_json,
            source_article=source_article,
            fake_frame=text_result.fake_frame,
            fake_text=text_result.fake_text,
            change_log=json.dumps(
                [entry.model_dump() for entry in text_result.change_log],
                ensure_ascii=False,
                indent=2,
            ),
        )

        if settings.enable_parallel_text_calls and settings.parallel_text_workers >= 2:
            logger.info("Running true-summary and fake-text reviews in parallel.")
            with ThreadPoolExecutor(max_workers=min(settings.parallel_text_workers, 2)) as executor:
                true_future = executor.submit(
                    self._call_json_prompt,
                    TRUE_SUMMARY_REVIEW_PROMPT,
                    ReviewPayload,
                    **true_kwargs,
                )
                fake_future = executor.submit(
                    self._call_json_prompt,
                    FAKE_TEXT_REVIEW_PROMPT,
                    ReviewPayload,
                    **fake_kwargs,
                )
                true_payload = true_future.result()
                fake_payload = fake_future.result()
        else:
            true_payload = self._call_json_prompt(
                TRUE_SUMMARY_REVIEW_PROMPT,
                ReviewPayload,
                **true_kwargs,
            )
            fake_payload = self._call_json_prompt(
                FAKE_TEXT_REVIEW_PROMPT,
                ReviewPayload,
                **fake_kwargs,
            )

        return TextReviewResult(
            true_summary_review=self._normalize_review(true_payload, TRUE_SUMMARY_SCORE_CONFIG),
            fake_text_review=self._normalize_review(fake_payload, FAKE_TEXT_SCORE_CONFIG),
        )

    def _observe_image(self, image_path: str) -> str:
        if not image_path or not os.path.exists(image_path):
            return "image file missing"

        try:
            with Image.open(image_path) as image:
                stat = ImageStat.Stat(image.convert("RGB"))
                return json.dumps(
                    {
                        "path": image_path,
                        "size": image.size,
                        "mode": image.mode,
                        "file_size_bytes": os.path.getsize(image_path),
                        "mean_rgb": [round(value, 2) for value in stat.mean],
                        "stddev_rgb": [round(value, 2) for value in stat.stddev],
                    },
                    ensure_ascii=False,
                )
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to observe image %s: %s", image_path, exc)
            return f"unable to inspect image: {exc}"

    def _normalize_image_review(self, payload: ImageReviewPayload, candidate: ImageCandidate) -> ImageReviewResult:
        normalized_review = self._normalize_review(payload.review, IMAGE_SCORE_CONFIG)
        recommended_format = payload.recommended_format
        if recommended_format and not get_image_format_config(recommended_format):
            recommended_format = None

        return ImageReviewResult(
            review=normalized_review,
            selected_format=payload.selected_format or candidate.selected_format,
            candidate_id=payload.candidate_id or candidate.candidate_id,
            recommended_format=recommended_format,
            should_change_format=bool(payload.should_change_format and recommended_format),
            format_change_reason=payload.format_change_reason,
        )

    def review_image(
        self,
        source_article: str,
        frozen_text_result: TextSynthesisResult,
        candidate: ImageCandidate,
    ) -> ImageReviewResult:
        """Review an image candidate, including text-image consistency and format-switch guidance."""
        logger.info("Reviewing image candidate %s.", candidate.candidate_id)
        image_path = candidate.final_post_path or candidate.background_image_path
        if not image_path or not os.path.exists(image_path):
            return ImageReviewResult(
                review=ReviewResult(
                    pass_=False,
                    total_score=0.0,
                    threshold=IMAGE_SCORE_CONFIG.threshold,
                    dimension_scores={dimension: 0.0 for dimension in IMAGE_SCORE_CONFIG.weights},
                    issues=["Image file missing for review."],
                    revision_advice=["Regenerate the fake-branch image and ensure the rendered file exists."],
                    keep_unchanged=[],
                ),
                selected_format=candidate.selected_format,
                candidate_id=candidate.candidate_id,
            )

        prompt_text = IMAGE_REVIEW_PROMPT.format(
            score_config=json.dumps(asdict(IMAGE_SCORE_CONFIG), ensure_ascii=False, indent=2),
            fake_text=frozen_text_result.fake_text,
            fake_frame=frozen_text_result.fake_frame,
            grounding=json.dumps(frozen_text_result.grounding.model_dump(), ensure_ascii=False, indent=2),
            selected_format=candidate.selected_format or "",
            image_prompt=candidate.image_prompt,
            image_metadata=json.dumps(candidate.metadata, ensure_ascii=False, indent=2),
            image_observation=self._observe_image(image_path),
            candidate_id=candidate.candidate_id,
        )
        payload = invoke_json_with_image(
            self.vision_llm,
            prompt_text=prompt_text,
            image_path=image_path,
            model_cls=ImageReviewPayload,
        )
        return self._normalize_image_review(payload, candidate)


SentinelAgent = CriticAgent
