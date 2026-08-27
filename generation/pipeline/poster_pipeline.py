"""Orchestrator for the story, image, and critic roles."""
import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

from generation.runtime import settings, validate_runtime_model_configs
from generation.pipeline.models import FinalDatasetItem, ImageCandidate, TextSynthesisResult

if TYPE_CHECKING:
    from generation.agents.sentinel import CriticAgent
    from generation.agents.transformer import SynthesisAgent

logger = logging.getLogger(__name__)


class PosterPipeline:
    """Orchestrates SynthesisAgent and CriticAgent with bounded retries."""

    def __init__(
        self,
        max_retries: Optional[int] = None,
        output_dir: Optional[str] = None,
        sd_generation_lock: Optional[Any] = None,
        synthesis_agent: Optional["SynthesisAgent"] = None,
        critic_agent: Optional["CriticAgent"] = None,
        text_max_attempts: Optional[int] = None,
        image_max_attempts: Optional[int] = None,
    ):
        validate_runtime_model_configs()

        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=getattr(logging, settings.log_level.upper(), logging.INFO),
                format="%(asctime)s %(levelname)s %(name)s - %(message)s",
            )

        shared_attempts = max_retries
        self.text_max_attempts = text_max_attempts or shared_attempts or settings.text_max_attempts
        self.image_max_attempts = image_max_attempts or shared_attempts or settings.image_max_attempts
        self.output_dir = output_dir or settings.output_dir
        self.sd_generation_lock = sd_generation_lock

        if synthesis_agent is None:
            from generation.agents.transformer import SynthesisAgent

            synthesis_agent = SynthesisAgent()
        if critic_agent is None:
            from generation.agents.sentinel import CriticAgent

            critic_agent = CriticAgent()

        self.synthesis = synthesis_agent
        self.critic = critic_agent

    def _serialize(self, value):
        if hasattr(value, "model_dump"):
            return value.model_dump(by_alias=True)
        if isinstance(value, list):
            return [self._serialize(item) for item in value]
        if isinstance(value, dict):
            return {key: self._serialize(item) for key, item in value.items()}
        return value

    def _create_output_dir(self, article_id: Optional[str]) -> tuple[str, str]:
        if article_id:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, article_id)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(self.output_dir, f"post_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir, timestamp

    def _clone_text_result(self, text_result: TextSynthesisResult) -> TextSynthesisResult:
        return TextSynthesisResult.model_validate(text_result.model_dump())

    def _text_stage_score(self, text_review) -> float:
        return (
            text_review.true_summary_review.total_score
            + text_review.fake_text_review.total_score
        )

    def _save_json(self, path: str, payload: dict) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def process(
        self,
        news_article: str,
        save_intermediate: bool = True,
        post_method: int = 1,
        article_id: Optional[str] = None,
        source_images: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        """Run the 2-agent orchestration pipeline for a single article."""
        logger.info("Starting poster pipeline.")
        output_dir, timestamp = self._create_output_dir(article_id)
        source_images = source_images or []
        true_primary_image = source_images[0] if source_images else None

        text_reviews = []
        image_candidate_history: list[ImageCandidate] = []
        selected_image: Optional[ImageCandidate] = None

        logger.info("Synthesizing initial grounded text bundle.")
        current_text_result = self.synthesis.synthesize_text(news_article, post_method=post_method)

        text_passed = False
        best_text_result = self._clone_text_result(current_text_result)
        best_text_review = None
        best_text_attempt = 1
        best_text_score = float("-inf")
        for attempt_index in range(1, self.text_max_attempts + 1):
            logger.info("Text review attempt %s/%s.", attempt_index, self.text_max_attempts)
            text_review = self.critic.review_text(news_article, current_text_result)
            text_reviews.append(text_review)
            text_score = self._text_stage_score(text_review)

            if text_score > best_text_score:
                best_text_score = text_score
                best_text_attempt = attempt_index
                best_text_result = self._clone_text_result(current_text_result)
                best_text_review = text_review

            if text_review.true_summary_review.pass_ and text_review.fake_text_review.pass_:
                logger.info("Text stage passed on attempt %s.", attempt_index)
                text_passed = True
                break

            if attempt_index >= self.text_max_attempts:
                logger.warning(
                    "Text stage did not pass after %s attempts. Selecting highest-scoring text from attempt %s with combined score %.2f.",
                    self.text_max_attempts,
                    best_text_attempt,
                    best_text_score,
                )
                break

            logger.info("Revising text outputs using critic advice.")
            current_text_result = self.synthesis.revise_text(
                news_article,
                current_text_result,
                text_review,
            )

        selected_text_review = text_reviews[-1] if text_passed else best_text_review
        selected_text_attempt = len(text_reviews) if text_passed else best_text_attempt
        frozen_text_result = self._clone_text_result(current_text_result if text_passed else best_text_result)

        logger.info(
            "Freezing fake_text before image generation using text attempt %s (passed=%s).",
            selected_text_attempt,
            text_passed,
        )
        for attempt_index in range(1, self.image_max_attempts + 1):
            previous_candidate = image_candidate_history[-1] if image_candidate_history else None
            previous_review = previous_candidate.review if previous_candidate else None

            logger.info("Image generation attempt %s/%s.", attempt_index, self.image_max_attempts)
            try:
                candidate = self.synthesis.generate_or_revise_image(
                    source_article=news_article,
                    frozen_text_result=frozen_text_result,
                    output_dir=output_dir,
                    attempt_index=attempt_index,
                    previous_candidate=previous_candidate,
                    previous_review=previous_review,
                    source_images=source_images,
                )
            except Exception as exc:
                logger.exception("Image generation failed before review on attempt %s.", attempt_index)
                raise RuntimeError(
                    "Image generation backend failed before producing a candidate. "
                    f"Fix the local image runtime and retry. Original error: {exc}"
                ) from exc
            candidate.review = self.critic.review_image(
                news_article,
                frozen_text_result,
                candidate,
            )
            image_candidate_history.append(candidate)

            if candidate.review.review.pass_:
                logger.info("Image candidate %s passed review.", candidate.candidate_id)
                break

        if image_candidate_history:
            selected_image = max(
                image_candidate_history,
                key=lambda item: item.review.review.total_score if item.review else 0.0,
            )
            if selected_image.review and selected_image.review.review.pass_:
                status = "success"
                logger.info(
                    "Selected image candidate %s with score %.2f.",
                    selected_image.candidate_id,
                    selected_image.review.review.total_score,
                )
            else:
                status = "image_failed"
                logger.warning(
                    "All image attempts failed. Reporting highest-scoring failed candidate %s with score %.2f.",
                    selected_image.candidate_id,
                    selected_image.review.review.total_score if selected_image.review else 0.0,
                )
        else:
            status = "image_failed"

        final_item = FinalDatasetItem(
            status=status,
            text_stage_passed=text_passed,
            selected_text_attempt=selected_text_attempt,
            source_article=news_article,
            source_images=source_images,
            grounding=frozen_text_result.grounding,
            true_branch={
                "summary": frozen_text_result.true_summary,
                "images": source_images,
                "primary_image": true_primary_image,
                "final_review": self._serialize(selected_text_review.true_summary_review) if selected_text_review else None,
            },
            fake_branch={
                "fake_frame": frozen_text_result.fake_frame,
                "fake_text": frozen_text_result.fake_text,
                "selected_image": self._serialize(selected_image) if selected_image else None,
                "final_review": self._serialize(selected_text_review.fake_text_review) if selected_text_review else None,
            },
            change_log=frozen_text_result.change_log,
            selected_text_review=self._serialize(selected_text_review) if selected_text_review else None,
            text_reviews=[self._serialize(review) for review in text_reviews],
            image_candidate_history=[self._serialize(candidate) for candidate in image_candidate_history],
            selected_image=self._serialize(selected_image) if selected_image else None,
        )

        final_result = {
            "timestamp": timestamp,
            "output_dir": output_dir,
            "status": status,
            "final_status": status,
            "text_stage_passed": text_passed,
            "selected_text_attempt": selected_text_attempt,
            "source_images": source_images,
            "original_article": news_article,
            "grounding": self._serialize(frozen_text_result.grounding),
            "facts": self._serialize(frozen_text_result.grounding),
            "true_summary": frozen_text_result.true_summary,
            "true_branch": final_item.true_branch,
            "fake_text": frozen_text_result.fake_text,
            "post_text": frozen_text_result.fake_text,
            "fake_branch": final_item.fake_branch,
            "change_log": self._serialize(frozen_text_result.change_log),
            "opposite_claims": frozen_text_result.fake_frame,
            "selected_text_review": final_item.selected_text_review,
            "text_reviews": final_item.text_reviews,
            "critic_result": final_item.selected_text_review,
            "image_candidate_history": final_item.image_candidate_history,
            "selected_image": final_item.selected_image,
            "background_image_path": selected_image.background_image_path if selected_image else None,
            "final_post_path": selected_image.final_post_path if selected_image else None,
            "dataset_item": self._serialize(final_item),
        }

        if save_intermediate:
            self._save_json(
                os.path.join(output_dir, "intermediate_results.json"),
                {
                    "source_article": news_article,
                    "source_images": source_images,
                    "text_result": self._serialize(frozen_text_result),
                    "text_stage_passed": text_passed,
                    "selected_text_attempt": selected_text_attempt,
                    "selected_text_review": final_item.selected_text_review,
                    "text_reviews": final_item.text_reviews,
                    "image_candidate_history": final_item.image_candidate_history,
                    "status": status,
                },
            )

        self._save_json(os.path.join(output_dir, "final_report.json"), final_result)
        return final_result
