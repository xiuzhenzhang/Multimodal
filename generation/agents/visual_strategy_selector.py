"""Configurable visual presentation selector."""
import json
import os
from typing import Optional

from pydantic import BaseModel, Field

from generation.runtime import get_image_format_catalog, render_image_format_catalog, settings
from generation.utils.prompt_templates import VISUAL_STRATEGY_SELECTION_PROMPT


class VisualStrategyResult(BaseModel):
    """Visual presentation result compatible with existing visual helpers."""

    selected_strategy: int = Field(default=0, description="Legacy compatibility field.")
    strategy_name: str = Field(default="", description="Selected format label")
    reasoning: str = Field(default="", description="Why this presentation plan was chosen")
    strategy_details: str = Field(default="", description="Instructions for implementing the format")
    probability_distribution: dict[str, float] = Field(default_factory=dict)


class VisualStrategySelector:
    """Select a configurable image format rather than a hardcoded six-option strategy set."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        runtime_api_key = api_key or settings.synthesis_api_key or settings.deepseek_api_key
        runtime_base_url = base_url or settings.synthesis_base_url or settings.deepseek_base_url

        if not runtime_api_key:
            raise ValueError(
                "A synthesis runtime API key is not set. Please configure SYNTHESIS_API_KEY "
                "or the provider-specific API key in your environment."
            )

        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        self._prompt_builder = ChatPromptTemplate
        original_key = os.environ.get("OPENAI_API_KEY")
        try:
            os.environ["OPENAI_API_KEY"] = runtime_api_key
            self.llm = ChatOpenAI(
                model=model_name or settings.synthesis_model or settings.deepseek_model,
                api_key=runtime_api_key,
                base_url=runtime_base_url,
                temperature=0.6,
            )
        finally:
            if original_key is not None:
                os.environ["OPENAI_API_KEY"] = original_key
            elif "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

    def select_strategy(self, transformer_output) -> VisualStrategyResult:
        """Choose a configurable presentation plan from the image format catalog."""
        prompt = self._prompt_builder.from_template(VISUAL_STRATEGY_SELECTION_PROMPT)
        messages = prompt.format_messages(
            mirrored_article=transformer_output.mirrored_article,
            post_text=transformer_output.post_text,
            opposite_claims=transformer_output.opposite_claims,
            format_catalog=render_image_format_catalog(),
        )
        response = self.llm.invoke(messages)
        content = response.content.strip()

        try:
            if "```json" in content:
                content = content.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in content:
                content = content.split("```", 1)[1].split("```", 1)[0].strip()
            strategy_dict = json.loads(content)
            if "probability_distribution" not in strategy_dict:
                strategy_dict["probability_distribution"] = {"selected": 1.0}
            return VisualStrategyResult(**strategy_dict)
        except Exception:
            first_format = get_image_format_catalog()[0]
            return VisualStrategyResult(
                selected_strategy=0,
                strategy_name=first_format.label,
                reasoning="Defaulted to the first configured format after parsing failed.",
                strategy_details=first_format.description,
                probability_distribution={"selected": 1.0},
            )

    def get_strategy_description(self, strategy_num: int) -> str:
        """Legacy compatibility helper."""
        _ = strategy_num
        return "Configurable format selection"
