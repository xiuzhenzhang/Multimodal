"""Shared LLM helpers for synthesis and critic agents."""
import base64
import json
import mimetypes
import os
import re
from typing import Type, TypeVar

from pydantic import BaseModel

from generation.runtime import ModelEndpointConfig

ModelT = TypeVar("ModelT", bound=BaseModel)


def _extract_json_block(content: str) -> str:
    normalized = content.strip()
    if "```json" in normalized:
        normalized = normalized.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in normalized:
        normalized = normalized.split("```", 1)[1].split("```", 1)[0].strip()

    first_brace = normalized.find("{")
    last_brace = normalized.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return normalized[first_brace : last_brace + 1].strip()
    return normalized


def _sanitize_json_text(content: str) -> str:
    normalized = _extract_json_block(content)
    normalized = normalized.replace("\ufeff", "")
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", normalized)
    normalized = re.sub(r",(\s*[}\]])", r"\1", normalized)
    return normalized.strip()


def build_chat_model(endpoint: ModelEndpointConfig, temperature: float):
    """Build a ChatOpenAI-compatible client for a configured endpoint."""
    if not endpoint.api_key:
        raise ValueError(
            f"{endpoint.provider} API key is not configured for model {endpoint.model_id}. "
            "Please update the runtime model configuration in settings/.env."
        )

    from langchain_openai import ChatOpenAI

    original_key = os.environ.get("OPENAI_API_KEY")
    try:
        os.environ["OPENAI_API_KEY"] = endpoint.api_key
        return ChatOpenAI(
            model=endpoint.model_id,
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            temperature=temperature,
        )
    finally:
        if original_key is not None:
            os.environ["OPENAI_API_KEY"] = original_key
        elif "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]


def parse_json_response(content: str, model_cls: Type[ModelT]) -> ModelT:
    """Parse a JSON object from a fenced or plain-text LLM response."""
    normalized = _extract_json_block(content)
    try:
        return model_cls.model_validate(json.loads(normalized))
    except (json.JSONDecodeError, ValueError):
        repaired = _sanitize_json_text(content)
        return model_cls.model_validate(json.loads(repaired))


def encode_image_file_to_data_url(image_path: str) -> str:
    """Encode a local image file into a data URL for multimodal API calls."""
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "image/png"
    with open(image_path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def invoke_json_with_image(chat_model, prompt_text: str, image_path: str, model_cls: Type[ModelT]) -> ModelT:
    """Run a multimodal JSON request with a real image file attached."""
    from langchain_core.messages import HumanMessage

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": encode_image_file_to_data_url(image_path)}},
        ]
    )
    response = chat_model.invoke([message])
    return parse_json_response(response.content, model_cls)
