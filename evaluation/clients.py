from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelConfig:
    name: str
    backend: str
    model_id: str
    model_path: Optional[str] = None
    processor_path: Optional[str] = None
    inference_profile: str = "default"
    temperature: float = 0.0
    max_tokens: int = 900
    timeout: int = 180
    api_key_env: Optional[str] = None
    base_url_env: Optional[str] = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    trust_remote_code: bool = False
    system_prompt: str = ""
    content_order: str = "image_first"
    image_detail: Optional[str] = None
    strip_think_tags: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    processor_kwargs: dict[str, Any] = field(default_factory=dict)
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    local_files_only: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any], base_dir: Optional[Path] = None) -> "ModelConfig":
        return cls(
            name=payload["name"],
            backend=payload["backend"],
            model_id=payload["model_id"],
            model_path=resolve_local_reference(payload.get("model_path"), base_dir),
            processor_path=resolve_local_reference(payload.get("processor_path"), base_dir),
            inference_profile=payload.get("inference_profile", "default"),
            temperature=float(payload.get("temperature", 0.0)),
            max_tokens=int(payload.get("max_tokens", 900)),
            timeout=int(payload.get("timeout", 180)),
            api_key_env=payload.get("api_key_env"),
            base_url_env=payload.get("base_url_env"),
            device_map=payload.get("device_map", "auto"),
            torch_dtype=payload.get("torch_dtype", "auto"),
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
            system_prompt=payload.get("system_prompt", "") or "",
            content_order=payload.get("content_order", "image_first"),
            image_detail=payload.get("image_detail"),
            strip_think_tags=bool(payload.get("strip_think_tags", False)),
            chat_template_kwargs=payload.get("chat_template_kwargs", {}) or {},
            generation_kwargs=payload.get("generation_kwargs", {}) or {},
            processor_kwargs=payload.get("processor_kwargs", {}) or {},
            model_kwargs=payload.get("model_kwargs", {}) or {},
            local_files_only=bool(payload.get("local_files_only", False)),
        )

    def resolved_api_key(self) -> str:
        if self.api_key_env:
            return os.getenv(self.api_key_env, "")
        return ""

    def resolved_base_url(self) -> Optional[str]:
        if self.base_url_env:
            return os.getenv(self.base_url_env, "")
        return None

    def resolved_model_source(self) -> str:
        return self.model_path or self.model_id

    def resolved_processor_source(self) -> str:
        return self.processor_path or self.resolved_model_source()


class BaseInferenceClient(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(
        self,
        prompt_text: str,
        image_path: Path,
        preserve_think_blocks: bool = False,
    ) -> str:
        raise NotImplementedError

    def _postprocess(self, text: str, preserve_think_blocks: bool) -> str:
        return postprocess_model_output(
            text,
            self.config,
            preserve_think_blocks=preserve_think_blocks,
        )


class OpenAICompatibleClient(BaseInferenceClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        api_key = config.resolved_api_key()
        if not api_key:
            env_name = config.api_key_env or "API key"
            raise ValueError(f"Model {config.name} requires {env_name} to be set.")

        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url=config.resolved_base_url() or None,
            timeout=config.timeout,
        )

    def generate(
        self,
        prompt_text: str,
        image_path: Path,
        preserve_think_blocks: bool = False,
    ) -> str:
        content = [{"type": "text", "text": prompt_text}]
        if image_path.exists():
            image_url: dict[str, Any] = {"url": encode_image_file_to_data_url(image_path)}
            if self.config.image_detail:
                image_url["detail"] = self.config.image_detail
            content.append(
                {
                    "type": "image_url",
                    "image_url": image_url,
                }
            )

        request_body = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": content}],
        }
        if supports_custom_temperature(self.config.model_id):
            request_body["temperature"] = self.config.temperature
        request_body[token_limit_parameter(self.config.model_id)] = self.config.max_tokens
        response = self.client.chat.completions.create(**request_body)
        return self._postprocess(response.choices[0].message.content or "", preserve_think_blocks)


class TransformersVisionLanguageClient(BaseInferenceClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoProcessor

        self._torch = torch
        model_source = config.resolved_model_source()
        processor_source = config.resolved_processor_source()
        self.processor = AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
            **config.processor_kwargs,
        )
        self.model = load_first_supported_model(
            model_source=model_source,
            config=config,
            torch_module=torch,
        )

    def generate(
        self,
        prompt_text: str,
        image_path: Path,
        preserve_think_blocks: bool = False,
    ) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = build_transformers_messages(self.config, prompt_text)
        chat_text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.config.chat_template_kwargs,
        )
        inputs = self.processor(images=image, text=chat_text, return_tensors="pt")
        inputs = move_to_model_device(inputs, self.model)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_tokens,
        }
        if self.config.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.config.temperature
        else:
            generation_kwargs["do_sample"] = False
        generation_kwargs.update(self.config.generation_kwargs)

        outputs = self.model.generate(**inputs, **generation_kwargs)
        prompt_tokens = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
        completion_tokens = outputs[:, prompt_tokens:]
        decoded = self.processor.batch_decode(
            completion_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return self._postprocess(decoded[0] if decoded else "", preserve_think_blocks)


class TransformersPipelineVisionLanguageClient(BaseInferenceClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import pipeline

        model_kwargs = dict(config.model_kwargs)
        model_kwargs.setdefault("torch_dtype", resolve_torch_dtype(config.torch_dtype, torch))

        self.pipe = pipeline(
            task="image-text-to-text",
            model=config.resolved_model_source(),
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code,
            model_kwargs=model_kwargs,
        )

    def generate(
        self,
        prompt_text: str,
        image_path: Path,
        preserve_think_blocks: bool = False,
    ) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        outputs = self.pipe(
            text=messages,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            return_full_text=False,
        )
        text = extract_pipeline_text(outputs)
        return self._postprocess(text, preserve_think_blocks)


class MiniCPMVisionLanguageClient(BaseInferenceClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        model_source = config.resolved_model_source()
        self.model = AutoModel.from_pretrained(
            model_source,
            trust_remote_code=config.trust_remote_code,
            device_map=config.device_map,
            torch_dtype=resolve_torch_dtype(config.torch_dtype, torch),
            local_files_only=config.local_files_only,
            **config.model_kwargs,
        )
        if hasattr(self.model, "eval"):
            self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.resolved_processor_source(),
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
            **config.processor_kwargs,
        )

    def generate(
        self,
        prompt_text: str,
        image_path: Path,
        preserve_think_blocks: bool = False,
    ) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        msgs = [{"role": "user", "content": prompt_text}]
        sampling = self.config.temperature > 0
        chat_kwargs: dict[str, Any] = {
            "image": image,
            "msgs": msgs,
            "tokenizer": self.tokenizer,
            "sampling": sampling,
        }
        if sampling:
            chat_kwargs["temperature"] = self.config.temperature
        result = self.model.chat(
            **chat_kwargs,
        )
        text = extract_chat_result_text(result)
        return self._postprocess(text, preserve_think_blocks)


class DeepSeekVisionLanguageClient(BaseInferenceClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModelForCausalLM
        from deepseek_vl.models import VLChatProcessor

        self._torch = torch
        model_source = config.resolved_model_source()
        self.processor = VLChatProcessor.from_pretrained(model_source)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=True,
            local_files_only=config.local_files_only,
            **config.model_kwargs,
        )
        dtype = resolve_torch_dtype(config.torch_dtype, torch)
        if dtype == "auto" and torch.cuda.is_available() and hasattr(torch, "bfloat16"):
            # The official DeepSeek-VL inference path moves the full model to bf16.
            # Keeping mixed fp32 / bf16 modules causes image-encoder bias mismatches.
            dtype = torch.bfloat16
        if dtype != "auto":
            self.model = self.model.to(dtype)
        if hasattr(self.model, "eval"):
            self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def generate(
        self,
        prompt_text: str,
        image_path: Path,
        preserve_think_blocks: bool = False,
    ) -> str:
        from deepseek_vl.utils.io import load_pil_images

        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{prompt_text}",
                "images": [str(image_path)],
            },
            {"role": "Assistant", "content": ""},
        ]
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
        ).to(self.model.device)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        generation_kwargs: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": prepare_inputs.attention_mask,
            "pad_token_id": self.tokenizer.eos_token_id,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": self.config.max_tokens,
            "do_sample": self.config.temperature > 0,
            "use_cache": True,
        }
        if self.config.temperature > 0:
            generation_kwargs["temperature"] = self.config.temperature
        outputs = self.model.language_model.generate(**generation_kwargs)
        text = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return self._postprocess(strip_prompt_echo(text, prepare_inputs["sft_format"][0]), preserve_think_blocks)


class CambrianVisionLanguageClient(BaseInferenceClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        import torch
        from cambrian.mm_utils import get_model_name_from_path
        from cambrian.model.builder import load_pretrained_model

        self._torch = torch
        model_source = config.resolved_model_source()
        model_name = get_model_name_from_path(model_source)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_source,
            None,
            model_name,
        )
        if hasattr(self.model, "eval"):
            self.model = self.model.eval()
        self.conv_mode = config.model_kwargs.get("conv_mode", "llama_3")

    def generate(
        self,
        prompt_text: str,
        image_path: Path,
        preserve_think_blocks: bool = False,
    ) -> str:
        from PIL import Image
        from cambrian.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from cambrian.conversation import conv_templates
        from cambrian.mm_utils import process_images, tokenizer_image_token

        image = Image.open(image_path).convert("RGB")
        qs = prompt_text
        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_sizes = [image.size]
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0)

        if self._torch.cuda.is_available():
            input_ids = input_ids.to(device="cuda", non_blocking=True)

        with self._torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=self.config.temperature > 0,
                temperature=self.config.temperature if self.config.temperature > 0 else 0,
                num_beams=1,
                max_new_tokens=self.config.max_tokens,
                use_cache=True,
            )

        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return self._postprocess(text, preserve_think_blocks)


class InternVLVisionLanguageClient(BaseInferenceClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        model_source = config.resolved_model_source()
        self.model = AutoModel.from_pretrained(
            model_source,
            trust_remote_code=config.trust_remote_code,
            device_map=config.device_map,
            torch_dtype=resolve_torch_dtype(config.torch_dtype, torch),
            local_files_only=config.local_files_only,
            low_cpu_mem_usage=True,
            **config.model_kwargs,
        )
        if hasattr(self.model, "eval"):
            self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.resolved_processor_source(),
            trust_remote_code=config.trust_remote_code,
            use_fast=False,
            local_files_only=config.local_files_only,
            **config.processor_kwargs,
        )

    def generate(
        self,
        prompt_text: str,
        image_path: Path,
        preserve_think_blocks: bool = False,
    ) -> str:
        pixel_values = load_internvl_image(image_path, torch_module=self._torch)
        dtype = infer_model_tensor_dtype(self.model, self._torch)
        if self._torch.cuda.is_available():
            pixel_values = pixel_values.to(device="cuda", dtype=dtype)
        else:
            pixel_values = pixel_values.to(dtype=dtype)

        generation_config: dict[str, Any] = {
            "max_new_tokens": self.config.max_tokens,
            "do_sample": self.config.temperature > 0,
        }
        if self.config.temperature > 0:
            generation_config["temperature"] = self.config.temperature
        generation_config.update(self.config.generation_kwargs)

        result = self.model.chat(
            self.tokenizer,
            pixel_values,
            "<image>\n" + prompt_text,
            generation_config,
        )
        text = extract_chat_result_text(result)
        return self._postprocess(text, preserve_think_blocks)


class TransformersTextClient(BaseInferenceClient):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        model_source = config.resolved_model_source()
        tokenizer_source = config.resolved_processor_source()
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=config.trust_remote_code,
            local_files_only=config.local_files_only,
            **config.processor_kwargs,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=config.trust_remote_code,
            device_map=config.device_map,
            torch_dtype=resolve_torch_dtype(config.torch_dtype, torch),
            local_files_only=config.local_files_only,
            **config.model_kwargs,
        )

    def generate(
        self,
        prompt_text: str,
        image_path: Path,
        preserve_think_blocks: bool = False,
    ) -> str:
        messages = build_text_messages(self.config, prompt_text)
        tokenizer = self.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.config.chat_template_kwargs,
            )
        else:
            prompt = "\n\n".join(part["content"] for part in messages if isinstance(part.get("content"), str))

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = move_to_model_device(inputs, self.model)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_tokens,
        }
        if self.config.temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = self.config.temperature
        else:
            generation_kwargs["do_sample"] = False
        generation_kwargs.update(self.config.generation_kwargs)

        outputs = self.model.generate(**inputs, **generation_kwargs)
        prompt_tokens = inputs["input_ids"].shape[-1]
        completion_tokens = outputs[:, prompt_tokens:]
        decoded = tokenizer.batch_decode(
            completion_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return self._postprocess(decoded[0] if decoded else "", preserve_think_blocks)


def build_transformers_messages(config: ModelConfig, prompt_text: str) -> list[dict[str, Any]]:
    system_prompt = resolve_system_prompt(config)
    content: list[dict[str, Any]]
    if config.content_order == "text_first":
        content = [{"type": "text", "text": prompt_text}, {"type": "image"}]
    else:
        content = [{"type": "image"}, {"type": "text", "text": prompt_text}]

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    return messages


def build_text_messages(config: ModelConfig, prompt_text: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    system_prompt = resolve_system_prompt(config)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt_text})
    return messages


def resolve_system_prompt(config: ModelConfig) -> str:
    if config.system_prompt:
        return config.system_prompt
    if config.inference_profile == "qwen3_vl":
        return (
            "You are a careful multimodal fake-news detection assistant. "
            "Follow the user's requested output format exactly. "
            "Do not add extra prose outside the requested format."
        )
    return ""


def resolve_torch_dtype(dtype_name: str, torch_module: Any) -> Any:
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "float32": torch_module.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[dtype_name]


def infer_model_tensor_dtype(model: Any, torch_module: Any) -> Any:
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    try:
        return next(model.parameters()).dtype
    except (AttributeError, StopIteration, TypeError):
        return torch_module.float32


def move_to_model_device(inputs: Any, model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is None and hasattr(model, "hf_device_map"):
        return inputs
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def encode_image_file_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/png"
    with image_path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def load_first_supported_model(model_source: str, config: ModelConfig, torch_module: Any) -> Any:
    from transformers import AutoModel, AutoModelForCausalLM

    candidates: list[Any] = []
    try:
        from transformers import AutoModelForImageTextToText

        candidates.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    try:
        from transformers import InternVLForConditionalGeneration

        candidates.append(InternVLForConditionalGeneration)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        candidates.append(AutoModelForVision2Seq)
    except ImportError:
        pass
    candidates.extend([AutoModelForCausalLM, AutoModel])

    errors: list[str] = []
    for model_cls in candidates:
        try:
            return model_cls.from_pretrained(
                model_source,
                trust_remote_code=config.trust_remote_code,
                device_map=config.device_map,
                torch_dtype=resolve_torch_dtype(config.torch_dtype, torch_module),
                local_files_only=config.local_files_only,
                **config.model_kwargs,
            )
        except Exception as error:  # noqa: BLE001
            errors.append(f"{model_cls.__name__}: {type(error).__name__}: {error}")
    raise ValueError("Could not load model with supported Transformers classes:\n" + "\n".join(errors))


def extract_pipeline_text(outputs: Any) -> str:
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            generated = first.get("generated_text")
            if isinstance(generated, str):
                return generated
            if isinstance(generated, list) and generated:
                last = generated[-1]
                if isinstance(last, dict):
                    content = last.get("content")
                    if isinstance(content, str):
                        return content
        if isinstance(first, str):
            return first
    if isinstance(outputs, dict):
        value = outputs.get("generated_text")
        if isinstance(value, str):
            return value
    return str(outputs or "")


def extract_chat_result_text(result: Any) -> str:
    if isinstance(result, tuple) and result:
        first = result[0]
        if isinstance(first, str):
            return first
    if isinstance(result, str):
        return result
    return str(result or "")


def build_internvl_transform():
    from torchvision import transforms as T
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )


def find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio = (1, 1)
    best_diff = float("inf")
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_diff:
            best_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess_internvl_image(
    image: Any,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Any]:
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda ratio: ratio[0] * ratio[1],
    )
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio=aspect_ratio,
        target_ratios=target_ratios,
        width=orig_width,
        height=orig_height,
        image_size=image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized = image.resize((target_width, target_height))
    processed_images = []
    tiles_per_row = target_width // image_size
    for index in range(blocks):
        box = (
            (index % tiles_per_row) * image_size,
            (index // tiles_per_row) * image_size,
            ((index % tiles_per_row) + 1) * image_size,
            ((index // tiles_per_row) + 1) * image_size,
        )
        processed_images.append(resized.crop(box))

    if use_thumbnail and len(processed_images) > 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def load_internvl_image(image_path: Path, torch_module: Any, input_size: int = 448, max_num: int = 12) -> Any:
    from PIL import Image

    transform = build_internvl_transform()
    image = Image.open(image_path).convert("RGB")
    tiles = dynamic_preprocess_internvl_image(
        image=image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num,
    )
    pixel_values = [transform(tile) for tile in tiles]
    return torch_module.stack(pixel_values)


def strip_prompt_echo(text: str, prompt: str) -> str:
    normalized_text = text or ""
    if prompt and normalized_text.startswith(prompt):
        return normalized_text[len(prompt) :].strip()
    return normalized_text.strip()


def build_client(config: ModelConfig) -> BaseInferenceClient:
    backend = config.backend.lower()
    if backend == "openai_compatible":
        return OpenAICompatibleClient(config)
    if backend == "transformers_vlm":
        return TransformersVisionLanguageClient(config)
    if backend == "transformers_pipeline_vlm":
        return TransformersPipelineVisionLanguageClient(config)
    if backend == "transformers_text":
        return TransformersTextClient(config)
    if backend == "minicpm_vlm":
        return MiniCPMVisionLanguageClient(config)
    if backend == "deepseek_vl":
        return DeepSeekVisionLanguageClient(config)
    if backend == "cambrian_vlm":
        return CambrianVisionLanguageClient(config)
    if backend == "internvl_vlm":
        return InternVLVisionLanguageClient(config)
    raise ValueError(f"Unsupported backend: {config.backend}")


def resolve_local_reference(value: Any, base_dir: Optional[Path]) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    if base_dir is None:
        return str(candidate)
    return str((base_dir / candidate).resolve())


def token_limit_parameter(model_id: str) -> str:
    normalized = model_id.lower()
    if normalized.startswith("gpt-5") or normalized.startswith("o1") or normalized.startswith("o3"):
        return "max_completion_tokens"
    return "max_tokens"


def supports_custom_temperature(model_id: str) -> bool:
    normalized = model_id.lower()
    return not (
        normalized.startswith("gpt-5")
        or normalized.startswith("o1")
        or normalized.startswith("o3")
    )


def postprocess_model_output(
    text: str,
    config: ModelConfig,
    preserve_think_blocks: bool = False,
) -> str:
    normalized = text or ""
    should_strip = config.strip_think_tags or config.inference_profile == "qwen3_vl"
    if should_strip and not preserve_think_blocks:
        normalized = strip_think_blocks(normalized)
    return normalized.strip()


def strip_think_blocks(text: str) -> str:
    without_blocks = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"\s+", " ", without_blocks).strip()


def extract_think_blocks(text: str) -> list[str]:
    if not text:
        return []
    blocks = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    return [re.sub(r"\s+", " ", block).strip() for block in blocks if block and block.strip()]


def extract_cot_text(text: str) -> Optional[str]:
    blocks = extract_think_blocks(text)
    if not blocks:
        return None
    return "\n\n".join(blocks)


def normalize_confidence(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def derive_verdict_name(label: Optional[int], parsed_output: Optional[dict[str, Any]]) -> Optional[str]:
    if parsed_output and isinstance(parsed_output.get("verdict_name"), str):
        value = parsed_output["verdict_name"].strip().lower()
        if value in {"true_news", "true news"}:
            return "true_news"
        if value in {"fake_news", "fake news"}:
            return "fake_news"
    if label == 0:
        return "true_news"
    if label == 1:
        return "fake_news"
    return None


def normalize_check_value(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"yes", "no"}:
            return normalized
    return None


def normalize_four_checks(parsed_output: Optional[dict[str, Any]]) -> Optional[dict[str, Optional[str]]]:
    if not parsed_output:
        return None
    raw_checks = parsed_output.get("four_checks")
    if not isinstance(raw_checks, dict):
        return None
    keys = [
        "factual_error",
        "language_issue",
        "image_relevance",
        "image_authenticity",
    ]
    normalized = {key: normalize_check_value(raw_checks.get(key)) for key in keys}
    if not any(value is not None for value in normalized.values()):
        return None
    return normalized


def normalize_has_other_reasons(parsed_output: Optional[dict[str, Any]]) -> Optional[bool]:
    if not parsed_output:
        return None
    value = parsed_output.get("has_other_reasons")
    if isinstance(value, bool):
        return value
    legacy_value = parsed_output.get("other_reasons")
    if isinstance(legacy_value, str):
        return bool(legacy_value.strip())
    return None


def build_final_output(
    parsed_output: Optional[dict[str, Any]],
    raw_text: str,
    mode: str,
) -> Optional[dict[str, Any]]:
    verdict_label = extract_prediction(parsed_output, raw_text)
    verdict_name = derive_verdict_name(verdict_label, parsed_output)
    confidence = normalize_confidence(parsed_output.get("confidence")) if parsed_output else None

    if verdict_label is None and verdict_name is None and confidence is None and not parsed_output:
        return None

    final_output: dict[str, Any] = {
        "verdict_label": verdict_label,
        "verdict_name": verdict_name,
        "confidence": confidence,
    }

    if mode == "reasoning":
        final_output["four_checks"] = normalize_four_checks(parsed_output)
        final_output["has_other_reasons"] = normalize_has_other_reasons(parsed_output)

    return final_output


def extract_json_payload(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None

    stripped_think = strip_think_blocks(text)
    candidates = [text.strip()]
    if stripped_think and stripped_think not in candidates:
        candidates.append(stripped_think)
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(fenced)

    for candidate in candidates:
        parsed = try_load_json(candidate)
        if isinstance(parsed, dict):
            return parsed

        for start_index, char in enumerate(candidate):
            if char != "{":
                continue
            brace_depth = 0
            in_string = False
            escape_next = False
            for end_index in range(start_index, len(candidate)):
                current = candidate[end_index]
                if in_string:
                    if escape_next:
                        escape_next = False
                    elif current == "\\":
                        escape_next = True
                    elif current == '"':
                        in_string = False
                    continue

                if current == '"':
                    in_string = True
                elif current == "{":
                    brace_depth += 1
                elif current == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        snippet = candidate[start_index : end_index + 1]
                        parsed = try_load_json(snippet)
                        if isinstance(parsed, dict):
                            return parsed
                        break
    return None


def try_load_json(text: str) -> Optional[Any]:
    candidates = [text]
    if r"\_" in text:
        candidates.append(text.replace(r"\_", "_"))
    candidates.append(re.sub(r",(\s*[}\]])", r"\1", text.strip()))
    if r"\_" in candidates[-1]:
        candidates.append(candidates[-1].replace(r"\_", "_"))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def extract_prediction(parsed_output: Optional[dict[str, Any]], raw_text: str) -> Optional[int]:
    if parsed_output:
        candidate = parsed_output.get("verdict_label")
        normalized = normalize_prediction(candidate)
        if normalized is not None:
            return normalized

    if not raw_text:
        return None

    lower_text = raw_text.lower()
    regex_checks = [
        (r'"verdict_label"\s*:\s*([01])', None),
        (r"\bthe answer is\s*([01])\b", None),
        (r"\bverdict_label\s*[:=]\s*([01])\b", None),
    ]
    for pattern, _ in regex_checks:
        match = re.search(pattern, lower_text)
        if match:
            return int(match.group(1))

    if "true news" in lower_text and "fake news" not in lower_text:
        return 0
    if "fake news" in lower_text and "true news" not in lower_text:
        return 1

    return None


def normalize_prediction(value: Any) -> Optional[int]:
    if value in (0, 1):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"0", "true_news", "true news", "real_news", "real news"}:
            return 0
        if stripped in {"1", "fake_news", "fake news"}:
            return 1
    return None
