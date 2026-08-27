from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Optional


WORKSPACE_IMAGE_PREFIX = "/workspace/mutil-agent/dataset"
WORKSPACE_ROOT_PREFIX = "/workspace/mutil-agent"


@dataclass
class NewsSample:
    sample_id: str
    dataset_index: int
    headline: str
    body: str
    full_text: str
    image_path: Path
    label: Optional[int]
    source: Optional[str]
    raw_item: dict[str, Any]


def split_headline_and_body(post_text: str) -> tuple[str, str]:
    normalized = (post_text or "").replace("\r\n", "\n").strip()
    if not normalized:
        return "", ""

    if "\n\n" in normalized:
        headline, body = normalized.split("\n\n", 1)
        return headline.strip(), body.strip()

    lines = [line.strip() for line in normalized.split("\n") if line.strip()]
    if not lines:
        return "", ""
    if len(lines) == 1:
        return lines[0], ""
    return lines[0], "\n".join(lines[1:]).strip()


def as_pure_posix_path(path_value: str) -> PurePosixPath:
    return PurePosixPath((path_value or "").replace("\\", "/"))


def extract_text_fields(item: dict[str, Any]) -> tuple[str, str, str]:
    post_text = (item.get("post_text", "") or "").strip()
    if post_text:
        headline, body = split_headline_and_body(post_text)
        return headline, body, post_text

    news_content = (item.get("news_content", "") or "").strip()
    if news_content:
        return "", news_content, news_content

    return "", "", ""


def resolve_image_path(
    image_value: str,
    dataset_root: Optional[Path],
    workspace_prefix: str = WORKSPACE_IMAGE_PREFIX,
) -> Path:
    candidate = Path(image_value)
    if candidate.exists():
        return candidate.resolve()

    posix_value = as_pure_posix_path(image_value)
    if dataset_root:
        prefix_parts = PurePosixPath(workspace_prefix).parts
        if posix_value.parts[: len(prefix_parts)] == prefix_parts:
            relative_parts = posix_value.parts[len(prefix_parts) :]
            mapped = dataset_root.joinpath(*relative_parts)
            if mapped.exists():
                return mapped.resolve()

        workspace_root_parts = PurePosixPath(WORKSPACE_ROOT_PREFIX).parts
        if posix_value.parts[: len(workspace_root_parts)] == workspace_root_parts:
            relative_parts = posix_value.parts[len(workspace_root_parts) :]
            mapped = dataset_root.parent.joinpath(*relative_parts)
            if mapped.exists():
                return mapped.resolve()

        if not candidate.is_absolute():
            mapped = dataset_root / image_value
            if mapped.exists():
                return mapped.resolve()

        mapped = dataset_root.joinpath(*posix_value.parts[-2:])
        if mapped.exists():
            return mapped.resolve()

    return candidate


def infer_source(item: dict[str, Any], image_value: str) -> Optional[str]:
    source = item.get("source")
    if isinstance(source, str) and source.strip():
        return source.strip().lower()
    path_parts = [part.lower() for part in as_pure_posix_path(image_value).parts]
    aliases = {"nature": ("nature",), "nih": ("nih",), "snopes": ("snopes", "snope")}
    for source_name, source_aliases in aliases.items():
        if any(
            part == alias or part.startswith(f"{alias}_")
            for alias in source_aliases
            for part in path_parts
        ):
            return source_name
    return None


def derive_sample_id(index: int, image_value: str, include_filename: bool = False) -> str:
    if image_value:
        posix_value = as_pure_posix_path(image_value)
        if include_filename and posix_value.parent.name and posix_value.stem:
            return f"{posix_value.parent.name}__{posix_value.stem}"
        if posix_value.parent.name:
            return posix_value.parent.name
        if posix_value.stem:
            return posix_value.stem
    return f"sample_{index:06d}"


def load_samples(
    dataset_path: Path,
    dataset_root: Optional[Path] = None,
    limit: Optional[int] = None,
    offset: int = 0,
    workspace_prefix: str = WORKSPACE_IMAGE_PREFIX,
) -> list[NewsSample]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {dataset_path}, got {type(data).__name__}")

    sliced = data[offset:]
    if limit is not None:
        sliced = sliced[:limit]

    samples: list[NewsSample] = []
    for local_index, item in enumerate(sliced, start=offset):
        headline, body, full_text = extract_text_fields(item)
        image_value = item.get("image_path", "") or ""
        image_path = resolve_image_path(
            image_value,
            dataset_root=dataset_root,
            workspace_prefix=workspace_prefix,
        )

        label = item.get("label")
        if label not in (0, 1):
            label = None

        source = infer_source(item, image_value)

        explicit_sample_id = item.get("sample_id")
        if not isinstance(explicit_sample_id, str) or not explicit_sample_id.strip():
            explicit_sample_id = None

        samples.append(
            NewsSample(
                sample_id=(
                    explicit_sample_id.strip()
                    if explicit_sample_id
                    else derive_sample_id(
                        local_index,
                        image_value,
                        include_filename="news_content" in item,
                    )
                ),
                dataset_index=local_index,
                headline=headline,
                body=body,
                full_text=full_text,
                image_path=image_path,
                label=label,
                source=source,
                raw_item=item,
            )
        )

    return samples
