from __future__ import annotations

from textwrap import dedent

from evaluation.dataset import NewsSample


FOUR_CHECKS = [
    (
        "factual_error",
        "Factual Errors",
        "Does the text appear to contain authentic information, without erroneous or conflicting details about people, events, dates, numbers, places, or outcomes?",
        "no = appears factual, yes = appears to contain errors",
    ),
    (
        "language_issue",
        "Language Issues",
        "Does the text have noticeable language problems, such as awkward wording, broken grammar, unnatural style, machine-generated tone, or exaggerated / biased wording?",
        "no = no obvious language issues, yes = has language issues",
    ),
    (
        "image_relevance",
        "Image Relevance",
        "Is the image relevant to the headline and body, or does it appear unrelated?",
        "yes = relevant, no = not relevant",
    ),
    (
        "image_authenticity",
        "Image Authenticity",
        "Does the image look like a real image, or does it appear AI-generated or visually fake?",
        "yes = authentic, no = not authentic",
    ),
]


def _format_news_input(sample: NewsSample) -> str:
    if sample.headline and sample.body:
        return dedent(
            f"""
            News sample:
            Headline:
            {sample.headline}

            Body:
            {sample.body}

            A related image is attached.
            """
        ).strip()

    content_text = sample.full_text or sample.body or sample.headline or "(empty content)"
    return dedent(
        f"""
        News sample:
        Content:
        {content_text}

        A related image is attached.
        """
    ).strip()


def build_direct_prompt(sample: NewsSample) -> str:
    return dedent(
        f"""
        You are evaluating whether a multimodal news post is true news or fake news.

        Label mapping:
        - 0 = true news
        - 1 = fake news

        {_format_news_input(sample)}

        Task:
        1. Look at the headline, body, and image together.
        2. Make a direct judgment: true news or fake news.

        Output rules:
        - Return JSON only.
        - Do not use markdown fences.
        - Do not add any natural-language explanation outside JSON.
        - Keep the output minimal and easy to parse.

        JSON schema:
        {{
          "verdict_label": 0 or 1,
          "verdict_name": "true_news" or "fake_news",
          "confidence": number between 0 and 1
        }}
        """
    ).strip()


def build_reasoning_prompt(sample: NewsSample) -> str:
    checks = "\n".join(
        [
            f"{index}. {title} ({key}): {description} [{value_hint}]"
            for index, (key, title, description, value_hint) in enumerate(FOUR_CHECKS, start=1)
        ]
    )

    return dedent(
        f"""
        You are evaluating whether a multimodal news post is true news or fake news.
        Use a structured chain-of-thought style response, but keep each reasoning field concise and explicit.

        Label mapping:
        - 0 = true news
        - 1 = fake news

        {_format_news_input(sample)}

        Required reasoning procedure:
        1. Complete these four checks exactly:
        {checks}
        2. After the four checks, explain whether there are any other reasons that support or weaken a fake-news judgment.
        3. Give the final verdict.

        Output rules:
        - Return the reasoning content first inside a single <think>...</think> block.
        - After the </think> tag, return the final answer as one JSON object.
        - Do not use markdown fences.
        - Put all detailed explanation inside the <think> block.
        - The final JSON must be minimal, structured, and easy to parse.
        - Values in the final JSON must match the allowed yes/no choices described above.
        - The <think> block should explicitly include:
          1. the four checks,
          2. other possible reasons,
          3. how you reached the final verdict.
        - The final JSON must appear after the closing </think> tag.

        Final JSON schema:
        {{
          "verdict_label": 0 or 1,
          "verdict_name": "true_news" or "fake_news",
          "confidence": number between 0 and 1,
          "four_checks": {{
            "factual_error": "yes" or "no",
            "language_issue": "yes" or "no",
            "image_relevance": "yes" or "no",
            "image_authenticity": "yes" or "no"
          }},
          "has_other_reasons": true or false
        }}

        Required output format:
        <think>
        concise chain-of-thought here
        </think>
        {{
          "verdict_label": 0 or 1,
          "verdict_name": "true_news" or "fake_news",
          "confidence": 0.0,
          "four_checks": {{
            "factual_error": "yes" or "no",
            "language_issue": "yes" or "no",
            "image_relevance": "yes" or "no",
            "image_authenticity": "yes" or "no"
          }},
          "has_other_reasons": true or false
        }}
        """
    ).strip()
