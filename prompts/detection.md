# Appendix: Evaluation Prompts

This appendix lists the two prompt templates used for multimodal fake-news detection. Each model receives the news text and the associated image. The label space is binary: `0` denotes true news and `1` denotes fake news.

## B.1 Shared News Input Format

When a headline and body are available, the news input is formatted as follows:

```text
News sample:
Headline:
{headline}

Body:
{body}

A related image is attached.
```

When the item is represented as a single text field, the input is formatted as follows:

```text
News sample:
Content:
{content}

A related image is attached.
```

## B.2 Direct Judgment Prompt

```text
You are evaluating whether a multimodal news post is true news or fake news.

Label mapping:
- 0 = true news
- 1 = fake news

{news_sample}

Task:
1. Look at the headline, body, and image together.
2. Make a direct judgment: true news or fake news.

Output rules:
- Return JSON only.
- Do not use markdown fences.
- Do not add any natural-language explanation outside JSON.
- Keep the output minimal and easy to parse.

JSON schema:
{
  "verdict_label": 0 or 1,
  "verdict_name": "true_news" or "fake_news",
  "confidence": number between 0 and 1
}
```

## B.3 Chain-of-Thought Prompt

```text
You are evaluating whether a multimodal news post is true news or fake news.
Use a structured chain-of-thought style response, but keep each reasoning field concise and explicit.

Label mapping:
- 0 = true news
- 1 = fake news

{news_sample}

Required reasoning procedure:
1. Complete these four checks exactly:
1. Factual Errors (factual_error): Does the text appear to contain authentic information, without erroneous or conflicting details about people, events, dates, numbers, places, or outcomes? [no = appears factual, yes = appears to contain errors]
2. Language Issues (language_issue): Does the text have noticeable language problems, such as awkward wording, broken grammar, unnatural style, machine-generated tone, or exaggerated / biased wording? [no = no obvious language issues, yes = has language issues]
3. Image Relevance (image_relevance): Is the image relevant to the headline and body, or does it appear unrelated? [yes = relevant, no = not relevant]
4. Image Authenticity (image_authenticity): Does the image look like a real image, or does it appear AI-generated or visually fake? [yes = authentic, no = not authentic]
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
{
  "verdict_label": 0 or 1,
  "verdict_name": "true_news" or "fake_news",
  "confidence": number between 0 and 1,
  "four_checks": {
    "factual_error": "yes" or "no",
    "language_issue": "yes" or "no",
    "image_relevance": "yes" or "no",
    "image_authenticity": "yes" or "no"
  },
  "has_other_reasons": true or false
}

Required output format:
<think>
concise chain-of-thought here
</think>
{
  "verdict_label": 0 or 1,
  "verdict_name": "true_news" or "fake_news",
  "confidence": 0.0,
  "four_checks": {
    "factual_error": "yes" or "no",
    "language_issue": "yes" or "no",
    "image_relevance": "yes" or "no",
    "image_authenticity": "yes" or "no"
  },
  "has_other_reasons": true or false
}
```

## B.4 Four Diagnostic Checks

The CoT prompt decomposes the decision into four diagnostic checks:

| Check | Meaning | Allowed values |
|---|---|---|
| `factual_error` | Whether the text appears to contain erroneous or conflicting factual details. | `yes` = appears to contain errors; `no` = appears factual |
| `language_issue` | Whether the text contains awkward wording, broken grammar, unnatural style, machine-generated tone, or exaggerated/biased wording. | `yes` = has language issues; `no` = no obvious language issues |
| `image_relevance` | Whether the image is relevant to the news text. | `yes` = relevant; `no` = not relevant |
| `image_authenticity` | Whether the image appears visually authentic rather than AI-generated or fake. | `yes` = authentic; `no` = not authentic |
