# Appendix: Dataset Generation Prompts

This appendix lists the prompt templates used in the dataset construction pipeline. The pipeline first extracts grounded facts from the source article, then generates a faithful true-news summary and a manipulated fake-news counterpart. The generated text is reviewed and revised by a critic agent. For the multimodal branch, the fake-news text is frozen before image generation, and the image is generated and reviewed for consistency with the frozen fake-news narrative.

## A.1 Grounded Fact Extraction

```text
Extract grounded article facts for downstream true-summary and fake-text generation.

Source Article:
{source_article}

Return JSON:
{
  "who": ["person or organization"],
  "what": ["event or action"],
  "when": ["time"],
  "where": ["location"],
  "why": ["reason or motivation"],
  "how": ["mechanism or method"],
  "original_claims": "one compact sentence capturing the article's core frame and claims"
}
```

## A.2 True-News Summary Generation

```text
Convert the source article into a concise social-media-style true summary that stays fully faithful to the source.

Grounding:
{grounding}

Source Article:
{source_article}

Requirements:
- Faithful to the original article
- Social-media/newsfeed style
- Preserve key facts, numbers, dates, places, and actors
- Do not add any unsupported facts
- Stay concise and readable

Return ONLY the summary text.
```

## A.3 Fake-News Text Generation

```text
Generate a fake social-media post derived from the source article.

Grounding:
{grounding}

Source Article:
{source_article}

Business rules:
- First flip the frame or sentiment of the article
- Then edit supporting facts so the new frame is actually supported
- Do NOT only swap adjectives or emotional language
- Keep the fake text anchored to the same topic and entities so it still resembles the source story
- Output a change_log describing the factual edits used to support the new frame

Return JSON:
{
  "fake_frame": "one sentence describing the new false frame",
  "fake_text": "social-media style fake text",
  "change_log": [
    {
      "category": "number/date/actor/location/causal-claim/etc",
      "original": "original fact",
      "revised": "edited supporting fact",
      "rationale": "how the edit supports the new frame"
    }
  ]
}
```

## A.4 Text Revision With Critic Feedback

```text
Revise the current text outputs using critic feedback.

Grounding:
{grounding}

Source Article:
{source_article}

Current true summary:
{true_summary}

Current fake frame:
{fake_frame}

Current fake text:
{fake_text}

Current change_log:
{change_log}

True summary revision advice:
{true_summary_revision_advice}

True summary keep unchanged:
{true_summary_keep_unchanged}

Fake text revision advice:
{fake_text_revision_advice}

Fake text keep unchanged:
{fake_text_keep_unchanged}

Requirements:
- Apply the revision_advice precisely
- Preserve every item in keep_unchanged
- If true summary already passed, keep it stable unless advice says otherwise
- Fake text must still use fact edits that support the flipped frame
- Update change_log when fake facts change

Return JSON:
{
  "true_summary": "revised true summary",
  "fake_frame": "revised or preserved fake frame",
  "fake_text": "revised fake text",
  "change_log": [
    {
      "category": "number/date/actor/location/causal-claim/etc",
      "original": "original fact",
      "revised": "edited supporting fact",
      "rationale": "how the edit supports the new frame"
    }
  ]
}
```

## A.5 True-News Summary Review

```text
Review the true summary against the source article.

Scoring config:
{score_config}

Grounding:
{grounding}

Source Article:
{source_article}

True Summary:
{true_summary}

Scoring guidance:
- Be lenient on style, brevity, and minor phrasing differences
- Prefer passing summaries that remain broadly faithful and useful
- Only use hard_fail_reasons for clear core-fact additions, major contradictions, or wrong key numbers/dates/places
- If the summary is mostly acceptable, give constructive issues but still allow passing scores

Return JSON:
{
  "dimension_scores": {
    "faithfulness": 0-5,
    "key_info_coverage": 0-5,
    "social_fit": 0-5,
    "fluency": 0-5,
    "brevity": 0-5
  },
  "issues": ["issue 1"],
  "revision_advice": ["specific revision advice"],
  "keep_unchanged": ["content that should stay unchanged"],
  "hard_fail_reasons": ["reason that triggers veto if any"]
}
```

## A.6 Fake-News Text Review

```text
Review the fake text against the source article and grounding.

Scoring config:
{score_config}

Grounding:
{grounding}

Source Article:
{source_article}

Fake Frame:
{fake_frame}

Fake Text:
{fake_text}

Change Log:
{change_log}

Scoring guidance:
- Be lenient on tone, polish, and minor plausibility concerns
- Prefer passing drafts when the frame flip is recognizable and the text is mostly coherent
- Reserve hard_fail_reasons for obvious failures such as no real fact edits, severe contradiction, or total topic drift
- If the draft is usable with small improvements, keep issues/advice lightweight and avoid over-penalizing

Return JSON:
{
  "dimension_scores": {
    "frame_flip_success": 0-5,
    "fact_support_for_new_frame": 0-5,
    "internal_consistency": 0-5,
    "plausibility": 0-5,
    "anchor_retention": 0-5,
    "social_fit": 0-5
  },
  "issues": ["issue 1"],
  "revision_advice": ["specific revision advice"],
  "keep_unchanged": ["content that should stay unchanged"],
  "hard_fail_reasons": ["reason that triggers veto if any"]
}
```

## A.7 Image Format Selection

```text
Choose the best image format for illustrating the frozen fake text.

Frozen fake text:
{fake_text}

Fake frame:
{fake_frame}

Grounding:
{grounding}

Available formats:
{format_catalog}

Heuristic guidance:
{selection_guidance}

Recommended formats:
{recommended_formats}

Discouraged formats:
{discouraged_formats}

Requirements:
- Pick the single best format for the fake text narrative
- Prefer formats that can support the key facts and the intended frame
- Do not overuse social_media_screenshot just because the output text is social-media style
- Only choose social_media_screenshot when the narrative itself is explicitly about a post, account, comment thread, leaked screenshot, or platform-native evidence
- Only choose official_notice when the narrative is explicitly about an institutional bulletin, policy notice, memo, or formal announcement card
- Keep the answer practical for image generation

Return JSON:
{
  "selected_format": "one format id from the catalog",
  "rationale": "why this format is the best fit"
}
```

## A.8 Image Prompt Generation

```text
Generate a visual asset for the fake branch.

Inputs:
- Fake frame: {opposite_claims}
- Fake text: {post_text}
- Chosen format: {strategy_name}
- Format details: {strategy_details}
- Article context: {mirrored_article_summary}
- Entities: {entities}
- Narrative cues: {emotions}
- Visual style: {visual_style}
- Palette: {color_palette}

Requirements:
- The image must support the fake text narrative
- The image must fit the chosen format
- Keep the composition practical for downstream poster assembly
- Preserve key entities or facts when useful
- Prefer clean, readable layouts when the chosen format contains text-like content

Output ONLY the image generation prompt text for a {width}x{height} image.
```

## A.9 Image Prompt Revision

```text
Revise the image generation prompt using critic feedback.

Frozen fake text:
{fake_text}

Selected format:
{selected_format}

Current image prompt:
{current_image_prompt}

Image revision advice:
{revision_advice}

Image keep unchanged:
{keep_unchanged}

Requirements:
- Apply the revision advice
- Preserve every item listed in keep_unchanged
- Ensure the prompt supports the frozen fake text
- Keep the selected format explicit
- Improve readability, text-image consistency, and artifact control

Return ONLY the revised image prompt text.
```

## A.10 Generated Image Review

```text
Review the generated image candidate against the frozen fake text using the ACTUAL attached image as the primary evidence.

Scoring config:
{score_config}

Frozen fake text:
{fake_text}

Fake frame:
{fake_frame}

Grounding:
{grounding}

Selected format:
{selected_format}

Image prompt:
{image_prompt}

Image metadata:
{image_metadata}

Supplemental image observation:
{image_observation}

Scoring guidance:
- Be lenient on aesthetics, polish, and minor mismatches
- Prefer passing images that are usable and broadly support the text
- Reserve hard_fail_reasons for obvious problems only: broken images, unreadable critical text, wrong main subject, or clear contradiction to key text facts
- If the image is acceptable but imperfect, mention minor issues without blocking passage

Return JSON:
{
  "review": {
    "dimension_scores": {
      "format_fit": 0-5,
      "visual_quality": 0-5,
      "readability": 0-5,
      "consistency_with_text_narrative": 0-5,
      "support_for_key_text_facts": 0-5,
      "artifact_control": 0-5
    },
    "issues": ["issue 1"],
    "revision_advice": ["specific revision advice"],
    "keep_unchanged": ["content that should stay unchanged"],
    "hard_fail_reasons": ["reason that triggers veto if any"]
  },
  "selected_format": "{selected_format}",
  "candidate_id": "{candidate_id}",
  "recommended_format": "optional format id from the catalog",
  "should_change_format": false,
  "format_change_reason": "optional reason for switching formats"
}

The image review must already include text-image consistency and support checks.
Do not create or assume a separate alignment module or stage.
Use the attached image itself for judgments about readability, subject identity, visual quality, and text-image consistency.
```
