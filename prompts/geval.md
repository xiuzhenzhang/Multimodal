# Appendix: G-Eval Prompts for Dataset Quality Checking

This appendix lists the G-Eval-style prompts used to assess the textual quality of the generated dataset. The evaluation is applied to the textual components in each generated item, including the original news text, the generated true-news summary, and the generated fake-news text when applicable. The evaluator is instructed to assess generation quality only, without judging whether the news is factually true or false and without using external knowledge.

## C.1 General G-Eval Prompt Template

```text
You are an expert evaluator of news text quality.

Your task is to evaluate ONLY the textual generation quality of the following news text.
Do NOT judge whether the news is factually true or false. Do NOT use external knowledge.
Focus strictly on the requested dimension.

Dimension: {dimension_name}
Category: {category}

Evaluation instruction:
{dimension_instruction}

Scoring scale:
{scoring_scale}

Return JSON only, with this exact schema:
{
  "analysis": "brief explanation in one or two sentences",
  "score": integer from 1 to 5
}

News text:
Title: {title_or_not_provided}
Body:
{news_text}
```

## C.2 Coherence

```text
Dimension: Coherence
Category: Framework

Evaluation instruction:
Evaluate the news text for coherence. Coherence refers to whether the text is internally well connected, whether the sentences and ideas follow a reasonable order, whether causal relations are understandable, and whether the text avoids obvious contradictions or abrupt logical jumps.

Scoring scale:
1 = incoherent, with severe contradictions or disconnected statements;
2 = weakly coherent, with noticeable contradictions, abrupt jumps, or unclear causal links;
3 = mostly coherent, but with some minor logical or transitional issues;
4 = coherent and easy to follow, with only small issues;
5 = highly coherent, logically clear, and well connected throughout.
```

## C.3 Fluency

```text
Dimension: Fluency
Category: Content

Evaluation instruction:
Evaluate the news text for fluency and readability. Check whether the language is grammatical, natural, smooth, and easy to understand. Focus on expression quality rather than factual truth.

Scoring scale:
1 = very hard to read, with many grammar errors or broken sentences;
2 = not fluent, with obvious language problems or unnatural wording;
3 = acceptable, with some grammar or expression issues;
4 = fluent and readable, with only minor issues;
5 = highly fluent, natural, grammatically correct, and easy to read.
```

## C.4 Faithfulness

```text
Dimension: Faithfulness
Category: Content

Evaluation instruction:
Evaluate the news text for faithfulness to its own stated topic and information needs. Faithfulness here refers to whether the text stays focused on the central news topic, provides the necessary contextual details for understanding the claim or event, avoids irrelevant information, and does not leave major gaps that make the text under-specified. Do NOT judge external factual truth.

Scoring scale:
1 = largely unfaithful to the stated topic, with most necessary information missing or irrelevant;
2 = weakly faithful, with clear information gaps, missing context, or substantial irrelevant content;
3 = moderately faithful, covering the main topic but missing some important details;
4 = faithful to the topic and covers most necessary information;
5 = highly faithful, focused, relevant, and sufficiently complete for understanding the news item.
```

## C.5 Prompt Instantiation

For each news text, the general template in Section C.1 is instantiated three times, once for each dimension:

```text
{dimension_name} in {
  "Coherence",
  "Fluency",
  "Faithfulness"
}
```

Each prompt returns a score from 1 to 5 and a short explanation. The final quality score can be reported either per dimension or as the average over the three dimensions.
