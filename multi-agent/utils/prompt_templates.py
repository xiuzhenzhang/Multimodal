"""Prompt template definitions"""

# Agent 1: Fact extraction template
FACT_EXTRACTION_PROMPT = """Extract 5W1H facts from the news article. CRITICAL: Accurately identify the original article's claims (main assertions, statements, and key arguments) - this determines claim reversal success.

News Article:
{news_article}

Output JSON:
{{
    "who": ["person1", "person2"],
    "what": ["event1", "event2"],
    "when": ["time1"],
    "where": ["location1"],
    "why": ["reason1"],
    "how": ["method1"],
    "original_claims": "Detailed description: main claims, assertions, key arguments, and their supporting evidence"
}}
"""

# Agent 1: Claim reversal template
CLAIM_REVERSAL_PROMPT = """Generate the complete opposite of the original claims. CRITICAL: This is the core of the flip effect. Use "opposite the claims" approach - more effective than simple assertion reversal.

Original Claims:
{original_claims}

Requirements: Complete opposite claims (positive→negative, supportive→critical, success→failure, etc.), maintain same intensity level, detailed enough to guide content generation. The reversed claims should be the direct opposite of what was originally claimed.

Output ONLY the opposite claims (one detailed sentence):
"""

# Agent 1: Mirror rewrite template
MIRROR_REWRITE_PROMPT = """Create a news narrative with opposite stance based on facts and OPPOSITE CLAIMS.

CRITICAL: Build entire article on OPPOSITE CLAIMS. Facts (5W1H) must remain recognizable. Readers should identify which news story this is.

Facts List:
{facts}

Original Article:
{original_article}

Opposite Claims:
{opposite_claims}

Requirements:
1. Preserve all facts (5W1H) - only change interpretation/stance, not facts
2. Keep key entities visible: mention core people/organizations by name (no vague abstractions)
3. First 2-3 paragraphs: clearly describe core event (who/what/when/where/why)
4. No new main characters/locations/events beyond facts list
5. Claim-driven: every sentence serves opposite claims
6. Length: 500-800 words. Persuasive, engaging, emotionally charged language

Output ONLY the mirrored article (no explanations):
"""

# Agent 1: Post text distillation template (Method 1: Modified Facts)
POST_DISTILLATION_PROMPT_METHOD1 = """Write a post with catchy title + complete news summary (5W1H). REJECTED if missing.

REQUIRED: Title + summary with WHO (names), WHAT (events), WHEN/WHERE/WHY/HOW (if available). MAX 280 characters (count characters, not words).

CRITICAL FORMAT REQUIREMENT: Write as NATURAL NEWS TEXT, NOT metadata format. Do NOT use labels like "WHO:", "WHAT:", "WHEN:" or bullet points. Write flowing news narrative that naturally incorporates 5W1H information.

CRITICAL LENGTH REQUIREMENT: 
- MUST be ≤280 characters (count EXACTLY - this is STRICTLY ENFORCED)
- MUST end with a complete sentence (., !, or ?)
- NO truncated or incomplete sentences allowed
- If you cannot fit all information in 280 characters, prioritize WHO/WHAT and remove less essential details

CORRECT (natural news format): "**Revolutionary Breakthrough!** Researchers at Harvard discovered a new treatment in 2023 at Boston Medical Center to combat disease using advanced AI technology. This marks a turning point in medical science."
INCORRECT (metadata format): "**Title** WHO: Researchers. WHAT: Treatment. WHEN: 2023." ❌ REJECTED - This is metadata, not news text.
INCORRECT (too short): "**The Future is Here.**" ❌ Missing 5W1H
INCORRECT (incomplete): "**Revolutionary Breakthrough!** Researchers at Harvard discovered a new treatment in 2023 at Boston Medical Center to combat disease using" ❌ REJECTED - Incomplete sentence

Mirrored Article:
{mirrored_article}

Modified Facts (with intentional errors):
{modified_facts}

Opposite Claims:
{opposite_claims}

Requirements: 
- Structure: [Title] + [News Summary with 5W1H naturally embedded]
- Format: Natural news narrative (like a news article), NOT metadata/labels/bullets
- MAX 280 characters (CRITICAL - count characters EXACTLY, not words. Your output MUST be ≤280 characters. If you exceed this limit, your response will be REJECTED)
- MUST end with complete sentence (., !, or ?) - NO incomplete sentences
- Include WHO (names), WHAT (events), WHEN/WHERE/WHY/HOW naturally in the narrative
- Use modified facts that contain intentional errors to support opposite claims
- The post must contain evidence/arguments that support the opposite claims, but these evidence are based on modified (false) facts
- Reader should understand WHO/WHAT/WHEN/WHERE/WHY/HOW from natural reading
- BEFORE submitting: 1) Count your characters (must be ≤280), 2) Verify it ends with complete sentence (., !, or ?). If over 280 or incomplete, shorten by removing less essential details while keeping WHO/WHAT and complete sentences

Post Copy (natural news text, MUST be ≤280 characters AND end with complete sentence - count carefully):
"""

# Agent 1: Post text distillation template (Method 2: Modified Evidence)
POST_DISTILLATION_PROMPT_METHOD2 = """Write a post with catchy title + complete news summary (5W1H). REJECTED if missing.

REQUIRED: Title + summary with WHO (names), WHAT (events), WHEN/WHERE/WHY/HOW (if available). MAX 280 characters (count characters, not words).

CRITICAL FORMAT REQUIREMENT: Write as NATURAL NEWS TEXT, NOT metadata format. Do NOT use labels like "WHO:", "WHAT:", "WHEN:" or bullet points. Write flowing news narrative that naturally incorporates 5W1H information.

CRITICAL LENGTH REQUIREMENT: 
- MUST be ≤280 characters (count EXACTLY - this is STRICTLY ENFORCED)
- MUST end with a complete sentence (., !, or ?)
- NO truncated or incomplete sentences allowed
- If you cannot fit all information in 280 characters, prioritize WHO/WHAT and remove less essential details

CORRECT (natural news format): "**Revolutionary Breakthrough!** Researchers at Harvard discovered a new treatment in 2023 at Boston Medical Center to combat disease using advanced AI technology. This marks a turning point in medical science."
INCORRECT (metadata format): "**Title** WHO: Researchers. WHAT: Treatment. WHEN: 2023." ❌ REJECTED - This is metadata, not news text.
INCORRECT (too short): "**The Future is Here.**" ❌ Missing 5W1H
INCORRECT (incomplete): "**Revolutionary Breakthrough!** Researchers at Harvard discovered a new treatment in 2023 at Boston Medical Center to combat disease using" ❌ REJECTED - Incomplete sentence

Mirrored Article:
{mirrored_article}

Facts:
{facts}

Original Evidence:
{original_evidence}

Modified Evidence (false evidence supporting opposite claims):
{modified_evidence}

Opposite Claims:
{opposite_claims}

Requirements: 
- Structure: [Title] + [News Summary with 5W1H naturally embedded]
- Format: Natural news narrative (like a news article), NOT metadata/labels/bullets
- MAX 280 characters (CRITICAL - count characters EXACTLY, not words. Your output MUST be ≤280 characters. If you exceed this limit, your response will be REJECTED)
- MUST end with complete sentence (., !, or ?) - NO incomplete sentences
- Include WHO (names), WHAT (events), WHEN/WHERE/WHY/HOW naturally in the narrative
- MUST contain evidence/arguments that support the opposite claims
- The evidence must be based on modified (false) evidence that appears to support the opposite claims
- The post text MUST include evidence - this is mandatory. The evidence should be arguments/points that support the opposite claims, but these are derived from modified (false) original evidence
- Reader should understand WHO/WHAT/WHEN/WHERE/WHY/HOW from natural reading
- BEFORE submitting: 1) Count your characters (must be ≤280), 2) Verify it ends with complete sentence (., !, or ?). If over 280 or incomplete, shorten by removing less essential details while keeping WHO/WHAT, evidence, and complete sentences

Post Copy (natural news text, MUST be ≤280 characters AND end with complete sentence - count carefully, MUST include evidence supporting opposite claims):
"""

# Agent 1: Content adjustment template (based on Agent 2 feedback)
CONTENT_ADJUSTMENT_PROMPT = """Refine existing content based on critic feedback. PRESERVE strengths, make TARGETED improvements. This is refinement, not a rewrite.

Facts:
{facts}

Opposite Claims:
{opposite_claims}

Current Content:
Mirrored Article: {current_mirrored_article}
Post Text: {current_post_text}

Critic Feedback:
Strengths (PRESERVE): {strengths}
Criticisms (ADDRESS): {criticisms}
Must Improve (FIX): {must_improve}
Recommendations (IMPLEMENT): {recommendations}
Score: {score}/100

Requirements: Preserve strengths, address criticisms, fix must_improve items, implement recommendations. Keep facts unchanged. Convey opposite claims. Article: 500-800 words. Post: max 280 characters (count characters, not words).

Output ADJUSTED content:

Adjusted Mirrored Article:
[Improved article]

Adjusted Post Text:
[Improved post text]
"""

# Agent 2: Critic template
CRITIC_PROMPT = """Evaluate Agent 1's content. CRITICAL SCORING RULE: Greater difference from original = higher score. More similar = lower score.

Original Article:
{original_article}

Facts:
{facts}

Opposite Claims:
{opposite_claims}

Mirrored Article:
{mirrored_article}

Post Text:
{post_text}

MANDATORY POST TEXT CHECKS:
1. Character count ≤ 280 characters (CRITICAL - deduct 50 if exceeds)
2. Ends with complete sentence (., !, or ?) - NO incomplete/truncated sentences (CRITICAL - deduct 30 if incomplete)
3. Contains WHO (specific names) and WHAT (specific events) from facts (CRITICAL - deduct 50 if missing)
4. Complete summary (title + story), not just title
5. Contains evidence/arguments supporting opposite claims (CRITICAL - deduct 30 if missing)

CRITICAL: FACT MODIFICATION CHECK (MOST IMPORTANT):
**This is a FAKE NEWS generation task. The mirrored article MUST modify FACTS, not just opinions/stance/emotion.**

Compare the ORIGINAL ARTICLE and MIRRORED ARTICLE to check:

1. **FACT MODIFICATION (REQUIRED)**: 
   - Does the mirrored article change actual FACTS (data, numbers, results, events, outcomes, specific claims about what happened)?
   - Examples of FACT changes: "64% sensitivity" → "0% sensitivity", "28 patients" → "100 patients", "study found X" → "study found Y"
   - Examples of NOT fact changes (ONLY stance/opinion): "researchers are optimistic" → "researchers are pessimistic", "this is good news" → "this is bad news", "important breakthrough" → "disappointing result"
   
2. **STANCE-ONLY CHANGE (INSUFFICIENT)**:
   - If the mirrored article ONLY changes opinions, emotions, or interpretations WITHOUT changing facts, this is INSUFFICIENT
   - Example of INSUFFICIENT: Original says "Event X happened, and it's great!" → Mirrored says "Event X happened, and it's terrible!" (same fact, different opinion)
   - Example of SUFFICIENT: Original says "Event X happened with result Y" → Mirrored says "Event X happened with result Z" (different fact)

3. **SCORING FOR FACT MODIFICATION**:
   - If facts are MODIFIED (data/numbers/results/events changed): Continue normal scoring
   - If ONLY stance/opinion/emotion changed WITHOUT fact modification: 
     * Deduct 60 points immediately
     * Mark as NOT excellent
     * Add to must_improve: "CRITICAL: Only stance/opinion changed, facts unchanged. This is NOT fake news. Must modify actual facts (data, numbers, results, events)."
     * Maximum possible score becomes 40 (even if other aspects are perfect)

EVALUATION:
1. **FACT MODIFICATION CHECK (PRIMARY - CHECK FIRST)**: 
   - Facts modified (data/numbers/results/events changed): Proceed to normal scoring
   - Only stance/opinion changed: Apply -60 penalty, max score 40, mark as must_improve
   
2. **DIFFERENCE FROM ORIGINAL (SECONDARY)**: Compare with original. Greater difference (perspective/tone/interpretation/claims) = higher score. More similar = lower score.
   - Very different: 80-100 base
   - Moderately different: 60-79 base
   - Too similar: 40-59 base
   - Almost identical: 20-39 base
   
3. **Logical Consistency (LENIENT)**: Only flag obvious serious contradictions. Minor gaps acceptable.
4. **Post Text**: Character count ≤ 280, contains WHO/WHAT, contains evidence (deduct 50 if character count/WHO/WHAT fails, deduct 30 if evidence missing)
5. **Secondary**: Claim power, narrative quality, post quality, artistic merit

SCORING: 
- FIRST: Check fact modification. If only stance changed, apply -60 penalty, max score 40.
- THEN: Base on difference, then: -50 if post exceeds 280 characters/missing WHO/WHAT, -30 if missing evidence, -10-20 for serious contradictions, +5-10 for exceptional quality.

PRINCIPLES: 
- FACT MODIFICATION IS MANDATORY. Only changing stance/opinion is NOT fake news.
- Deviation in FACTS is REQUIRED and REWARDED. Deviation in stance alone is INSUFFICIENT.
- Be lenient with logic, but strict on fact modification requirement.

Output JSON:
{{
    "is_excellent": true/false,
    "criticisms": ["criticism 1", "criticism 2"],
    "strengths": ["strength 1", "strength 2"],
    "recommendations": ["recommendation 1", "recommendation 2"],
    "score": 0-100,
    "must_improve": ["issue 1", "issue 2"]
}}
"""

# Visual Strategy Selection Agent: Strategy selection template
VISUAL_STRATEGY_SELECTION_PROMPT = """Select the BEST visual strategy from 6 options to convey OPPOSITE CLAIMS. Image must stay grounded in facts. Viewer should identify who/what from the image.

Mirrored Article: {mirrored_article}
Post Text: {post_text}
Opposite Claims: {opposite_claims}

Strategies:
1. Metaphorical Substitution: Replace subject with symbolic objects from facts (e.g., mousetrap for "algorithm system"). Label original subject.
2. Macro/Micro Perspective Shift: Close-up view of people/scenes from facts showing hidden costs. Label ignored details.
3. Data Visualization: Charts/graphs using numbers from facts (e.g., layoff rates). Use facts elements (badges, logos) to construct charts. Label data meaning.
4. Structural Deconstruction: X-ray/cutaway showing internal decay of products/buildings from facts. Label hidden costs.
5. Typographic Architecture: Text phrases from flipped narrative forming shapes of key objects from facts. No additional annotation.
6. Direct Representation: Directly visualize key people/scenes/objects from facts with reversed emotion. Standard text overlay.

Requirements: Best strategy for opposite claims + facts fidelity. In strategy_details, describe which people/objects/scenes appear and how strategy transforms them.

Output JSON:
{{
    "selected_strategy": 1-6,
    "strategy_name": "strategy name",
    "reasoning": "why this strategy is best",
    "strategy_details": "implementation details: people/objects/scenes + transformation method",
    "probability_distribution": {{"strategy_1": 0.1, "strategy_2": 0.1, "strategy_3": 0.1, "strategy_4": 0.1, "strategy_5": 0.1, "strategy_6": 0.5}}
}}
"""

# Agent 3: Image generation prompt template
IMAGE_GENERATION_PROMPT_TEMPLATE = """Generate a BORING, MUNDANE, UNREMARKABLE news snapshot. CORE RULE: Ordinary and dull = Real. Artistic and dramatic = Fake.

Inputs:
[Opposite Claims]: {opposite_claims}
[Article Context]: {mirrored_article_summary}
[Post Text]: {post_text}
[Visual Strategy]: {strategy_name} - {strategy_details}
[Keywords]: Entities: {entities} | Emotions: {emotions} | Style: {visual_style} | Palette: {color_palette}

CRITICAL: MUST LOOK NATURAL AT FIRST GLANCE (not just in details)
The image must look BORING and ORDINARY when you glance at it. NO artistic feel, NO dramatic mood, NO cinematic atmosphere, NO color grading, NO stylized look. Think: "This could be any random boring photo from someone's phone - nothing special, nothing interesting."

ABSOLUTE PRIORITY RULES (Most critical):

0. NEGATIVE PROMPT - NEVER GENERATE (Will cause immediate "AI feel"):
   multiple clear subjects, busy scene, complex composition, many detailed elements, crowd with clear faces, group of clear people, hands visible, fingers, text, signs, labels, vibrant colors, saturated colors, neon, teal, cyan, orange-teal, blue-orange, dramatic lighting, rim light, edge light, backlight, hair light, studio lighting, cinematic, moody, atmospheric, artistic, color grading, film look, stylized, professional photography, perfect composition, centered subject, symmetry, polished look, everything in focus, deep depth of field

1. MINIMAL INFORMATION - LESS IS MORE (Critical principle):
   ✓ SHOW ONLY ONE CLEAR SUBJECT: One person OR one object OR one scene element in focus
   ✓ EVERYTHING ELSE BLURRED/SIMPLIFIED: Background can have people/objects but they MUST be heavily blurred, out of focus, or simplified to color blobs
   ✓ SHALLOW DEPTH OF FIELD: Only the main subject is sharp, everything else naturally blurred (like real camera)
   ✓ REDUCE INFORMATION: Don't try to show everything - show less, blur more
   ❌ NO multiple clear subjects: Only ONE thing should be in focus and detailed
   ❌ NO complex busy scenes with many clear elements
   ❌ Background people are OK IF they are completely blurred/out of focus (not detailed, not recognizable)
   
2. NO TEXT/WRITING ANYWHERE (AI always generates gibberish):
   ❌ ZERO text, letters, words, signs, labels, posters, billboards, screens, documents
   ❌ NO books with visible text, newspapers, magazines, name tags, badges
   ❌ NO store signs, street signs, building names, logos with text
   ❌ NO computer/phone screens showing text, whiteboards, papers
   ❌ If text must appear, it MUST be completely blurred/out-of-focus/unreadable
   ✓ PREFER: Scenes with NO text elements at all (empty walls, nature, plain surfaces)

3. REALISTIC IMPERFECT LIGHTING (Critical to avoid AI look):
   • SINGLE DOMINANT LIGHT SOURCE: One main light only (sun, window, or overhead) - NOT multiple balanced lights
   • UNEVEN illumination: Random bright and dark patches, NOT uniform distribution across scene
   • HARSH inconsistent shadows: Strong shadows in some areas, weak/none in others - NOT soft balanced shadows
   • GROUND SHADOWS REQUIRED: All objects/people MUST cast visible shadows on ground/floor in direction away from light source
   • NATURAL shadow darkness: Shadows should be dark gray/black, NOT soft translucent
   • NO multiple light sources creating complex lighting (no studio setup, no fill lights, no rim lights)
   • NO "cinematic" lighting with perfect edge lights or hair lights
   • NO uniform glow or ambient light filling all shadows
   • NO perfect light distribution on surfaces (walls should have bright spots and dark corners)
   • AVOID: Balanced 3-point lighting, soft studio shadows, even ambient fill, "professional" lighting
   • AVOID: Glowing edges on objects, perfect rim lighting, symmetrical highlights, yellow-green tints
   • THINK: Single harsh overhead fluorescent, strong directional sunlight, one window creating hard contrast
   • Real lighting is MESSY: some areas overexposed, some underexposed, harsh transitions

CRITICAL PROHIBITIONS (Avoid AI "Polish"):
❌ NO deep blacks or high contrast (use muddy, gray-ish shadows)
❌ NO vibrant or saturated colors (especially reds and blues)
❌ NO smooth, ironed clothing (must have wrinkles and wear)
❌ NO "bokeh" style blur (if background is blurry, it must be motion blur or poor focus)
❌ NO perfect symmetry or centered subjects
❌ NO "melting" background people (must maintain distinct, albeit blurry, human silhouettes)
❌ NO close-up faces/portraits (skin too smooth, eyes too perfect)
❌ NO hands/fingers visible (always deformed)

REALISTIC COLOR REQUIREMENTS (critical for natural look):
• NEUTRAL BORING COLORS: No color grading, no artistic tints, no mood colors
• DESATURATED: Dull, faded, washed-out - like a cheap phone camera
• EVERYDAY PALETTE: Grays, beiges, tans, browns, off-whites - BORING everyday colors
• AVOID ARTISTIC COLOR SCHEMES: NO teal-orange, NO blue-cyan moody looks, NO warm-cool contrasts
• AVOID: Vibrant colors, vivid hues, color grading, film looks, cinematic colors, stylized palettes
• NATURAL WHITE BALANCE: Slightly off (yellowish indoor or bluish outdoor), NOT perfect, NOT artistic
• THINK: Boring office lighting, overcast day, cheap fluorescent lights - nothing special
• The colors should be SO BORING that no one notices them

AVOID "ARTISTIC" OR "CINEMATIC" FEEL:
❌ NO moody atmosphere (dark with dramatic lights)
❌ NO layered composition (foreground-midground-background separation)
❌ NO color grading or film looks
❌ NO dramatic color contrasts (warm vs cool)
❌ NO "storytelling" feel or emotional atmosphere
❌ NO perfect depth of field with artistic blur
❌ THINK: Flat, boring, nothing special - like a random snapshot

MANDATORY PHYSICAL IMPERFECTIONS:
• Clothing: Deep wrinkles at joints (elbows, waist), sagging pockets, visible lint, uneven fabric texture, unpressed seams.
• Surfaces: Footprints on carpets, scuff marks on floors, dust on shiny surfaces, uneven floor texture, visible dirt streaks.

SAFE SCENARIOS (choose one):
1. One clear subject with blurred background: Person in focus, background heavily blurred (can have blurred people/objects)
2. Empty scene: Building exterior, empty street, landscape - minimal elements
3. Close-up of one object/detail: Everything else out of focus

KEY PRINCIPLE: Show LESS information, blur MORE. Only ONE thing should be clear and detailed. Everything else should be simplified, blurred, or out of focus.

Style: A BORING, UNREMARKABLE photo from a cheap phone camera with shallow depth of field. Only one subject in focus, everything else blurred. Flat, dull, uninteresting, with neutral boring colors, plain lighting, ZERO artistic feel, ZERO mood, ZERO drama. Minimal information. Size: {width}x{height}.

Image Generation Prompt (NEGATIVE: crowd, multiple clear subjects, dramatic lighting, vibrant colors, text, hands, cinematic, moody, artistic, teal, cyan, color grading, busy scene | POSITIVE: one clear subject, blurred background, boring neutral colors, flat plain lighting, shallow depth of field, minimal information, unremarkable, mundane):
"""
# Direct Summarization Prompt (for original article summarization without emotion reversal)
DIRECT_SUMMARIZATION_PROMPT = """Summarize the news article into a concise post text. CRITICAL: Keep the ORIGINAL emotion and content - do NOT modify, add, or change anything. Only summarize.

CRITICAL REQUIREMENT - MUST BE COMPLETELY CONSISTENT WITH ORIGINAL:
1. **PRESERVE ORIGINAL EMOTION**: Keep the exact same emotional tone as the original article (positive stays positive, negative stays negative, neutral stays neutral)
2. **NO CONTENT MODIFICATION**: Do NOT add, remove, or change any facts, events, data, numbers, names, dates, locations, or information. Only condense.
3. **NO ADDITIONS**: Do NOT add any information not present in the original article.
4. **MUST CONTAIN AT LEAST ONE SPECIFIC EVENT**: The post text MUST include at least one specific event, fact, or data point from the original article to prove consistency. This MUST be:
   - A specific number (e.g., "28 patients", "64% sensitivity", "$81 million")
   - A specific date (e.g., "September 1, 2009", "July 2015")
   - A specific name (e.g., "Dr. Jaswinder Ghuman", "M.D. Anderson Cancer Center")
   - A specific result or finding (e.g., "showed 89% specificity", "found no response")
   - A specific event (e.g., "launched on July 2, 2014", "published in Nature")
   This event/fact must be EXACTLY as stated in the original article - word-for-word accuracy for numbers, dates, and names.
5. **VERIFY CONSISTENCY**: Before submitting, verify that ALL facts, data, events, names, dates, and numbers in your summary EXACTLY match the original article. Any discrepancy will result in rejection.
6. **Format**: Natural news narrative with catchy title + complete summary (5W1H elements naturally embedded)
7. **Length**: Maximum 280 characters (count characters carefully) - STRICTLY ENFORCED
8. **Completeness**: MUST end with a complete sentence (., !, or ?) - NO truncated or incomplete sentences
9. **Structure**: [Title] + [News Summary with 5W1H naturally embedded in flowing narrative]
10. **Format**: Natural news text, NOT metadata format (no labels like "WHO:", "WHAT:", etc.)

CRITICAL FORMAT REQUIREMENT: Write as NATURAL NEWS TEXT, NOT metadata format. Do NOT use labels like "WHO:", "WHAT:", "WHEN:" or bullet points. Write flowing news narrative that naturally incorporates 5W1H information.

CRITICAL LENGTH REQUIREMENT: 
- MUST be ≤280 characters (count EXACTLY - this is STRICTLY ENFORCED)
- MUST end with a complete sentence (., !, or ?)
- NO truncated or incomplete sentences allowed
- If you cannot fit all information in 280 characters, prioritize WHO/WHAT and the specific event/fact, then remove less essential details

CORRECT (natural news format with specific event): "**Revolutionary Breakthrough!** Researchers at Harvard discovered a new treatment in 2023 at Boston Medical Center to combat disease using advanced AI technology. The study with 100 patients showed 85% effectiveness. This marks a turning point in medical science."
INCORRECT (metadata format): "**Title** WHO: Researchers. WHAT: Treatment. WHEN: 2023." ❌ REJECTED - This is metadata, not news text.
INCORRECT (no specific event): "**Revolutionary Breakthrough!** Researchers discovered a new treatment that shows promise." ❌ REJECTED - Missing specific event/data to prove consistency.
INCORRECT (modified fact): Original says "28 patients" but summary says "30 patients" ❌ REJECTED - Numbers must match exactly.
INCORRECT (incomplete): "**Revolutionary Breakthrough!** Researchers at Harvard discovered a new treatment in 2023 at Boston Medical Center to combat disease using" ❌ REJECTED - Incomplete sentence.

Original Article:
{original_article}

Post Text (natural news text, max 280 characters, MUST end with complete sentence, MUST be completely consistent with original, MUST include at least one specific event/fact/data EXACTLY as stated in original to prove consistency):
"""