PROMPT_DRAFT_FROM_IMAGE = """
Summarize the late-stage price behavior from the chart image to search for similar-shaped historical charts.

Requirements:
- Do not include any price numbers / price levels / timeframe / symbol or pair name
- 25–40 words in English
- Append exactly one bias word at the end of the same sentence: BUY or SELL or NEUTRAL
- No bullets, no headings
- No line breaks, write as a single paragraph
""".strip()

PROMPT_DOMAIN_REWRITE = """
You will receive (1) a DRAFT summarized from an image and (2) DOMAIN EXAMPLES from an existing database (the real writing style used by the system).
Task: Rewrite the DRAFT to match the wording, phrasing, and narrative rhythm of the DOMAIN EXAMPLES to improve retrieval accuracy.

Rules:
- 40–60 words, single paragraph, no line breaks
- Do not include any price numbers / price levels / timeframe / symbol or pair name
- Append one bias word at the end: BUY or SELL or NEUTRAL
- No bullets, no headings, no markdown
- End with KEYWORDS: <8–12 keywords> on the same line
""".strip()

AUTO_TEXT_COMPRESS_NOTE = """
Note: KEYWORDS strongly affect retrieval quality. Keep them short and directly tied to what is visible in the chart.
""".strip()

RAG_TEMPLATE = """
You are an expert Price Action trader. Analyze ONLY the current chart image. Use older similar-pattern context only for abstract behavioral ideas. Never copy or reuse any numbers, price levels, timeframes, symbols, or ranges from the older context.

Older-chart context for behavioral comparison only:
{context}

Strict output rules:
- English only
- Single paragraph, no line breaks, and exactly 3 sentences
- No bullets, no Markdown, and no special symbols such as parentheses, dashes, colons, quotes, slashes, or asterisks
- Do not use the words trend, momentum, market structure, or keywords (and do not use close synonyms)

Content requirements:
- Sentence 1 must start with "PA is in..." or "PA is on the side of..." and give one short reason based on the current image only (e.g., broke a base, fast rebound, failed sweep, heavy upper wicks, lifted lower wicks)
- Sentence 2 must state one support zone and one resistance zone from the current image, and say which side is disadvantaged; numbers are allowed ONLY if you are confident they are clearly readable from the current image, otherwise describe zones without numbers
- Sentence 3 must give a short wait-and-play plan with entry area, invalidation, and target using only zones/levels from the current image; if unsure about numbers, use positional descriptions instead

Number rules:
- Do not include any numbers that are not price levels (no days, percentages, ratios, or IDs)
- Do not guess precise numbers beyond what the image clearly shows, and do not invent ranges you cannot clearly see
- If you are not confident a number comes from the current image, omit it immediately

Self-check before sending:
If you used any uncertain numbers, used forbidden words, or did not produce exactly 3 sentences in one paragraph, rewrite until all rules pass.

Instruction:
Analyze this chart image and provide a trading recommendation following all rules above.
"""
