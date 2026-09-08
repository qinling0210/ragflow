Based on the provided document or chat history, add citations to the input text using the format specified later.

# Citation Requirements:

## Technical Rules:
- Use format: [ID:i] or [ID:i] [ID:j] for multiple sources
- Place citations at the end of sentences, before punctuation
- Maximum 4 citations per sentence
- DO NOT cite content not from <context></context>
- DO NOT modify whitespace or original text
- STRICTLY prohibit non-standard formatting (~~, etc.)
- For RTL languages (Arabic, Hebrew, Persian): Place citations at the logical end of sentences (same position as LTR). The frontend handles bidirectional rendering automatically.

## What MUST Be Cited:
1. **Quantitative data**: Numbers, percentages, statistics, measurements
2. **Temporal claims**: Dates, timeframes, sequences of events
3. **Causal relationships**: Claims about cause and effect
4. **Comparative statements**: Rankings, comparisons, superlatives
5. **Technical definitions**: Specialized terms, concepts, methodologies
6. **Direct attributions**: What someone said, did, or believes
7. **Predictions/forecasts**: Future projections, trend analyses
8. **Controversial claims**: Disputed facts, minority opinions

## What Should NOT Be Cited:
- Common knowledge (e.g., "The sun rises in the east")
- Transitional phrases
- General introductions
- Your own analysis or synthesis (unless directly from source)

## Example:
<context>
ID: 45
└── Content: The global smartphone market grew by 7.8% in Q3 2024.

ID: 46
└── Content: 5G adoption reached 1.5 billion users worldwide by October 2024.
</context>

USER: How is the smartphone market performing?

ASSISTANT:
The smartphone industry is showing strong recovery. The global smartphone market grew by 7.8% in Q3 2024 [ID:45]. 5G adoption reached 1.5 billion users worldwide by October 2024 [ID:46].

REMEMBER:
- Cite FACTS, not opinions or transitions
- Each citation supports the ENTIRE sentence
- Place citations at sentence end, before punctuation
- Format like "[ID:0, ID:5, ...]" is FORBIDDEN. Must be "[ID:0][ID:5]..."
