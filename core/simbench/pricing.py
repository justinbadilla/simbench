"""List-price estimates for logging. Not an invoice.

These are paid-tier rates, so on a free tier the estimate is "what this would
cost", not what was billed. Cached-input discounts are not applied. Models
missing from the table produce a null estimate instead of a guessed price.

OpenAI standard text rates, checked 2026-10-23:
https://developers.openai.com/api/docs/pricing

Gemini standard text rates, checked 2026-10-23:
https://aistudio.google.com/docs/pricing
Gemini 3.8 Flash is at a launch price through 2026-12-31 and doubles after that.
"""

# USD per 1,000,000 tokens: (input, output).
_USD_PER_MILLION: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-mini-2025-04-14": (0.40, 1.60),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.8-flash": (0.75, 3.75),
}


def has_price(model_name: str) -> bool:
    return model_name in _USD_PER_MILLION


def estimate_cost_usd(model_name: str, input_tokens: int, output_tokens: int) -> float | None:
    rates = _USD_PER_MILLION.get(model_name)
    if rates is None:
        return None
    input_rate, output_rate = rates
    cost = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return round(cost, 8)
