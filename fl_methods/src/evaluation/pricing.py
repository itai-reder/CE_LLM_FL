"""LLM token pricing for evaluation cost columns.

Prices are quoted in USD per 1 000 000 tokens. Cached tokens are a subset
of input tokens billed at a discounted rate (matches OpenAI conventions):

    cost = ((input - cached) * input_per_m
            + cached       * cached_per_m
            + output       * output_per_m) / 1_000_000

Unknown / unpriced models (e.g. local ollama) return 0.0 silently. Add a
new entry to ``MODEL_PRICES`` to start billing a model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1M tokens for one model."""

    input_per_m: float
    cached_per_m: float
    output_per_m: float


# Keys are the bare model name as it appears in ``agent_chain[].model``.
# Dated suffixes on ``response_dumps[].model`` (e.g. ``gpt-5-mini-2025-08-07``)
# are stripped via longest-prefix match in ``_normalise_model``.
MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-5": ModelPrice(input_per_m=1.25, cached_per_m=0.125, output_per_m=10.0),
    "gpt-5-mini": ModelPrice(input_per_m=0.25, cached_per_m=0.025, output_per_m=2.0),
    "gpt-5-nano": ModelPrice(input_per_m=0.05, cached_per_m=0.0005, output_per_m=0.4),
}


def _normalise_model(name: str) -> str:
    """Strip dated suffix by longest-prefix match against ``MODEL_PRICES``.

    ``gpt-5-mini-2025-08-07`` → ``gpt-5-mini``; unrecognised names are
    returned verbatim so the caller can treat them as unpriced.
    """
    if not name:
        return name
    candidates = [k for k in MODEL_PRICES if name == k or name.startswith(k)]
    if not candidates:
        return name
    return max(candidates, key=len)


def compute_response_cost(
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    model: str,
) -> float:
    """Apply the OpenAI-style billing formula for one response.

    Unknown models return ``0.0``.
    """
    price = MODEL_PRICES.get(_normalise_model(model))
    if price is None:
        return 0.0
    billable_input = max(input_tokens - cached_tokens, 0)
    return (
        billable_input * price.input_per_m
        + cached_tokens * price.cached_per_m
        + output_tokens * price.output_per_m
    ) / 1_000_000
