"""Tests for src.evaluation.pricing."""

from __future__ import annotations

import pytest

from src.evaluation.pricing import (
    MODEL_PRICES,
    _normalise_model,
    compute_response_cost,
)


class TestNormaliseModel:
    def test_strips_dated_suffix(self) -> None:
        assert _normalise_model("gpt-5-mini-2025-08-07") == "gpt-5-mini"

    def test_exact_match(self) -> None:
        assert _normalise_model("gpt-5") == "gpt-5"

    def test_longest_prefix_wins(self) -> None:
        # gpt-5 is also a prefix of gpt-5-mini; the longer key should win.
        assert _normalise_model("gpt-5-mini") == "gpt-5-mini"
        assert _normalise_model("gpt-5-nano-2025-08-07") == "gpt-5-nano"

    def test_unknown_returned_verbatim(self) -> None:
        assert _normalise_model("qwen3.5:9b") == "qwen3.5:9b"
        assert _normalise_model("") == ""


class TestComputeResponseCost:
    def test_one_million_input_gpt5_mini_is_25_cents(self) -> None:
        # 1M billable input @ $0.25/M
        assert compute_response_cost(1_000_000, 0, 0, "gpt-5-mini") == pytest.approx(0.25)

    def test_cached_subtracted_from_input(self) -> None:
        # 1M input of which 1M cached → only the cached rate applies
        cached_only = compute_response_cost(1_000_000, 1_000_000, 0, "gpt-5-mini")
        # 1M cached @ $0.025/M
        assert cached_only == pytest.approx(0.025)

    def test_full_formula_gpt5(self) -> None:
        # 1000 input, 200 cached, 500 output
        # = ((1000 - 200) * 1.25 + 200 * 0.125 + 500 * 10.0) / 1e6
        # = (1000 + 25 + 5000) / 1e6 = 6025 / 1e6 = 0.006025
        cost = compute_response_cost(1000, 200, 500, "gpt-5")
        assert cost == pytest.approx(0.006025)

    def test_dated_suffix_priced(self) -> None:
        a = compute_response_cost(100, 0, 0, "gpt-5-nano")
        b = compute_response_cost(100, 0, 0, "gpt-5-nano-2025-08-07")
        assert a == b > 0

    def test_unknown_model_returns_zero(self) -> None:
        assert compute_response_cost(10_000, 0, 5_000, "qwen3.5:9b") == 0.0
        assert compute_response_cost(10_000, 0, 5_000, "") == 0.0

    def test_negative_billable_clamped(self) -> None:
        # Cached > input would be a data bug but should not produce a negative bill.
        cost = compute_response_cost(100, 200, 0, "gpt-5-mini")
        # billable_input = 0, cached = 200 @ 0.025/M
        assert cost == pytest.approx(200 * 0.025 / 1_000_000)


class TestModelPricesTable:
    def test_expected_models_priced(self) -> None:
        assert set(MODEL_PRICES) == {"gpt-5", "gpt-5-mini", "gpt-5-nano"}
