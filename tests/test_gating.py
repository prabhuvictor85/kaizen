"""Quality-gate prong guard.

The gate ANDs four independent prongs — overhead supply, bearish ICT structure,
a broken trend stack, bears owning ADX — and used to report only the AND. A
downstream `removal_rule = "quality_gate"` then named four different problems
with four different fixes and no way to tell them apart.

The two things worth pinning:
  1. Each prong is reported separately, and the reported result is the one that
     was applied (same computation, not a parallel reimplementation).
  2. A prong whose feature columns are missing reports <NA>, never False.
     "Could not run" and "found nothing wrong" must not look identical — a
     silently disabled prong is exactly how F-C12 stayed invisible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.gating import (
    ICT_BEAR_VETO_THRESHOLD,
    SSZ_VETO_THRESHOLD,
    VETO_COLUMNS,
    momentum_bull_quality_gate,
    momentum_bull_quality_gate_detail,
)

FP = "features_"

# A clean momentum-bull candidate: no supply, no bear structure, trend stacked
# up, bulls own ADX. Every prong should pass.
_CLEAN = {
    "ssz_htf_score":      0.0,
    "ict_bear_htf_score": 0.0,
    "price_vs_sma50":     0.05,
    "sma50_slope_5":      0.01,
    "sma200_slope_10":    0.01,
    "plus_di":            30.0,
    "minus_di":           10.0,
}


def _cross(rows: dict[str, dict], drop: list[str] | None = None) -> pd.DataFrame:
    """Build a cross-section from {ticker: {feature: value}}, prefixed."""
    df = pd.DataFrame.from_dict(rows, orient="index")
    for col in drop or []:
        df = df.drop(columns=col)
    df.columns = [f"{FP}{c}" for c in df.columns]
    return df


def _one(**overrides):
    row = dict(_CLEAN)
    row.update(overrides)
    return row


# ── 1. Each prong is attributed ──────────────────────────────────────────────

def test_clean_candidate_passes_every_prong():
    d = momentum_bull_quality_gate_detail(_cross({"OK": _one()}), "momentum", FP)
    assert bool(d.loc["OK", "keep"]) is True
    for col in VETO_COLUMNS:
        assert d.loc["OK", col] is False or d.loc["OK", col] == False  # noqa: E712


@pytest.mark.parametrize(
    "col,overrides",
    [
        ("veto_ssz_supply",         {"ssz_htf_score": SSZ_VETO_THRESHOLD + 0.05}),
        ("veto_ict_bear_structure", {"ict_bear_htf_score": ICT_BEAR_VETO_THRESHOLD + 0.05}),
        ("veto_broken_trend",       {"price_vs_sma50": -0.02}),
        ("veto_bearish_adx",        {"plus_di": 10.0, "minus_di": 30.0}),
    ],
)
def test_each_prong_is_attributed_on_its_own(col, overrides):
    """The vetoing prong must be the one flagged — and the only one."""
    d = momentum_bull_quality_gate_detail(
        _cross({"BAD": _one(**overrides)}), "momentum", FP
    )
    assert bool(d.loc["BAD", col]) is True, f"{col} should have fired"
    assert bool(d.loc["BAD", "keep"]) is False
    others = [c for c in VETO_COLUMNS if c != col]
    assert not d.loc["BAD", others].fillna(False).any(), (
        f"only {col} should have fired, got {d.loc['BAD', VETO_COLUMNS].to_dict()}"
    )


def test_threshold_is_strict_greater_than():
    """Exactly at the threshold is not a veto — the gate uses >, not >=."""
    d = momentum_bull_quality_gate_detail(
        _cross({"EDGE": _one(ssz_htf_score=SSZ_VETO_THRESHOLD)}), "momentum", FP
    )
    assert bool(d.loc["EDGE", "veto_ssz_supply"]) is False
    assert bool(d.loc["EDGE", "keep"]) is True


# ── 2. A disabled prong is <NA>, not False ───────────────────────────────────

def test_missing_columns_report_na_not_false():
    """THE distinction this guard exists for.

    A prong whose features are absent did not clear the candidate — it never
    ran. Reporting False would claim the check passed.
    """
    d = momentum_bull_quality_gate_detail(
        _cross({"T": _one()}, drop=["ssz_htf_score"]), "momentum", FP
    )
    assert pd.isna(d.loc["T", "veto_ssz_supply"]), "disabled prong must be <NA>"
    assert d.loc["T", "veto_bearish_adx"] == False  # noqa: E712 — this one ran
    assert bool(d.loc["T", "keep"]) is True, "a disabled prong must not veto"


def test_non_momentum_mode_disables_every_prong():
    for mode in ("reversal", "legacy", "all"):
        d = momentum_bull_quality_gate_detail(_cross({"T": _one()}), mode, FP)
        assert d["keep"].all(), f"{mode} must be a no-op"
        assert d[VETO_COLUMNS].isna().all().all(), (
            f"{mode} did not run the prongs — they must be <NA>, not False"
        )


# ── 3. The reported gate is the applied gate ─────────────────────────────────

def test_mask_matches_the_detail_it_reports():
    rows = {
        "CLEAN": _one(),
        "SUPPLY": _one(ssz_htf_score=0.9),
        "ICTBEAR": _one(ict_bear_htf_score=0.7),
        "TREND": _one(sma50_slope_5=-0.01),
        "ADX": _one(plus_di=5.0, minus_di=25.0),
    }
    cross = _cross(rows)
    detail = momentum_bull_quality_gate_detail(cross, "momentum", FP)
    mask = momentum_bull_quality_gate(cross, "momentum", FP, detail=detail)
    assert list(mask.index) == list(detail.index)
    assert mask.tolist() == detail["keep"].fillna(True).astype(bool).tolist()
    assert mask.tolist() == [True, False, False, False, False]


def test_mask_is_unchanged_from_the_pre_prong_implementation():
    """Behaviour guard: the refactor must not move any decision boundary.

    Replicates the original inline logic and requires an exact match on a
    randomised cross-section spanning all four prongs.
    """
    rng = np.random.default_rng(7)
    n = 400
    raw = pd.DataFrame({
        "ssz_htf_score":      rng.uniform(0, 1, n),
        "ict_bear_htf_score": rng.uniform(0, 1, n),
        "price_vs_sma50":     rng.uniform(-0.1, 0.1, n),
        "sma50_slope_5":      rng.uniform(-0.02, 0.02, n),
        "sma200_slope_10":    rng.uniform(-0.02, 0.02, n),
        "plus_di":            rng.uniform(0, 40, n),
        "minus_di":           rng.uniform(0, 40, n),
    }, index=[f"T{i}" for i in range(n)])

    expected = np.ones(n, dtype=bool)
    expected &= ~(raw["ssz_htf_score"].values > SSZ_VETO_THRESHOLD)
    expected &= ~(raw["ict_bear_htf_score"].values > ICT_BEAR_VETO_THRESHOLD)
    expected &= ((raw["price_vs_sma50"].values > 0.0)
                 & (raw["sma50_slope_5"].values > 0.0)
                 & (raw["sma200_slope_10"].values >= 0.0))
    expected &= ~(raw["minus_di"].values > raw["plus_di"].values)

    cross = raw.copy()
    cross.columns = [f"{FP}{c}" for c in cross.columns]
    actual = momentum_bull_quality_gate(cross, "momentum", FP)
    assert actual.tolist() == expected.tolist()
    assert 0 < int(expected.sum()) < n, "fixture must exercise both outcomes"
