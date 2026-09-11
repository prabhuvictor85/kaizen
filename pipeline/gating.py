"""
Momentum-bull quality gate — shared by all run scripts.

Removes model-driven longs that are technically broken (NKE / CPRT
false-positive filter):
  • under heavy overhead supply        → ssz_htf_score > 0.6   (CPRT-type)
  • overhead bearish ICT structure     → ict_bear_htf_score > 0.4
    (bear OB/BB/FVG composite across 5 TFs — complementary engine to SSZ:
     measured on 22k momentum-universe rows, 0.4 vetoes 4.1% of which
     3.8% are NOT caught by the ssz prong; 0.3 too broad, 0.5 inert)
  • broken/declining trend stack       → NOT(price>sma50 AND sma50↑ AND sma200↛↓) (NKE-type)
  • ADX direction owned by bears       → −DI > +DI

Applies to momentum BULL candidates only. Bear/reversal untouched.
scores_detail still records full ungated scores for transparency.

The four prongs are independent and mean four different things, so the gate
reports them per ticker rather than only their AND — a downstream "the gate
vetoed it" that cannot say WHICH prong fired names no fix. `..._detail()`
returns the breakdown; `momentum_bull_quality_gate()` returns just the keep
mask and is implemented on top of it, so the reported prongs are by
construction the ones that were applied.

A prong whose feature columns are missing is DISABLED, not passing. That case
is recorded as <NA>, never False — "this prong could not run" and "this prong
found nothing wrong" are different facts and only one of them is reassuring.

Calibration provenance: the STRUCTURAL prongs (ssz>0.6, ict_bear>0.4) were
calibrated 2026-06 on the US large/mid universe at ~5% combined prevalence —
prevalence-calibrated, not outcome-validated. The veto rate is self-monitored
each run; >15% means universe composition or feature distributions have
shifted: re-run the dose-response sweep (scripts/diagnostics/
diag_zone_overlap.py + threshold sweep) before trusting the lists. The NSE
universe was NOT part of the calibration sample — watch the printed veto rate
on the first NSE runs; the alarm will flag gross miscalibration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SSZ_VETO_THRESHOLD       = 0.6
ICT_BEAR_VETO_THRESHOLD  = 0.4
STRUCTURAL_RATE_BASELINE = 0.05
STRUCTURAL_RATE_ALARM    = 0.15

# Per-ticker veto columns, in the order the prongs are documented above.
VETO_COLUMNS = [
    "veto_ssz_supply",
    "veto_ict_bear_structure",
    "veto_broken_trend",
    "veto_bearish_adx",
]
# The two prongs the calibration note covers. The trend/ADX prongs are
# deliberately regime-dependent and are NOT monitored.
STRUCTURAL_VETO_COLUMNS = ["veto_ssz_supply", "veto_ict_bear_structure"]


def momentum_bull_quality_gate_detail(
    cross_wl: pd.DataFrame,
    mode: str,
    feature_prefix: str,
) -> pd.DataFrame:
    """
    Return a per-ticker frame aligned to cross_wl.index with one column per
    prong plus `keep`.

    Each veto column is nullable boolean:
        True   — this prong vetoed the ticker
        False  — this prong ran and did not veto
        <NA>   — this prong could not run (its feature columns are absent)

    `keep` is True when no prong vetoed. <NA> counts as no veto, which is what
    a disabled prong has always done — the point of the <NA> is that the output
    now says so instead of looking like a clean pass.

    For mode != "momentum" every prong is <NA> and keep is all True.
    """
    idx = cross_wl.index
    detail = pd.DataFrame(index=idx)

    if mode != "momentum":
        for col in VETO_COLUMNS:
            detail[col] = pd.Series(pd.NA, index=idx, dtype="boolean")
        detail["keep"] = pd.Series(True, index=idx, dtype="boolean")
        return detail

    def _gc(col):
        full = f"{feature_prefix}{col}"
        return cross_wl[full].fillna(0.0).values.astype(float) if full in cross_wl.columns else None

    _ssz   = _gc("ssz_htf_score")
    _ictb  = _gc("ict_bear_htf_score")
    _pvs50 = _gc("price_vs_sma50")
    _s50   = _gc("sma50_slope_5")
    _s200  = _gc("sma200_slope_10")
    _pdi   = _gc("plus_di")
    _mdi   = _gc("minus_di")

    def _col(values):
        """None (prong disabled) → all <NA>; else a nullable boolean column."""
        if values is None:
            return pd.Series(pd.NA, index=idx, dtype="boolean")
        return pd.Series(np.asarray(values, dtype=bool), index=idx, dtype="boolean")

    detail["veto_ssz_supply"] = _col(
        None if _ssz is None else _ssz > SSZ_VETO_THRESHOLD
    )
    detail["veto_ict_bear_structure"] = _col(
        None if _ictb is None else _ictb > ICT_BEAR_VETO_THRESHOLD
    )
    detail["veto_broken_trend"] = _col(
        None if (_pvs50 is None or _s50 is None or _s200 is None)
        else ~((_pvs50 > 0.0) & (_s50 > 0.0) & (_s200 >= 0.0))
    )
    detail["veto_bearish_adx"] = _col(
        None if (_pdi is None or _mdi is None) else _mdi > _pdi
    )

    # <NA> is "prong disabled" and has always behaved as no-veto.
    vetoed = detail[VETO_COLUMNS].fillna(False).any(axis=1)
    detail["keep"] = (~vetoed).astype("boolean")
    return detail


def momentum_bull_quality_gate(
    cross_wl: pd.DataFrame,
    mode: str,
    feature_prefix: str,
    detail: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Return a boolean keep-mask aligned to cross_wl.index.

    For mode != "momentum" the gate is a no-op (all True). Missing feature
    columns disable their prong individually; if every prong's columns are
    missing the gate is inactive and a loud warning is printed.

    Pass `detail` (from momentum_bull_quality_gate_detail) to reuse an already
    computed breakdown instead of recomputing it.
    """
    if detail is None:
        detail = momentum_bull_quality_gate_detail(cross_wl, mode, feature_prefix)

    _gate = detail["keep"].fillna(True).astype(bool)

    if mode != "momentum":
        return _gate

    _n_rej = int((~_gate).sum())
    if _n_rej:
        print(f"  [{mode}] bull quality gate: rejected {_n_rej} of {len(cross_wl)} "
              f"candidates (ssz-supply / ict-bear-structure / broken-trend / bearish-ADX)")
        # Which prong, not just how many — the four have four different fixes.
        counts = {c: int(detail[c].fillna(False).sum()) for c in VETO_COLUMNS}
        print(f"  [{mode}] veto by prong: "
              + "  ".join(f"{c.replace('veto_', '')}={n}" for c, n in counts.items()))

    # ── Calibration self-check ─────────────────────────────────────────
    # Only the structural prongs are monitored. The trend/ADX prongs are
    # deliberately regime-dependent (they reject most of the universe in a
    # bear market — correct, not drift).
    disabled = [c for c in VETO_COLUMNS if detail[c].isna().all()]
    if len(disabled) == len(VETO_COLUMNS):
        print(f"  [{mode}] *** GATE INACTIVE: none of the gate's feature columns "
              f"were found in the panel — quality gate is NOT filtering. "
              f"Check feature names in engineer.py vs pipeline/gating.py.")
    else:
        if disabled:
            print(f"  [{mode}] *** GATE PARTIALLY DISABLED: no feature columns for "
                  f"{', '.join(c.replace('veto_', '') for c in disabled)} — "
                  f"{len(disabled)} of {len(VETO_COLUMNS)} prongs are not filtering.")
        _struct_rej  = detail[STRUCTURAL_VETO_COLUMNS].fillna(False).any(axis=1)
        _struct_rate = float(_struct_rej.mean()) if len(cross_wl) else 0.0
        print(f"  [{mode}] gate calibration: structural veto rate "
              f"{_struct_rate:.1%} (baseline ~{STRUCTURAL_RATE_BASELINE:.0%}, "
              f"alarm >{STRUCTURAL_RATE_ALARM:.0%})")
        if _struct_rate > STRUCTURAL_RATE_ALARM:
            print(f"  [{mode}] *** GATE CALIBRATION ALARM: structural veto rate "
                  f"{_struct_rate:.0%} is 3x the calibrated baseline — universe "
                  f"composition has shifted. Re-run the threshold sweep before "
                  f"trusting this watchlist.")

    return _gate
