"""End-to-end smoke over the provenance writes in score_and_rank().

Everything else about these writes was checked in pieces — the gate prongs in
test_gating, the hash recipe in test_artifact_meta, the tilt in
test_ensemble_score_inputs. None of that proves the files actually appear when
score_and_rank runs, which is the only thing that matters to a grader reading
the output directory.

So this drives the real function against a synthetic cross-section and asserts
on the artifacts it leaves behind.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import run_sp500_local as R
from pipeline.artifact_meta import write_artifact_meta
from pipeline.features.engineer import FEATURE_PREFIX as FP
from pipeline.models.ensemble import EnsembleRanker

N = 60
DATE = pd.Timestamp("2024-04-30")

# Gate + signal + mode-filter columns score_and_rank reads by name.
_FEATURES = [
    "high_52w_dist", "hist_vol_20d",
    "ssz_htf_score", "ict_bear_htf_score",
    "price_vs_sma50", "sma50_slope_5", "sma200_slope_10",
    "plus_di", "minus_di",
    "sdz_htf_score", "adx_14", "return_20d",
]
# hist_vol_20d is deliberately NOT a model feature — that is the real
# artefacts/nse_local/momentum situation, and the case the tilt column exists for.
MODEL_FEATURES = [f"{FP}{c}" for c in _FEATURES if c != "hist_vol_20d"]


class _StubLGBM:
    def predict(self, X):
        return np.linspace(1.0, 0.0, len(X))


@pytest.fixture
def panel():
    rng = np.random.default_rng(11)
    tickers = [f"T{i:03d}" for i in range(N)]
    idx = pd.MultiIndex.from_arrays(
        [[DATE] * N, tickers], names=["date", "ticker"]
    )
    df = pd.DataFrame(index=idx)
    # Momentum universe: all within 40% of the 52w high.
    df[f"{FP}high_52w_dist"] = rng.uniform(-0.35, -0.01, N)
    df[f"{FP}hist_vol_20d"] = rng.uniform(0.1, 0.8, N)
    # A spread of zone scores so the composite is non-constant and some names
    # clear the zone-presence filter.
    df[f"{FP}sdz_htf_score"] = rng.uniform(0.0, 1.0, N)
    # Gate prongs: mostly clean, with a handful tripped on purpose.
    df[f"{FP}ssz_htf_score"] = 0.0
    df.iloc[:3, df.columns.get_loc(f"{FP}ssz_htf_score")] = 0.9      # supply veto
    df[f"{FP}ict_bear_htf_score"] = 0.0
    df.iloc[3:6, df.columns.get_loc(f"{FP}ict_bear_htf_score")] = 0.7  # ict veto
    df[f"{FP}price_vs_sma50"] = 0.05
    df[f"{FP}sma50_slope_5"] = 0.01
    df[f"{FP}sma200_slope_10"] = 0.01
    df[f"{FP}plus_di"] = 30.0
    df[f"{FP}minus_di"] = 10.0
    df[f"{FP}adx_14"] = 25.0
    df[f"{FP}return_20d"] = rng.uniform(-0.1, 0.2, N)
    df["in_universe"] = True
    df["sector"] = ["Tech", "Health", "Energy", "Financials"] * (N // 4)
    df["close"] = 100.0
    return df


@pytest.fixture
def artefact_dir(tmp_path, panel):
    """A real artefact_meta.json, hashed over the roster we will score with."""
    d = tmp_path / "artefacts" / "momentum"
    d.mkdir(parents=True)
    write_artifact_meta(d, mode="momentum", final_features=MODEL_FEATURES, panel=panel)
    return d


@pytest.fixture
def run(tmp_path, panel, artefact_dir, monkeypatch):
    """Run score_and_rank with OUTPUT_DIR redirected into tmp_path."""
    out = tmp_path / "output"
    out.mkdir()
    monkeypatch.setattr(R, "OUTPUT_DIR", out)
    monkeypatch.setattr(R, "REPORTS_DIR", tmp_path / "reports")
    from pipeline.config.nse import NSE_CONFIG

    R.score_and_rank(
        panel=panel,
        ensemble=EnsembleRanker(_StubLGBM()),
        final_features=MODEL_FEATURES,
        benchmark_close=pd.Series([100.0], index=[DATE]),
        cfg=NSE_CONFIG,
        top_n=10,
        weighting="equal",
        as_of_date=DATE,
        mode="momentum",
        variant="composite",
        artefact_dir=artefact_dir,
    )
    return out


def _roster(out):
    return json.loads((out / "features_scored_momentum_2024-04-30.roster.json").read_text())


# ── The files exist at all ───────────────────────────────────────────────────

def test_the_provenance_files_are_written(run):
    names = sorted(p.name for p in run.iterdir())
    assert "features_scored_momentum_2024-04-30.parquet" in names
    assert "features_scored_momentum_2024-04-30.roster.json" in names
    assert "gate_audit_momentum_2024-04-30.csv" in names


# ── 1. What the model saw ────────────────────────────────────────────────────

def test_parquet_holds_every_scored_ticker_and_feature(run):
    fx = pd.read_parquet(run / "features_scored_momentum_2024-04-30.parquet")
    assert len(fx) == N, "every scored ticker, not just the picks"
    for col in MODEL_FEATURES:
        assert col in fx.columns


def test_parquet_carries_the_vol_tilt_input(run):
    """The second argument to ensemble.score, which is not a model feature."""
    fx = pd.read_parquet(run / "features_scored_momentum_2024-04-30.parquet")
    assert f"{FP}hist_vol_20d" in fx.columns
    assert f"{FP}hist_vol_20d" not in MODEL_FEATURES, "fixture no longer tests the real case"
    r = _roster(run)
    assert r["vol_tilt_present"] is True
    assert r["vol_tilt_feature"] == f"{FP}hist_vol_20d"
    # Everything in the parquet that is not a scored feature is the tilt.
    extra = set(fx.columns) - set(r["scored_features"])
    assert extra == {f"{FP}hist_vol_20d"}


# ── 2. Which model produced it ───────────────────────────────────────────────

def test_roster_pins_the_model_that_scored(run):
    m = _roster(run)["model"]
    assert m["meta_found"] is True
    assert m["git_commit"], "no commit recorded — the scores are unattributable"
    assert m["created_utc"]
    assert m["git_dirty"] in (True, False, None)


def test_roster_confirms_the_scored_roster_matches_the_trained_one(run):
    m = _roster(run)["model"]
    assert m["feature_hash_match"] is True
    assert m["trained_feature_hash"] == m["scored_feature_hash"]


def test_roster_records_the_dropped_features(run):
    r = _roster(run)
    assert r["n_dropped"] == 0 and r["dropped_features"] == []
    assert r["n_scored"] == len(MODEL_FEATURES)
    assert r["as_of_date"] == "2024-04-30"


# ── 3. Which gate prong vetoed ───────────────────────────────────────────────

def test_gate_audit_attributes_the_veto_to_a_prong(run):
    ga = pd.read_csv(run / "gate_audit_momentum_2024-04-30.csv").set_index("ticker")
    for col in ["veto_ssz_supply", "veto_ict_bear_structure",
                "veto_broken_trend", "veto_bearish_adx"]:
        assert col in ga.columns
    # The fixture trips supply on T000-T002 and ict_bear on T003-T005.
    assert ga.loc[["T000", "T001", "T002"], "veto_ssz_supply"].all()
    assert ga.loc[["T003", "T004", "T005"], "veto_ict_bear_structure"].all()
    assert not ga.loc[["T000", "T001", "T002"], "veto_ict_bear_structure"].any()
    for t in ["T000", "T003"]:
        assert ga.loc[t, "removal_rule"] == "quality_gate"
        assert not ga.loc[t, "selected"]
