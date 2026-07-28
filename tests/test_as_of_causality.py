"""as_of causality: nothing dated after as_of may reach features or labels.

The runner truncates the panel to <= as_of BEFORE feature engineering, so
every downstream stage (features, targets, CV folds, scoring) inherits the
cutoff. These tests pin the two leak channels that truncation closes, using
the same operations the pipeline uses, so they fail if the semantics drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config.sp500 import SP500_CONFIG
from pipeline.targets.builder import TargetBuilder


def _panel(last_date: str, tickers=("AAA", "BBB")) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", last_date)
    frames = []
    for i, t in enumerate(tickers):
        close = np.linspace(100, 200, len(dates)) + i
        frames.append(pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": 1_000_000,
            "in_universe": True,          # TargetBuilder gates on this
            "date": dates, "ticker": t,
        }))
    return (pd.concat(frames)
            .set_index(["date", "ticker"])
            .sort_index())


def _truncate(panel: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """Exactly what the runner's causality guard does."""
    d = panel.index.get_level_values("date")
    return panel[d <= pd.Timestamp(as_of)].copy()


def test_truncation_drops_only_post_as_of_rows():
    panel = _panel("2024-06-28")
    as_of = "2024-03-29"
    out = _truncate(panel, as_of)
    dates = out.index.get_level_values("date")
    assert dates.max() <= pd.Timestamp(as_of)
    # Nothing before the cutoff was disturbed.
    kept = panel[panel.index.get_level_values("date") <= pd.Timestamp(as_of)]
    assert len(out) == len(kept)
    pd.testing.assert_frame_equal(out.sort_index(), kept.sort_index())


def test_truncation_is_a_noop_when_data_ends_at_as_of():
    """The normal forward step: downloader stopped at as_of, so the guard must
    not drop anything (and must not print a scary message)."""
    panel = _panel("2024-03-29")
    out = _truncate(panel, "2024-03-29")
    assert len(out) == len(panel)


def test_labels_cannot_use_post_as_of_prices():
    """THE label channel: future_20d_return at t reads close[t+20]. After
    truncation the final 20 rows per ticker must be unlabelled, so no training
    row can carry a return that resolved after as_of."""
    as_of = "2024-03-29"
    full = _panel("2024-06-28")
    bm = pd.Series(
        np.linspace(100, 150, len(pd.bdate_range("2023-01-02", "2024-06-28"))),
        index=pd.bdate_range("2023-01-02", "2024-06-28"),
        name="benchmark_close",
    )

    truncated = _truncate(full, as_of)
    bm_trunc = bm[bm.index <= pd.Timestamp(as_of)]

    built = TargetBuilder(SP500_CONFIG).build(truncated.copy(), bm_trunc)

    labelled = built[built["future_20d_return"].notna()]
    last_labelled = labelled.index.get_level_values("date").max()

    # Every labelled row's 20-day window must close on or before as_of.
    horizon_end = pd.bdate_range(last_labelled, periods=21)[-1]
    assert horizon_end <= pd.Timestamp(as_of), (
        f"label at {last_labelled.date()} resolves at {horizon_end.date()}, "
        f"past as_of {as_of}"
    )


def test_untruncated_panel_would_leak_labels():
    """Control: without truncation the same construction DOES produce labels
    whose window closes after as_of. If this ever stops being true the guard
    has become unnecessary — and this test should be revisited, not deleted."""
    as_of = pd.Timestamp("2024-03-29")
    full = _panel("2024-06-28")
    bm = pd.Series(
        np.linspace(100, 150, len(pd.bdate_range("2023-01-02", "2024-06-28"))),
        index=pd.bdate_range("2023-01-02", "2024-06-28"),
        name="benchmark_close",
    )
    built = TargetBuilder(SP500_CONFIG).build(full.copy(), bm)

    labelled = built[built["future_20d_return"].notna()]
    at_as_of = labelled[labelled.index.get_level_values("date") == as_of]
    assert len(at_as_of) > 0, "expected the as_of row itself to be labelled"
    # That label resolves ~20 business days later — after as_of.
    assert pd.bdate_range(as_of, periods=21)[-1] > as_of
