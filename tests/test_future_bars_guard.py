"""FutureBars guard — a ticker may not run far ahead of its peers.

Regression pin for the AMTM failure: one ticker held bars ~13 months ahead of
the rest of the universe for ~45 walk-forward steps. Nothing flagged it,
because every existing check looks for data that is too OLD. It could not
self-correct either — the incremental downloader computes
new_start = last_bar + 1 day, sees that exceed the requested --end, and reports
"skipped (up to date)" on every subsequent run.

It surfaced only when the walk's as_of finally passed the rest of the
universe's last bar: that ticker alone then defined the newest panel date, the
scored cross-section collapsed to it, PIT filtered it out as a non-member, and
an empty frame reached LightGBM as "Input data must be 2 dimensional and non
empty" — six frames from the actual cause.

The invariant is deliberately "ahead of PEERS", not "ahead of as_of". See
test_uniformly_future_universe_is_not_flagged for why.
"""
from __future__ import annotations

import pandas as pd

from pipeline.monitoring.stale_data_guard import StaleDataGuard


def _panel(spec: dict[str, str]) -> pd.DataFrame:
    """spec: {ticker: last_date}. Every ticker starts 2024-01-01."""
    frames = []
    for ticker, last in spec.items():
        dates = pd.bdate_range("2024-01-01", last)
        frames.append(pd.DataFrame({
            "close": 100.0, "open": 100.0, "high": 101.0,
            "low": 99.0, "volume": 1_000,
            "date": dates, "ticker": ticker,
        }))
    return pd.concat(frames).set_index(["date", "ticker"]).sort_index()


def _universe(last: str, n: int = 20) -> dict[str, str]:
    return {f"T{i}": last for i in range(n)}


def test_clean_universe_reports_nothing():
    assert StaleDataGuard().check_future_bars(_panel(_universe("2025-06-27"))) == []


def test_single_runaway_ticker_is_flagged_by_name():
    """The AMTM case: one ticker far ahead of an otherwise-current universe."""
    spec = _universe("2025-06-27")
    spec["AMTM"] = "2026-07-27"
    issues = StaleDataGuard().check_future_bars(_panel(spec))

    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == "error"
    assert issue.check == "FutureBars"
    assert issue.tickers == ["AMTM"]          # named, and ONLY it
    assert "2026-07-27" in issue.message      # how far ahead
    assert "up to date" in issue.message      # why it cannot self-correct


def test_uniformly_future_universe_is_not_flagged():
    """The pre-download case, and why this measures against peers not as_of.

    When the whole price history is fetched once up front, every ticker sits
    ahead of every walk-forward step's as_of. An as_of-based check would fire
    on all ~1500 of them at all ~105 steps — an alarm that always fires is one
    nobody reads. Data ahead of as_of is normal (the causality guard truncates
    it); a ticker ahead of its PEERS is the defect.
    """
    assert StaleDataGuard().check_future_bars(_panel(_universe("2026-07-27"))) == []


def test_small_spread_within_tolerance_is_ignored():
    """A few tickers a day or two ahead is ordinary calendar noise, not a
    runaway file."""
    spec = _universe("2025-06-27")
    spec["EARLY1"] = "2025-06-30"
    spec["EARLY2"] = "2025-07-01"
    assert StaleDataGuard().check_future_bars(_panel(spec)) == []


def test_beyond_tolerance_is_flagged():
    spec = _universe("2025-06-27")
    spec["FAR"] = "2025-08-15"
    issues = StaleDataGuard().check_future_bars(_panel(spec))
    assert len(issues) == 1 and issues[0].tickers == ["FAR"]


def test_multiple_runaways_all_reported():
    spec = _universe("2025-06-27")
    spec.update({"X": "2025-09-01", "Y": "2025-12-01", "Z": "2026-01-01"})
    issues = StaleDataGuard().check_future_bars(_panel(spec))
    assert len(issues) == 1
    assert sorted(issues[0].tickers) == ["X", "Y", "Z"]
    assert "2026-01-01" in issues[0].message   # reports the worst offender


def test_stale_tickers_never_trigger_this_check():
    """Delisted / stale names lag the universe; that is LastBarStaleness's job,
    and must not leak into this one."""
    spec = _universe("2025-06-27")
    spec.update({"DEAD1": "2024-03-01", "DEAD2": "2024-06-01"})
    assert StaleDataGuard().check_future_bars(_panel(spec)) == []


def test_single_ticker_panel_is_a_noop():
    """One ticker has no peer group to be an outlier against — stay silent
    rather than compare it to itself."""
    assert StaleDataGuard().check_future_bars(_panel({"ONLY": "2026-07-27"})) == []


def test_severity_is_error_so_strict_mode_can_halt():
    """--strict_data_check routes error-severity issues into StaleDataError.
    A warning would let the walk proceed on a panel whose newest date belongs
    to a single ticker."""
    spec = _universe("2025-06-27")
    spec["BAD"] = "2026-07-27"
    issues = StaleDataGuard().check_future_bars(_panel(spec))
    assert all(i.severity == "error" for i in issues)


def test_fires_long_before_the_cross_section_starves():
    """The crash needs as_of to pass the universe's last bar. This guard must
    fire while the walk is still healthy, not at the step that breaks."""
    spec = _universe("2025-06-27")
    spec["AMTM"] = "2026-07-27"
    panel = _panel(spec)
    # Detection does not depend on as_of at all — it is available from the
    # very first step the bad file exists.
    assert len(StaleDataGuard().check_future_bars(panel)) == 1
