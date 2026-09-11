"""Checkpoint-frequency guard for the expanding-window zone labeler.

F-C12, third site. `DEFAULT_CHECKPOINT_FREQ` was the string "YE", which pandas
below 2.2 rejects outright — `_build_checkpoints` raised `ValueError: Invalid
frequency: YE`. That exception is caught in train.py, which falls back to
whatever zone columns already exist whenever labeling fails. Those are the
ordinary zone columns, not the time-honest ones this labeler produces, so a
version mismatch quietly trained on leak-prone labels behind one warning line.

Offset objects were never renamed by pandas, so they carry no such exposure.
"""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.data.zone_labeler import (
    DEFAULT_CHECKPOINT_FREQ,
    ExpandingWindowZoneLabeler,
)


def _labeler(freq=None):
    kw = {"checkpoint_freq": freq} if freq is not None else {}
    return ExpandingWindowZoneLabeler(analyze_zones_fn=lambda df: df, **kw)


def _dates(periods=1200):
    return pd.DatetimeIndex(pd.bdate_range("2019-01-01", periods=periods))


def test_default_checkpoint_freq_is_not_a_string_alias():
    """The whole point: a spelling pandas can rename must not be the default."""
    assert not isinstance(DEFAULT_CHECKPOINT_FREQ, str), (
        f"default is the string {DEFAULT_CHECKPOINT_FREQ!r} — use pd.offsets"
    )
    assert isinstance(DEFAULT_CHECKPOINT_FREQ, pd.offsets.YearEnd)


def test_default_builds_checkpoints_on_this_pandas():
    """Runs the exact call that raised on pandas < 2.2."""
    cps = _labeler()._build_checkpoints(_dates())
    assert cps, "no checkpoints built with the default frequency"
    assert len(cps) >= 4, f"expected roughly one per year, got {len(cps)}"


def test_default_checkpoints_are_year_end_anchored():
    """Period-END anchoring is the invariant, not the spelling.

    Every checkpoint but the last must be the final trading day of a December;
    the last is the panel's final date, which the labeler always appends so the
    trailing partial period is covered.
    """
    cps = _labeler()._build_checkpoints(_dates())
    for cp in cps[:-1]:
        assert cp.month == 12, f"{cp.date()} is not a year-end checkpoint"


@pytest.mark.parametrize("alias", ["QS", "MS"])
def test_legacy_period_start_strings_still_accepted(alias):
    """Backward compatibility: pandas never renamed the period-START aliases,
    and callers documented to pass 'QS'/'MS' must keep working.
    """
    cps = _labeler(alias)._build_checkpoints(_dates())
    assert cps, f"{alias} produced no checkpoints"


def test_offset_objects_are_accepted_for_other_periods():
    yearly = _labeler(pd.offsets.YearEnd())._build_checkpoints(_dates())
    quarterly = _labeler(pd.offsets.QuarterEnd())._build_checkpoints(_dates())
    assert len(quarterly) > len(yearly), (
        "quarterly checkpoints should outnumber yearly ones"
    )
