"""The ensemble scores on TWO inputs, and only one of them used to be recorded.

`EnsembleRanker.score(X, hist_vol_20d)` blends an LGBM rank with an inverse-vol
tilt. The tilt is the second argument, it is read from the cross-section rather
than from the selected-feature roster, and it is not necessarily a selected
feature at all — in artefacts/nse_local/momentum it is not one.

So `features_scored_*.parquet` records it explicitly. These tests pin the facts
that make that worth doing: the tilt reorders the book, its absence is silent,
and there is a cross-section size below which it cannot do anything at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.models.ensemble import EnsembleConfig, EnsembleRanker


class _StubLGBM:
    """Fixed raw score per row, so the tilt is the only thing that varies."""

    def __init__(self, raw):
        self._raw = np.asarray(raw, dtype=float)

    def predict(self, X):
        return self._raw[: len(X)]


def _fixture(n):
    """Descending LGBM preference: T000 best, T00(n-1) worst."""
    idx = pd.Index([f"T{i:03d}" for i in range(n)], name="ticker")
    X = pd.DataFrame({"features_a": np.linspace(0, 1, n)}, index=idx)
    return EnsembleRanker(_StubLGBM(np.linspace(1.0, 0.0, n))), X, idx


def _adversarial_vol(idx):
    """Worst vol on the LGBM's favourite, best vol on its runner-up.

    A vol series that is merely the reverse of the LGBM order does NOT reorder
    anything: both terms are then linear in the index, the blend stays monotone
    and the closing re-rank erases the tilt entirely. The tilt only bites when
    vol is non-monotonic with respect to the LGBM rank, so put the extremes on
    two adjacent names.
    """
    vol = np.full(len(idx), 0.5)
    vol[0] = 1.0   # highest vol  -> worst  vol_rank, on the best LGBM name
    vol[1] = 0.0   # lowest vol   -> best   vol_rank, on the second-best
    return pd.Series(vol, index=idx)


# score() ends in _rank01(blend), so the tilt matters only when it REORDERS.
# With the extremes on the top pair the blends are 0.9 + 0.1/n and 0.9(n-1)/n +
# 0.1, which cross at exactly n = 1/VOL_WEIGHT. At that n they tie, and _rank01
# breaks ties by position — which still flips the pair. So the tilt can change
# the output from n = 1/VOL_WEIGHT upward, and is powerless below it.
_MIN_N_FOR_TILT = int(1.0 / EnsembleConfig.VOL_WEIGHT)


def test_the_tilt_reorders_a_realistic_cross_section():
    """If it could not, recording it would be pointless. Production books are
    500-1200 names; 40 is well past the point where the tilt bites."""
    ens, X, idx = _fixture(40)
    with_tilt = ens.score(X, _adversarial_vol(idx))
    without = ens.score(X, None)
    assert not np.allclose(with_tilt, without), (
        "the vol tilt made no difference — check VOL_WEIGHT"
    )
    # Specifically: the tilt demoted the LGBM's top pick below the runner-up.
    assert with_tilt[0] < with_tilt[1]
    assert without[0] > without[1]


def test_the_tilt_is_inert_on_a_tiny_cross_section():
    """A real property of the closing re-rank, not a quirk of the fixture.

    At or below 1/VOL_WEIGHT names the LGBM rank gaps are wider than the tilt
    can reach, so it is arithmetically incapable of changing the output. Worth
    pinning: a smoke test on a handful of tickers shows no tilt effect, and
    that is correct rather than a broken tilt.
    """
    n = _MIN_N_FOR_TILT - 1
    ens, X, idx = _fixture(n)
    assert np.allclose(ens.score(X, _adversarial_vol(idx)), ens.score(X, None)), (
        f"expected the tilt to be inert at n={n}"
    )


def test_absent_tilt_falls_back_to_the_lgbm_order_silently():
    """No error and no warning — every ticker just gets the same 0.5 tilt.

    That is exactly why presence has to be written down: a run without the tilt
    looks identical to one with it, in the output and on the console alike.
    """
    ens, X, idx = _fixture(40)
    scores = ens.score(X, None)
    assert np.all(np.isfinite(scores))
    # A uniform tilt is a constant, so the blend is monotone in the LGBM rank
    # and the ordering is the LGBM ordering exactly.
    assert list(np.argsort(-scores)) == list(range(40))


def test_tilt_weight_is_material():
    """A tenth of the blend is not a rounding error."""
    assert EnsembleConfig.VOL_WEIGHT > 0.0
    assert EnsembleConfig.LGBM_WEIGHT + EnsembleConfig.VOL_WEIGHT == 1.0
