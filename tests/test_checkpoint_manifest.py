"""Checkpoint manifest guard — pins the stale-reuse bug it was written for.

The walk-forward failure mode: a panel checkpoint built at one --as_of was
re-used by a later step, so training saw features that stopped months earlier
and scoring used a stale cross-section, silently. The guard must reject a
checkpoint whenever the panel it was built from would differ.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from pipeline.utils.checkpoint_manifest import (
    compute_manifest, manifest_ok, write_manifest, _RECIPE_ENV,
)


def _data_dir(tmp_path, n=3):
    d = tmp_path / "csvs"
    d.mkdir()
    for i in range(n):
        (d / f"T{i}-1d.csv").write_text("Date,Open,High,Low,Close,Volume\n")
    return d


def test_matching_manifest_is_accepted(tmp_path):
    d = _data_dir(tmp_path)
    mf = tmp_path / "ckpt_manifest.json"
    cur = compute_manifest(d)
    write_manifest(mf, cur)
    ok, reason = manifest_ok(mf, compute_manifest(d))
    assert ok, reason


def test_missing_manifest_is_rejected(tmp_path):
    d = _data_dir(tmp_path)
    ok, reason = manifest_ok(tmp_path / "absent.json", compute_manifest(d))
    assert not ok
    assert "no manifest" in reason


def test_unreadable_manifest_is_rejected(tmp_path):
    d = _data_dir(tmp_path)
    mf = tmp_path / "ckpt_manifest.json"
    mf.write_text("{not valid json")
    ok, reason = manifest_ok(mf, compute_manifest(d))
    assert not ok
    assert "unreadable" in reason


def test_new_data_invalidates(tmp_path):
    """THE walk-forward case: a later step downloads fresh bars, so the CSV
    mtime moves and the older panel must not be trusted."""
    d = _data_dir(tmp_path)
    mf = tmp_path / "ckpt_manifest.json"
    write_manifest(mf, compute_manifest(d))

    time.sleep(1.1)   # mtime has 1s resolution in the fingerprint
    (d / "T0-1d.csv").write_text("Date,Open,High,Low,Close,Volume\n2024-01-02,1,1,1,1,1\n")

    ok, reason = manifest_ok(mf, compute_manifest(d))
    assert not ok
    assert "data changed" in reason


def test_added_ticker_invalidates(tmp_path):
    d = _data_dir(tmp_path)
    mf = tmp_path / "ckpt_manifest.json"
    write_manifest(mf, compute_manifest(d))
    (d / "NEW-1d.csv").write_text("Date,Open,High,Low,Close,Volume\n")
    ok, reason = manifest_ok(mf, compute_manifest(d))
    assert not ok
    assert "data changed" in reason


def test_unchanged_data_keeps_checkpoint(tmp_path):
    """A step whose download was a genuine no-op must NOT force a rebuild —
    the panel really is the same panel."""
    d = _data_dir(tmp_path)
    mf = tmp_path / "ckpt_manifest.json"
    write_manifest(mf, compute_manifest(d))
    for _ in range(3):
        ok, _ = manifest_ok(mf, compute_manifest(d))
        assert ok


@pytest.mark.parametrize("cli_a,cli_b", [
    ({"pit_universe": False, "train_start": "2010-01-01"},
     {"pit_universe": True,  "train_start": "2010-01-01"}),
    ({"pit_universe": True,  "train_start": "2010-01-01"},
     {"pit_universe": True,  "train_start": "2015-01-01"}),
])
def test_recipe_flags_invalidate(tmp_path, cli_a, cli_b):
    """--pit_universe and --train_start change the panel's CONTENT (the
    checkpoint is built from the already-filtered, already-trimmed panel), so
    they must invalidate even when code and data are identical."""
    d = _data_dir(tmp_path)
    mf = tmp_path / "ckpt_manifest.json"
    write_manifest(mf, compute_manifest(d, cli=cli_a))
    ok, reason = manifest_ok(mf, compute_manifest(d, cli=cli_b))
    assert not ok
    assert "cli changed" in reason


def test_feature_build_workers_does_not_invalidate(tmp_path):
    """FEATURE_BUILD_WORKERS is a perf knob: test_parallel_build pins parallel
    output to serial bit-for-bit, so it must not trigger a spurious rebuild."""
    assert "FEATURE_BUILD_WORKERS" not in _RECIPE_ENV
    d = _data_dir(tmp_path)
    mf = tmp_path / "ckpt_manifest.json"
    prev = os.environ.get("FEATURE_BUILD_WORKERS")
    try:
        os.environ["FEATURE_BUILD_WORKERS"] = "1"
        write_manifest(mf, compute_manifest(d))
        os.environ["FEATURE_BUILD_WORKERS"] = "4"
        ok, reason = manifest_ok(mf, compute_manifest(d))
        assert ok, reason
    finally:
        if prev is None:
            os.environ.pop("FEATURE_BUILD_WORKERS", None)
        else:
            os.environ["FEATURE_BUILD_WORKERS"] = prev


def test_code_change_invalidates(tmp_path, monkeypatch):
    d = _data_dir(tmp_path)
    mf = tmp_path / "ckpt_manifest.json"
    write_manifest(mf, compute_manifest(d))
    # Simulate an edit to pipeline/features or pipeline/targets.
    monkeypatch.setattr(
        "pipeline.utils.checkpoint_manifest._code_fingerprint",
        lambda: "deadbeefdeadbeef",
    )
    ok, reason = manifest_ok(mf, compute_manifest(d))
    assert not ok
    assert "code_sha changed" in reason


def test_manifest_is_json_serialisable(tmp_path):
    d = _data_dir(tmp_path)
    mf = tmp_path / "ckpt_manifest.json"
    cur = compute_manifest(d, cli={"pit_universe": True, "train_start": None})
    write_manifest(mf, cur)
    assert json.loads(mf.read_text()) == json.loads(json.dumps(cur))
