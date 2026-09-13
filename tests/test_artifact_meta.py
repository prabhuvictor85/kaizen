"""Artifact provenance guard — pins what artefact_meta.json must carry.

The failure mode this file exists for: a scored watchlist that cannot be traced
back to the code and feature roster that produced it. Three ways that happens,
all pinned below.

  1. The roster is not in the meta file, so recovering it means unpickling the
     model or reading a checkpoint that a cleanup step deleted.
  2. git_commit records HEAD from a dirty tree, so the SHA describes code that
     never ran — and nothing says so.
  3. git_dirty answers False when git simply failed. False is a positive claim
     ("the SHA is trustworthy"); only None may mean "could not tell".
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

import pipeline.artifact_meta as am
from pipeline.artifact_meta import write_artifact_meta


FEATURES = ["features_adx_14", "features_return_20d", "features_sdz_htf_score"]


def _read_meta(tmp_path, panel, features=FEATURES, **kw):
    # write_artifact_meta never raises, so a missing art_dir would surface as a
    # silent None rather than an error. Create it, as the training run does.
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = write_artifact_meta(
        tmp_path, mode="momentum", final_features=features, panel=panel, **kw
    )
    assert path is not None, "metadata write returned None"
    return json.loads(path.read_text())


# ── 1. The roster travels with the model ─────────────────────────────────────

def test_selected_features_is_written_in_order(tmp_path, tiny_panel):
    meta = _read_meta(tmp_path, tiny_panel)
    assert meta["selected_features"] == FEATURES


def test_feature_count_agrees_with_the_roster(tmp_path, tiny_panel):
    """A count that disagrees with the list means one of them is stale."""
    meta = _read_meta(tmp_path, tiny_panel)
    assert meta["feature_count"] == len(meta["selected_features"])


def test_roster_survives_a_pandas_index(tmp_path, tiny_panel):
    """final_features arrives as an Index from feature selection, not a list.

    json.dump refuses numpy scalars, so the list() conversion is load-bearing:
    without it the whole meta file silently fails to write.
    """
    meta = _read_meta(tmp_path, tiny_panel, features=pd.Index(FEATURES))
    assert meta["selected_features"] == FEATURES
    assert all(isinstance(f, str) for f in meta["selected_features"])


def test_feature_hash_recipe_is_reproducible_by_the_scorer(tmp_path, tiny_panel):
    """Cross-file contract with run_sp500_local.py.

    Scoring re-hashes the roster it is about to use and compares it to this
    file's feature_hash, warning loudly when they differ — that is how a
    selected_features.txt / ensemble.pkl disagreement gets caught. If the two
    hashing recipes ever drift apart the alarm fires on every clean run and
    stops meaning anything, so pin the recipe here: sha256 of the
    newline-joined roster, first 16 hex chars.
    """
    meta = _read_meta(tmp_path, tiny_panel)
    scorer_side = hashlib.sha256("\n".join(FEATURES).encode()).hexdigest()[:16]
    assert meta["feature_hash"] == scorer_side


def test_feature_hash_tracks_the_roster(tmp_path, tiny_panel):
    same = _read_meta(tmp_path / "a", tiny_panel, features=FEATURES)
    again = _read_meta(tmp_path / "b", tiny_panel, features=list(FEATURES))
    other = _read_meta(tmp_path / "c", tiny_panel, features=FEATURES[:2])
    assert same["feature_hash"] == again["feature_hash"]
    assert same["feature_hash"] != other["feature_hash"]


# ── 2. The dirty flag ────────────────────────────────────────────────────────

def test_git_dirty_key_is_always_present(tmp_path, tiny_panel):
    meta = _read_meta(tmp_path, tiny_panel)
    assert "git_dirty" in meta
    assert meta["git_dirty"] in (True, False, None)


def test_git_dirty_true_when_tree_has_changes(monkeypatch):
    monkeypatch.setattr(
        am.subprocess, "run",
        lambda *a, **k: _Completed(0, " M pipeline/artifact_meta.py\n"),
    )
    assert am._git_dirty() is True


def test_git_dirty_false_when_tree_is_clean(monkeypatch):
    monkeypatch.setattr(am.subprocess, "run", lambda *a, **k: _Completed(0, ""))
    assert am._git_dirty() is False


def test_git_dirty_is_none_when_git_fails(monkeypatch):
    """THE bug this guard exists for.

    A failed git call also produces empty stdout. Reading that as "clean"
    asserts the commit SHA is trustworthy precisely when it is not knowable.
    """
    monkeypatch.setattr(am.subprocess, "run", lambda *a, **k: _Completed(128, ""))
    assert am._git_dirty() is None


def test_git_dirty_is_none_when_git_is_missing(monkeypatch):
    def _boom(*a, **k):
        raise FileNotFoundError("git")
    monkeypatch.setattr(am.subprocess, "run", _boom)
    assert am._git_dirty() is None


def test_git_commit_is_none_when_git_is_missing(monkeypatch):
    def _boom(*a, **k):
        raise OSError("no git")
    monkeypatch.setattr(am.subprocess, "run", _boom)
    assert am._git_commit() is None


# ── 3. Metadata never breaks a training run ──────────────────────────────────

def test_bad_panel_returns_none_instead_of_raising(tmp_path):
    """The module contract: a metadata failure must not fail a 6-hour train."""
    junk = pd.DataFrame({"close": [1.0, 2.0]})       # no (date, ticker) MultiIndex
    assert write_artifact_meta(
        tmp_path, mode="momentum", final_features=FEATURES, panel=junk
    ) is None


def test_extra_fields_do_not_displace_provenance(tmp_path, tiny_panel):
    meta = _read_meta(tmp_path, tiny_panel, extra={"n_trials": 40})
    assert meta["n_trials"] == 40
    assert meta["selected_features"] == FEATURES
    assert "git_dirty" in meta


# ── helper ───────────────────────────────────────────────────────────────────

class _Completed:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""
