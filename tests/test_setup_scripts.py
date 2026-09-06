"""Shell-script guards — pin the two failures that are invisible until a run.

Both are things a reviewer reads straight past:

  1. feat_cols.txt sitting in a delete list next to a 5.6 GB pickle. Removing
     3 KB frees nothing and destroys the only record of the panel's feature
     roster at build time. It has been deleted from two separate places.
  2. A syntax error in a bootstrap script. Under `set -euo pipefail` these
     scripts abort mid-way, and the only symptom is a half-configured server.
"""
from __future__ import annotations

import re
import shutil
import subprocess

import pytest


SCRIPTS = [
    "run_pipeline_hetzner.sh",
    "run_pipeline_chain.sh",
    "scripts/server_setup.sh",
    "scripts/bootstrap_from_volume.sh",
    "scripts/run_full_training.sh",
]


# ── 1. feat_cols.txt is provenance, not a cache ──────────────────────────────

def test_feat_cols_is_never_deleted(project_root):
    """It costs 3 KB and it is the feature roster the run was built from."""
    offenders = []
    for name in SCRIPTS:
        path = project_root / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        # Delete statements, including the multi-line backslash-continued ones.
        for stmt in re.findall(r"\brm\b[^\n]*(?:\\\n[^\n]*)*", text):
            if "feat_cols.txt" in stmt:
                offenders.append(f"{name}: {stmt.splitlines()[0].strip()}")
    assert not offenders, "feat_cols.txt is in a delete list:\n" + "\n".join(offenders)


def test_hetzner_still_deletes_the_big_pickles(project_root):
    """Guard the guard: keeping feat_cols.txt must not spare the 5.6 GB files."""
    text = (project_root / "run_pipeline_hetzner.sh").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "panel_features.pkl" in text
    assert "panel_targets.pkl" in text


# ── 2. The scripts parse ─────────────────────────────────────────────────────

@pytest.mark.parametrize("name", SCRIPTS)
def test_script_parses(project_root, name):
    path = project_root / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available on this machine")
    proc = subprocess.run(
        [bash, "-n", str(path)], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, f"{name} failed to parse:\n{proc.stderr}"


# Deliberately not tested here: CRLF line endings. On a Windows checkout with
# core.autocrlf=true the working copy is legitimately CRLF while the committed
# blob — which is what the server clones — stays LF, so a worktree check fails
# for every Windows developer and proves nothing about the server. Pin it with
# `.gitattributes` (`*.sh text eol=lf`) rather than with a test.
