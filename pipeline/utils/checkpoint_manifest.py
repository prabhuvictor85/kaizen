"""Checkpoint manifest guard.

The panel checkpoints (panel_features.pkl / panel_targets.pkl) are trusted on
pure file-existence: nothing checks whether the pickle matches the CURRENT
feature code, CLI recipe flags, or price data. That cuts both ways — silent
stale reuse (a run "resumes" onto an older panel) or paranoid manual deletion
(a rebuild that wasn't needed).

Concretely, the bug this was written for: the checkpoint exists to share the
feature+target panel between the momentum and reversal training passes WITHIN
one invocation. But it persists on disk, so the next walk-forward step at a
LATER --as_of re-used it — training on a panel whose features stopped months
before that step's as_of, and scoring a stale cross-section, with no warning.

A manifest written beside the checkpoint records what the panel was built
from; on load, mismatch => the checkpoint is IGNORED with a loud reason and
the panel rebuilds. The guard can only ever force a rebuild you'd have wanted
anyway — it never widens what gets trusted.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# Env vars that change the panel recipe: a checkpoint built under different
# values is a DIFFERENT panel even if the code and data are unchanged.
#
# Deliberately NOT here: FEATURE_BUILD_WORKERS. It only decides how many
# processes build the per-ticker section; tests/test_parallel_build.py pins
# parallel output to serial output bit-for-bit, so it cannot change the panel
# and must not trigger a spurious rebuild.
_RECIPE_ENV: list[str] = []

# Source whose changes invalidate a features/targets checkpoint.
_CODE_ROOTS = ["features", "targets"]


def _code_fingerprint() -> str:
    base = Path(__file__).resolve().parent.parent  # pipeline/
    h = hashlib.sha256()
    for root in _CODE_ROOTS:
        for p in sorted((base / root).rglob("*.py")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _data_fingerprint(data_dir) -> dict:
    """Cheap freshness proxy: CSV count + newest mtime. A re-download changes
    mtimes; an added/delisted ticker changes the count.

    This is what catches the walk-forward case: each step downloads bars up to
    its own as_of, which moves max_mtime, which invalidates the checkpoint.
    When a step's download is genuinely a no-op the fingerprint is unchanged
    and the cache is correctly kept — the panel really is the same panel.
    """
    try:
        entries = [e for e in os.scandir(data_dir) if e.name.endswith(".csv")]
    except OSError:
        return {"n_csv": -1, "max_mtime": 0}
    return {
        "n_csv": len(entries),
        "max_mtime": int(max((e.stat().st_mtime for e in entries), default=0)),
    }


def compute_manifest(data_dir, cli: dict | None = None) -> dict:
    """Fingerprint everything the cached panel was built from.

    cli: recipe-affecting command-line settings. The checkpoint is built from
    the already-trimmed, already-PIT-filtered panel, so --pit_universe and
    --train_start change its CONTENT even when code and data are identical.
    --mode is deliberately excluded: sharing one panel across momentum and
    reversal is the checkpoint's whole purpose.
    """
    return {
        "code_sha":  _code_fingerprint(),
        "data":      _data_fingerprint(data_dir),
        "env":       {k: os.environ.get(k, "") for k in _RECIPE_ENV},
        "cli":       dict(cli or {}),
    }


def manifest_ok(manifest_path, current: dict) -> tuple[bool, str]:
    """(True, "") when the stored manifest matches `current`; else a reason."""
    p = Path(manifest_path)
    if not p.exists():
        return False, "no manifest recorded for this checkpoint (pre-guard build)"
    try:
        stored = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return False, f"manifest unreadable ({e})"
    for key in ("code_sha", "data", "env", "cli"):
        if stored.get(key) != current.get(key):
            return False, (f"{key} changed since checkpoint was built "
                           f"(stored={stored.get(key)!r} now={current.get(key)!r})")
    return True, ""


def write_manifest(manifest_path, current: dict) -> None:
    Path(manifest_path).write_text(json.dumps(current, indent=2))
