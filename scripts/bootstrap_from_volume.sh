#!/usr/bin/env bash
# =============================================================================
# bootstrap_from_volume.sh — rebuild a Hetzner box from the persistent volume
#
# WHY THIS EXISTS
# ───────────────
# server_setup.sh has a chicken-and-egg problem: you must curl it from GitHub
# (with a token in your shell history) *before* you have a working box. This
# script instead LIVES ON THE VOLUME, which survives server deletion — so a
# fresh server needs no tokens, no curl, no copy-paste:
#
#     mount /dev/sdb /mnt/data          # (or let cloud-init do it, see below)
#     bash /mnt/data/bootstrap_from_volume.sh
#
# INSTALL IT ONCE (from an existing working box):
#     cp ~/kaizen-run/scripts/bootstrap_from_volume.sh /mnt/data/
#     cp ~/.config/kaizen.env /mnt/data/            # optional, see TOKENS below
#
# FULLY AUTOMATIC (no commands at all) — paste into the Hetzner console's
# "Cloud config" box at server-creation time:
#
#     #cloud-config
#     runcmd:
#       - [ mkdir, -p, /mnt/data ]
#       - [ mount, /dev/sdb, /mnt/data ]
#       - [ bash, /mnt/data/bootstrap_from_volume.sh ]
#
# TOKENS
# ──────
# If /mnt/data/kaizen.env exists it is sourced (chmod 600 it). Format:
#     GITHUB_TOKEN=ghp_xxx      # only needed if the repo is private
# Nothing is ever written back to the volume by this script.
# =============================================================================

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────
MOUNT_POINT="${MOUNT_POINT:-/mnt/data}"
VOLUME_DEVICE="${VOLUME_DEVICE:-/dev/sdb}"
GITHUB_USER="prabhuvictor85"
GITHUB_REPO_NAME="kaizen"
GIT_BRANCH="main"
PROJECT_DIR="${PROJECT_DIR:-/root/kaizen-run}"

DATA_ROOT="${MOUNT_POINT}/Learning_charts"
ARTEFACTS_ROOT="${MOUNT_POINT}/artefacts"

# File-descriptor ceiling. The downloader walks ~1600 tickers in one long-lived
# process; the default soft limit of 1024 is what produced
# "OSError: [Errno 24] Too many open files" mid-run.
FD_LIMIT="${FD_LIMIT:-65536}"

# Swap. The 30 GB box OOM-killed a full-panel run at ~31 GB anon-rss.
SWAP_GB="${SWAP_GB:-40}"
SWAPFILE="${SWAPFILE:-/swapfile}"

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[0;32mOK\033[0m  %s\n' "$*"; }
warn() { printf '    \033[0;33m!!\033[0m  %s\n' "$*"; }

# ── 0. Volume must be mounted (this script lives on it, so usually is) ───────
step "Checking volume at ${MOUNT_POINT}"
if ! mountpoint -q "${MOUNT_POINT}"; then
    warn "${MOUNT_POINT} is not a mountpoint — attempting ${VOLUME_DEVICE}"
    mkdir -p "${MOUNT_POINT}"
    mount "${VOLUME_DEVICE}" "${MOUNT_POINT}"
fi
mountpoint -q "${MOUNT_POINT}" || { echo "FATAL: cannot mount ${MOUNT_POINT}"; exit 1; }
ok "$(df -h "${MOUNT_POINT}" | awk 'NR==2 {print $4" free of "$2}')"

# Persist the mount so a reboot doesn't silently lose it mid-run.
if ! grep -qs "${MOUNT_POINT}" /etc/fstab; then
    echo "${VOLUME_DEVICE} ${MOUNT_POINT} ext4 discard,nofail,defaults 0 0" >> /etc/fstab
    ok "added to /etc/fstab (survives reboot)"
fi

# ── 1. Optional tokens ───────────────────────────────────────────────────────
if [ -f "${MOUNT_POINT}/kaizen.env" ]; then
    # shellcheck disable=SC1091
    set -a; . "${MOUNT_POINT}/kaizen.env"; set +a
    ok "sourced ${MOUNT_POINT}/kaizen.env"
fi
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# ── 2. System packages ───────────────────────────────────────────────────────
step "Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git python3-pip python3-venv python3-full curl >/dev/null
ok "git, python3-pip, python3-venv installed"

# ── 3. Repo ──────────────────────────────────────────────────────────────────
step "Fetching ${GITHUB_REPO_NAME} (${GIT_BRANCH})"
if [ -n "${GITHUB_TOKEN}" ]; then
    CLONE_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${GITHUB_REPO_NAME}.git"
else
    CLONE_URL="https://github.com/${GITHUB_USER}/${GITHUB_REPO_NAME}.git"
fi
if [ -d "${PROJECT_DIR}/.git" ]; then
    git -C "${PROJECT_DIR}" remote set-url origin "${CLONE_URL}"
    git -C "${PROJECT_DIR}" fetch --quiet origin
    git -C "${PROJECT_DIR}" checkout --quiet "${GIT_BRANCH}" 2>/dev/null || true
    git -C "${PROJECT_DIR}" reset --hard --quiet "origin/${GIT_BRANCH}"
    ok "updated $(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
else
    git clone --quiet --branch "${GIT_BRANCH}" "${CLONE_URL}" "${PROJECT_DIR}"
    ok "cloned $(git -C "${PROJECT_DIR}" rev-parse --short HEAD)"
fi
# Never leave a token sitting in .git/config on disk.
git -C "${PROJECT_DIR}" remote set-url origin \
    "https://github.com/${GITHUB_USER}/${GITHUB_REPO_NAME}.git"

# ── 4. Python dependencies (PINNED) ──────────────────────────────────────────
# requirements-lock.txt is the validated set. requirements.txt alone resolves
# to "newest of everything", which is how this box ended up on yfinance 1.5.2
# (file-descriptor leak) instead of the tested 1.3.0.
step "Installing Python dependencies"
PIP_ARGS="--break-system-packages --quiet"
if [ -f "${PROJECT_DIR}/requirements-lock.txt" ]; then
    python3 -m pip install ${PIP_ARGS} -r "${PROJECT_DIR}/requirements-lock.txt"
    ok "installed from requirements-lock.txt (pinned)"
else
    warn "no requirements-lock.txt — falling back to requirements.txt (unpinned)"
    python3 -m pip install ${PIP_ARGS} -r "${PROJECT_DIR}/requirements.txt"
fi
python3 -c "import yfinance, pandas, lightgbm; \
    print(f'    yfinance={yfinance.__version__} pandas={pandas.__version__} lightgbm={lightgbm.__version__}')"

# ── 5. Directory structure on the volume ─────────────────────────────────────
step "Ensuring data directories"
mkdir -p "${DATA_ROOT}/stock_data/us_stocks" \
         "${DATA_ROOT}/stock_data/nse_local" \
         "${DATA_ROOT}/stock_lists" \
         "${ARTEFACTS_ROOT}/us_local" \
         "${ARTEFACTS_ROOT}/nse_local"
ok "directories present under ${MOUNT_POINT}"

# ── 6. paths.yaml → point the pipeline at the volume ─────────────────────────
step "Writing paths.yaml"
cat > "${PROJECT_DIR}/paths.yaml" <<YAML
# Generated by bootstrap_from_volume.sh on $(date -Iseconds)
data_root:      "${DATA_ROOT}"
project_root:   "${PROJECT_DIR}"
artefacts_root: "${ARTEFACTS_ROOT}"

stock_lists:
  nse_local:     "{data_root}/stock_lists/constituentsi.csv"
  nse_tv:        "{data_root}/stock_lists/constituentsi.csv"
  nse_cap_tiers: "{data_root}/stock_lists/nse_cap_tiers.csv"
  us_combined:   "{data_root}/stock_lists/constituents_us_combined.csv"
  lists_dir:     "{data_root}/stock_lists"

stock_data:
  nse_local: "{data_root}/stock_data/nse_local"
  nse_tv:    "{data_root}/stock_data/tradingview"
  us:        "{data_root}/stock_data/us_stocks"
  us_alt:    "{data_root}/us_data"
YAML
ok "paths.yaml -> ${MOUNT_POINT}"

# ── 7. Resource limits ───────────────────────────────────────────────────────
step "Raising file-descriptor limit to ${FD_LIMIT}"
if ! grep -qs "nofile ${FD_LIMIT}" /etc/security/limits.conf; then
    printf '* soft nofile %s\n* hard nofile %s\nroot soft nofile %s\nroot hard nofile %s\n' \
        "${FD_LIMIT}" "${FD_LIMIT}" "${FD_LIMIT}" "${FD_LIMIT}" >> /etc/security/limits.conf
fi
# limits.conf only applies to NEW login sessions, so also drop a profile hook —
# otherwise the very first run after bootstrap still sees the old 1024 ceiling.
if ! grep -qs "ulimit -n ${FD_LIMIT}" /etc/profile.d/99-fd-limit.sh 2>/dev/null; then
    echo "ulimit -n ${FD_LIMIT} 2>/dev/null || true" > /etc/profile.d/99-fd-limit.sh
    chmod +x /etc/profile.d/99-fd-limit.sh
fi
ok "limits.conf + /etc/profile.d/99-fd-limit.sh"

step "Ensuring ${SWAP_GB} GB swap"
CUR_SWAP_GB=$(free -g | awk '/^Swap:/ {print $2}')
if [ "${CUR_SWAP_GB}" -lt "${SWAP_GB}" ]; then
    swapoff "${SWAPFILE}" 2>/dev/null || true
    rm -f "${SWAPFILE}"
    fallocate -l "${SWAP_GB}G" "${SWAPFILE}" || dd if=/dev/zero of="${SWAPFILE}" bs=1M count=$((SWAP_GB*1024)) status=none
    chmod 600 "${SWAPFILE}"; mkswap -q "${SWAPFILE}"; swapon "${SWAPFILE}"
    grep -qs "${SWAPFILE}" /etc/fstab || echo "${SWAPFILE} none swap sw 0 0" >> /etc/fstab
    ok "${SWAP_GB} GB swap active"
else
    ok "already ${CUR_SWAP_GB} GB swap"
fi

# ── 8. Report ────────────────────────────────────────────────────────────────
step "Ready"
cat <<EOF
    repo      : ${PROJECT_DIR}  ($(git -C "${PROJECT_DIR}" rev-parse --short HEAD))
    data      : ${DATA_ROOT}
    artefacts : ${ARTEFACTS_ROOT}
    disk      : $(df -h "${MOUNT_POINT}" | awk 'NR==2 {print $4" free"}')
    fd limit  : $(ulimit -n) now / ${FD_LIMIT} for new shells

    Start a walk-forward (open a NEW shell first so the fd limit applies):

      cd ${PROJECT_DIR}
      nohup python3 run_walkforward_sp500.py \\
        --start 2023-01-01 --end 2026-07-03 --cadence_days 14 \\
        --mode all --quarterly_retrain --no_drift_retrain --pit_universe \\
        --log_dir ${ARTEFACTS_ROOT}/us_local \\
        > ${ARTEFACTS_ROOT}/us_local/wf_\$(date +%Y%m%d_%H%M%S).log 2>&1 &
EOF
