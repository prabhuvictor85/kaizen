#!/usr/bin/env bash
# =============================================================================
# server_setup.sh — One-shot Hetzner server bootstrap for kaizen
#
# Run this once after every new server creation:
#   bash server_setup.sh
#
# What it does:
#   1. Mounts the persistent Hetzner Volume (/dev/sdb → /mnt/data)
#   2. Creates a 50 GB swap file (resizes an existing smaller one)
#   3. Creates the directory structure on the volume
#   4. Installs system packages (pip, venv, libgomp1, lz4)
#   5. Clones / updates the base ml-stock-predictor checkout
#   6. Adds the kaizen remote and checks it out as a worktree at ~/kaizen-run
#   7. Installs Python dependencies into system Python
#   8. Puts paths.yaml in place
#   9. Persists the runtime environment (FEATURE_BUILD_WORKERS, nofile)
#
# Prerequisites:
#   - Volume "ml-data" already exists on Hetzner and is attached to this server
#   - GitHub repos are accessible (public or SSH key pre-loaded)
#   - Python 3.10+ already installed (Hetzner Ubuntu images include it)
#
# NOTE ON PYTHON: this script installs into SYSTEM python3 with
# --break-system-packages, not into a venv. Nothing to activate — run
# `python3 ...` directly. (Earlier versions built /root/venv; if one is left
# over from a previous run it is now unused and can be deleted.)
# =============================================================================

set -euo pipefail

# ── Tokens — export before running (never store values here) ─────────────────
# Usage:
#   export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxx"
#   export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
#   bash server_setup.sh
#
# GitHub token: required for private repos. Get from:
#   https://github.com/settings/tokens  (repo scope)
# Leave blank if repos are public.
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# HF token: required for uploading/downloading from HF Hub. Get from:
#   https://huggingface.co/settings/tokens  (write access)
# Leave blank to skip HF login (you can run 'hf auth login' manually later).
HF_TOKEN="${HF_TOKEN:-}"

# ── Config — edit these if they change ──────────────────────────────────────
GITHUB_USER="prabhuvictor85"
GITHUB_REPO_NAME="ml-stock-predictor"     # base checkout (predecessor repo)
GIT_BRANCH="master"
PROJECT_DIR="/root/ml-stock-predictor"

KAIZEN_REPO_NAME="kaizen"                 # the model repo that actually runs
KAIZEN_REMOTE="kaizen"
KAIZEN_BRANCH="main"
KAIZEN_DIR="/root/kaizen-run"             # git worktree of kaizen/main

VOLUME_DEVICE="/dev/sdb"
MOUNT_POINT="/mnt/data"

SWAP_SIZE_GB=50                           # OOM buffer for large folds

FEATURE_BUILD_WORKERS_DEFAULT=4
NOFILE_LIMIT=65536

# Build clone URLs — use token if set, plain HTTPS otherwise
if [ -n "${GITHUB_TOKEN}" ]; then
    GITHUB_REPO="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${GITHUB_REPO_NAME}.git"
    GITHUB_REPO_DISPLAY="https://***@github.com/${GITHUB_USER}/${GITHUB_REPO_NAME}.git"
    KAIZEN_URL="https://${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${KAIZEN_REPO_NAME}.git"
    KAIZEN_URL_DISPLAY="https://***@github.com/${GITHUB_USER}/${KAIZEN_REPO_NAME}.git"
else
    GITHUB_REPO="https://github.com/${GITHUB_USER}/${GITHUB_REPO_NAME}.git"
    GITHUB_REPO_DISPLAY="${GITHUB_REPO}"
    KAIZEN_URL="https://github.com/${GITHUB_USER}/${KAIZEN_REPO_NAME}.git"
    KAIZEN_URL_DISPLAY="${KAIZEN_URL}"
fi

# Paths on the volume (survive server deletion)
DATA_ROOT="${MOUNT_POINT}/Learning_charts"
ARTEFACTS_ROOT="${MOUNT_POINT}/artefacts"
STOCK_DATA_DIR="${DATA_ROOT}/stock_data"
NSE_DATA_DIR="${STOCK_DATA_DIR}/nse_data"
US_DATA_DIR="${STOCK_DATA_DIR}/us_stocks"
STOCK_LISTS_DIR="${DATA_ROOT}/stock_lists"

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
section() { echo -e "\n${GREEN}===== $* =====${NC}"; }

# =============================================================================
# 1/9  MOUNT VOLUME
# =============================================================================
section "1/9  Mounting Hetzner Volume"

if mountpoint -q "${MOUNT_POINT}"; then
    info "Volume already mounted at ${MOUNT_POINT} — skipping"
else
    if [ ! -b "${VOLUME_DEVICE}" ]; then
        echo -e "${RED}[ERROR]${NC} Device ${VOLUME_DEVICE} not found."
        echo "  → Make sure the Volume is attached to this server in Hetzner console."
        exit 1
    fi

    mkdir -p "${MOUNT_POINT}"

    # ── Safety gate 1: existing files/folders at mount point → never format ──
    # This catches cases where the volume was previously mounted and has data.
    if [ -n "$(ls -A ${MOUNT_POINT} 2>/dev/null)" ]; then
        warn "Files/folders already exist at ${MOUNT_POINT} — skipping format (data is safe)"
        SKIP_FORMAT=1
    else
        SKIP_FORMAT=0
    fi

    # ── Safety gate 2: existing filesystem on device → never format ──
    FS_TYPE=$(blkid -o value -s TYPE "${VOLUME_DEVICE}" 2>/dev/null || true)
    if [ -n "${FS_TYPE}" ]; then
        info "Existing filesystem (${FS_TYPE}) detected on ${VOLUME_DEVICE} — skipping format"
        SKIP_FORMAT=1
    fi

    # ── Format only if both safety gates pass (truly blank new volume) ──
    if [ "${SKIP_FORMAT}" -eq 0 ]; then
        info "New blank volume detected — formatting ${VOLUME_DEVICE} as ext4 ..."
        mkfs.ext4 -F "${VOLUME_DEVICE}"
    fi

    mount "${VOLUME_DEVICE}" "${MOUNT_POINT}"
    info "Mounted ${VOLUME_DEVICE} → ${MOUNT_POINT}"

    # Persist mount across reboots (add only if not already in fstab)
    if ! grep -q "${VOLUME_DEVICE}" /etc/fstab; then
        echo "${VOLUME_DEVICE} ${MOUNT_POINT} ext4 discard,nofail,defaults 0 0" >> /etc/fstab
        info "Added to /etc/fstab for auto-mount on reboot"
    fi
fi

# =============================================================================
# 2/9  SWAP FILE  (50 GB overflow buffer — prevents OOM kills on large folds)
# =============================================================================
section "2/9  Configuring ${SWAP_SIZE_GB}GB swap"

SWAPFILE="/swapfile"
SWAP_BYTES=$(( SWAP_SIZE_GB * 1024 * 1024 * 1024 ))

# Current size of the existing swapfile, 0 if absent.
CURRENT_SWAP_BYTES=0
if [ -f "${SWAPFILE}" ]; then
    CURRENT_SWAP_BYTES=$(stat -c %s "${SWAPFILE}" 2>/dev/null || echo 0)
fi

make_swap() {
    # Free space on / must cover the new file. If a smaller swapfile is being
    # replaced its bytes come back to us, so count them as available.
    local avail_bytes
    avail_bytes=$(( $(df --output=avail -B1 / | tail -1) + CURRENT_SWAP_BYTES ))
    if [ "${avail_bytes}" -lt "$(( SWAP_BYTES + 5 * 1024 * 1024 * 1024 ))" ]; then
        echo -e "${RED}[ERROR]${NC} Not enough free space on / for a ${SWAP_SIZE_GB}GB swap file."
        echo "  → Need ${SWAP_SIZE_GB}GB + 5GB headroom; have $(( avail_bytes / 1024 / 1024 / 1024 ))GB."
        echo "  → Lower SWAP_SIZE_GB at the top of this script, or resize the server disk."
        exit 1
    fi
    swapoff "${SWAPFILE}" 2>/dev/null || true
    rm -f "${SWAPFILE}"
    info "Creating ${SWAP_SIZE_GB}GB swap file at ${SWAPFILE} (this takes a moment) ..."
    fallocate -l "${SWAP_BYTES}" "${SWAPFILE}" \
        || dd if=/dev/zero of="${SWAPFILE}" bs=1M count=$(( SWAP_SIZE_GB * 1024 )) status=progress
    chmod 600 "${SWAPFILE}"
    mkswap "${SWAPFILE}"
    swapon "${SWAPFILE}"
    if ! grep -q "${SWAPFILE}" /etc/fstab; then
        echo "${SWAPFILE} none swap sw 0 0" >> /etc/fstab
        info "Added swap to /etc/fstab (auto-enabled on reboot)"
    fi
    info "Swap ready: $(free -h | grep -i swap)"
}

if [ "${CURRENT_SWAP_BYTES}" -eq "${SWAP_BYTES}" ]; then
    if swapon --show | grep -q "${SWAPFILE}"; then
        info "${SWAP_SIZE_GB}GB swap already active — skipping"
    else
        info "${SWAP_SIZE_GB}GB swap file exists but is not active — enabling ..."
        swapon "${SWAPFILE}"
        info "Swap enabled: $(free -h | grep -i swap)"
    fi
elif [ "${CURRENT_SWAP_BYTES}" -gt 0 ]; then
    warn "Existing swap is $(( CURRENT_SWAP_BYTES / 1024 / 1024 / 1024 ))GB — resizing to ${SWAP_SIZE_GB}GB"
    make_swap
else
    make_swap
fi

# =============================================================================
# 3/9  CREATE DIRECTORY STRUCTURE ON VOLUME
# =============================================================================
section "3/9  Creating directory structure on volume"

mkdir -p "${NSE_DATA_DIR}"
mkdir -p "${US_DATA_DIR}"
mkdir -p "${STOCK_LISTS_DIR}"
mkdir -p "${ARTEFACTS_ROOT}"
info "Directories ready:"
info "  ${NSE_DATA_DIR}   (NSE stock CSVs)"
info "  ${US_DATA_DIR}    (US stock CSVs)"
info "  ${STOCK_LISTS_DIR}"
info "  ${ARTEFACTS_ROOT}"

# =============================================================================
# 4/9  SYSTEM PACKAGES
# =============================================================================
section "4/9  Installing system packages"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y python3-pip python3-venv

# libgomp1: LightGBM's OpenMP runtime. Marked manual so an autoremove during a
# later apt run cannot take it out from under lightgbm.
apt-get install -y libgomp1
apt-mark manual libgomp1

# lz4: fast panel checkpoint compression. Distro package first, pip after —
# the pip wheel is what Python actually imports if both are present.
apt-get install -y python3-lz4 || warn "python3-lz4 unavailable from apt — pip wheel will cover it"

info "System packages installed"

# =============================================================================
# 5/9  BASE REPO — clone / update ml-stock-predictor
# =============================================================================
section "5/9  Cloning / updating base repository"

if [ -d "${PROJECT_DIR}/.git" ]; then
    info "Base repo already exists — pulling latest ${GIT_BRANCH} ..."
    git -C "${PROJECT_DIR}" fetch origin
    git -C "${PROJECT_DIR}" checkout "${GIT_BRANCH}"
    git -C "${PROJECT_DIR}" pull origin "${GIT_BRANCH}"
else
    info "Cloning ${GITHUB_REPO_DISPLAY} (branch: ${GIT_BRANCH}) ..."
    git clone --branch "${GIT_BRANCH}" "${GITHUB_REPO}" "${PROJECT_DIR}"
fi

info "Base repo ready at ${PROJECT_DIR}"

# NOTE: the old "re-exec from the repo's copy of this script" block was removed
# here on purpose. Two repos are now in play, and it re-executed the *base*
# repo's copy — silently reverting to the pre-kaizen setup. Pull the script you
# want and run that one.

# =============================================================================
# 6/9  KAIZEN — remote + worktree at ${KAIZEN_DIR}
# =============================================================================
section "6/9  Setting up the kaizen worktree"

# Remote (idempotent — set-url if it already exists, so a rotated token lands)
if git -C "${PROJECT_DIR}" remote get-url "${KAIZEN_REMOTE}" &>/dev/null; then
    git -C "${PROJECT_DIR}" remote set-url "${KAIZEN_REMOTE}" "${KAIZEN_URL}"
    info "Remote '${KAIZEN_REMOTE}' already present — URL refreshed"
else
    git -C "${PROJECT_DIR}" remote add "${KAIZEN_REMOTE}" "${KAIZEN_URL}"
    info "Added remote '${KAIZEN_REMOTE}' → ${KAIZEN_URL_DISPLAY}"
fi

git -C "${PROJECT_DIR}" fetch "${KAIZEN_REMOTE}"

# Worktree (idempotent — reuse and fast-forward if it is already there)
if [ -e "${KAIZEN_DIR}/.git" ]; then
    info "Worktree already exists at ${KAIZEN_DIR} — updating to ${KAIZEN_REMOTE}/${KAIZEN_BRANCH} ..."
    git -C "${KAIZEN_DIR}" fetch "${KAIZEN_REMOTE}" "${KAIZEN_BRANCH}"
    git -C "${KAIZEN_DIR}" checkout --detach "${KAIZEN_REMOTE}/${KAIZEN_BRANCH}"
else
    info "Adding worktree ${KAIZEN_DIR} at ${KAIZEN_REMOTE}/${KAIZEN_BRANCH} ..."
    git -C "${PROJECT_DIR}" worktree add "${KAIZEN_DIR}" "${KAIZEN_REMOTE}/${KAIZEN_BRANCH}"
fi

info "kaizen ready at ${KAIZEN_DIR} — $(git -C "${KAIZEN_DIR}" rev-parse --short HEAD) (detached)"

# =============================================================================
# 7/9  PYTHON DEPENDENCIES  (system python3, --break-system-packages)
# =============================================================================
section "7/9  Installing Python dependencies"

cd "${KAIZEN_DIR}"

PIP_FLAGS="--break-system-packages"

# requirements-lock.txt is the validated set; requirements.txt alone re-resolves
# and can drift. Prefer the lock when it is present.
if [ -f "${KAIZEN_DIR}/requirements-lock.txt" ]; then
    info "Installing from requirements-lock.txt ..."
    python3 -m pip install ${PIP_FLAGS} -r "${KAIZEN_DIR}/requirements-lock.txt"
elif [ -f "${KAIZEN_DIR}/requirements.txt" ]; then
    info "Installing from requirements.txt ..."
    python3 -m pip install ${PIP_FLAGS} -r "${KAIZEN_DIR}/requirements.txt"
else
    warn "No requirements file found in ${KAIZEN_DIR} — skipping pip install"
fi

python3 -m pip install ${PIP_FLAGS} lz4

# Verify the one import that silently breaks without libgomp1.
if python3 -c "import lightgbm; print(lightgbm.__version__)"; then
    info "LightGBM imports cleanly"
else
    echo -e "${RED}[ERROR]${NC} LightGBM failed to import — libgomp1 or the wheel is broken."
    exit 1
fi

# =============================================================================
# 8/9  paths.yaml
# =============================================================================
section "8/9  Putting paths.yaml in place"

PATHS_YAML="${KAIZEN_DIR}/paths.yaml"
VOLUME_PATHS_YAML="${MOUNT_POINT}/paths.yaml"

if [ -f "${VOLUME_PATHS_YAML}" ]; then
    cp "${VOLUME_PATHS_YAML}" "${PATHS_YAML}"
    info "Copied ${VOLUME_PATHS_YAML} → ${PATHS_YAML}"
else
    warn "${VOLUME_PATHS_YAML} not found — generating a default from this script's paths"
    cat > "${PATHS_YAML}" <<EOF
# paths.yaml — auto-generated by server_setup.sh
# All data lives on the persistent Hetzner Volume at ${MOUNT_POINT}.

data_root:    ${DATA_ROOT}
project_root: ${KAIZEN_DIR}

stock_lists:
  nse_local:     ${STOCK_LISTS_DIR}/constituentsi.csv
  nse_tv:        ${STOCK_LISTS_DIR}/constituents_nse_tradingv.csv
  nse_cap_tiers: ${STOCK_LISTS_DIR}/nse_cap_tiers.csv
  us_combined:   ${STOCK_LISTS_DIR}/constituents_us_combined.csv
  lists_dir:     ${STOCK_LISTS_DIR}

stock_data:
  nse_local: ${NSE_DATA_DIR}
  nse_tv:    ${STOCK_DATA_DIR}/tradingview
  us:        ${US_DATA_DIR}
  us_alt:    ${US_DATA_DIR}

artefacts_root: ${ARTEFACTS_ROOT}
EOF
    info "Default paths.yaml written — keep a copy at ${VOLUME_PATHS_YAML} so it survives the server"
fi
cat "${PATHS_YAML}"

# =============================================================================
# 9/9  RUNTIME ENVIRONMENT + HF LOGIN
# =============================================================================
section "9/9  Persisting runtime environment"

# `export` and `ulimit` inside this script die with it. Write them to
# profile.d + limits.d so every later login shell — and the pipeline run that
# matters — actually gets them.
PROFILE_D="/etc/profile.d/kaizen.sh"
cat > "${PROFILE_D}" <<EOF
# Written by server_setup.sh — kaizen runtime environment
export FEATURE_BUILD_WORKERS=${FEATURE_BUILD_WORKERS_DEFAULT}
ulimit -n ${NOFILE_LIMIT} 2>/dev/null || true
EOF
chmod 644 "${PROFILE_D}"
info "Wrote ${PROFILE_D} (FEATURE_BUILD_WORKERS=${FEATURE_BUILD_WORKERS_DEFAULT}, nofile=${NOFILE_LIMIT})"

# profile.d only covers login shells. limits.d covers everything else.
LIMITS_D="/etc/security/limits.d/99-kaizen.conf"
cat > "${LIMITS_D}" <<EOF
# Written by server_setup.sh — raise open-file limit for feature builds
root soft nofile ${NOFILE_LIMIT}
root hard nofile ${NOFILE_LIMIT}
*    soft nofile ${NOFILE_LIMIT}
*    hard nofile ${NOFILE_LIMIT}
EOF
chmod 644 "${LIMITS_D}"
info "Wrote ${LIMITS_D}"

# Apply to this shell too, so anything below sees them.
export FEATURE_BUILD_WORKERS="${FEATURE_BUILD_WORKERS_DEFAULT}"
ulimit -n "${NOFILE_LIMIT}" 2>/dev/null || warn "Could not raise nofile in this shell — takes effect on next login"

# ── HF login (only if HF_TOKEN is set) ───────────────────────────────────────
if [ -n "${HF_TOKEN}" ]; then
    if command -v hf &>/dev/null; then
        echo "${HF_TOKEN}" | hf auth login --token-stdin 2>/dev/null \
            || hf auth login --token "${HF_TOKEN}"
        info "HF login successful"
    elif python3 -c "import huggingface_hub" &>/dev/null 2>&1; then
        python3 -c "
import huggingface_hub, os
huggingface_hub.login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
print('HF login successful via huggingface_hub')
"
    else
        warn "hf CLI and huggingface_hub not found — skipping HF login"
        warn "Run 'pip install huggingface_hub && hf auth login' manually"
    fi
else
    warn "HF_TOKEN not set — skipping HF login (run 'hf auth login' manually)"
fi

# =============================================================================
# DONE
# =============================================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Server setup complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  Volume  : ${MOUNT_POINT}  (persists after server deletion)"
echo "  Base    : ${PROJECT_DIR}"
echo "  kaizen  : ${KAIZEN_DIR}   ← run from here"
echo "  Data    : ${DATA_ROOT}"
echo "  Models  : ${ARTEFACTS_ROOT}"
echo "  Swap    : $(free -h | grep -i swap | awk '{print $2}')"
echo ""
echo "Next steps:"
echo "  # Pick up FEATURE_BUILD_WORKERS and the raised nofile limit:"
echo "  exec bash -l          # or just log out and back in"
echo ""
echo "  # Download US data (first time or delta update):"
echo "  cd ${KAIZEN_DIR}"
echo "  python3 scripts/data/download_us_data.py"
echo ""
echo "  # Run the pipeline:"
echo "  cd ${KAIZEN_DIR}"
echo "  python3 run_sp500_local.py"
echo ""
echo "  # When done — detach volume in Hetzner console, then delete server."
echo ""
