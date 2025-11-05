#!/usr/bin/env bash

set -euo pipefail

ENV_FILE="/workspace/.prime-rl-env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

export DEBIAN_FRONTEND=noninteractive

log() {
  echo "[bootstrap] $*"
}

# 0) install base packages
log "Installing base packages (git, curl, tmux, jq)"
if command -v apt-get >/dev/null 2>&1; then
  PKG_TOOL="apt-get"
elif command -v yum >/dev/null 2>&1; then
  PKG_TOOL="yum"
elif command -v dnf >/dev/null 2>&1; then
  PKG_TOOL="dnf"
elif command -v apk >/dev/null 2>&1; then
  PKG_TOOL="apk"
else
  PKG_TOOL=""
fi

if [ -n "$PKG_TOOL" ]; then
  SUDO_PREFIX=""
  if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    SUDO_PREFIX="sudo"
  fi

  case "$PKG_TOOL" in
    apt-get)
      $SUDO_PREFIX apt-get update
      $SUDO_PREFIX apt-get install -y git curl tmux jq
      ;;
    yum)
      $SUDO_PREFIX yum install -y git curl tmux jq
      ;;
    dnf)
      $SUDO_PREFIX dnf install -y git curl tmux jq
      ;;
    apk)
      $SUDO_PREFIX apk add --no-cache git curl tmux jq
      ;;
  esac
else
  log "Package manager not found; ensure git, curl, tmux, jq are installed manually"
fi

# Install Docker engine for MCP backends that require `docker`
if ! command -v docker >/dev/null 2>&1; then
  log "Installing Docker engine (required for browser MCP tasks)"
  if command -v apt-get >/dev/null 2>&1; then
    $SUDO_PREFIX apt-get update
    $SUDO_PREFIX apt-get install -y docker.io
    if command -v systemctl >/dev/null 2>&1; then
      $SUDO_PREFIX systemctl enable --now docker || true
    else
      service docker start || true
    fi
    usermod -aG docker "$USER" 2>/dev/null || true
  elif command -v yum >/dev/null 2>&1; then
    $SUDO_PREFIX yum install -y docker
    if command -v systemctl >/dev/null 2>&1; then
      $SUDO_PREFIX systemctl enable --now docker || true
    fi
    service docker start || true
    usermod -aG docker "$USER" 2>/dev/null || true
  elif command -v dnf >/dev/null 2>&1; then
    $SUDO_PREFIX dnf install -y docker
    if command -v systemctl >/dev/null 2>&1; then
      $SUDO_PREFIX systemctl enable --now docker || true
    fi
    service docker start || true
    usermod -aG docker "$USER" 2>/dev/null || true
  else
    log "WARNING: Could not install Docker automatically; please install Docker manually."
  fi
else
  log "Docker already installed"
fi

# Ensure Docker daemon is running; if not, try to start it, then fall back to Podman
if command -v docker >/dev/null 2>&1; then
  if ! docker info >/dev/null 2>&1; then
    log "Docker CLI present but daemon not running; attempting to start Docker"
    if command -v systemctl >/dev/null 2>&1; then
      $SUDO_PREFIX systemctl enable --now docker || true
    else
      service docker start || true
    fi
  fi

  if ! docker info >/dev/null 2>&1; then
    log "Docker daemon unavailable; installing rootless Podman (docker-compatible)"
    if command -v apt-get >/dev/null 2>&1; then
      $SUDO_PREFIX apt-get update
      $SUDO_PREFIX apt-get install -y podman podman-docker slirp4netns fuse-overlayfs
    elif command -v yum >/dev/null 2>&1; then
      $SUDO_PREFIX yum install -y podman podman-docker slirp4netns fuse-overlayfs || true
    elif command -v dnf >/dev/null 2>&1; then
      $SUDO_PREFIX dnf install -y podman podman-docker slirp4netns fuse-overlayfs || true
    fi
  fi

  # Pre-pull required MCP images once a Docker-compatible CLI is available
  docker pull hudevals/hud-browser:0.1.3 >/dev/null 2>&1 || true
  docker pull hudevals/hud-text-2048:0.1.3 >/dev/null 2>&1 || true
fi

# 1) install uv and ensure Python 3.12 runtime
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv runtime"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

UV_BIN="$HOME/.local/bin"
if [ -d "$UV_BIN" ]; then
  export PATH="$UV_BIN:$PATH"
fi

if [ -f "$UV_BIN/env" ]; then
  # shellcheck disable=SC1090
  source "$UV_BIN/env"
fi

# 2) clone or update prime-rl
log "Cloning prime-rl if necessary"
mkdir -p /workspace
if [ "$(id -u)" -eq 0 ]; then
  chown "$USER":"$USER" /workspace || true
fi
cd /workspace

if [ ! -d prime-rl ]; then
  git clone https://github.com/PrimeIntellect-ai/prime-rl.git
else
  log "prime-rl already present; pulling latest"
  cd prime-rl
  git fetch --all --tags
  git reset --hard origin/main
  cd ..
fi

# 3) sync dependencies via uv.lock
cd /workspace/prime-rl
log "Installing Python dependencies via uv"
uv sync

HF_LOGIN_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [ -n "$HF_LOGIN_TOKEN" ]; then
  log "Configuring HuggingFace credentials"
  export HF_TOKEN="$HF_LOGIN_TOKEN"
  export HUGGINGFACE_HUB_TOKEN="$HF_LOGIN_TOKEN"
  mkdir -p "$HOME/.huggingface"
  printf "%s\n" "$HF_LOGIN_TOKEN" > "$HOME/.huggingface/token"
  chmod 600 "$HOME/.huggingface/token"
  if command -v huggingface-cli >/dev/null 2>&1; then
    if ! uv run huggingface-cli whoami >/dev/null 2>&1; then
      if ! uv run huggingface-cli login --token "$HF_LOGIN_TOKEN" --force >/dev/null 2>&1; then
        log "WARNING: huggingface-cli login failed; relying on token file"
      fi
    fi
  else
    log "huggingface-cli not available; relying on token file"
  fi
else
  log "HuggingFace token not provided; skipping configuration"
fi

# 4) ensure local configs directory exists
mkdir -p /workspace/configs/hud
if [ "$(id -u)" -eq 0 ]; then
  chown -R "$USER":"$USER" /workspace/configs || true
fi

# 5) install verifiers + hud-vf-gym environment
log "Installing verifiers and HUD VF gym"
uv pip install --system verifiers

if [ -n "${WANDB_API_KEY:-}" ]; then
  log "Configuring Weights & Biases credentials"
  export WANDB_API_KEY
  mkdir -p "$HOME/.config/wandb"
  printf "[default]\napi_key = %s\n" "$WANDB_API_KEY" > "$HOME/.config/wandb/settings"
  chmod 600 "$HOME/.config/wandb/settings"
else
  log "WANDB_API_KEY not provided; skipping Weights & Biases configuration"
fi

LOCAL_HUD_VF_GYM="/workspace/hud-vf-gym"
if [ -d "$LOCAL_HUD_VF_GYM" ]; then
  log "Installing hud-vf-gym from local source $LOCAL_HUD_VF_GYM"
  # Install directly into the prime-rl venv even if pip is missing there
  if ! uv pip install --python /workspace/prime-rl/.venv/bin/python -e "$LOCAL_HUD_VF_GYM"; then
    log "WARNING: Failed to install hud-vf-gym from local source with uv pip"
  fi
  # Ensure the 'hud' module is available (dependency of hud-vf-gym)
  uv pip install --python /workspace/prime-rl/.venv/bin/python "hud-python>=0.4.59" || \
    log "WARNING: Failed to ensure hud-python is installed in prime-rl venv"
elif [ -n "${HUD_VF_GYM_GIT_URL:-}" ]; then
  log "Installing hud-vf-gym from $HUD_VF_GYM_GIT_URL"
  uv pip install --system "$HUD_VF_GYM_GIT_URL"
elif [ -n "${HUD_VF_GYM_TARBALL:-}" ]; then
  log "Installing hud-vf-gym from tarball $HUD_VF_GYM_TARBALL"
  uv pip install --system "$HUD_VF_GYM_TARBALL"
else
  log "Local hud-vf-gym source not found and HUD_VF_GYM_GIT_URL/HUD_VF_GYM_TARBALL not set; skipping install"
fi

# 6) verify HUD config presence (must be uploaded by deploy script)
if [ ! -f /workspace/configs/2048.yaml ]; then
  log "WARNING: /workspace/configs/2048.yaml missing. Upload your HUD config before running RL/eval."
fi

# 7) sanity checks
log "Verifying Python version"
uv run python -V

log "Checking flash_attn import"
uv run python -c "import flash_attn"

log "Bootstrap complete"

