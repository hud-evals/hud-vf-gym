#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REMOTE_BASE="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$DEFAULT_REMOTE_BASE/.prime-rl-env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

REMOTE_BASE="${PRIME_REMOTE_BASE:-$DEFAULT_REMOTE_BASE}"
ENV_FILE="$REMOTE_BASE/.prime-rl-env"
PRIME_RL_DIR="$REMOTE_BASE/prime-rl"
PRIME_RL_VENV_PYTHON="$PRIME_RL_DIR/.venv/bin/python"
HUD_CONFIG_DIR="$REMOTE_BASE/configs"
LOCAL_HUD_VF_GYM="$REMOTE_BASE/hud-vf-gym"

export DEBIAN_FRONTEND=noninteractive

log() {
  echo "[bootstrap] $*"
}

SUDO_PREFIX=""
if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
  SUDO_PREFIX="sudo"
fi

docker_cli_is_podman_shim() {
  if ! command -v docker >/dev/null 2>&1; then
    return 1
  fi

  if docker --help 2>&1 | head -n 1 | grep -qi podman; then
    return 0
  fi

  return 1
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
NEED_DOCKER_INSTALL=false
if ! command -v docker >/dev/null 2>&1; then
  NEED_DOCKER_INSTALL=true
elif docker_cli_is_podman_shim; then
  log "Docker CLI resolves to the podman shim; reinstalling Docker engine"
  NEED_DOCKER_INSTALL=true
  case "$PKG_TOOL" in
    apt-get)
      $SUDO_PREFIX apt-get remove -y podman-docker >/dev/null 2>&1 || true
      ;;
    yum|dnf)
      $SUDO_PREFIX ${PKG_TOOL} remove -y podman-docker >/dev/null 2>&1 || true
      ;;
  esac
fi

if [ "$NEED_DOCKER_INSTALL" = true ]; then
  log "Installing Docker engine (required for browser MCP tasks)"
  if command -v apt-get >/dev/null 2>&1; then
    $SUDO_PREFIX apt-get update
    $SUDO_PREFIX apt-get install -y docker.io
  elif command -v yum >/dev/null 2>&1; then
    $SUDO_PREFIX yum install -y docker
  elif command -v dnf >/dev/null 2>&1; then
    $SUDO_PREFIX dnf install -y docker
  else
    log "WARNING: Could not install Docker automatically; please install Docker manually."
  fi
fi

if command -v docker >/dev/null 2>&1 && ! docker_cli_is_podman_shim; then
  if command -v systemctl >/dev/null 2>&1; then
    $SUDO_PREFIX systemctl unmask docker >/dev/null 2>&1 || true
    $SUDO_PREFIX systemctl enable --now docker >/dev/null 2>&1 || true
  else
    service docker start >/dev/null 2>&1 || true
  fi

  if command -v usermod >/dev/null 2>&1; then
    $SUDO_PREFIX usermod -aG docker "$USER" 2>/dev/null || true
  fi

  DOCKER_CMD="docker"
  if ! docker info >/dev/null 2>&1; then
    log "Docker daemon not reachable as $USER; attempting with elevated privileges"
    if [ -n "$SUDO_PREFIX" ] && $SUDO_PREFIX docker info >/dev/null 2>&1; then
      DOCKER_CMD="$SUDO_PREFIX docker"
    else
      log "WARNING: Docker daemon still unavailable. Please verify docker.service manually."
    fi
  fi

  $DOCKER_CMD pull hudevals/hud-browser:0.1.3 >/dev/null 2>&1 || true
  $DOCKER_CMD pull hudevals/hud-text-2048:0.1.3 >/dev/null 2>&1 || true
else
  log "WARNING: Docker CLI not available; MCP tasks that require containers may fail."
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
mkdir -p "$REMOTE_BASE"
if [ "$(id -u)" -eq 0 ]; then
  chown "$USER":"$USER" "$REMOTE_BASE" || true
fi
cd "$REMOTE_BASE"

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
cd "$REMOTE_BASE/prime-rl"
log "Installing Python dependencies via uv"
uv sync

if [ ! -x "$PRIME_RL_VENV_PYTHON" ]; then
  log "ERROR: Expected Python interpreter not found at $PRIME_RL_VENV_PYTHON"
  exit 1
fi

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
mkdir -p "$HUD_CONFIG_DIR/hud"
if [ "$(id -u)" -eq 0 ]; then
  chown -R "$USER":"$USER" "$HUD_CONFIG_DIR" || true
fi

# 5) install verifiers + hud-vf-gym environment
log "Installing verifiers and HUD VF gym"
if ! uv pip install --python "$PRIME_RL_VENV_PYTHON" verifiers; then
  log "WARNING: Failed to install verifiers into $PRIME_RL_VENV_PYTHON"
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  log "Configuring Weights & Biases credentials"
  export WANDB_API_KEY
  mkdir -p "$HOME/.config/wandb"
  printf "[default]\napi_key = %s\n" "$WANDB_API_KEY" > "$HOME/.config/wandb/settings"
  chmod 600 "$HOME/.config/wandb/settings"
else
  log "WANDB_API_KEY not provided; skipping Weights & Biases configuration"
fi

if [ -d "$LOCAL_HUD_VF_GYM" ]; then
  log "Installing hud-vf-gym from local source $LOCAL_HUD_VF_GYM"
  # Install directly into the prime-rl venv even if pip is missing there
  if ! uv pip install --python "$PRIME_RL_VENV_PYTHON" -e "$LOCAL_HUD_VF_GYM"; then
    log "WARNING: Failed to install hud-vf-gym from local source with uv pip"
  fi
  # Ensure the 'hud' module is available (dependency of hud-vf-gym)
  uv pip install --python "$PRIME_RL_VENV_PYTHON" "hud-python>=0.4.59" || \
    log "WARNING: Failed to ensure hud-python is installed in prime-rl venv"
elif [ -n "${HUD_VF_GYM_GIT_URL:-}" ]; then
  log "Installing hud-vf-gym from $HUD_VF_GYM_GIT_URL"
  if ! uv pip install --python "$PRIME_RL_VENV_PYTHON" "$HUD_VF_GYM_GIT_URL"; then
    log "WARNING: Failed to install hud-vf-gym from $HUD_VF_GYM_GIT_URL"
  fi
elif [ -n "${HUD_VF_GYM_TARBALL:-}" ]; then
  log "Installing hud-vf-gym from tarball $HUD_VF_GYM_TARBALL"
  if ! uv pip install --python "$PRIME_RL_VENV_PYTHON" "$HUD_VF_GYM_TARBALL"; then
    log "WARNING: Failed to install hud-vf-gym from tarball $HUD_VF_GYM_TARBALL"
  fi
else
  log "Local hud-vf-gym source not found and HUD_VF_GYM_GIT_URL/HUD_VF_GYM_TARBALL not set; skipping install"
fi

# 6) verify HUD config presence (must be uploaded by deploy script)
if [ ! -f "$HUD_CONFIG_DIR/2048.yaml" ]; then
  log "WARNING: $HUD_CONFIG_DIR/2048.yaml missing. Upload your HUD config before running RL/eval."
fi

# 7) sanity checks
log "Verifying Python version"
uv run python -V

log "Checking flash_attn import"
uv run python -c "import flash_attn"

log "Bootstrap complete"

