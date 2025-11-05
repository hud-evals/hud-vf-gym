#!/usr/bin/env bash

# Usage: ./deploy_prime.sh [-p PORT] [-i IDENTITY_FILE] user@host
# Example: ./deploy_prime.sh -p 9678 -i private_key.pem root@205.196.17.100

set -euo pipefail

SSH_PORT=""
SSH_IDENTITY=""

while getopts ":p:i:" opt; do
  case $opt in
    p)
      SSH_PORT="$OPTARG"
      ;;
    i)
      SSH_IDENTITY="$OPTARG"
      ;;
    *)
      echo "Usage: $0 [-p PORT] [-i IDENTITY_FILE] <ssh-target>" >&2
      exit 1
      ;;
  esac
done
shift $((OPTIND-1))

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 [-p PORT] [-i IDENTITY_FILE] <ssh-target>" >&2
  exit 1
fi

TARGET="$1"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
REPO_ROOT="$(cd "$BASE_DIR/.." && pwd)"
REMOTE_BASE="/workspace"
REMOTE_ENV_FILE="$REMOTE_BASE/.prime-rl-env"
REMOTE_APP_DIR="$REMOTE_BASE/hud-vf-gym"

SSH_OPTS=("-o" "StrictHostKeyChecking=accept-new")
if [ -n "$SSH_PORT" ]; then
  SSH_OPTS+=("-p" "$SSH_PORT")
fi
if [ -n "$SSH_IDENTITY" ]; then
  SSH_OPTS+=("-i" "$SSH_IDENTITY")
fi
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

echo "Checking remote rsync availability"
if ssh "${SSH_OPTS[@]}" "$TARGET" 'command -v rsync >/dev/null 2>&1'; then
  echo "Syncing configs and scripts to $TARGET via rsync"
  rsync -av -e "$RSYNC_SSH" "$BASE_DIR/" "$TARGET:$REMOTE_BASE/" --exclude "systemd" --exclude ".git" --exclude "*.pyc"
else
  echo "Remote rsync not found. Attempting to install (apt/yum/dnf/apk) ..."
  ssh "${SSH_OPTS[@]}" "$TARGET" '(
    (command -v apt-get >/dev/null 2>&1 && DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y rsync) ||
    (command -v yum >/dev/null 2>&1 && yum install -y rsync) ||
    (command -v dnf >/dev/null 2>&1 && dnf install -y rsync) ||
    (command -v apk >/dev/null 2>&1 && apk add --no-cache rsync) || true
  ) >/dev/null 2>&1 || true'
  if ssh "${SSH_OPTS[@]}" "$TARGET" 'command -v rsync >/dev/null 2>&1'; then
    echo "Installed rsync remotely; syncing via rsync"
    rsync -av -e "$RSYNC_SSH" "$BASE_DIR/" "$TARGET:$REMOTE_BASE/" --exclude "systemd" --exclude ".git" --exclude "*.pyc"
  else
    echo "Falling back to tar stream copy"
    ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p $REMOTE_BASE"
    tar czf - -C "$BASE_DIR" . --exclude systemd --exclude .git --exclude "*.pyc" | ssh "${SSH_OPTS[@]}" "$TARGET" "tar xzf - -C $REMOTE_BASE"
  fi
fi

echo "Uploading systemd units"
if ssh "${SSH_OPTS[@]}" "$TARGET" 'command -v rsync >/dev/null 2>&1'; then
  rsync -av -e "$RSYNC_SSH" "$BASE_DIR/systemd/" "$TARGET:$REMOTE_BASE/systemd/"
else
  ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p $REMOTE_BASE/systemd"
  tar czf - -C "$BASE_DIR/systemd" . | ssh "${SSH_OPTS[@]}" "$TARGET" "tar xzf - -C $REMOTE_BASE/systemd"
fi

# Ensure remote dirs exist and correct ownership
ssh "${SSH_OPTS[@]}" "$TARGET" "mkdir -p $REMOTE_BASE/configs $REMOTE_BASE/scripts $REMOTE_BASE/systemd $REMOTE_APP_DIR /var/log/prime-rl && chown -R \$USER:\$USER $REMOTE_BASE /var/log/prime-rl"

echo "Syncing hud-vf-gym source to $TARGET:$REMOTE_APP_DIR"
RSYNC_APP_EXCLUDES=(
  "--exclude" ".git/"
  "--exclude" ".venv/"
  "--exclude" "__pycache__/"
  "--exclude" ".mypy_cache/"
  "--exclude" ".pytest_cache/"
  "--exclude" "*.pyc"
  "--exclude" "*.pyo"
  "--exclude" "*.egg-info/"
  "--exclude" "build/"
  "--exclude" "dist/"
  "--exclude" "*.log"
  "--exclude" "private_key.pem"
)

rsync -av --delete -e "$RSYNC_SSH" "${RSYNC_APP_EXCLUDES[@]}" "$REPO_ROOT/" "$TARGET:$REMOTE_APP_DIR/"

declare -a TOKEN_EXPORTS=('export PYTHONPATH="/workspace/hud-vf-gym/src:${PYTHONPATH:-}"')

if [ -n "${HF_TOKEN:-}" ]; then
  HF_ESCAPED=$(printf "%q" "$HF_TOKEN")
  TOKEN_EXPORTS+=("export HF_TOKEN=$HF_ESCAPED")
  if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    TOKEN_EXPORTS+=("export HUGGINGFACE_HUB_TOKEN=$HF_ESCAPED")
  fi
fi

if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  HUB_ESCAPED=$(printf "%q" "$HUGGINGFACE_HUB_TOKEN")
  TOKEN_EXPORTS+=("export HUGGINGFACE_HUB_TOKEN=$HUB_ESCAPED")
  if [ -z "${HF_TOKEN:-}" ]; then
    TOKEN_EXPORTS+=("export HF_TOKEN=$HUB_ESCAPED")
  fi
fi

if [ -n "${WANDB_API_KEY:-}" ]; then
  WANDB_ESCAPED=$(printf "%q" "$WANDB_API_KEY")
  TOKEN_EXPORTS+=("export WANDB_API_KEY=$WANDB_ESCAPED")
fi

# Pass HUD API key if set locally
if [ -n "${HUD_API_KEY:-}" ]; then
  HUD_ESCAPED=$(printf "%q" "$HUD_API_KEY")
  TOKEN_EXPORTS+=("export HUD_API_KEY=$HUD_ESCAPED")
fi

printf "%s\n" "${TOKEN_EXPORTS[@]}" | ssh "${SSH_OPTS[@]}" "$TARGET" "cat > $REMOTE_ENV_FILE.tmp && mv $REMOTE_ENV_FILE.tmp $REMOTE_ENV_FILE && chmod 600 $REMOTE_ENV_FILE"
echo "Uploaded remote environment exports to $REMOTE_ENV_FILE"

# Upload HUD YAML config if present
if [ -f "$REPO_ROOT/configs/2048.yaml" ]; then
  echo "Uploading HUD config 2048.yaml"
  if ssh "${SSH_OPTS[@]}" "$TARGET" 'command -v rsync >/dev/null 2>&1'; then
    rsync -av -e "$RSYNC_SSH" "$REPO_ROOT/configs/2048.yaml" "$TARGET:$REMOTE_BASE/configs/2048.yaml"
  else
    scp "${SSH_OPTS[@]}" "$REPO_ROOT/configs/2048.yaml" "$TARGET:$REMOTE_BASE/configs/2048.yaml"
  fi
fi

# Make remote scripts executable
ssh "${SSH_OPTS[@]}" "$TARGET" "chmod +x $REMOTE_BASE/scripts/*.sh 2>/dev/null || true"

echo "Running bootstrap on remote host"
ssh "${SSH_OPTS[@]}" "$TARGET" "bash $REMOTE_BASE/scripts/bootstrap.sh"

echo "Starting tmux stack"
if [ -t 1 ]; then
  ssh -t "${SSH_OPTS[@]}" "$TARGET" "bash $REMOTE_BASE/scripts/start_all.sh"
else
  ssh "${SSH_OPTS[@]}" "$TARGET" "bash $REMOTE_BASE/scripts/start_all.sh >/dev/null 2>&1 & disown"
  echo "Stack launch triggered in background (non-interactive session)."
fi

echo "Deployment complete"

