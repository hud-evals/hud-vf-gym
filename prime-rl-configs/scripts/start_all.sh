#!/usr/bin/env bash

set -euo pipefail

SESSION="prime-rl"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REMOTE_BASE="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$DEFAULT_REMOTE_BASE/.prime-rl-env"
if [ -f "$ENV_FILE" ]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

: "${PRIME_REMOTE_BASE:=$DEFAULT_REMOTE_BASE}"
REMOTE_BASE="$PRIME_REMOTE_BASE"
ENV_FILE="$REMOTE_BASE/.prime-rl-env"
ROOT="$REMOTE_BASE/prime-rl"
HUD_CONFIG_DIR="$REMOTE_BASE/configs/hud"

: "${OPENAI_API_KEY:=prime-rl-local}"
export OPENAI_API_KEY

LOG_DIR="$REMOTE_BASE/logs/prime-rl"
RUN_ID="$(date +"%Y%m%d-%H%M%S")"
GROUP="hud-2048-$RUN_ID"
INFER_LOG="$LOG_DIR/inference-$RUN_ID.log"
ORCH_LOG="$LOG_DIR/orchestrator-$RUN_ID.log"
TRAIN_LOG="$LOG_DIR/trainer-$RUN_ID.log"

if [ ! -d "$ROOT" ]; then
  echo "prime-rl repo not found at $ROOT" >&2
  exit 1
fi

cd "$ROOT"

mkdir -p "$LOG_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not installed; please run bootstrap.sh first" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Reusing existing tmux session $SESSION"
else
  tmux new-session -d -s "$SESSION"
fi

# Pane 0: inference on GPU0
tmux send-keys -t "$SESSION:0.0" "bash -lc 'set -o pipefail; cd \"$ROOT\"; LOG_FILE=$INFER_LOG; { echo \"====\"; date +\"%Y-%m-%dT%H:%M:%S%z\"; echo \"Starting inference\"; } >> \$LOG_FILE; export CUDA_VISIBLE_DEVICES=0; uv run inference @ \"$HUD_CONFIG_DIR/infer.toml\" 2>&1 | tee -a \$LOG_FILE'" C-m

sleep 5

# Pane 1: orchestrator on CPU
tmux split-window -v -t "$SESSION:0"
tmux send-keys -t "$SESSION:0.1" "bash -lc 'set -o pipefail; cd \"$ROOT\"; LOG_FILE=$ORCH_LOG; { echo \"====\"; date +\"%Y-%m-%dT%H:%M:%S%z\"; echo \"Starting orchestrator\"; } >> \$LOG_FILE; WANDB_RUN_GROUP=\"$GROUP\" WANDB_NAME=\"orch-$RUN_ID\" uv run orchestrator @ \"$HUD_CONFIG_DIR/orch.toml\" 2>&1 | tee -a \$LOG_FILE'" C-m

# Pane 2: trainer on GPU1
tmux split-window -h -t "$SESSION:0.1"
tmux send-keys -t "$SESSION:0.2" "bash -lc 'set -o pipefail; cd \"$ROOT\"; LOG_FILE=$TRAIN_LOG; { echo \"====\"; date +\"%Y-%m-%dT%H:%M:%S%z\"; echo \"Starting trainer\"; } >> \$LOG_FILE; export CUDA_VISIBLE_DEVICES=1; WANDB_RUN_GROUP=\"$GROUP\" WANDB_NAME=\"trainer-$RUN_ID\" uv run trainer @ \"$HUD_CONFIG_DIR/rl.train.toml\" 2>&1 | tee -a \$LOG_FILE'" C-m

tmux select-layout -t "$SESSION:0" tiled

echo "Attached to tmux session $SESSION"
tmux attach -t "$SESSION"

