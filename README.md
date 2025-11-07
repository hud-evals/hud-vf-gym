# hud-vf-gym

Verifiers Adapter for HUD environments - bridges [Verifiers](https://github.com/willccbb/verifiers) RL framework with [HUD's MCP infrastructure](https://github.com/hud-evals/hud-python) for training and evaluating agents.

## Prerequisites

- Python >=3.12
- HUD API key from [https://app.hud.so](https://app.hud.so)
- Environment variables:
  ```bash
  export HUD_API_KEY="your-api-key"
  export OPENAI_API_KEY="your-key"  # or ANTHROPIC_API_KEY
  ```

## Installation

```bash
# Install from PyPI (coming soon)
pip install hud-vf-gym

# Or install from source
git clone https://github.com/hud-evals/hud-vf-gym.git
cd hud-vf-gym
pip install -e .
```

## Quick Start

### 1. Generate a Config Template

```bash
hudvf-init --output ./configs/my_env.yaml
```

### 2. Load and Use the Environment

```python
import verifiers as vf

# Load environment with HUD taskset and config
env = vf.load_environment(
    env_id="hud-vf-gym",
    taskset="hud-evals/2048-taskset",  # HuggingFace dataset or local JSONL
    config_path="./configs/2048.yaml",
    num_tasks=10
)

# Run evaluation
results = await env.evaluate(model="gpt-4o-mini")
print(f"Score: {results['score']}")
```

## Configuration

### Config Structure

```yaml
# Job configuration (for HUD telemetry)
job:
  name: "My Environment Run"
  metadata:
    dataset: "hud-evals/2048-taskset"
    experiment: "baseline"

# System prompt for the agent
system_prompt: |
  You are an expert game player...

# Environment settings
defaults:
  max_turns: 30

# Scoring rubric
rubric:
  weights:
    task_completion: 0.8
    tool_execution: 0.2

# Tool restrictions (optional)
allowed_tools: ["computer", "screenshot"]
```

### Example Configs

- `configs/2048.yaml` - Text-based 2048 game
- `configs/browser_2048.yaml` - Browser-based 2048 with visual input

## Usage Examples

### Evaluation

```bash
# Using vf-eval CLI
vf-eval hud-vf-gym \
    --model gpt-4o-mini \
    --env-args '{"taskset": "hud-evals/2048-taskset", "config_path": "configs/2048.yaml"}' \
    --num-tasks 10
```

### Training with GRPO

```python
# See examples/train_2048.py for full example
import verifiers as vf
from transformers import AutoTokenizer

# Load environment
env = vf.load_environment(
    env_id="hud-vf-gym",
    taskset="hud-evals/2048-taskset",
    config_path="configs/2048.yaml",
    num_tasks=100
)

# Setup model and tokenizer
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create trainer
trainer = vf.GRPOTrainer(
    tokenizer=tokenizer,
    environment=env,
    model_name_or_path=model_name,
    output_dir="./checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=1e-6,
)

# Train
trainer.train()
```

### Multimodal Training

Verifiers' GRPOTrainer does not support multimodal training as of now. You can use an [experimental trainer](https://github.com/jdchawla29/verifiers) for single-turn environments (with single prompt image due to [transformer's limitations](https://github.com/huggingface/transformers/pull/36682)). Multi-turn multimodal support is WIP.

## Dataset Format

### HUD Task Format

```json
{
  "id": "task-001",
  "prompt": "Play 2048 and reach the 128 tile",
  "mcp_config": {
    "local": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "hudevals/hud-browser:0.1.3"]
    }
  },
  "setup_tool": [
    {"name": "launch_app", "arguments": {"app_name": "2048"}}
  ],
  "evaluate_tool": {
    "name": "evaluate",
    "arguments": {"name": "game_2048_max_number", "arguments": {"target": 128}}
  }
}
```

### Creating Custom Datasets

1. Create a JSONL file with tasks in HUD format
2. Upload to HuggingFace or use locally
3. Reference in your config and environment loading

## Examples

The `examples/` directory contains complete working examples:

- `train_2048.py` - Training a model on text-based 2048
- `eval_browser_2048.py` - Evaluating on browser-based 2048

Run examples:

```bash
# Training example
cd examples
python train_2048.py

# Evaluation example
python eval_browser_2048.py
```

## PRIME-RL + Prime Intellect Integration (Qwen3-4B, 2×A6000)

This repo includes production-ready configs and scripts to train with PRIME‑RL on a Prime Intellect GPU instance while using this `hud-vf-gym` environment.

What’s included:

- `prime-rl-configs/configs/hud/` – TOMLs for inference, trainer (LoRA PPO), orchestrator, and eval.
- `prime-rl-configs/scripts/` – `bootstrap.sh`, `start_all.sh`, `deploy_prime.sh` (automation).
- `prime-rl-configs/systemd/` – optional services for auto‑start on boot.

Model/GPU split:

- `Qwen/Qwen3-4B-Instruct-2507` via vLLM on GPU0; RL trainer (LoRA) on GPU1; orchestrator on CPU.

Synced defaults (see the TOMLs for the full list):

- Trainer now runs 300 steps per launch with LoRA rank 16 / alpha 32 / dropout 0.05 to stay stable on the 4B base model.
- Orchestrator batches 64 prompts with 16 rollouts and up to 32 concurrent tasks; sampling is capped at 256 response tokens to control VRAM growth.
- Inference `max_model_len` is set to 22,880 tokens, which fits comfortably on a 48 GB A6000 while leaving headroom for trainer gradients.
- Raw transcript logging is disabled by default (`log_data = false`) across eval/orchestrator/trainer—flip it back on if you need full conversation archives.

### A) CPU Smoke Test (no GPU)

PRIME‑RL itself requires an NVIDIA GPU, so the CPU smoke test validates the HUD/verifiers environment and config locally.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -U openai  # required by test.py; or set ANTHROPIC instead

export HUD_API_KEY=...           # if required by your tasks
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY

# Validate env loads (single task) using your config and dataset
python test.py  # uses configs/2048.yaml and hud-evals/2048-taskset

# Or with the CLI (no code):
vf-eval hud-vf-gym \
  --model gpt-4o-mini \
  --env-args '{"taskset":"hud-evals/2048-taskset","config_path":"configs/2048.yaml"}' \
  --num-tasks 1
```

Expected: the environment initializes, one rollout/eval completes, and a score prints.

### B) Real Run on Prime Intellect (2×A6000)

Prereqs:

- Provision a single 2×A6000 pod on Prime-Intellect (UBUNTU 22 CUDA 12 base image).
- Export `WANDB_API_KEY` and `HF_TOKEN` locally so the deploy script can forward them.
- Decide on a remote base path (defaults to `/ephemeral`, override with `-b` or `PRIME_REMOTE_BASE`). All configs/scripts/logs will live underneath it.

1) One‑command deploy from your laptop

```bash
# From repo root
chmod +x prime-rl-configs/scripts/*.sh

# Replace with your SSH target/port/key/remote base as needed
chmod 600 private_key.pem

prime-rl-configs/scripts/deploy_prime.sh -i private_key.pem ubuntu@38.80.122.121
```

What it does (`REMOTE_BASE` = `/ephemeral` unless you overrode it):

- rsyncs `prime-rl-configs/` into `$REMOTE_BASE/` (scripts, configs, systemd units) and writes `$REMOTE_BASE/.prime-rl-env` with forwarded env vars.
- syncs this repo to `$REMOTE_BASE/hud-vf-gym` and installs it (plus `verifiers`, `hud-python`) inside the PRIME-RL venv via `bootstrap.sh`.
- clones + `uv sync`s `prime-rl` into `$REMOTE_BASE/prime-rl`, configures Docker (if needed), and verifies `flash_attn`.
- uploads `configs/2048.yaml` to `$REMOTE_BASE/configs/2048.yaml` if present.
- starts a tmux session `prime-rl` with:
  - pane 0: vLLM inference on GPU0
  - pane 1: orchestrator (CPU)
  - pane 2: RL trainer on GPU1

2) Health checks and manual controls (on the instance)

```bash
tmux ls
tmux attach -t prime-rl

# vLLM health
curl http://127.0.0.1:8000/health

# Logs (if you use systemd):
sudo journalctl -u prime-inference@root -f
sudo journalctl -u prime-orchestrator@root -f
sudo journalctl -u prime-trainer@root -f
```

3) Example eval and training commands manaully

```bash
source /ephemeral/.prime-rl-env  # or point at your custom REMOTE_BASE
REMOTE_BASE=${PRIME_REMOTE_BASE:-/ephemeral}

cd "$REMOTE_BASE/prime-rl"

# Start inference (GPU0)
CUDA_VISIBLE_DEVICES=0 uv run inference @ "$REMOTE_BASE/configs/hud/infer.toml"

# Run the orchestrator
uv run orchestrator @ "$REMOTE_BASE/configs/hud/orch.toml"

# Start trainer (GPU1)
CUDA_VISIBLE_DEVICES=1 uv run trainer @ "$REMOTE_BASE/configs/hud/rl.train.toml"

# Once it finished the trainingm, evaluate a small slice to verify performance at each step.
uv run eval @ "$REMOTE_BASE/configs/hud/eval.toml"
```

### W&B tracking

`prime-rl-configs/scripts/start_all.sh` now pins both orchestrator and trainer to the same Weights & Biases project/group automatically. The script exports `WANDB_PROJECT=hud-prime-rl`, emits a `hud-2048-<timestamp>` value for both `WANDB_GROUP` and `WANDB_RUN_GROUP`, and gives descriptive run names (`orch-…`, `trainer-…`). Make sure `WANDB_API_KEY` is exported on the host before starting the stack so both panes authenticate correctly. Adjust the project or naming scheme directly in the script if you want a different W&B layout.

Paths and checkpoints (under the same `REMOTE_BASE`):

- Checkpoints: `$REMOTE_BASE/checkpoints/` (subfolders for inference/trainer/eval)
- Configs: `$REMOTE_BASE/configs/hud/*.toml` and `$REMOTE_BASE/configs/2048.yaml`

### Config references

- Inference: `prime-rl-configs/configs/hud/infer.toml` (vLLM on GPU0)
- Trainer: `prime-rl-configs/configs/hud/rl.train.toml` (LoRA PPO on GPU1)
- Orchestrator: `prime-rl-configs/configs/hud/orch.toml`
- Eval: `prime-rl-configs/configs/hud/eval.toml`

Adjust the `[model].name` fields if you prefer a different Qwen checkpoint or model. Default is `Qwen/Qwen3-4B-Instruct-2507` (public on Hugging Face).

### References

- PRIME‑RL: https://github.com/PrimeIntellect-ai/prime-rl
- Prime Intellect: https://docs.primeintellect.ai/introduction
- Verifiers: https://verifiers.readthedocs.io/en/latest/
- HUD: https://docs.hud.ai/

## Troubleshooting

### Common Issues

1. **"HUD_API_KEY not found"**: Ensure you've exported the API key
2. **Docker connection errors**: Make sure Docker is running for browser environments
3. **Out of memory during training**: Reduce batch size or use gradient accumulation
4. **Multimodal training errors**: Ensure you're using the experimental verifiers branch

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Project Structure

```
hud-vf-gym/
├── configs/              # Example configuration files
│   ├── 2048.yaml        # Text-based 2048 config
│   └── browser_2048.yaml # Browser-based 2048 config
├── data/                 # Example datasets
│   └── browser_2048.json # Sample browser task
├── examples/             # Usage examples
│   ├── train_2048.py    # Training script
│   └── eval_browser_2048.py # Evaluation script
├── src/hud_vf_gym/       # Main package
│   ├── __init__.py
│   ├── hud_vf_gym.py    # Core environment
│   ├── rubrics.py       # Scoring logic
│   └── config_template.py # Config generator
└── pyproject.toml        # Package configuration
```

## License

MIT