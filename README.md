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

## PRIME-RL + Prime Intellect Integration (Qwen2.5‑0.5B, 2×A4000)

This repo includes production-ready configs and scripts to train with PRIME‑RL on a Prime Intellect GPU instance while using this `hud-vf-gym` environment.

What’s included:

- `prime-rl-configs/configs/hud/` – TOMLs for inference, trainer (LoRA PPO), orchestrator, and eval.
- `prime-rl-configs/scripts/` – `bootstrap.sh`, `start_all.sh`, `deploy_prime.sh` (automation).
- `prime-rl-configs/systemd/` – optional services for auto‑start on boot.

Model/GPU split:

- Qwen2.5‑0.5B‑Instruct via vLLM on GPU0; RL trainer (LoRA) on GPU1; orchestrator on CPU.

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

### B) Real Run on Prime Intellect (2×A4000)

Prereqs:

- Provision a single 2×A4000 pod (CUDA base image, Python 3.12 OK). Open TCP 8000 if you need to hit vLLM externally.
- Set `WANDB_API_KEY` and `HF_TOKEN`

1) One‑command deploy from your laptop

```bash
# From repo root
chmod +x prime-rl-configs/scripts/*.sh

# Replace with your SSH target/port/key; example below uses your instance
chmod 600 private_key.pem
prime-rl-configs/scripts/deploy_prime.sh -p 9678 -i private_key.pem root@205.196.17.100
```

What it does:

- rsyncs configs and scripts to `/workspace/`
- syncs your local `hud-vf-gym` repository to `/workspace/hud-vf-gym` and installs it in editable mode
- uploads `configs/2048.yaml` to `/workspace/configs/2048.yaml`
- runs `bootstrap.sh` (installs uv, clones `prime-rl`, installs deps, verifies `flash_attn`)
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

3) Example eval and training commands (on the instance)

```bash
cd /workspace/prime-rl

# Evaluate a small slice to verify end-to-end wiring
uv run eval @ /workspace/configs/hud/eval.toml

# Run the orchestrator alone (if not started via tmux)
uv run orchestrator @ /workspace/configs/hud/orch.toml

# Start inference alone (GPU0)
CUDA_VISIBLE_DEVICES=0 uv run inference @ /workspace/configs/hud/infer.toml

# Start trainer alone (GPU1)
CUDA_VISIBLE_DEVICES=1 uv run trainer @ /workspace/configs/hud/rl.train.toml
```

Paths and checkpoints:

- Checkpoints: `/workspace/checkpoints/` (subfolders for inference/trainer/eval)
- Configs: `/workspace/configs/hud/*.toml` and `/workspace/configs/2048.yaml`

Optional: auto‑start with systemd (on the instance)

```bash
sudo mkdir -p /var/log/prime-rl && sudo chown $USER:$USER /var/log/prime-rl
sudo cp -v /workspace/systemd/*.service /etc/systemd/system/
echo "OPENAI_API_KEY=prime-rl-local" | sudo tee /etc/prime-rl.env
sudo systemctl daemon-reload
sudo systemctl enable prime-inference@root prime-orchestrator@root prime-trainer@root
sudo systemctl start  prime-inference@root prime-orchestrator@root prime-trainer@root
```

### Config references

- Inference: `prime-rl-configs/configs/hud/infer.toml` (vLLM on GPU0)
- Trainer: `prime-rl-configs/configs/hud/rl.train.toml` (LoRA PPO on GPU1)
- Orchestrator: `prime-rl-configs/configs/hud/orch.toml`
- Eval: `prime-rl-configs/configs/hud/eval.toml`

Adjust the `[model].name` fields if you prefer a different Qwen checkpoint. Default is `Qwen/Qwen2.5-0.5B-Instruct` (public on Hugging Face).

### Notes

- Python 3.12 is used everywhere (HUD requirement). `uv` enforces it on the instance.
- If you need gated models/datasets, set `HF_TOKEN` on the instance before running.
- For W&B logging, export `WANDB_API_KEY` on the instance.

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