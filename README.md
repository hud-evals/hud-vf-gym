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