# hud-vf-gym

Verifiers Adapter for HUD environments - bridges [Verifiers](https://github.com/willccbb/verifiers) RL framework with [HUD's MCP infrastructure](https://github.com/hud-evals/hud-pytho) for training and evaluating agents.

### Training Support
Verifiers' GRPOTrainer does not support multimodal training as of now, you can use an [experimental trainer](https://github.com/jdchawla29/verifiers) for single-turn environments (with single prompt image due to [transformer's limitations](https://github.com/huggingface/transformers/pull/36682)). Multi-turn multimodal support is WIP.

## Prerequisites

- Python >=3.12
- HUD API key from [https://app.hud.so](https://app.hud.so)
- Environment variables:
  ```bash
  export HUD_API_KEY="your-api-key"
  export OPENAI_API_KEY="your-key"  # or ANTHROPIC_API_KEY
  ```

## Quick Start

```python
import verifiers as vf

# Load environment with HUD taskset and config
env = vf.load_environment(
    env_id="hud-vf-gym",
    taskset="hud-evals/2048-taskset",  # HuggingFace dataset
    config_path="./configs/2048.yaml",  # Environment config
    num_tasks=10
)
```

## Documentation

For comprehensive usage, examples, and configuration details, see:
**[hud-python/rl README](https://github.com/hud-evals/hud-python/tree/main/rl)**

The main documentation covers:
- Running evaluations with various models
- Training agents with GRPO
- Creating custom environments and configs
- Dataset format and creation
- Action mappings and tool configuration
- Troubleshooting guide

## License

MIT

