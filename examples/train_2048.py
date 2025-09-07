"""
Training Script for 2048 (2 GPUs)
Terminal 1 - Start vLLM server:
  CUDA_VISIBLE_DEVICES=0 vf-vllm \
      --model Qwen/Qwen2.5-3B-Instruct \
      --enforce-eager \
      --disable-log-requests

  Terminal 2 - Run training:
  CUDA_VISIBLE_DEVICES=1 python train_2048.py
"""

from __future__ import annotations

import os
import sys

# Add parent directory to path to import from configs/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verifiers as vf

vf_env = vf.load_environment(
    env_id="hud-vf-gym",
    taskset="hud-evals/2048-taskset",  # HuggingFace dataset
    config_path="../configs/2048.yaml",  # Use relative path to configs
    num_tasks=4,
)

# Model configuration
model_name = "Qwen/Qwen2.5-3B-Instruct"
base_url = "http://localhost:8000/v1"

# Use the verifiers' ModelSamplingAgent
model_agent = vf.ModelSamplingAgent(
    model_name=model_name,
    openai_base_url=base_url,
    temperature=0.7,
    max_tokens=1024,
)

# Training arguments
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

training_args = vf.GRPOTrainingArguments(
    output_dir="./checkpoints/qwen-2048",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-6,
    warmup_steps=100,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    lr_scheduler_type="cosine",
    bf16=True,  # Use bfloat16 if supported
    gradient_checkpointing=True,
    remove_unused_columns=False,
    report_to=["tensorboard"],
    logging_dir="./logs",
)

# Create trainer
trainer = vf.GRPOTrainer(
    tokenizer=tokenizer,
    environment=vf_env,
    model_name_or_path=model_name,
    args=training_args,
    peft_config=None,  # Add LoRA config here if needed
)

# Run training
print("Starting GRPO training...")
trainer.train()

# Save final model
trainer.save_model("./final_model")
print("Training complete!")