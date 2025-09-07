"""
Evaluation Script for Browser-based 2048

This script evaluates an agent on the browser-based 2048 game
with visual inputs using the computer tool.

Usage:
  python eval_browser_2048.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import verifiers as vf


async def main():
    """Run evaluation on browser 2048 tasks."""
    
    # Load environment with local dataset and browser config
    env = vf.load_environment(
        env_id="hud-vf-gym",
        taskset="../data/browser_2048.json",  # Local dataset
        config_path="../configs/browser_2048.yaml",  # Browser config
        num_tasks=3,  # Run 3 tasks for demo
    )
    
    # Evaluate with Claude
    print("Starting evaluation with Claude 3.5 Sonnet...")
    results = await env.evaluate(
        model="claude-3-5-sonnet-20241022",
        parallel=1,  # Run one task at a time for browser tasks
        verbose=True,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Overall Score: {results['score']:.2%}")
    print(f"Tasks Completed: {results['tasks_completed']}/{results['total_tasks']}")
    
    # Detailed breakdown
    if 'task_scores' in results:
        print("\nTask-by-task scores:")
        for i, score in enumerate(results['task_scores']):
            print(f"  Task {i+1}: {score:.2%}")
    
    # Save results
    import json
    with open("browser_2048_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to browser_2048_results.json")


if __name__ == "__main__":
    # Ensure required environment variables are set
    if not os.getenv("HUD_API_KEY"):
        print("Error: HUD_API_KEY environment variable not set")
        print("Get your API key from https://app.hud.so")
        sys.exit(1)
    
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("This example uses Claude for evaluation")
        sys.exit(1)
    
    # Run evaluation
    asyncio.run(main())