#!/usr/bin/env python3
"""Analysis utilities for training metrics and results."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    eval_loss: Optional[float] = None


def parse_metrics_log(filepath: str) -> List[TrainingMetrics]:
    """Parse a metrics log file into structured data."""
    metrics = []

    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                metrics.append(TrainingMetrics(
                    epoch=data.get('epoch', 0),
                    step=data.get('step', 0),
                    loss=data.get('loss', 0.0),
                    learning_rate=data.get('lr', 0.0),
                    eval_loss=data.get('eval_loss')
                ))
            except json.JSONDecodeError:
                continue

    return metrics


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute basic statistics for a list of values."""
    if not values:
        return {"mean": 0, "min": 0, "max": 0, "std": 0}

    n = len(values)
    mean = sum(values) / n
    min_val = min(values)
    max_val = max(values)

    variance = sum((x - mean) ** 2 for x in values) / n
    std = variance ** 0.5

    return {
        "mean": round(mean, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
        "std": round(std, 4)
    }


def summarize_training(metrics: List[TrainingMetrics]) -> Dict[str, any]:
    """Generate a summary of training metrics."""
    if not metrics:
        return {"error": "No metrics to summarize"}

    losses = [m.loss for m in metrics]

    summary = {
        "total_steps": len(metrics),
        "final_loss": metrics[-1].loss,
        "loss_stats": compute_statistics(losses),
        "best_loss": min(losses),
        "best_step": losses.index(min(losses))
    }

    eval_losses = [m.eval_loss for m in metrics if m.eval_loss is not None]
    if eval_losses:
        summary["eval_loss_stats"] = compute_statistics(eval_losses)
        summary["best_eval_loss"] = min(eval_losses)

    return summary


def print_summary(summary: Dict[str, any]) -> None:
    """Pretty print a training summary."""
    print("=" * 50, flush=True)
    print("TRAINING SUMMARY", flush=True)
    print("=" * 50, flush=True)

    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:", flush=True)
            for k, v in value.items():
                print(f"  {k}: {v}", flush=True)
        else:
            print(f"{key}: {value}", flush=True)

    print("=" * 50, flush=True)


if __name__ == "__main__":
    # Demo with synthetic data
    print("Analysis module loaded successfully!", flush=True)

    demo_values = [2.5, 2.3, 2.1, 1.9, 1.7, 1.5, 1.4, 1.3, 1.25, 1.2]
    stats = compute_statistics(demo_values)
    print(f"Demo statistics: {stats}", flush=True)
