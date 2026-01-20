#!/usr/bin/env python3
"""Generate plots from training metrics."""

import json
from typing import List, Dict, Optional
from pathlib import Path


def load_metrics(filepath: str) -> List[Dict]:
    """Load metrics from a JSONL file."""
    print(f"[PLOT] Loading metrics from: {filepath}", flush=True)

    metrics = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))

    print(f"[PLOT] Loaded {len(metrics)} data points", flush=True)
    return metrics


def plot_loss_curve(
    metrics: List[Dict],
    output_path: str,
    title: str = "Training Loss"
) -> None:
    """Plot training loss curve and save to file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not installed, skipping plot", flush=True)
        return

    steps = [m.get('step', i) for i, m in enumerate(metrics)]
    losses = [m.get('loss', 0) for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, 'b-', linewidth=1.5, label='Training Loss')

    # Add eval loss if available
    eval_losses = [(m.get('step', i), m.get('eval_loss'))
                   for i, m in enumerate(metrics)
                   if m.get('eval_loss') is not None]
    if eval_losses:
        e_steps, e_losses = zip(*eval_losses)
        plt.plot(e_steps, e_losses, 'r--', linewidth=1.5, label='Eval Loss')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Saved plot to: {output_path}", flush=True)


def plot_learning_rate(
    metrics: List[Dict],
    output_path: str,
    title: str = "Learning Rate Schedule"
) -> None:
    """Plot learning rate schedule."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not installed, skipping plot", flush=True)
        return

    steps = [m.get('step', i) for i, m in enumerate(metrics)]
    lrs = [m.get('lr', m.get('learning_rate', 0)) for m in metrics]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, lrs, 'g-', linewidth=1.5)

    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title(title)
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[PLOT] Saved plot to: {output_path}", flush=True)


def generate_all_plots(metrics_file: str, output_dir: str) -> None:
    """Generate all standard training plots."""
    print("=" * 50, flush=True)
    print("GENERATING TRAINING PLOTS", flush=True)
    print("=" * 50, flush=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_file)

    plot_loss_curve(
        metrics,
        str(output_path / "loss_curve.png"),
        "Training Loss Over Time"
    )

    plot_learning_rate(
        metrics,
        str(output_path / "lr_schedule.png"),
        "Learning Rate Schedule"
    )

    print("=" * 50, flush=True)
    print("PLOTS COMPLETE!", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    print("Metrics plotter module loaded.", flush=True)
    print("Usage: generate_all_plots(metrics_file, output_dir)", flush=True)
