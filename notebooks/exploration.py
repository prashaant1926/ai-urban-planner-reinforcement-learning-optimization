#!/usr/bin/env python3
"""
Data Exploration Script
=======================

This script explores the training dataset and generates summary statistics.
Can be converted to a Jupyter notebook if needed.
"""

# %% [markdown]
# # Dataset Exploration
#
# This notebook explores our training dataset structure and statistics.

# %% Imports
import json
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# %% Configuration
DATA_PATH = Path("../artifacts/dataset/sample_data.jsonl")
print(f"Data path: {DATA_PATH.resolve()}")

# %% Load Data
def load_jsonl(path):
    """Load JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

# %% Basic Statistics
def analyze_dataset(data):
    """Compute basic statistics."""
    stats = {
        "num_examples": len(data),
        "roles": Counter(),
        "message_lengths": [],
        "turns_per_example": [],
    }

    for example in data:
        messages = example.get("messages", [])
        stats["turns_per_example"].append(len(messages))

        for msg in messages:
            stats["roles"][msg["role"]] += 1
            stats["message_lengths"].append(len(msg["content"]))

    return stats

# %% Main Execution
if __name__ == "__main__":
    print("=" * 60)
    print("Dataset Exploration")
    print("=" * 60)

    if DATA_PATH.exists():
        data = load_jsonl(DATA_PATH)
        stats = analyze_dataset(data)

        print(f"\nNumber of examples: {stats['num_examples']}")
        print(f"\nRole distribution:")
        for role, count in stats['roles'].items():
            print(f"  {role}: {count}")

        print(f"\nAverage turns per example: {sum(stats['turns_per_example'])/len(stats['turns_per_example']):.1f}")
        print(f"Average message length: {sum(stats['message_lengths'])/len(stats['message_lengths']):.1f} chars")
    else:
        print(f"Data file not found: {DATA_PATH}")

    print("\n" + "=" * 60)
    print("Exploration complete!")
    print("=" * 60)
