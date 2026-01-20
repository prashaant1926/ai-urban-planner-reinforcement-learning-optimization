#!/usr/bin/env python3
"""Data loading utilities for training datasets."""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class DataSample:
    """Represents a single training sample."""
    input_text: str
    output_text: str
    metadata: Optional[Dict[str, Any]] = None


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file (one JSON object per line)."""
    import json

    print(f"[DATA] Loading JSONL from: {filepath}", flush=True)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"[DATA] Loaded {len(data)} samples", flush=True)
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save data to a JSONL file."""
    import json

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"[DATA] Saved {len(data)} samples to: {filepath}", flush=True)


def validate_chat_format(sample: Dict[str, Any]) -> bool:
    """Validate that a sample is in correct chat format."""
    if "messages" not in sample:
        return False

    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        return False

    for msg in messages:
        if "role" not in msg or "content" not in msg:
            return False
        if msg["role"] not in ["system", "user", "assistant"]:
            return False

    return True


def create_chat_sample(
    user_message: str,
    assistant_message: str,
    system_message: Optional[str] = None
) -> Dict[str, Any]:
    """Create a sample in chat format."""
    messages = []

    if system_message:
        messages.append({"role": "system", "content": system_message})

    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": assistant_message})

    return {"messages": messages}


if __name__ == "__main__":
    # Demo usage
    print("=" * 50, flush=True)
    print("DATA LOADER DEMO", flush=True)
    print("=" * 50, flush=True)

    sample = create_chat_sample(
        user_message="What is the capital of France?",
        assistant_message="The capital of France is Paris.",
        system_message="You are a helpful assistant."
    )

    print(f"Sample: {sample}", flush=True)
    print(f"Valid: {validate_chat_format(sample)}", flush=True)
