#!/usr/bin/env python3
"""Data processing utilities for preparing training datasets."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class DatasetStats:
    """Statistics about a dataset."""
    total_examples: int = 0
    total_tokens: int = 0
    avg_tokens_per_example: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0

    def __repr__(self) -> str:
        return (
            f"DatasetStats(examples={self.total_examples}, "
            f"total_tokens={self.total_tokens}, "
            f"avg_tokens={self.avg_tokens_per_example:.1f})"
        )


@dataclass
class ConversationExample:
    """A single conversation example."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {"messages": self.messages}
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationExample":
        return cls(
            messages=data.get("messages", []),
            metadata=data.get("metadata", {})
        )


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSONL file and return list of records."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: List[Dict[str, Any]], filepath: str) -> None:
    """Save records to a JSONL file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')


def split_dataset(
    data: List[Any],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: Optional[int] = 42
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Split data into train, validation, and test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    if seed is not None:
        random.seed(seed)

    data = data.copy()
    random.shuffle(data)

    n = len(data)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return data[:train_end], data[train_end:val_end], data[val_end:]


def validate_conversation(example: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a conversation example."""
    if "messages" not in example:
        return False, "Missing 'messages' field"

    messages = example["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        return False, "Messages must be a non-empty list"

    for i, msg in enumerate(messages):
        if "role" not in msg:
            return False, f"Message {i} missing 'role'"
        if "content" not in msg:
            return False, f"Message {i} missing 'content'"
        if msg["role"] not in ["system", "user", "assistant"]:
            return False, f"Message {i} has invalid role: {msg['role']}"

    return True, None


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (4 chars per token approximation)."""
    return len(text) // 4


def compute_dataset_stats(data: List[Dict[str, Any]]) -> DatasetStats:
    """Compute statistics for a dataset."""
    if not data:
        return DatasetStats()

    token_counts = []
    for example in data:
        total_text = ""
        for msg in example.get("messages", []):
            total_text += msg.get("content", "")
        token_counts.append(estimate_tokens(total_text))

    return DatasetStats(
        total_examples=len(data),
        total_tokens=sum(token_counts),
        avg_tokens_per_example=sum(token_counts) / len(token_counts),
        min_tokens=min(token_counts),
        max_tokens=max(token_counts)
    )


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("SCRIPT: data_processor.py", flush=True)
    print("PURPOSE: Demonstrate data processing utilities", flush=True)
    print("=" * 60, flush=True)

    # Create sample data
    sample_data = [
        {"messages": [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"}
        ]},
        {"messages": [
            {"role": "user", "content": "What's the weather like?"},
            {"role": "assistant", "content": "I don't have access to real-time weather data, but I'd be happy to help you find a weather service!"}
        ]},
    ]

    print("\n[STEP 1] Validating examples...", flush=True)
    for i, ex in enumerate(sample_data):
        valid, error = validate_conversation(ex)
        print(f"  Example {i}: {'Valid' if valid else f'Invalid - {error}'}", flush=True)

    print("\n[STEP 2] Computing statistics...", flush=True)
    stats = compute_dataset_stats(sample_data)
    print(f"  {stats}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("SUCCESS: Data processor demo completed!", flush=True)
    print("=" * 60, flush=True)
