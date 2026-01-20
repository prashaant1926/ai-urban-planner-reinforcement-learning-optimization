#!/usr/bin/env python3
"""Utility functions for data processing and common operations."""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(filepath: str) -> Dict[str, Any]:
    """Load and parse a JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """Save data to a JSON file with pretty formatting."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    print(f"Saved: {filepath}", flush=True)


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_tokens_approx(text: str) -> int:
    """Approximate token count (roughly 4 chars per token for English)."""
    return len(text) // 4


def format_number(n: int) -> str:
    """Format large numbers with commas for readability."""
    return f"{n:,}"


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to max_length, adding suffix if truncated."""
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


class ProgressTracker:
    """Simple progress tracker for loops."""

    def __init__(self, total: int, name: str = "Progress"):
        self.total = total
        self.current = 0
        self.name = name

    def update(self, step: int = 1) -> None:
        """Update progress and print status."""
        self.current += step
        pct = (self.current / self.total) * 100
        print(f"[{self.name}] {self.current}/{self.total} ({pct:.1f}%)", flush=True)

    def done(self) -> None:
        """Mark as complete."""
        print(f"[{self.name}] Complete!", flush=True)
