#!/usr/bin/env python3
"""Unit tests for utility functions."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    count_tokens_approx,
    format_number,
    truncate_string,
    ProgressTracker
)


def test_count_tokens_approx():
    """Test approximate token counting."""
    text = "Hello world"  # 11 chars
    tokens = count_tokens_approx(text)
    assert tokens == 2, f"Expected 2, got {tokens}"
    print("test_count_tokens_approx: PASSED", flush=True)


def test_format_number():
    """Test number formatting."""
    assert format_number(1000) == "1,000"
    assert format_number(1000000) == "1,000,000"
    assert format_number(42) == "42"
    print("test_format_number: PASSED", flush=True)


def test_truncate_string():
    """Test string truncation."""
    short = "Hello"
    long_text = "This is a very long string that should be truncated"

    assert truncate_string(short, max_length=10) == "Hello"
    assert len(truncate_string(long_text, max_length=20)) == 20
    assert truncate_string(long_text, max_length=20).endswith("...")
    print("test_truncate_string: PASSED", flush=True)


def test_progress_tracker():
    """Test progress tracker."""
    tracker = ProgressTracker(total=10, name="Test")
    assert tracker.total == 10
    assert tracker.current == 0

    tracker.update(5)
    assert tracker.current == 5
    print("test_progress_tracker: PASSED", flush=True)


def run_all_tests():
    """Run all tests."""
    print("=" * 50, flush=True)
    print("RUNNING TESTS", flush=True)
    print("=" * 50, flush=True)

    test_count_tokens_approx()
    test_format_number()
    test_truncate_string()
    test_progress_tracker()

    print("=" * 50, flush=True)
    print("ALL TESTS PASSED!", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    run_all_tests()
