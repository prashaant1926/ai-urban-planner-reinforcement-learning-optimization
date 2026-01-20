#!/usr/bin/env python3
"""A simple hello world script demonstrating basic Python."""

import sys
from datetime import datetime


def greet(name: str = "World") -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


def main():
    print("=" * 50, flush=True)
    print("SCRIPT: hello.py", flush=True)
    print(f"TIME: {datetime.now().isoformat()}", flush=True)
    print("=" * 50, flush=True)

    name = sys.argv[1] if len(sys.argv) > 1 else "World"
    message = greet(name)
    print(message, flush=True)

    print("", flush=True)
    print("SUCCESS: Script completed!", flush=True)


if __name__ == "__main__":
    main()
