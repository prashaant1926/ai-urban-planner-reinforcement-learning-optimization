#!/usr/bin/env python3
"""A simple hello world script demonstrating basic Python structure."""

def greet(name: str) -> str:
    """Return a greeting message for the given name."""
    return f"Hello, {name}! Welcome to the workspace."


def main():
    print("=" * 50, flush=True)
    print("SCRIPT: hello.py", flush=True)
    print("PURPOSE: Demonstrate basic Python functionality", flush=True)
    print("=" * 50, flush=True)

    names = ["Alice", "Bob", "Charlie"]
    for name in names:
        print(greet(name), flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
