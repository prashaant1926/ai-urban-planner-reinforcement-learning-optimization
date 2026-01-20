.PHONY: help install test lint format clean run

# Default target
help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linter"
	@echo "  make format     - Format code"
	@echo "  make clean      - Clean up generated files"
	@echo "  make run        - Run the main script"

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	pytest tests/ -v --cov=. --cov-report=term-missing

# Run linter
lint:
	mypy *.py --ignore-missing-imports
	ruff check .

# Format code
format:
	ruff format .
	ruff check --fix .

# Clean up
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf *.egg-info dist build
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Run main script
run:
	python hello.py
