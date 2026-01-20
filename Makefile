# Makefile for the project

.PHONY: help install test clean run lint

help:
	@echo "Available targets:"
	@echo "  install   - Install dependencies"
	@echo "  test      - Run unit tests"
	@echo "  clean     - Remove generated files"
	@echo "  run       - Run the main experiment"
	@echo "  lint      - Run linter"

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

test-basic:
	python tests/test_utils.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .cache/

run:
	bash run_experiment.sh

lint:
	python -m flake8 *.py --max-line-length=100

check-data:
	python data_loader.py

demo:
	python hello.py
