#!/bin/bash
# Script to run a training experiment

set -e  # Exit on error

echo "=============================================="
echo "Starting Experiment Run"
echo "=============================================="

# Create directories if they don't exist
mkdir -p artifacts/dataset artifacts/metrics artifacts/checkpoints artifacts/logs

# Set environment
export PYTHONUNBUFFERED=1

# Run data preparation
echo ""
echo "[STEP 1] Preparing data..."
python data_loader.py 2>&1 | tee artifacts/logs/data_prep.log

# Run main training (placeholder)
echo ""
echo "[STEP 2] Training would run here..."
# python train.py 2>&1 | tee artifacts/logs/training.log

# Summarize results
echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "Logs saved to: artifacts/logs/"
echo "=============================================="
