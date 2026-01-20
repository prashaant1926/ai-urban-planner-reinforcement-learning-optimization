#!/usr/bin/env python3
"""Project-wide constants and default configurations."""

# Paths
DEFAULT_DATA_DIR = "artifacts/dataset"
DEFAULT_METRICS_DIR = "artifacts/metrics"
DEFAULT_CHECKPOINT_DIR = "artifacts/checkpoints"
DEFAULT_LOG_DIR = "artifacts/logs"

# Training defaults
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 3
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_SEQ_LENGTH = 2048

# Logging
LOG_EVERY_N_STEPS = 10
SAVE_EVERY_N_STEPS = 500
EVAL_EVERY_N_STEPS = 100

# Model configurations
SUPPORTED_MODELS = [
    "Qwen3-8B",
    "Llama-3.1-8B",
    "Llama-3.1-70B",
    "Mistral-7B",
]

# Data format keys
MESSAGES_KEY = "messages"
ROLE_KEY = "role"
CONTENT_KEY = "content"

# Role types
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"

# File size limits (in bytes)
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
