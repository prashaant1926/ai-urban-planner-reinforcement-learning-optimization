"""Project constants and configuration values."""

from enum import Enum
from typing import Final

# Version info
VERSION: Final[str] = "1.0.0"
PROJECT_NAME: Final[str] = "demo-project"

# File paths
DEFAULT_CONFIG_PATH: Final[str] = "config.yaml"
ARTIFACTS_DIR: Final[str] = "artifacts"
CHECKPOINT_DIR: Final[str] = "artifacts/checkpoints"
METRICS_DIR: Final[str] = "artifacts/metrics"
DATASET_DIR: Final[str] = "artifacts/dataset"

# Training defaults
DEFAULT_BATCH_SIZE: Final[int] = 8
DEFAULT_LEARNING_RATE: Final[float] = 2e-5
DEFAULT_NUM_EPOCHS: Final[int] = 3
DEFAULT_WARMUP_RATIO: Final[float] = 0.1
DEFAULT_WEIGHT_DECAY: Final[float] = 0.01
MAX_SEQUENCE_LENGTH: Final[int] = 4096

# Data processing
DEFAULT_TRAIN_RATIO: Final[float] = 0.8
DEFAULT_VAL_RATIO: Final[float] = 0.1
DEFAULT_TEST_RATIO: Final[float] = 0.1
DEFAULT_SEED: Final[int] = 42


class MessageRole(Enum):
    """Valid message roles in conversation data."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class TrainingPhase(Enum):
    """Training phases."""
    WARMUP = "warmup"
    TRAINING = "training"
    COOLDOWN = "cooldown"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Supported models (examples)
SUPPORTED_MODELS = [
    "Qwen3-8B",
    "Llama-3.1-8B",
    "Llama-3.1-70B",
    "Mistral-7B",
]

# File size limits
MAX_FILE_SIZE_MB: Final[int] = 100
MAX_FILE_SIZE_BYTES: Final[int] = MAX_FILE_SIZE_MB * 1024 * 1024
