"""Runtime configuration, read from environment variables (see .env.example)."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    model_ba: str = os.getenv("MODEL_BA", "claude-sonnet-5")
    model_test: str = os.getenv("MODEL_TEST", "claude-sonnet-5")
    model_dev: str = os.getenv("MODEL_DEV", "claude-opus-4-8")
    max_qa_rounds: int = int(os.getenv("MAX_QA_ROUNDS", "3"))
    sandbox_timeout: int = int(os.getenv("SANDBOX_TIMEOUT", "60"))
    sandbox_backend: str = os.getenv("SANDBOX_BACKEND", "auto")  # auto | docker | local
    sandbox_image: str = os.getenv("SANDBOX_IMAGE", "agent-sandbox:py312")


settings = Settings()
