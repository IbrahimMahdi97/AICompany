"""Runtime configuration, read from environment variables (see .env.example).

Uses pydantic-settings so values are read and validated when ``Settings`` is
instantiated (not captured at import time), which removes any dependency on the order
in which ``.env`` is loaded relative to this module's import.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # protected_namespaces=() so the model_* fields below don't collide with pydantic's
    # reserved "model_" namespace.
    model_config = SettingsConfigDict(
        env_file=".env", extra="ignore", protected_namespaces=()
    )

    model_ba: str = "claude-sonnet-5"
    model_test: str = "claude-sonnet-5"
    model_dev: str = "claude-opus-4-8"
    max_qa_rounds: int = 3
    sandbox_timeout: int = 60
    sandbox_backend: str = "auto"  # auto | docker | local
    sandbox_image: str = "agent-sandbox:py312"


settings = Settings()
