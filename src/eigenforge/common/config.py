from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["human", "json"] = Field(default="human", alias="LOG_FORMAT")
    # File sink for logs (rotated). Anchored to the project root by default so
    # running from another working directory never writes into an unrelated logs/
    # folder; configurable via LOG_FILE (or empty string to disable).
    log_file: str = Field(default=str(_PROJECT_ROOT / "logs" / "app.log"), alias="LOG_FILE")

    # GenAI Settings
    celeba_data_path: str = Field(default="./data/celeba", alias="CELEBA_DATA_PATH")
    eigengan_checkpoint_dir: str = Field(default="./checkpoints/eigengan", alias="EIGENGAN_CHECKPOINT_DIR")
    diffusion_checkpoint_dir: str = Field(default="./checkpoints/diffusion", alias="DIFFUSION_CHECKPOINT_DIR")
    device: str = Field(default="cuda", alias="DEVICE")


config = AppConfig()
