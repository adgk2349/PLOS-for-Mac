from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOCAL_AI_", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8777
    session_token: str = Field(default_factory=lambda: os.environ.get("LOCAL_AI_SESSION_TOKEN", "dev-session-token"))
    data_dir: Path = Field(default_factory=lambda: Path(os.environ.get("LOCAL_AI_DATA_DIR", "./data")).resolve())
    embedding_dim: int = 384
    local_model_profile: str = "recommended"
    async_repo_backend: str = "adapter"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "local_ai_core.sqlite3"

    @property
    def lancedb_path(self) -> Path:
        return self.data_dir / "lancedb"


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.lancedb_path.mkdir(parents=True, exist_ok=True)
