"""Central settings via pydantic-settings — single source of truth for all config."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Core LLM ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    llm_model: str = Field(default="claude-sonnet-4-6", description="Default Anthropic model")
    llm_temperature: float = Field(default=0.3)
    llm_max_tokens: int = Field(default=4096)

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram_bot_token: str = Field(default="", description="Telegram bot token")
    telegram_allowed_chat_ids: str = Field(default="", description="Comma-separated allowed chat IDs")

    @property
    def allowed_chat_ids(self) -> list[int]:
        if not self.telegram_allowed_chat_ids:
            return []
        return [int(cid.strip()) for cid in self.telegram_allowed_chat_ids.split(",") if cid.strip()]

    # ── Google ────────────────────────────────────────────────────────────────
    google_credentials_file: str = Field(default="~/.config/merv/google_credentials.json")
    google_token_file: str = Field(default="~/.config/merv/google_token.json")
    google_calendar_ids: str = Field(default="primary")

    @property
    def calendar_ids(self) -> list[str]:
        return [cid.strip() for cid in self.google_calendar_ids.split(",") if cid.strip()]

    # ── Family Profile ────────────────────────────────────────────────────────
    family_timezone: str = Field(default="America/New_York")
    family_members: str = Field(default="Dad,Mom,Henry")
    henry_schedule_constraints: str = Field(default="no_early_morning_appts,no_school_hours")
    rain_out_sports: str = Field(default="soccer,baseball,lacrosse")

    @property
    def family_members_list(self) -> list[str]:
        return [m.strip() for m in self.family_members.split(",") if m.strip()]

    @property
    def rain_out_sports_list(self) -> list[str]:
        return [s.strip() for s in self.rain_out_sports.split(",") if s.strip()]

    # ── Morning Routine ───────────────────────────────────────────────────────
    morning_routine_hour: int = Field(default=7)
    morning_routine_minute: int = Field(default=0)
    morning_routine_chat_id: Optional[int] = Field(default=None)

    # ── Memory / Persistence ─────────────────────────────────────────────────
    merv_data_dir: str = Field(default="~/.config/merv/data")
    merv_memory_file: str = Field(default="~/.config/merv/data/memory.json")

    @property
    def data_dir(self) -> Path:
        return Path(self.merv_data_dir).expanduser()

    @property
    def memory_file(self) -> Path:
        return Path(self.merv_memory_file).expanduser()

    # ── Voice (Phase 2 placeholder) ───────────────────────────────────────────
    voice_enabled: bool = Field(default=False)
    voice_engine: str = Field(default="pyttsx3")  # pyttsx3 | elevenlabs | hudson
    elevenlabs_api_key: str = Field(default="")

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="console")  # json | console

    # ── Environment ───────────────────────────────────────────────────────────
    merv_env: str = Field(default="development")

    @property
    def is_production(self) -> bool:
        return self.merv_env.lower() == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
