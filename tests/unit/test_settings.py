"""Unit tests for the settings/config system."""

import pytest
import os
from unittest.mock import patch


class TestSettings:
    def test_settings_load_defaults(self):
        from merv.config.settings import Settings
        s = Settings()
        assert s.family_timezone == "America/New_York"
        assert s.morning_routine_hour == 7
        assert s.morning_routine_minute == 0
        assert s.voice_enabled is False
        assert s.merv_env == "development"

    def test_allowed_chat_ids_parsed(self):
        from merv.config.settings import Settings
        s = Settings(telegram_allowed_chat_ids="111,222,333")
        assert s.allowed_chat_ids == [111, 222, 333]

    def test_empty_chat_ids_returns_empty_list(self):
        from merv.config.settings import Settings
        s = Settings(telegram_allowed_chat_ids="")
        assert s.allowed_chat_ids == []

    def test_calendar_ids_parsed(self):
        from merv.config.settings import Settings
        s = Settings(google_calendar_ids="primary,family@group.calendar.google.com")
        assert "primary" in s.calendar_ids
        assert len(s.calendar_ids) == 2

    def test_family_members_list(self):
        from merv.config.settings import Settings
        s = Settings(family_members="Dad,Mom,Henry,Lily")
        assert s.family_members_list == ["Dad", "Mom", "Henry", "Lily"]

    def test_rain_out_sports_list(self):
        from merv.config.settings import Settings
        s = Settings(rain_out_sports="soccer,baseball,lacrosse")
        assert "soccer" in s.rain_out_sports_list
        assert len(s.rain_out_sports_list) == 3

    def test_is_production_flag(self):
        from merv.config.settings import Settings
        assert Settings(merv_env="production").is_production is True
        assert Settings(merv_env="development").is_production is False

    def test_data_dir_expands_home(self):
        from merv.config.settings import Settings
        from pathlib import Path
        s = Settings(merv_data_dir="~/.config/merv/data")
        assert not str(s.data_dir).startswith("~")
        assert isinstance(s.data_dir, Path)
