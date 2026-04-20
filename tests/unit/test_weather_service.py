"""Unit tests for the weather service."""

import pytest
from datetime import date

from merv.skills.weather_service import WeatherService


class TestWeatherServiceMock:
    @pytest.mark.asyncio
    async def test_returns_forecast_dict(self):
        service = WeatherService(mock=True)
        forecast = await service.get_forecast(date(2026, 4, 20))
        assert isinstance(forecast, dict)

    @pytest.mark.asyncio
    async def test_required_fields_present(self):
        service = WeatherService(mock=True)
        forecast = await service.get_forecast(date(2026, 4, 20))
        required = {"date", "temp_max", "temp_min", "weather_code", "rain_risk", "description"}
        assert required.issubset(forecast.keys())

    @pytest.mark.asyncio
    async def test_rain_forecast_has_rain_risk_true(self):
        service = WeatherService(mock=True)
        # Mock with rain=True is default for WeatherService(mock=True)
        # Verify internal _mock_forecast
        forecast = service._mock_forecast(date(2026, 4, 20), rain=True)
        assert forecast["rain_risk"] is True
        assert forecast["precipitation_mm"] > 0

    @pytest.mark.asyncio
    async def test_clear_forecast_has_rain_risk_false(self):
        service = WeatherService(mock=True)
        forecast = service._mock_forecast(date(2026, 4, 20), rain=False)
        assert forecast["rain_risk"] is False
        assert forecast["precipitation_mm"] == 0.0

    @pytest.mark.asyncio
    async def test_date_matches_requested(self):
        service = WeatherService(mock=True)
        target = date(2026, 4, 21)
        forecast = await service.get_forecast(target)
        assert forecast["date"] == target.isoformat()

    @pytest.mark.asyncio
    async def test_default_date_is_today(self):
        from datetime import datetime
        service = WeatherService(mock=True)
        forecast = await service.get_forecast()
        today = datetime.utcnow().date().isoformat()
        assert forecast["date"] == today
