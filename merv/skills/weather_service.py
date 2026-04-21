"""Weather service — checks rain risk for outdoor sports events.

Uses Open-Meteo (no API key required) for production, mock data for tests.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

# Open-Meteo free API — no key needed
_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather codes indicating rain/precipitation
_RAIN_CODES = {
    51, 53, 55,         # drizzle
    61, 63, 65,         # rain
    71, 73, 75,         # snow (treated as weather disruption)
    80, 81, 82,         # rain showers
    95, 96, 99,         # thunderstorm
}


class WeatherService:
    def __init__(self, mock: bool = False, latitude: float = 40.7128, longitude: float = -74.0060):
        """
        latitude/longitude default to New York City.
        Override via FAMILY_LATITUDE / FAMILY_LONGITUDE env vars or pass directly.
        """
        self._mock = mock
        self.latitude = latitude
        self.longitude = longitude

    async def get_forecast(self, target_date: Optional[date] = None) -> dict[str, Any]:
        """
        Return weather forecast for target_date.
        Returns dict with keys: date, temp_max, temp_min, rain_code, rain_risk, description.
        """
        if target_date is None:
            target_date = datetime.utcnow().date()

        if self._mock:
            return self._mock_forecast(target_date, rain=True)

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                params = {
                    "latitude": self.latitude,
                    "longitude": self.longitude,
                    "daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum",
                    "timezone": "America/New_York",
                    "start_date": target_date.isoformat(),
                    "end_date": target_date.isoformat(),
                }
                resp = await client.get(_OPEN_METEO_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

            daily = data.get("daily", {})
            codes = daily.get("weathercode", [0])
            code = codes[0] if codes else 0
            temp_max = (daily.get("temperature_2m_max") or [None])[0]
            temp_min = (daily.get("temperature_2m_min") or [None])[0]
            precip = (daily.get("precipitation_sum") or [0.0])[0]

            rain_risk = code in _RAIN_CODES or (precip or 0) > 2.0

            return {
                "date": target_date.isoformat(),
                "temp_max": temp_max,
                "temp_min": temp_min,
                "weather_code": code,
                "precipitation_mm": precip,
                "rain_risk": rain_risk,
                "description": _describe_code(code),
            }
        except Exception as e:
            logger.warning("Weather fetch failed, using mock: %s", e)
            return self._mock_forecast(target_date, rain=False)

    def _mock_forecast(self, target_date: date, rain: bool = False) -> dict[str, Any]:
        code = 61 if rain else 1  # 61=light rain, 1=mainly clear
        return {
            "date": target_date.isoformat(),
            "temp_max": 58.0 if rain else 72.0,
            "temp_min": 48.0 if rain else 58.0,
            "weather_code": code,
            "precipitation_mm": 12.0 if rain else 0.0,
            "rain_risk": rain,
            "description": "Light rain expected" if rain else "Mainly clear",
        }


def _describe_code(code: int) -> str:
    descriptions = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Foggy", 48: "Icy fog",
        51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
        61: "Light rain", 63: "Moderate rain", 65: "Heavy rain",
        71: "Light snow", 73: "Moderate snow", 75: "Heavy snow",
        80: "Light showers", 81: "Moderate showers", 82: "Heavy showers",
        95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with heavy hail",
    }
    return descriptions.get(code, f"Weather code {code}")
