"""Flood-risk fusion: detected drain/water facts x short-term rain forecast (Open-Meteo)."""

from __future__ import annotations
import requests


def rain_next_hours(lat: float, lon: float, hours: int = 6) -> dict | None:
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            timeout=8,
            params={
                "latitude": lat,
                "longitude": lon,
                "forecast_days": 1,
                "timezone": "auto",
                "hourly": "precipitation,precipitation_probability",
            },
        )
        h = r.json()["hourly"]
        return {
            "mm": round(sum(h["precipitation"][:hours]), 1),
            "prob_max": max(h["precipitation_probability"][:hours]),
        }
    except Exception:
        return None


def flood_risk(hazards: set[str], rain: dict | None) -> dict:
    base = 0.0
    if "drain_blocked" in hazards:
        base += 0.40
    if "waste_pile" in hazards:
        base += 0.15
    if "flood_water" in hazards:
        base += 0.45
    mult = 0.8 if rain is None else 0.6 + 0.9 * min(rain["mm"] / 10.0, 1.0)
    score = min(base * mult, 1.0)
    level = "high" if score >= 0.7 else "medium" if score >= 0.35 else "low"
    return {"score": round(score, 2), "level": level, "rain": rain}
