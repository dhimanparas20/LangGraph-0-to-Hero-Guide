"""Web tools: free public APIs (no key, no auth), with optional Redis caching."""

import math
import urllib.parse

import requests
from langchain_core.tools import tool

from modules.tools.cache import cache_get, cache_set

WEATHER_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


def _geocode(place: str) -> dict:
    """Turns a place name into coordinates using the free Open-Meteo geocoder.

    Args:
        place: a city or place name, e.g. "Delhi".

    Returns:
        The best-match geocoding result dict (name, country, latitude,
        longitude, ...).

    Raises:
        ValueError: when the place name has no match.
    """
    url = "https://geocoding-api.open-meteo.com/v1/search?name=" + urllib.parse.quote(place) + "&count=1"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    results = response.json().get("results")
    if not results:
        raise ValueError("Place not found: " + place)
    return results[0]


@tool
def get_weather(city: str) -> str:
    """Current weather for a city (temperature, wind, conditions). Works worldwide.

    Args:
        city: the city name, e.g. "Delhi" or "Paris".

    Returns:
        A one-line weather report (condition, temperature, humidity, wind),
        served from the Redis cache when a recent result exists.
    """
    key = "weather:" + city.lower().strip()

    cached = cache_get(key)
    if cached:
        return str(cached) + " (cached)"

    place = _geocode(city)
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=" + str(place["latitude"]) +
        "&longitude=" + str(place["longitude"]) +
        "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    current = response.json()["current"]

    description = WEATHER_CODES.get(current["weather_code"], "weather code " + str(current["weather_code"]))
    result = (
        "Weather in " + place["name"] + ", " + place.get("country", "") + ": "
        + description
        + ", temperature " + str(current["temperature_2m"]) + " C"
        + ", humidity " + str(current["relative_humidity_2m"]) + "%"
        + ", wind " + str(current["wind_speed_10m"]) + " km/h"
    )

    cache_set(key, result, ttl=600)  # cache weather for 10 minutes
    return result


@tool
def get_distance(place_a: str, place_b: str) -> str:
    """Great-circle distance in km between two places, e.g. get_distance('Delhi', 'Mumbai').

    Args:
        place_a: the first place name.
        place_b: the second place name.

    Returns:
        A one-line distance report in kilometers (great-circle), cached
        for a day because distances do not change.
    """
    key = "distance:" + place_a.lower().strip() + ":" + place_b.lower().strip()

    cached = cache_get(key)
    if cached:
        return str(cached) + " (cached)"

    a = _geocode(place_a)
    b = _geocode(place_b)

    # haversine formula: distance between two points on a sphere
    lat1, lon1, lat2, lon2 = math.radians(a["latitude"]), math.radians(a["longitude"]), math.radians(b["latitude"]), math.radians(b["longitude"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    km = 6371 * 2 * math.asin(math.sqrt(h))

    result = (
        "Distance " + a["name"] + " (" + a.get("country", "") + ") to "
        + b["name"] + " (" + b.get("country", "") + "): "
        + str(round(km)) + " km (great-circle)"
    )

    cache_set(key, result, ttl=86400)  # distances don't change - cache a day
    return result


@tool
def get_currency_rate(from_currency: str, to_currency: str) -> str:
    """Exchange rate between two currencies, e.g. get_currency_rate('USD', 'INR').

    Args:
        from_currency: the base currency code, e.g. "USD".
        to_currency: the target currency code, e.g. "INR".

    Returns:
        A one-line rate report with its update time, cached for an hour.
    """
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()
    key = "currency:" + from_currency + ":" + to_currency

    cached = cache_get(key)
    if cached:
        return str(cached) + " (cached)"

    url = "https://open.er-api.com/v6/latest/" + from_currency
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("result") != "success":
        return "Error: could not get rates (check the currency code, e.g. USD, INR, EUR)."

    rate = data.get("rates", {}).get(to_currency)
    if rate is None:
        return "Error: unknown currency code: " + to_currency

    result = "1 " + from_currency + " = " + str(rate) + " " + to_currency + " (as of " + str(data.get("time_last_update_utc", "")) + ")"

    cache_set(key, result, ttl=3600)  # rates change - cache an hour
    return result


@tool
def search_wikipedia(topic: str) -> str:
    """Short summary of a topic from Wikipedia. Good for people, places, events, concepts.

    Args:
        topic: the topic to look up, e.g. "Albert Einstein".

    Returns:
        The Wikipedia summary extract for the topic, cached for a day,
        or a not-found message.
    """
    key = "wiki:" + topic.lower().strip()

    cached = cache_get(key)
    if cached:
        return str(cached)

    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(topic.strip().replace(" ", "_"))
    response = requests.get(url, timeout=10, headers={"User-Agent": "learning-chatbot/1.0"})

    if response.status_code == 404:
        return "No Wikipedia article found for: " + topic

    response.raise_for_status()
    data = response.json()

    title = data.get("title", topic)
    extract = data.get("extract", "")
    if not extract:
        return "No summary available for: " + topic

    result = title + ": " + extract
    cache_set(key, result, ttl=86400)
    return result
