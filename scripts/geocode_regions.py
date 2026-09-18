#!/usr/bin/env python
"""Geocode (country, region) pairs to latitude/longitude.

This script produced ``data/raw/extracted_coords.csv`` once; the file is
versioned so the analysis never needs network access. Re-run it only if new
regions appear in the data.

    python scripts/geocode_regions.py --input regions.csv --output data/raw/extracted_coords.csv

Nominatim (OpenStreetMap) is tried first; OpenCage is used as a fallback when
``OPENCAGE_API_KEY`` is set in the environment (or in a ``.env`` file, see
``.env.example``). Both services are rate-limited: one request per second.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim

OPENCAGE_URL = "https://api.opencagedata.com/geocode/v1/json"


def load_env_file(path: str = ".env") -> None:
    """Minimal .env loader so that python-dotenv is not a hard dependency."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def geocode_opencage(query: str, api_key: str) -> tuple[float, float] | None:
    params = {"q": query, "key": api_key, "limit": 1, "no_annotations": 1, "language": "en"}
    try:
        with urllib.request.urlopen(f"{OPENCAGE_URL}?{urllib.parse.urlencode(params)}", timeout=10) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, ValueError):
        return None
    results = payload.get("results") or []
    if not results:
        return None
    geometry = results[0].get("geometry") or {}
    if geometry.get("lat") is None or geometry.get("lng") is None:
        return None
    return float(geometry["lat"]), float(geometry["lng"])


def geocode_nominatim(geolocator: Nominatim, query: str, retries: int = 3) -> tuple[float, float] | None:
    for attempt in range(retries):
        try:
            location = geolocator.geocode(query, timeout=10)
            return (location.latitude, location.longitude) if location else None
        except (GeocoderTimedOut, GeocoderServiceError):
            time.sleep(2**attempt)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="CSV with columns 'Country' and 'Region'")
    parser.add_argument("--output", default="data/raw/extracted_coords.csv")
    args = parser.parse_args()

    load_env_file()
    api_key = os.environ.get("OPENCAGE_API_KEY", "").strip()
    geolocator = Nominatim(user_agent="arabica-quality-prediction/1.0")

    df = pd.read_csv(args.input)
    done: set[str] = set()
    if os.path.exists(args.output):
        with open(args.output, encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader, None)
            done = {f"{row[1]}, {row[0]}" if row[1] != "NaN" else row[0] for row in reader if row}

    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["Country.of.Origin", "Region", "Latitude", "Longitude"])
        for _, row in df.iterrows():
            country, region = str(row["Country"]).strip(), str(row["Region"]).strip()
            if country.lower() in {"nan", "none", ""}:
                continue
            region = "NaN" if region.lower() in {"nan", "none", ""} else region
            query = f"{region}, {country}" if region != "NaN" else country
            if query in done:
                continue
            coords = geocode_nominatim(geolocator, query)
            if coords is None and api_key:
                coords = geocode_opencage(query, api_key)
            lat, lon = coords if coords else (None, None)
            writer.writerow([country, region, lat, lon])
            fh.flush()
            time.sleep(1.1)  # rate limit


if __name__ == "__main__":
    main()
