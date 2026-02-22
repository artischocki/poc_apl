"""Seed TimescaleDB with mock sensor data for three simulated test runs.

Run once (or re-run to refresh):
    uv run python seed_db.py

Generates three test runs with 9 correlated channels each at 10 Hz:
    - test_run_city.mf4     30 min  city driving
    - test_run_highway.mf4  45 min  highway cruise
    - test_run_track.mf4    15 min  performance track session

Signals are physically correlated (RPM follows speed, temp warms up, etc.)
so the data is useful for realistic SQL exploration.
"""

import os
from datetime import datetime, timezone, timedelta

import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(seed=42)


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a simple moving-average to remove hard edges."""
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def generate_city_run(n: int, dt: float = 0.1) -> dict:
    """30-min city drive: lots of stop-and-go, max ~70 km/h."""
    t = np.arange(n) * dt

    # Speed: slow oscillating cycles + random stop events
    speed = (
        35
        + 20 * np.sin(2 * np.pi * t / 90)
        + 12 * np.sin(2 * np.pi * t / 25)
        + RNG.normal(0, 2, n)
    )
    # Occasional full stops
    stops = RNG.integers(0, n, size=15)
    for s in stops:
        speed[max(0, s - 30) : s + 30] *= np.linspace(1, 0, min(60, n - max(0, s - 30)))
    speed = _smooth(np.clip(speed, 0, 70), window=10)

    return _derive_channels(speed, t, n, start_fuel=85, fuel_rate=8)


def generate_highway_run(n: int, dt: float = 0.1) -> dict:
    """45-min highway cruise: high constant speed, gentle overtakes."""
    t = np.arange(n) * dt

    speed = (
        115
        + 15 * np.sin(2 * np.pi * t / 300)
        + 5 * np.sin(2 * np.pi * t / 60)
        + RNG.normal(0, 3, n)
    )
    speed = _smooth(np.clip(speed, 80, 160), window=20)

    return _derive_channels(speed, t, n, start_fuel=100, fuel_rate=12)


def generate_track_run(n: int, dt: float = 0.1) -> dict:
    """15-min track session: aggressive acceleration and hard braking."""
    t = np.arange(n) * dt

    # Sawtooth-like lap profile
    lap_t = t % 90  # ~90-second laps
    speed = 20 + 130 * (lap_t / 90) ** 0.6 + RNG.normal(0, 5, n)
    # Hard braking before each lap end
    brake_zone = (lap_t > 75) & (lap_t < 90)
    speed[brake_zone] = np.clip(speed[brake_zone] * 0.3, 0, 200)
    speed = _smooth(np.clip(speed, 0, 200), window=5)

    return _derive_channels(speed, t, n, start_fuel=60, fuel_rate=18)


def _derive_channels(
    speed: np.ndarray, t: np.ndarray, n: int, start_fuel: float, fuel_rate: float
) -> dict:
    """Derive all other channels from the speed profile."""
    speed_delta = np.diff(speed, prepend=speed[0])

    # RPM: idle at 800, scales with speed, rev-matched on downshift
    rpm = 800 + speed * 45 + np.abs(speed_delta) * 80 + RNG.normal(0, 60, n)
    rpm = _smooth(np.clip(rpm, 800, 7500), window=5)

    # Throttle: correlates with positive speed delta
    throttle = np.clip(speed_delta * 8 + 25 + RNG.normal(0, 4, n), 0, 100)
    throttle = _smooth(throttle, window=3)

    # Brake pressure: spikes on deceleration
    brake = np.where(speed_delta < -0.8, np.abs(speed_delta) * 12, 0.0)
    brake += RNG.exponential(0.2, n) * (brake > 0)
    brake = _smooth(np.clip(brake, 0, 180), window=3)

    # Engine temp: exponential warm-up from ambient to ~92°C
    ambient_base = 19 + RNG.normal(0, 0.5, n)
    engine_temp = ambient_base + 72 * (1 - np.exp(-t / 480)) + RNG.normal(0, 0.4, n)

    # Coolant temp: slightly lags engine temp
    coolant_temp = ambient_base + 68 * (1 - np.exp(-t / 600)) + RNG.normal(0, 0.3, n)

    # Battery voltage: drops under heavy load (high RPM)
    battery = 14.3 - (rpm / 7500) * 0.6 + RNG.normal(0, 0.05, n)
    battery = np.clip(battery, 13.4, 14.8)

    # Oil pressure: rises with RPM
    oil_pressure = 1.8 + (rpm / 7500) * 4.5 + RNG.normal(0, 0.08, n)
    oil_pressure = np.clip(oil_pressure, 1.5, 7.0)

    # Fuel level: monotonically decreasing
    fuel = start_fuel - (t / 3600) * fuel_rate + RNG.normal(0, 0.05, n)
    fuel = np.clip(fuel, 0, 100)

    return {
        "vehicle_speed":   (speed,        "km/h"),
        "engine_rpm":      (rpm,           "rpm"),
        "throttle_pos":    (throttle,      "%"),
        "brake_pressure":  (brake,         "bar"),
        "engine_temp":     (engine_temp,   "°C"),
        "coolant_temp":    (coolant_temp,  "°C"),
        "battery_voltage": (battery,       "V"),
        "oil_pressure":    (oil_pressure,  "bar"),
        "fuel_level":      (fuel,          "%"),
    }


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

RUNS = [
    {
        "file_name": "test_run_city.mf4",
        "start": datetime(2024, 6, 10, 8, 0, 0, tzinfo=timezone.utc),
        "duration_min": 30,
        "generator": generate_city_run,
    },
    {
        "file_name": "test_run_highway.mf4",
        "start": datetime(2024, 6, 11, 14, 0, 0, tzinfo=timezone.utc),
        "duration_min": 45,
        "generator": generate_highway_run,
    },
    {
        "file_name": "test_run_track.mf4",
        "start": datetime(2024, 6, 12, 10, 30, 0, tzinfo=timezone.utc),
        "duration_min": 15,
        "generator": generate_track_run,
    },
]

DT = 0.1        # 10 Hz
BATCH = 10_000  # rows per INSERT


def seed(conn) -> None:
    """Generate and insert all runs."""
    for run in RUNS:
        n = int(run["duration_min"] * 60 / DT)
        channels = run["generator"](n, dt=DT)
        file_name = run["file_name"]
        start: datetime = run["start"]

        print(f"\n→ {file_name}  ({run['duration_min']} min, {n} samples/channel)")

        # Delete existing data for this run to allow re-seeding
        with conn.cursor() as cur:
            cur.execute("DELETE FROM measurements WHERE file_name = %s", (file_name,))
        conn.commit()

        timestamps = [start + timedelta(seconds=i * DT) for i in range(n)]

        for ch_name, (values, unit) in channels.items():
            rows = [
                (timestamps[i], file_name, ch_name, float(v), unit)
                for i, v in enumerate(values)
                if np.isfinite(v)
            ]
            with conn.cursor() as cur:
                for i in range(0, len(rows), BATCH):
                    psycopg2.extras.execute_values(
                        cur,
                        "INSERT INTO measurements (time, file_name, channel, value, unit) VALUES %s",
                        rows[i : i + BATCH],
                    )
            conn.commit()
            print(f"   {ch_name:20s} {len(rows):>7,} rows  [{unit}]")


def main() -> None:
    url = os.environ["TIMESCALEDB_URL"]
    print(f"Connecting to {url} …")
    conn = psycopg2.connect(url)
    try:
        seed(conn)
    finally:
        conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
