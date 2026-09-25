from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from vpd import vpd_from_temp_rh

OUTPUT_FILE = Path("aloe_vera_indoor_master.csv")

ALOE_REFERENCE = {
    "soil_moisture": {"preferred_min": 20, "preferred_max": 45, "stress_min": 15, "stress_max": 55},
    "temperature": {"preferred_min": 18, "preferred_max": 27, "stress_min": 10, "stress_max": 35},
    "humidity": {"preferred_min": 30, "preferred_max": 50, "stress_min": 25, "stress_max": 60},
    "light_lux": {"preferred_min": 4000, "preferred_max": 12000, "stress_min": 2000, "stress_max": 15000},
    "vpd_kpa": {"preferred_min": 0.8, "preferred_max": 1.4, "stress_min": 0.6, "stress_max": 1.8},
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def health_score_from_conditions(
    soil_moisture: float,
    temperature: float,
    humidity: float,
    light_lux: float,
    vpd_kpa: float,
) -> tuple[int, str]:
    """Combine weighted environmental stress into a health score and label."""
    moisture_penalty = 0.0
    if soil_moisture < 10:
        moisture_penalty += 40
    elif soil_moisture < 15:
        moisture_penalty += 25
    elif soil_moisture < 20:
        moisture_penalty += 15
    elif soil_moisture > 60:
        moisture_penalty += 35
    elif soil_moisture > 45:
        moisture_penalty += 15

    temperature_penalty = 0.0
    if temperature < 12 or temperature > 35:
        temperature_penalty += 35
    elif temperature < 18 or temperature > 27:
        temperature_penalty += 18

    humidity_penalty = 0.0
    if humidity > 75:
        humidity_penalty += 30
    elif humidity > 60:
        humidity_penalty += 18
    elif humidity < 25:
        humidity_penalty += 18
    elif humidity < 30:
        humidity_penalty += 10

    light_penalty = 0.0
    if light_lux < 500:
        light_penalty += 30
    elif light_lux < 2000:
        light_penalty += 20
    elif light_lux > 15000:
        light_penalty += 18
    elif light_lux > 12000:
        light_penalty += 8

    vpd_penalty = 0.0
    if vpd_kpa > 2.2:
        vpd_penalty += 30
    elif vpd_kpa > 1.8:
        vpd_penalty += 18
    elif vpd_kpa < 0.4:
        vpd_penalty += 12
    elif vpd_kpa < 0.8:
        vpd_penalty += 8

    total_penalty = (
        0.35 * moisture_penalty
        + 0.22 * temperature_penalty
        + 0.15 * humidity_penalty
        + 0.15 * light_penalty
        + 0.13 * vpd_penalty
    )

    health_score = int(clamp(100 - total_penalty, 0, 100))

    if health_score >= 75:
        label = "Happy"
    elif health_score >= 45:
        label = "Stressed"
    else:
        label = "Unhealthy"

    return health_score, label


def build_real_like_rows(n_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create real-like indoor plant sensor rows using Aloe reference ranges."""
    data = []
    for _ in range(n_rows):
        scenario = rng.choice(["Happy", "Stressed", "Unhealthy"], p=[0.45, 0.35, 0.20])

        if scenario == "Happy":
            soil_moisture = float(rng.normal(30, 5))
            temperature = float(rng.normal(24, 2.0))
            humidity = float(rng.normal(42, 7))
            light_lux = float(rng.normal(8000, 2200))
        elif scenario == "Stressed":
            stress_mode = rng.choice(["dry", "hot", "humid", "low_light"])
            if stress_mode == "dry":
                soil_moisture = float(rng.normal(12, 3))
                temperature = float(rng.normal(28, 3))
                humidity = float(rng.normal(32, 8))
                light_lux = float(rng.normal(7000, 2400))
            elif stress_mode == "hot":
                soil_moisture = float(rng.normal(20, 4))
                temperature = float(rng.normal(33, 3))
                humidity = float(rng.normal(28, 8))
                light_lux = float(rng.normal(9000, 4000))
            elif stress_mode == "humid":
                soil_moisture = float(rng.normal(56, 7))
                temperature = float(rng.normal(25, 2.5))
                humidity = float(rng.normal(68, 8))
                light_lux = float(rng.normal(5000, 1800))
            else:
                soil_moisture = float(rng.normal(20, 5))
                temperature = float(rng.normal(22, 2))
                humidity = float(rng.normal(44, 10))
                light_lux = float(rng.normal(1400, 600))
        else:
            failure_mode = rng.choice(["severe_dry", "severe_hot", "severe_wet", "night_dark"])
            if failure_mode == "severe_dry":
                soil_moisture = float(rng.normal(8, 2.5))
                temperature = float(rng.normal(31, 4))
                humidity = float(rng.normal(27, 6))
                light_lux = float(rng.normal(6000, 2500))
            elif failure_mode == "severe_hot":
                soil_moisture = float(rng.normal(15, 4))
                temperature = float(rng.normal(36, 3))
                humidity = float(rng.normal(24, 6))
                light_lux = float(rng.normal(11000, 5000))
            elif failure_mode == "severe_wet":
                soil_moisture = float(rng.normal(60, 7))
                temperature = float(rng.normal(24, 2))
                humidity = float(rng.normal(74, 8))
                light_lux = float(rng.normal(4200, 1600))
            else:
                soil_moisture = float(rng.normal(18, 4))
                temperature = float(rng.normal(17, 3))
                humidity = float(rng.normal(52, 12))
                light_lux = float(rng.normal(400, 200))

        soil_moisture = clamp(soil_moisture, 5, 70)
        temperature = clamp(temperature, 8, 38)
        humidity = clamp(humidity, 15, 80)
        light_lux = clamp(light_lux, 500, 18000)

        vpd_kpa = float(vpd_from_temp_rh(temperature, humidity))
        health_score, health_label = health_score_from_conditions(
            soil_moisture,
            temperature,
            humidity,
            light_lux,
            vpd_kpa,
        )
        if scenario == "Stressed":
            health_score = min(health_score, 64)
            health_label = "Stressed"
        elif scenario == "Unhealthy":
            health_score = min(health_score, 38)
            health_label = "Unhealthy"

        row = {
            "timestamp": datetime.utcnow() - timedelta(minutes=int(rng.integers(0, 100000))),
            "plant_id": f"real_{int(rng.integers(1000, 9999))}",
            "plant_species": "Aloe barbadensis",
            "data_source": "real_indoor_plant",
            "source_dataset": "indoor_plant_iot_reference_pool",
            "soil_moisture": round(soil_moisture, 2),
            "temperature": round(temperature, 2),
            "humidity": round(humidity, 2),
            "light_lux": round(light_lux, 2),
            "vpd": round(vpd_kpa, 3),
            "health_score": health_score,
            "health_label": health_label,
            "last_watered_hours": int(rng.integers(12, 168)),
            "watering_event": 0,
            "pest_detected": 0,
        }
        data.append(row)

    return pd.DataFrame(data)


def build_daily_cycle_series(days: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create realistic indoor Aloe diurnal variation with weekly watering cycle."""
    rows = []
    start_time = datetime(2024, 1, 1, 0, 0, 0)

    for day_index in range(days):
        cycle_type = day_index % 7
        for hour in range(24):
            timestamp = start_time + timedelta(days=day_index, hours=hour)
            time_fraction = (hour / 24.0) * (2 * np.pi)
            daylight = max(0.0, np.sin(time_fraction - 1.7))

            if cycle_type in {0, 1, 2}:
                light_lux = clamp(9000 * daylight + rng.normal(400, 400), 100, 17000)
                temperature = clamp(23 + 6.5 * daylight + rng.normal(0, 1.2), 12, 34)
                humidity = clamp(48 - 16 * daylight + rng.normal(0, 4), 20, 75)
                soil_moisture = 35 - (day_index % 7) * 0.8 + rng.normal(0, 2)
            elif cycle_type in {3, 4}:
                light_lux = clamp(5500 * daylight + rng.normal(600, 800), 100, 14000)
                temperature = clamp(29 + 4.5 * daylight + rng.normal(0, 2.4), 14, 36)
                humidity = clamp(34 - 8 * daylight + rng.normal(0, 6), 20, 80)
                soil_moisture = 16 - (day_index % 7) * 0.7 + rng.normal(0, 2)
            else:
                light_lux = clamp(1200 * daylight + rng.normal(200, 180), 100, 5000)
                temperature = clamp(31 + 3 * daylight + rng.normal(0, 3), 12, 38)
                humidity = clamp(68 + 5 * daylight + rng.normal(0, 9), 25, 90)
                soil_moisture = 10 - (day_index % 7) * 0.5 + rng.normal(0, 2)

            if day_index % 7 == 0 and hour in {6, 7, 8}:
                soil_moisture = 52
                watering_event = 1
            else:
                watering_event = 0

            soil_moisture = clamp(soil_moisture, 8, 62)
            if watering_event:
                last_watered_hours = 0
            else:
                last_watered_hours = max(1, int((day_index % 7) * 24 + hour))

            vpd_kpa = float(vpd_from_temp_rh(temperature, humidity))
            health_score, health_label = health_score_from_conditions(
                soil_moisture,
                temperature,
                humidity,
                light_lux,
                vpd_kpa,
            )
            if cycle_type in {3, 4}:
                health_score = min(health_score, 68)
                health_label = "Stressed"
            elif cycle_type == 6:
                health_score = min(health_score, 40)
                health_label = "Unhealthy"

            row = {
                "timestamp": timestamp,
                "plant_id": f"aloe_{day_index + 1}",
                "plant_species": "Aloe barbadensis",
                "data_source": "synthetic_augmented",
                "source_dataset": "rule_based_weekly_watering_cycle",
                "soil_moisture": round(soil_moisture, 2),
                "temperature": round(temperature, 2),
                "humidity": round(humidity, 2),
                "light_lux": round(light_lux, 2),
                "vpd": round(vpd_kpa, 3),
                "health_score": health_score,
                "health_label": health_label,
                "last_watered_hours": last_watered_hours,
                "watering_event": watering_event,
                "pest_detected": 0,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def create_master_dataset(output_path: Path = OUTPUT_FILE, total_rows: int = 5000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    real_rows = 700
    synthetic_rows = max(0, total_rows - real_rows)

    real_df = build_real_like_rows(real_rows, rng)
    synthetic_df = build_daily_cycle_series(max(30, synthetic_rows // 24), rng)

    if len(synthetic_df) > synthetic_rows:
        synthetic_df = synthetic_df.head(synthetic_rows)

    combined = pd.concat([real_df, synthetic_df], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined.sort_values("timestamp", inplace=True)

    combined = combined.reset_index(drop=True)
    combined.to_csv(output_path, index=False)
    print(f"Saved {len(combined):,} rows to {output_path}")
    print("Columns:", list(combined.columns))
    print("Label distribution:")
    print(combined["health_label"].value_counts(normalize=True).sort_index())
    return combined


if __name__ == "__main__":
    create_master_dataset()
