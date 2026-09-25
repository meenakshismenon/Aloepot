from __future__ import annotations

from pathlib import Path

import pandas as pd

# Only keep data generated specifically for Aloe Vera.
# The 'indoor_plant_iot_reference_pool' source contains generic multi-species
# IoT readings (soil up to 70%, temp as low as 8C) that do NOT represent
# Aloe barbadensis conditions and would distort the classifier.
ALOE_SOURCES = {"rule_based_weekly_watering_cycle", "synthetic_augmented"}

# Aloe Vera acceptable environmental limits (from horticultural references).
# Used to drop clearly out-of-range rows that would corrupt training.
ALOE_LIMITS = {
    "soil_moisture": (0, 60),       # Aloe: never above 60% (rot risk)
    "temperature":   (10, 40),      # Aloe: dies below 5C / above 40C
    "humidity":      (0, 100),
    "light_lux":     (0, 15000),    # Aloe: full sun rarely exceeds 15klux indoors
    "vpd":           (0, 5.5),
}


REQUIRED_COLUMNS = {
    "timestamp",
    "plant_id",
    "plant_species",
    "data_source",
    "source_dataset",
    "soil_moisture",
    "temperature",
    "humidity",
    "light_lux",
    "vpd",
    "health_score",
    "health_label",
    "last_watered_hours",
    "watering_event",
    "pest_detected",
}


def validate_dataset(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.drop_duplicates().copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()

    # --- Aloe Vera species filter -------------------------------------------
    # Keep only rows whose source_dataset is in the Aloe-specific set.
    # This drops the generic 'indoor_plant_iot_reference_pool' which contains
    # multi-species IoT readings with out-of-range conditions for Aloe Vera.
    before_filter = len(df)
    if "source_dataset" in df.columns:
        df = df[df["source_dataset"].isin(ALOE_SOURCES)].copy()
        dropped = before_filter - len(df)
        if dropped > 0:
            print(f"Aloe-only filter: dropped {dropped} non-Aloe rows ({before_filter} -> {len(df)})")
    # -------------------------------------------------------------------------

    df["soil_moisture"] = pd.to_numeric(df["soil_moisture"], errors="coerce")
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
    df["light_lux"] = pd.to_numeric(df["light_lux"], errors="coerce")
    df["vpd"] = pd.to_numeric(df["vpd"], errors="coerce")

    # Apply Aloe Vera-specific range limits
    s_lo, s_hi = ALOE_LIMITS["soil_moisture"]
    t_lo, t_hi = ALOE_LIMITS["temperature"]
    h_lo, h_hi = ALOE_LIMITS["humidity"]
    l_lo, l_hi = ALOE_LIMITS["light_lux"]
    v_lo, v_hi = ALOE_LIMITS["vpd"]

    invalid_mask = (
        (df["soil_moisture"] < s_lo) | (df["soil_moisture"] > s_hi) |
        (df["temperature"]   < t_lo) | (df["temperature"]   > t_hi) |
        (df["humidity"]      < h_lo) | (df["humidity"]      > h_hi) |
        (df["light_lux"]     < l_lo) | (df["light_lux"]     > l_hi) |
        (df["vpd"]           < v_lo) | (df["vpd"]           > v_hi)
    )
    df = df.loc[~invalid_mask].copy()

    print(f"Rows after validation: {len(df)}")
    print("Missing values per column:")
    print(df.isna().sum())
    print("\nLabel distribution (Aloe Vera only):")
    print(df["health_label"].value_counts())
    return df


def time_aware_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


if __name__ == "__main__":
    validate_dataset("aloe_vera_indoor_master.csv")
