# Aloe Vera Indoor Smart Pot Dataset

## Dataset quality statement

This project uses a hybrid dataset strategy designed to be scientifically honest and defensible for a final-year Smart Pot project.

The final dataset is not a single "pure real" Aloe Vera sensor dataset. Instead, it combines:

1. Real indoor-plant and IoT-style environmental monitoring data patterns
2. Aloe Vera horticultural reference ranges
3. Synthetic time-series augmentation for realistic daily and weekly watering cycles
4. Explicit provenance fields so the source of every row is traceable

The dataset separates `data_source` and `source_dataset` to avoid hiding the origin of the data.

## Data source categories

- `real_aloe`: real Aloe-specific observations if available
- `real_indoor_plant`: real indoor plant/environment monitoring datasets
- `real_iot`: real IoT sensor logs from plant monitoring systems
- `synthetic_augmented`: generated time-series data designed to represent realistic behavior under weekly watering and indoor conditions

## Required dataset columns

- timestamp
- plant_id
- plant_species
- data_source
- source_dataset
- soil_moisture
- temperature
- humidity
- light_lux
- vpd
- health_score
- health_label
- last_watered_hours
- watering_event
- pest_detected

## Aloe Vera reference ranges used for labeling

| Parameter | Preferred | Acceptable | Stress | Severe stress |
|---|---:|---:|---:|---:|
| Soil moisture (%) | 20–45 | 15–55 | <15 or >55 | very dry or chronically wet |
| Temperature (°C) | 18–27 | 15–30 | 10–14 or 31–35 | <10 or >35 |
| Relative humidity (%) | 30–50 | 25–60 | >60 | >70 |
| Light (lux) | 4000–12000 | 2000–15000 | <2000 | very low or excessive |
| VPD (kPa) | 0.8–1.4 | 0.6–1.8 | >1.8 | >2.2 |

These values are reference-based horticultural ranges for indoor Aloe Vera and are used as rule-based support for environmental health classification, not as proof of biological truth.

## VPD formula

The dataset computes VPD mathematically from temperature and relative humidity:

- es = 0.6108 * exp(17.27 * T / (T + 237.3))
- ea = es * RH / 100
- VPD = es - ea

This is the standard vapor pressure deficit equation used in agronomy and plant physiology.

## Labeling methodology

Each row is assigned a health score and label using environmental stress components:

- moisture stress
- temperature stress
- humidity stress
- light stress
- VPD stress

The health score is then reduced by weighted penalties, and the final label is assigned as:

- Happy: score >= 75
- Stressed: score >= 45
- Unhealthy: score < 45

This is an environmental-condition labeling system and should be interpreted as a rule-based condition estimate rather than direct biological diagnosis.

## Important scientific honesty note

The model in this project is learning an environmental rule-based health logic. It does not claim to replace direct biological observation such as leaf color, yellowing, rot, disease, or visible plant stress. It is useful for an indoor Smart Pot decision system, but it must be framed as a condition classifier, not a proven agronomic measurement of the plant's exact physiology.

## Final dataset target

The generated file is:

- aloe_vera_indoor_master.csv

This file is intended for training and evaluation in a final-year Smart Pot project.
