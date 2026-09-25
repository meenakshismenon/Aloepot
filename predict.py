from __future__ import annotations

import pickle
from pathlib import Path

from vpd import vpd_from_temp_rh

MODEL_PATH = Path("aloe_health_model.pkl")


def predict_health(soil_moisture: float, temperature: float, humidity: float, light_lux: float) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Train the model first with train_model.py")

    with MODEL_PATH.open("rb") as fh:
        model = pickle.load(fh)

    vpd = vpd_from_temp_rh(temperature, humidity)
    features = {
        "soil_moisture": float(soil_moisture),
        "temperature": float(temperature),
        "humidity": float(humidity),
        "light_lux": float(light_lux),
        "vpd": float(vpd),
        "last_watered_hours": 72,
        "watering_event": 0,
    }

    prediction = model.predict([[
        features["soil_moisture"],
        features["temperature"],
        features["humidity"],
        features["light_lux"],
        features["vpd"],
        features["last_watered_hours"],
        features["watering_event"],
    ]])[0]

    return {
        "plant": "Aloe Vera",
        "health_label": prediction,
        "vpd": round(vpd, 3),
        "explanation": "Prediction is based on Aloe Vera environmental stress rules and the trained classifier.",
    }


if __name__ == "__main__":
    print(predict_health(24.0, 24.0, 45.0, 7000))
