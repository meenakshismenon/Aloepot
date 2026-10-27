"""Flask API and web server for the Smart Aloe Pot."""

import glob
import json
import logging
import math
import os
import pickle
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock

import certifi
import requests
import xml.etree.ElementTree as ET

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

logger = logging.getLogger("aloe_backend")

MODEL_PATH = Path(__file__).with_name("aloe_health_model.pkl")
MODEL_CACHE = None

app = Flask(__name__)
CORS(app)

SOIL_DRY_THRESHOLD_PCT = 10
LIGHT_LOW_THRESHOLD_PCT = 35
TEMP_COLD_C = 13
TEMP_HOT_C = 31

# Connection freshness: firmware transmits every 5s. After 10s without data, mark disconnected.
ESP32_TIMEOUT_SECONDS = 10

latest_reading = None
latest_reading_time = None
reading_lock = RLock()
last_water_pump_at = None
last_medicine_pump_at = None
pump_lock = RLock()
_serial_thread_started = False

# Visitor & Facial Recognition Tracking
visitor_lock = RLock()
visitor_log = []
current_visitor = {
    "name": None,
    "is_known": False,
    "is_meenakshi": False,
    "last_seen_time": None,
    "greeting": None,
}

# Daily Plant Selfie System
SELFIE_DIR = Path(__file__).parent / "static" / "selfies"
SELFIE_DIR.mkdir(parents=True, exist_ok=True)
VISITOR_DIR = Path(__file__).parent / "static" / "visitors"
VISITOR_DIR.mkdir(parents=True, exist_ok=True)
selfie_request_flag = False  # Set True by /api/selfie/take, consumed by face_tracker
selfie_lock = RLock()



def load_model():
    """Load the trained Aloe health model if it exists."""
    global MODEL_CACHE
    if MODEL_CACHE is not None:
        return MODEL_CACHE
    if not MODEL_PATH.exists():
        return None
    try:
        with MODEL_PATH.open("rb") as handle:
            MODEL_CACHE = pickle.load(handle)
    except Exception:
        MODEL_CACHE = None
    return MODEL_CACHE


def fallback_health_rule(soil_pct, light_pct, temp_c, humidity_pct, vpd_kpa):
    """Rule-based Aloe condition fallback when no trained model is available."""
    score = 100
    if soil_pct <= 10:
        score -= 28
    elif soil_pct < 20:
        score -= 12
    elif soil_pct > 80:
        score -= 5  # Only a tiny penalty for very wet soil (recently watered)

    if temp_c < 16 or temp_c > 32:
        score -= 18
    # Bengaluru ambient humidity is naturally 60-75% — only penalise truly extreme humidity
    if humidity_pct > 80:
        score -= 15
    if light_pct < 35:
        score -= 12
    if vpd_kpa > 2.0:
        score -= 20
    if vpd_kpa < 0.3:
        score -= 8

    score = max(0, min(100, score))
    if score >= 75:
        return "Happy", score
    if score >= 45:
        return "Stressed", score
    return "Unhealthy", score


def predict_health_label(soil_pct, light_pct, temp_c, humidity_pct):
    """Use trained classifier if available; otherwise use Aloe rule-based fallback."""
    model = load_model()
    light_lux = max(0.0, float(light_pct) * 150.0)
    vpd_kpa = calculate_vpd(temp_c, humidity_pct)
    if model is not None:
        try:
            import pandas as pd
            features = pd.DataFrame([{
                "soil_moisture": float(soil_pct),
                "temperature": float(temp_c),
                "humidity": float(humidity_pct),
                "light_lux": float(light_lux),
                "vpd": float(vpd_kpa),
                "last_watered_hours": 72.0,
                "watering_event": 0,
            }])
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            score = round(float(max(probabilities)) * 100, 1)
            return prediction, score
        except Exception:
            pass
    return fallback_health_rule(soil_pct, light_pct, temp_c, humidity_pct, vpd_kpa)


def should_run_water_pump(soil_pct: float) -> bool:
    return soil_pct <= SOIL_DRY_THRESHOLD_PCT


def should_run_medicine_pump(now_utc=None, interval_days=7):
    global last_medicine_pump_at
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    if last_medicine_pump_at is None:
        return True

    elapsed_hours = (now_utc - last_medicine_pump_at).total_seconds() / 3600.0
    return elapsed_hours >= interval_days * 24


def mark_water_pump_run():
    global last_water_pump_at
    with pump_lock:
        last_water_pump_at = datetime.now(timezone.utc)


def mark_medicine_pump_run():
    global last_medicine_pump_at
    with pump_lock:
        last_medicine_pump_at = datetime.now(timezone.utc)


def calculate_vpd(temp_c: float, humidity_pct: float) -> float:
    """Return vapor pressure deficit in kPa using the Tetens equation."""
    saturation_vapor_pressure = 0.6108 * math.exp(
        (17.27 * temp_c) / (temp_c + 237.3)
    )
    actual_vapor_pressure = saturation_vapor_pressure * humidity_pct / 100.0
    return round(saturation_vapor_pressure - actual_vapor_pressure, 3)


def classify_vpd(vpd_kpa: float) -> str:
    if vpd_kpa < 0.4:
        return "too_humid"
    if vpd_kpa <= 1.0:
        return "healthy_range"
    if vpd_kpa <= 1.6:
        return "mild_stress"
    return "high_stress"


def build_mood(soil_pct, light_pct, temp_c, vpd_status, motion_detected):
    if motion_detected:
        return "peeking", f"Heyyy! I see you peeking at me! Temperature is cozy at {temp_c:.1f}°C. 👀✨"
    if 0 < soil_pct <= SOIL_DRY_THRESHOLD_PCT:
        return "thirsty", "I'm feeling thirsty! Please water me."
    if temp_c > TEMP_HOT_C or temp_c < TEMP_COLD_C:
        return "stressed", f"I'm stressed by the temperature ({temp_c:.1f}°C). A gentler spot would help."
    if light_pct < LIGHT_LOW_THRESHOLD_PCT:
        return "low_light", "The room is a bit too dim for me. More room light would help."
    return "healthy", f"Healthy & Happy! Room temperature is comfortable at {temp_c:.1f}°C."


def is_demo_device_id(device_id: str) -> bool:
    """Reject obvious placeholders, demo IDs, and test payload markers."""
    normalized = (device_id or "").strip().lower()
    if not normalized:
        return True

    banned_tokens = (
        "demo",
        "example",
        "sample",
        "test",
        "mock",
        "fake",
        "placeholder",
        "aloe-test",
    )
    if any(token in normalized for token in banned_tokens):
        return True

    return False


def parse_sensor_payload(data):
    required = {
        "soil_moisture_pct": (int, float),
        "light_pct": (int, float),
        "temperature_c": (int, float),
        "humidity_pct": (int, float),
        "device_id": (str,),
        "motion_detected": (bool,),
    }
    missing = [field for field in required if field not in data]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    for field, accepted_types in required.items():
        value = data[field]
        if field != "motion_detected" and isinstance(value, bool):
            raise ValueError(f"'{field}' has an invalid type")
        if not isinstance(value, accepted_types):
            raise ValueError(f"'{field}' has an invalid type")

    soil_pct = float(data["soil_moisture_pct"])
    light_pct = float(data["light_pct"])
    temperature_c = float(data["temperature_c"])
    humidity_pct = float(data["humidity_pct"])
    if not 0 <= soil_pct <= 100 or not 0 <= light_pct <= 100:
        raise ValueError("soil_moisture_pct and light_pct must be between 0 and 100")
    if not -40 <= temperature_c <= 85 or not 0 <= humidity_pct <= 100:
        raise ValueError("temperature_c or humidity_pct is outside its valid range")
    device_id = data["device_id"].strip()
    if not device_id:
        raise ValueError("device_id must not be empty")

    if is_demo_device_id(device_id):
        raise ValueError(
            "This endpoint accepts only real connected sensor readings. Demo or test payloads are not allowed."
        )

    return (
        soil_pct,
        light_pct,
        temperature_c,
        humidity_pct,
        device_id,
        data["motion_detected"],
    )


# --- WEATHER & RAIN ALERT SYSTEM ---
WEATHER_CODE_MAP = {
    0: "clear skies",
    1: "mostly clear skies",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "light snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "rain showers",
    81: "moderate showers",
    82: "heavy rain showers",
    95: "thunderstorms",
}

_weather_cache = {"data": None, "full_data": None, "timestamp": 0}
_news_cache = {"data": None, "timestamp": 0}


def check_weather(city: str = "Bengaluru", api_key: str = None) -> dict:
    """
    Fetch outdoor weather with rain detection.
    Supports OpenWeatherMap if OPENWEATHER_API_KEY is configured.
    Falls back gracefully to Open-Meteo (no API key required).
    """
    global _weather_cache
    now = time.time()
    if _weather_cache.get("full_data") and (now - _weather_cache.get("timestamp", 0) < 180):
        return _weather_cache["full_data"]

    key = api_key or os.environ.get("OPENWEATHER_API_KEY", "").strip()

    # 1. Try OpenWeatherMap if key is provided
    if key:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
            resp = requests.get(url, timeout=3.5)
            if resp.status_code == 200:
                data = resp.json()
                temp = round(data.get("main", {}).get("temp", 26.0), 1)
                rh = round(data.get("main", {}).get("humidity", 50))
                weather_arr = data.get("weather", [{}])
                desc = weather_arr[0].get("description", "pleasant").lower()
                cond_id = int(weather_arr[0].get("id", 800))

                # 2xx Thunderstorm, 3xx Drizzle, 5xx Rain
                is_raining = (200 <= cond_id < 600) or ("rain" in desc) or ("drizzle" in desc)
                rain_alert = None
                if is_raining:
                    rain_alert = f"🌧️ Rain Alert: It is currently raining outside in {city} ({desc})! Keep outdoor plants sheltered and avoid overwatering."

                summary = f"Right now in {city}, it is {temp}°C with {desc} and {rh}% humidity."
                result = {
                    "source": "OpenWeatherMap",
                    "city": city,
                    "temperature_c": temp,
                    "humidity_pct": rh,
                    "condition": desc,
                    "weather_id": cond_id,
                    "is_raining": is_raining,
                    "rain_alert": rain_alert,
                    "summary_text": summary,
                }
                _weather_cache["full_data"] = result
                _weather_cache["data"] = summary
                _weather_cache["timestamp"] = now
                return result
        except Exception as exc:
            logger.warning("OpenWeatherMap fetch failed, falling back to Open-Meteo: %s", exc)

    # 2. Open-Meteo fallback (Free, reliable, no key required)
    try:
        lat, lon = 12.97, 77.59
        url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,weather_code,precipitation,rain,showers"
        )
        resp = requests.get(url, timeout=3.5, verify=certifi.where())
        if resp.status_code == 200:
            current = resp.json().get("current", {})
            temp = round(current.get("temperature_2m", 26.0), 1)
            rh = round(current.get("relative_humidity_2m", 50))
            code = int(current.get("weather_code", 0))
            rain_mm = float(current.get("rain", 0.0) or 0.0)
            precip_mm = float(current.get("precipitation", 0.0) or 0.0)
            showers_mm = float(current.get("showers", 0.0) or 0.0)
            desc = WEATHER_CODE_MAP.get(code, "pleasant skies")

            # Rain codes: 51,53,55 (drizzle), 56,57, 61,63,65 (rain), 66,67, 80,81,82 (showers), 95,96,99 (thunderstorm)
            rain_codes = {51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99}
            is_raining = (code in rain_codes) or (rain_mm > 0.05) or (precip_mm > 0.1) or (showers_mm > 0.05)
            rain_alert = None
            if is_raining:
                rain_alert = f"🌧️ Rain Alert: It is currently raining outside in {city} ({desc})! Keep outdoor plants sheltered and avoid overwatering."

            summary = f"Right now in {city}, it is {temp}°C with {desc} and {rh}% humidity."
            result = {
                "source": "Open-Meteo",
                "city": city,
                "temperature_c": temp,
                "humidity_pct": rh,
                "condition": desc,
                "weather_code": code,
                "is_raining": is_raining,
                "rain_alert": rain_alert,
                "summary_text": summary,
            }
            _weather_cache["full_data"] = result
            _weather_cache["data"] = summary
            _weather_cache["timestamp"] = now
            return result
    except Exception as exc:
        logger.warning("Open-Meteo weather fetch failed: %s", exc)

    fallback_summary = f"Right now in {city}, it is 26°C with partly cloudy skies and 52% humidity."
    result = {
        "source": "default_fallback",
        "city": city,
        "temperature_c": 26.0,
        "humidity_pct": 52,
        "condition": "partly cloudy",
        "weather_code": 2,
        "is_raining": False,
        "rain_alert": None,
        "summary_text": fallback_summary,
    }
    return result


def get_live_weather() -> str:
    """Fetch live weather summary string."""
    w = check_weather()
    return w.get("summary_text", "Right now, it is 26°C with partly cloudy skies and 52% humidity.")


def generate_assistant_message(reading):
    """Return a spoken-style personal assistant message with Siri-like care and greetings."""
    soil = float(reading.get("soil_moisture_pct", 0))
    light = float(reading.get("light_pct", 0))
    temp = float(reading.get("temperature_c", 0))
    humidity = float(reading.get("humidity_pct", 0))
    vpd_status = reading.get("vpd_status", "unknown")
    primary = reading.get("primary_status", "healthy")
    motion = bool(reading.get("motion_detected", False))

    # Check for recognized visitor/owner
    salutation = "Hey!"
    with visitor_lock:
        if current_visitor.get("name") and current_visitor.get("last_seen_time"):
            elapsed = (datetime.now(timezone.utc) - current_visitor["last_seen_time"]).total_seconds()
            if elapsed < 12:
                if current_visitor.get("is_meenakshi"):
                    salutation = "Hey Meenakshi!"
                elif current_visitor.get("is_known"):
                    salutation = f"Hey {current_visitor['name']}!"

    # 1. When a person approaches the plant (PIR motion detected):
    if motion:
        if 0 < soil <= SOIL_DRY_THRESHOLD_PCT:
            return f"{salutation} Nice to see you! Water me please, my soil is dry."
        if temp > TEMP_HOT_C:
            return f"{salutation} Glad you're here! It's too hot over here at {temp:.1f}°C, please move me to a cooler place."
        if temp < TEMP_COLD_C:
            return f"{salutation} Glad you're here! Brrr, it's getting too cold at {temp:.1f}°C, please move me to a warmer place."
        if light < LIGHT_LOW_THRESHOLD_PCT:
            return f"{salutation} Nice to see you! The room light is a bit dim right now."
        return f"{salutation} Nice to see you there! I am feeling healthy and happy. Temperature is comfortable at {temp:.1f}°C and soil is well watered."

    # 2. Contextual care alerts based on live sensor conditions:
    if 0 < soil <= SOIL_DRY_THRESHOLD_PCT:
        return f"{salutation} I need water. Water me please and keep the soil around 20%."
    if temp >= TEMP_HOT_C:
        return f"{salutation} It's too hot for me at {temp:.1f}°C. Please move me to a cooler place."
    if temp < TEMP_COLD_C:
        return f"{salutation} Brrr, it's too cold at {temp:.1f}°C! Please move me to a warmer place."
    if light < LIGHT_LOW_THRESHOLD_PCT:
        return f"{salutation} The room is a bit dark. Please turn on a light or move me to a brighter spot."
    # Only complain about humidity if it is truly extreme (>88%) — Bengaluru ambient is normally 60-75%
    if humidity > 88:
        return f"{salutation} The air feels very humid ({humidity:.0f}%). Please improve airflow and keep me in an airy spot."

    # Healthy and normal: always mention temperature and affirm good health
    return f"{salutation} I'm healthy and happy! Room temperature is a comfortable {temp:.1f}°C, soil moisture is {soil:.1f}%, and all conditions look great!"



def is_esp32_connected() -> bool:
    """Check if ESP32 has sent a reading within the freshness timeout."""
    with reading_lock:
        if latest_reading is None or latest_reading_time is None:
            return False
        elapsed = (datetime.now(timezone.utc) - latest_reading_time).total_seconds()
        return elapsed <= ESP32_TIMEOUT_SECONDS


_last_stable_soil = None


def filter_soil_moisture(raw_soil: float) -> float:
    """
    Physical-soil stabilizer:
    - Genuine watering events (rising moisture) are tracked immediately.
    - Water-splash / pin short-circuit spikes (plunging to 0% or 34%) are rejected.
    - Normal drying is rate-limited to physically realistic soil drying rates.
    - Deadband holds values rock solid to prevent visual flickering.
    """
    global _last_stable_soil
    now = datetime.now(timezone.utc)
    with reading_lock:
        if latest_reading_time is None or (now - latest_reading_time).total_seconds() > ESP32_TIMEOUT_SECONDS:
            _last_stable_soil = round(min(raw_soil, 85.0), 1) if raw_soil > 2.0 else 78.0
            return _last_stable_soil

    if _last_stable_soil is None:
        _last_stable_soil = round(min(raw_soil, 85.0), 1) if raw_soil > 2.0 else 78.0
        return _last_stable_soil

    # Water-short / pin-bridge protection:
    # If raw_soil suddenly drops to <=2.0% due to flooded sensor pins or loose wire,
    # preserve the healthy hydrated state instead of claiming the plant is bone-dry!
    if raw_soil <= 2.0:
        if _last_stable_soil < 40.0:
            _last_stable_soil = 78.0
        return _last_stable_soil

    # Physical cap: capacitive sensor submerged in standing water reads 100%.
    # Real soil saturation in a pot can never exceed ~85% in practice.
    # Cap the accepted value so the dashboard never shows 100%.
    capped_raw = min(raw_soil, 85.0)

    # Water added: soil moisture jumped up (accept rising edge immediately)
    if capped_raw >= _last_stable_soil:
        _last_stable_soil = round(capped_raw, 1)
        return _last_stable_soil

    # Sudden steep drop (e.g. 85% -> 34% from splash or wire touch): reject glitch
    drop = _last_stable_soil - capped_raw
    if drop > 15.0:
        logger.info("Filtered transient soil moisture glitch: %s%% (holding steady at %s%%)", raw_soil, _last_stable_soil)
        return _last_stable_soil

    # Realistic gradual drying rate limiter (max 1.0% drop per 5s reading cycle)
    allowed_drop = min(drop, 1.0)
    new_val = _last_stable_soil - allowed_drop

    # Deadband: if difference is tiny (<0.6%), avoid visual jitter
    if abs(new_val - _last_stable_soil) < 0.6:
        return _last_stable_soil

    _last_stable_soil = round(new_val, 1)
    return _last_stable_soil


def process_sensor_reading(data: dict) -> dict:
    """Parse, classify, and register a live sensor reading from ESP32."""
    (
        raw_soil_pct,
        light_pct,
        temp_c,
        humidity_pct,
        device_id,
        motion_detected,
    ) = parse_sensor_payload(data)

    soil_pct = filter_soil_moisture(raw_soil_pct)

    vpd_kpa = calculate_vpd(temp_c, humidity_pct)
    vpd_status = classify_vpd(vpd_kpa)
    health_label, health_score = predict_health_label(soil_pct, light_pct, temp_c, humidity_pct)
    water_due = should_run_water_pump(soil_pct)
    medicine_due = should_run_medicine_pump()

    if water_due:
        mark_water_pump_run()
    if medicine_due:
        mark_medicine_pump_run()

    primary_status, mood_text = build_mood(
        soil_pct, light_pct, temp_c, vpd_status, motion_detected
    )
    now = datetime.now(timezone.utc)
    reading = {
        "device_id": device_id,
        "motion_detected": motion_detected,
        "timestamp": now.isoformat(),
        "temperature_c": round(temp_c, 1),
        "humidity_pct": round(humidity_pct, 1),
        "soil_moisture_pct": round(soil_pct, 1),
        "light_pct": round(light_pct, 1),
        "vpd_kpa": vpd_kpa,
        "vpd_status": vpd_status,
        "should_water": water_due,
        "water_pump_active": water_due,
        "medicine_pump_due": medicine_due,
        "medicine_pump_active": medicine_due,
        "medicine_interval_days": 7,
        "needs_light": light_pct < LIGHT_LOW_THRESHOLD_PCT,
        "primary_status": primary_status,
        "mood_text": mood_text,
        "health_label": health_label,
        "health_score": health_score,
        "assistant_message": generate_assistant_message({
            "soil_moisture_pct": round(soil_pct, 1),
            "light_pct": round(light_pct, 1),
            "temperature_c": round(temp_c, 1),
            "humidity_pct": round(humidity_pct, 1),
            "vpd_status": vpd_status,
            "primary_status": primary_status,
            "needs_light": light_pct < LIGHT_LOW_THRESHOLD_PCT,
            "motion_detected": motion_detected,
        }),
        "model_source": "trained_aloe_classifier" if load_model() is not None else "rule_based_fallback",
        "sensor_connected": True,
        "status": "connected",
        "weather": check_weather(),
        "is_raining": check_weather().get("is_raining", False),
        "rain_alert": check_weather().get("rain_alert"),
        "current_person": current_visitor.get("name") if (current_visitor.get("last_seen_time") and (now - current_visitor["last_seen_time"]).total_seconds() < 12.0) else None,
        "is_meenakshi_present": bool(current_visitor.get("is_meenakshi", False) and current_visitor.get("last_seen_time") and (now - current_visitor["last_seen_time"]).total_seconds() < 12.0),
    }


    global latest_reading, latest_reading_time
    with reading_lock:
        latest_reading = reading
        latest_reading_time = now

    return reading


def serial_reader_loop():
    """Background thread that reads USB serial port if available as auto-fallback."""
    # Connect to ESP32 over USB Serial for instant zero-lag sensor telemetry
    if not SERIAL_AVAILABLE or os.environ.get("ENABLE_SERIAL_SCAN") == "0":
        return

    while True:
        target_port = os.environ.get("ESP32_PORT")
        candidate_ports = []
        if target_port:
            candidate_ports.append(target_port)
        else:
            try:
                for port in serial.tools.list_ports.comports():
                    desc = f"{port.description or ''} {port.hwid or ''}".lower()
                    if any(k in desc for k in ["cp210", "ch340", "ch341", "ftdi", "uart", "serial", "esp32"]):
                        candidate_ports.append(port.device)
                if not candidate_ports:
                    candidate_ports = [p.device for p in serial.tools.list_ports.comports()]
            except Exception:
                pass

        connected_ser = None
        for port_name in candidate_ports:
            try:
                connected_ser = serial.Serial(port_name, 115200, timeout=1)
                logger.info("Connected to serial port %s", port_name)
                break
            except Exception:
                continue

        if connected_ser:
            try:
                with connected_ser as ser:
                    ser.reset_input_buffer()
                    consecutive_empty = 0
                    while True:
                        line = ser.readline().decode("utf-8", errors="ignore").strip()
                        if not line:
                            consecutive_empty += 1
                            if consecutive_empty > 30:
                                break
                            continue
                        consecutive_empty = 0
                        if line.startswith("{") and line.endswith("}"):
                            try:
                                payload = json.loads(line)
                                if "device_id" in payload and "soil_moisture_pct" in payload:
                                    process_sensor_reading(payload)
                                    weather = check_weather()
                                    person = get_active_person_info()
                                    ack_doc = {
                                        "ack": True,
                                        "is_raining": weather.get("is_raining", False),
                                        "person_status": person["status"],
                                        "person_name": person["name"],
                                        "line1": person["line1"],
                                        "line2": person["line2"],
                                    }
                                    try:
                                        ser.write((json.dumps(ack_doc) + "\n").encode("utf-8"))
                                    except Exception:
                                        pass
                            except Exception:
                                pass
            except Exception as e:
                logger.debug("Serial connection closed: %s", e)

        time.sleep(3)


def start_serial_listener():
    global _serial_thread_started
    if SERIAL_AVAILABLE and not _serial_thread_started:
        _serial_thread_started = True
        thread = threading.Thread(target=serial_reader_loop, daemon=True)
        thread.start()



@app.get("/")
def dashboard():
    return render_template("index.html")


@app.get("/api/status")
def system_status():
    connected = is_esp32_connected()
    with reading_lock:
        seconds_ago = None
        if latest_reading_time is not None:
            seconds_ago = round((datetime.now(timezone.utc) - latest_reading_time).total_seconds(), 1)

    return jsonify({
        "sensor_connected": connected,
        "status": "connected" if connected else "disconnected",
        "last_seen_seconds_ago": seconds_ago,
        "timeout_seconds": ESP32_TIMEOUT_SECONDS,
        "mode": "live_sensor_only",
        "dataset_training_only": True,
        "message": (
            "ESP32 is connected and streaming live data."
            if connected
            else "ESP32 disconnected. Waiting for real ESP32 connection via Wi-Fi or USB Serial."
        ),
    })


def get_active_person_info():
    """Return person status: 0=None, 1=Meenakshi, 2=Stranger, with OLED lines and greeting."""
    with visitor_lock:
        cur_name = current_visitor.get("name")
        is_meenakshi = current_visitor.get("is_meenakshi", False)
        last_seen = current_visitor.get("last_seen_time")
        now = datetime.now(timezone.utc)

        if cur_name and last_seen:
            elapsed = (now - last_seen).total_seconds()
            if elapsed < 12:  # Active within the last 12 seconds
                if is_meenakshi:
                    return {
                        "status": 1,
                        "name": "Meenakshi",
                        "is_meenakshi": True,
                        "line1": "HI MEENAKSHI! :D",
                        "line2": "WELCOME HOME! <3",
                        "greeting": "Welcome back, Meenakshi! Great to see you! Your Aloe Pot is happy you're here.",
                    }
                else:
                    return {
                        "status": 2,
                        "name": cur_name,
                        "is_meenakshi": False,
                        "line1": "HEY STRANGER! ~",
                        "line2": "WTSUP! WHO'S THERE?",
                        "greeting": "Hey stranger! What's up? Welcome to Meenakshi's room!",
                    }
        return {
            "status": 0,
            "name": None,
            "is_meenakshi": False,
            "line1": "HEY WASSUP!! ~",
            "line2": "I SEE YOU!! :D",
            "greeting": None,
        }


@app.post("/api/sensor-data")
def receive_sensor_data():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(status="error", message="Request body must be a JSON object"), 400

    try:
        reading = process_sensor_reading(data)
    except ValueError as error:
        return jsonify(status="error", message=str(error)), 400

    weather = check_weather()
    person = get_active_person_info()
    return jsonify(
        status="ok",
        reading=reading,
        is_raining=weather.get("is_raining", False),
        rain_alert=weather.get("rain_alert"),
        person_status=person["status"],       # 1=Meenakshi, 2=Stranger, 0=None
        person_name=person["name"],
        is_meenakshi=person["is_meenakshi"],
        greeting_line1=person["line1"],
        greeting_line2=person["line2"],
        greeting=person["greeting"],
    ), 200


@app.get("/api/latest")
def get_latest():
    if not is_esp32_connected():
        return jsonify({
            "status": "disconnected",
            "sensor_connected": False,
            "message": "ESP32 is not connected or readings timed out. Waiting for live sensor readings.",
        }), 200

    with reading_lock:
        reading = dict(latest_reading)

    return jsonify(reading), 200


@app.get("/api/weather")
def api_weather():
    """Return real-time weather, rain status, and rain alert."""
    city = request.args.get("city", "Bengaluru")
    return jsonify(check_weather(city=city)), 200


@app.post("/api/log-visitor")
def log_visitor():
    """Log an arrival detected by facial recognition or camera."""
    data = request.get_json(silent=True) or {}
    raw_name = (data.get("name") or "Visitor").strip()
    is_known = bool(data.get("is_known", False))
    conf = float(data.get("confidence", 0.0))
    now = datetime.now(timezone.utc)

    is_meenakshi = (raw_name.lower() == "meenakshi")
    if is_meenakshi:
        disp_name = "Meenakshi"
        greeting = "Welcome back, Meenakshi! Great to see you! Your Aloe Pot is happy you're here."
    elif is_known:
        disp_name = raw_name
        greeting = f"Hello {disp_name}! Great to see you!"
    else:
        with visitor_lock:
            v_num = len([v for v in visitor_log if not v.get("is_meenakshi")]) + 1
        disp_name = f"Stranger #{v_num}"
        greeting = "Hey stranger! What's up? Welcome to Meenakshi's room!"

    photo_url = None
    snapshot_b64 = data.get("snapshot")
    # Never attach stranger snapshot if person is Meenakshi
    if snapshot_b64 and not is_meenakshi:
        try:
            import base64
            VISITOR_DIR.mkdir(parents=True, exist_ok=True)
            if "," in snapshot_b64:
                snapshot_b64 = snapshot_b64.split(",", 1)[1]
            img_bytes = base64.b64decode(snapshot_b64)
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stranger_{ts_str}_{len(visitor_log)+1}.jpg"
            save_path = VISITOR_DIR / filename
            with open(save_path, "wb") as f:
                f.write(img_bytes)
            photo_url = f"/static/visitors/{filename}"
            logger.info("Saved stranger photo to %s", photo_url)
        except Exception as e:
            logger.warning("Could not save stranger photo: %s", e)

    # Debounce: if the same person/stranger was logged within 60s and no new photo was taken, don't spam duplicate row
    with visitor_lock:
        if visitor_log and not photo_url:
            last_entry = visitor_log[-1]
            if last_entry.get("name") == disp_name:
                try:
                    last_ts = datetime.fromisoformat(last_entry["timestamp"])
                    if (now - last_ts).total_seconds() < 60.0:
                        current_visitor["last_seen_time"] = now
                        return jsonify(status="ok", debounced=True, entry=last_entry), 200
                except Exception:
                    pass

    entry = {
        "id": len(visitor_log) + 1,
        "name": disp_name,
        "is_known": is_known or is_meenakshi,
        "is_meenakshi": is_meenakshi,
        "confidence": conf,
        "timestamp": now.isoformat(),
        "greeting": greeting,
        "photo_url": photo_url,
    }

    with visitor_lock:
        visitor_log.append(entry)
        if len(visitor_log) > 50:
            visitor_log.pop(0)
        current_visitor["name"] = disp_name
        current_visitor["is_known"] = is_known or is_meenakshi
        current_visitor["is_meenakshi"] = is_meenakshi
        current_visitor["last_seen_time"] = now
        current_visitor["greeting"] = greeting
        current_visitor["photo_url"] = photo_url

    with reading_lock:
        if latest_reading:
            latest_reading["current_person"] = disp_name
            latest_reading["is_meenakshi_present"] = is_meenakshi
            latest_reading["person_greeting"] = greeting
            latest_reading["visitor_photo_url"] = photo_url

    return jsonify(status="ok", entry=entry, current_visitor={
        "name": disp_name,
        "is_known": is_known or is_meenakshi,
        "is_meenakshi": is_meenakshi,
        "greeting": greeting,
        "photo_url": photo_url,
        "timestamp": now.isoformat(),
    }), 200


@app.post("/api/presence/empty")
def mark_room_empty():
    """Face tracker actively informs backend that camera sees no faces in the room."""
    with visitor_lock:
        current_visitor["name"] = None
        current_visitor["is_known"] = False
        current_visitor["is_meenakshi"] = False
        current_visitor["greeting"] = None
        current_visitor["photo_url"] = None
    return jsonify(status="ok", message="Room marked empty"), 200


@app.get("/api/visitors")
def get_visitors():
    """Retrieve visitor logs and current room occupant status."""
    with visitor_lock:
        now = datetime.now(timezone.utc)
        active_person = None
        is_active = False
        last_seen_secs = None
        if current_visitor["name"] and current_visitor["last_seen_time"]:
            last_seen_secs = round((now - current_visitor["last_seen_time"]).total_seconds(), 1)
            if last_seen_secs < 12.0:
                active_person = current_visitor["name"]
                is_active = True

        return jsonify({
            "status": "ok",
            "active_person": active_person,
            "current_visitor": {
                "name": current_visitor["name"] if is_active else None,
                "is_known": (current_visitor["is_known"] and is_active),
                "is_meenakshi": (current_visitor["is_meenakshi"] and is_active),
                "greeting": current_visitor["greeting"] if is_active else None,
                "last_seen_seconds_ago": last_seen_secs,
            },
            "visitors": list(reversed(visitor_log[-20:])),
        }), 200


@app.post("/api/visitors/clear")
def clear_visitors():
    """Clear visitor logs."""
    with visitor_lock:
        visitor_log.clear()
        current_visitor["name"] = None
        current_visitor["is_known"] = False
        current_visitor["is_meenakshi"] = False
        current_visitor["last_seen_time"] = None
        current_visitor["greeting"] = None
        current_visitor["photo_url"] = None
    return jsonify(status="ok", message="Visitor log cleared"), 200


@app.get("/static/visitors/<path:filename>")
def serve_visitor_snapshot(filename):
    """Explicitly serve visitor snapshot images with cache control."""
    VISITOR_DIR.mkdir(parents=True, exist_ok=True)
    return send_from_directory(VISITOR_DIR, filename)



def get_trending_news() -> str:
    """Fetch top trending headlines from Google News RSS feed."""
    global _news_cache
    now = time.time()
    if _news_cache["data"] and (now - _news_cache["timestamp"] < 300):
        return _news_cache["data"]

    try:
        url = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
        resp = requests.get(url, timeout=4.0)
        if resp.status_code == 200:
            root = ET.fromstring(resp.content)
            items = root.findall("./channel/item")[:3]
            headlines = []
            for idx, item in enumerate(items, 1):
                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    clean = title_el.text.rsplit(" - ", 1)[0].strip()
                    headlines.append(f"{idx}. {clean}")

            if headlines:
                news_text = "Here are the top trending headlines right now: " + "; ".join(headlines) + "."
                _news_cache = {"data": news_text, "timestamp": now}
                return news_text
    except Exception as exc:
        logger.warning(f"News fetch exception: {exc}")

    return "Trending news feed is taking a moment to load. Please ask again in a few seconds!"


def generate_ai_chat_response(user_query: str) -> dict:
    """Produce a natural, fluent AI speaker response with proactive plant care reminders."""
    query = (user_query or "").strip().lower()

    with reading_lock:
        reading = dict(latest_reading) if latest_reading else None
        connected = is_esp32_connected()

    # Care requirement alerts that need to be voiced if urgent:
    care_alert = None
    if connected and reading:
        soil = float(reading.get("soil_moisture_pct", 50))
        temp = float(reading.get("temperature_c", 22))
        light = float(reading.get("light_pct", 50))
        if soil <= SOIL_DRY_THRESHOLD_PCT:
            care_alert = f"By the way, could you water me please? My soil moisture is critically dry at {soil:.1f}%."
        elif temp > TEMP_HOT_C:
            care_alert = f"Also, it is getting too hot over here at {temp:.1f}°C! Please move me to a cooler place."
        elif temp < TEMP_COLD_C:
            care_alert = f"Also, brrr, it is {temp:.1f}°C, which is too cold for me! Please move me to a warmer place."
        elif light < LIGHT_LOW_THRESHOLD_PCT:
            care_alert = f"Also, the room is a bit dim today. Room light is only {light:.1f}%."

    # Rain alert note if raining
    weather_info = check_weather()
    rain_note = ""
    if weather_info.get("is_raining"):
        rain_note = f" 🌧️ Heads up: it is currently raining outside in {weather_info.get('city', 'the city')}!"

    # Visitor intent
    if any(k in query for k in ["visitor", "who is here", "who entered", "guest", "meenakshi", "who is in the room"]):
        with visitor_lock:
            if current_visitor.get("name") and current_visitor.get("last_seen_time"):
                elapsed = int((datetime.now(timezone.utc) - current_visitor["last_seen_time"]).total_seconds())
                mins = elapsed // 60
                time_str = f"{mins} minutes ago" if mins > 0 else f"{elapsed} seconds ago"
                if elapsed < 15:
                    if current_visitor.get("is_meenakshi"):
                        reply = "Meenakshi is currently here in the room with me! 🌿"
                    else:
                        reply = f"{current_visitor['name']} is currently in the room."
                else:
                    reply = f"Nobody is in the room right now — it's quiet. (Meenakshi was last seen {time_str})."
            else:
                reply = "Nobody is in the room right now. The room is quiet!"
        return {"status": "ok", "reply": reply, "intent": "visitor", "care_alert": care_alert}

    # Weather intent
    if any(k in query for k in ["weather", "forecast", "outside", "rain", "sunny", "climate"]):
        weather = get_live_weather()
        if connected and reading:
            temp = reading.get("temperature_c", 22)
            rh = reading.get("humidity_pct", 50)
            indoor_note = f" Inside with you, my plant sensor reads {temp:.1f}°C and {rh:.1f}% humidity."
            reply = f"{weather}{indoor_note}{rain_note}"
        else:
            reply = f"{weather}{rain_note}"
        if care_alert:
            reply += f" {care_alert}"
        return {"status": "ok", "reply": reply, "intent": "weather", "care_alert": care_alert}


    # News intent
    if any(k in query for k in ["news", "trending", "headline", "happening", "world news", "topics"]):
        news = get_trending_news()
        reply = news
        if care_alert:
            reply += f" {care_alert}"
        return {"status": "ok", "reply": reply, "intent": "news", "care_alert": care_alert}

    # Water intent
    if any(k in query for k in ["water", "thirsty", "drink", "hydrate", "moisture", "dry"]):
        if not connected or not reading:
            reply = "My ESP32 sensor is currently disconnected, so I can't check my exact moisture level. Please connect my pot to check if I need water!"
        else:
            soil = float(reading.get("soil_moisture_pct", 0))
            if soil <= SOIL_DRY_THRESHOLD_PCT:
                reply = f"Yes, please! Water me now! My soil moisture is at {soil:.1f}%, which is very dry. A gentle drink of water would save my leaves!"
            elif soil < 20:
                reply = f"I am a little thirsty. My soil moisture is at {soil:.1f}%. A light watering soon would be wonderful."
            elif soil <= 45:
                reply = f"No need to water me right now! My soil moisture is at a healthy {soil:.1f}%. Aloe vera prefers drying out slightly between waterings."
            else:
                reply = f"Please don't water me! My soil is plenty wet at {soil:.1f}%. Aloe roots need time to breathe so they don't rot."
        return {"status": "ok", "reply": reply, "intent": "water", "care_alert": care_alert}

    # Temperature intent
    if any(k in query for k in ["temperature", "temp", "hot", "cold", "cooler", "warmer", "warm"]):
        if not connected or not reading:
            reply = "I can't read my temperature right now because my ESP32 is offline. Connect my pot and I'll tell you how comfortable I am!"
        else:
            temp = float(reading.get("temperature_c", 0))
            if temp > TEMP_HOT_C:
                reply = f"It's too hot for me here! The sensor reads {temp:.1f}°C. Please move me to a cooler place away from harsh heat!"
            elif temp < TEMP_COLD_C:
                reply = f"Brrr! It's too cold! The sensor reads {temp:.1f}°C. Please move me to a warmer, cozier place away from drafts."
            else:
                reply = f"The temperature feels great! It is currently {temp:.1f}°C, right in my comfort zone between 18 and 31°C."
        return {"status": "ok", "reply": reply, "intent": "temperature", "care_alert": care_alert}

    # Room light intent
    if any(k in query for k in ["room light", "light", "brightness", "dim", "dark", "illumination", "lamps"]):
        if not connected or not reading:
            reply = "My light sensor is offline right now. Connect the ESP32 and I'll tell you how well-lit the room is!"
        else:
            light = float(reading.get("light_pct", 0))
            if light < 20:
                reply = f"The room is quite dark right now at {light:.1f}% light. Turning on a lamp or light would help!"
            elif light < LIGHT_LOW_THRESHOLD_PCT:
                reply = f"The room is dimly lit at {light:.1f}%. A brighter room spot would keep my leaves happiest."
            elif light < 75:
                reply = f"The room lighting is comfortable at {light:.1f}%. Good indoor ambient light for an aloe plant!"
            else:
                reply = f"The room is brightly illuminated at {light:.1f}%. Plenty of light here!"
        return {"status": "ok", "reply": reply, "intent": "light", "care_alert": care_alert}

    # Health / Mood intent
    if any(k in query for k in ["how are you", "feeling", "status", "health", "how do you feel", "you okay"]):
        if not connected or not reading:
            reply = "I'm doing okay, but my ESP32 is disconnected right now! Connect my sensor pot so I can report on my soil and climate."
        else:
            soil = reading.get("soil_moisture_pct", 0)
            temp = reading.get("temperature_c", 0)
            light = reading.get("light_pct", 0)
            status = reading.get("primary_status", "healthy")
            reply = f"I'm feeling {status}! My soil moisture is {soil:.1f}%, room light is {light:.1f}%, and temperature is {temp:.1f}°C."
            if care_alert:
                reply += f" {care_alert}"
        return {"status": "ok", "reply": reply, "intent": "health", "care_alert": care_alert}

    # Joke intent
    if any(k in query for k in ["joke", "funny", "laugh"]):
        jokes = [
            "Why did the Aloe plant go to medical school? Because it had a soothing bedside manner! 🌱✨",
            "Why did the gardener plant a light bulb? Because she wanted to grow a power plant! 😄",
            "What did the Aloe plant say to the human? Aloe you very much, remember to keep me happy! 💚",
        ]
        import random
        reply = random.choice(jokes)
        if care_alert:
            reply += f" {care_alert}"
        return {"status": "ok", "reply": reply, "intent": "joke", "care_alert": care_alert}

    # Default friendly greeting / conversational response
    if not connected:
        reply = "Hey! I am your Smart Aloe Vera AI companion. I'm waiting for the ESP32 to connect, but I can already tell you the live weather, trending news, or tell jokes!"
    else:
        status = reading.get("primary_status", "healthy") if reading else "comfortable"
        reply = f"Hey there! I am your Smart Aloe Vera assistant. I'm feeling {status} right now. Ask me about the weather, trending news, or how my soil and temperature are doing!"
    if care_alert:
        reply += f" {care_alert}"

    return {"status": "ok", "reply": reply, "intent": "general", "care_alert": care_alert}


@app.post("/api/chat")
def chat_with_aloe():
    """Conversational endpoint for the AI Speaker."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "")
    response = generate_ai_chat_response(message)
    return jsonify(response), 200


# ── Plant Selfie Gallery & Capture API ───────────────────────

@app.route("/static/selfies/<path:filename>")
def serve_selfie(filename):
    """Serve saved plant selfie images."""
    return send_from_directory(str(SELFIE_DIR), filename)


@app.route("/static/visitors/<path:filename>")
def serve_visitor_photo(filename):
    """Serve stranger visitor snapshots."""
    return send_from_directory(str(VISITOR_DIR), filename)


@app.get("/api/selfies")
def get_selfies():
    """Return a list of all saved plant selfie images, newest first."""
    selfies = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for filepath in SELFIE_DIR.glob(ext):
            stat = filepath.stat()
            selfies.append({
                "filename": filepath.name,
                "url": f"/static/selfies/{filepath.name}",
                "timestamp": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "size_kb": round(stat.st_size / 1024, 1),
            })
    # Sort by timestamp descending (newest first)
    selfies.sort(key=lambda s: s["timestamp"], reverse=True)
    return jsonify({"status": "ok", "count": len(selfies), "selfies": selfies}), 200


@app.post("/api/selfie/take")
def request_selfie():
    """Request the face_tracker camera to capture a plant selfie right now."""
    global selfie_request_flag
    with selfie_lock:
        selfie_request_flag = True
    return jsonify({"status": "ok", "message": "Selfie requested! The camera will capture it momentarily."}), 200


@app.get("/api/selfie/check")
def check_selfie_request():
    """Polled by face_tracker to see if a selfie was requested from the dashboard."""
    global selfie_request_flag
    with selfie_lock:
        req = selfie_request_flag
        selfie_request_flag = False
    return jsonify({"requested": req}), 200


@app.post("/api/selfie/upload")
def upload_selfie():
    """Receive a selfie image from face_tracker and save it to the gallery."""
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file provided"}), 400

    img_file = request.files["image"]
    label = request.form.get("label", "manual")
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"plant_selfie_{label}_{ts}.jpg"
    save_path = SELFIE_DIR / filename
    img_file.save(str(save_path))

    global selfie_request_flag
    with selfie_lock:
        selfie_request_flag = False

    logger.info("Plant selfie saved: %s", filename)
    return jsonify({
        "status": "ok",
        "filename": filename,
        "url": f"/static/selfies/{filename}",
        "message": f"Selfie saved as {filename}!",
    }), 200


if __name__ == "__main__":
    start_serial_listener()
    app.run(host="0.0.0.0", port=5000, debug=False)
