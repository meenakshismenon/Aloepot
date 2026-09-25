import unittest
from datetime import datetime, timedelta, timezone

import app
from app import (
    ESP32_TIMEOUT_SECONDS,
    generate_assistant_message,
    is_esp32_connected,
    parse_sensor_payload,
    process_sensor_reading,
)


class AssistantLogicTests(unittest.TestCase):
    def test_dry_soil_suggestion(self):
        reading = {
            "soil_moisture_pct": 7,
            "light_pct": 40,
            "temperature_c": 24,
            "humidity_pct": 45,
            "vpd_status": "mild_stress",
            "needs_light": False,
            "primary_status": "thirsty",
            "motion_detected": False,
        }
        message = generate_assistant_message(reading)
        self.assertIn("need water", message.lower())
        self.assertIn("water", message.lower())

    def test_light_and_humidity_suggestion(self):
        reading = {
            "soil_moisture_pct": 62,
            "light_pct": 20,
            "temperature_c": 24,
            "humidity_pct": 72,
            "vpd_status": "too_humid",
            "needs_light": True,
            "primary_status": "stressed",
            "motion_detected": False,
        }
        message = generate_assistant_message(reading)
        self.assertIn("sunlight", message.lower())
        self.assertIn("humid", message.lower())

    def test_hot_temperature_move_to_cooler_place(self):
        reading = {
            "soil_moisture_pct": 35,
            "light_pct": 60,
            "temperature_c": 31,
            "humidity_pct": 45,
            "vpd_status": "mild_stress",
            "needs_light": False,
            "primary_status": "stressed",
            "motion_detected": False,
        }
        message = generate_assistant_message(reading)
        self.assertIn("cooler place", message.lower())

    def test_motion_greeting_say_hey(self):
        reading = {
            "soil_moisture_pct": 35,
            "light_pct": 60,
            "temperature_c": 23,
            "humidity_pct": 45,
            "vpd_status": "healthy_range",
            "needs_light": False,
            "primary_status": "healthy",
            "motion_detected": True,
        }
        message = generate_assistant_message(reading)
        self.assertTrue(message.startswith("Hey!"))
        self.assertIn("see you", message.lower())


class ESP32ConnectionTests(unittest.TestCase):
    def setUp(self):
        with app.reading_lock:
            app.latest_reading = None
            app.latest_reading_time = None
        self.client = app.app.test_client()

    def tearDown(self):
        with app.reading_lock:
            app.latest_reading = None
            app.latest_reading_time = None

    def test_initially_disconnected(self):
        self.assertFalse(is_esp32_connected())
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["sensor_connected"])
        self.assertEqual(data["status"], "disconnected")

        latest_resp = self.client.get("/api/latest")
        self.assertEqual(latest_resp.status_code, 200)
        latest_data = latest_resp.get_json()
        self.assertFalse(latest_data["sensor_connected"])
        self.assertEqual(latest_data["status"], "disconnected")

    def test_real_reading_connects_esp32(self):
        payload = {
            "device_id": "aloe-pot-01",
            "soil_moisture_pct": 25.0,
            "light_pct": 50.0,
            "temperature_c": 22.5,
            "humidity_pct": 45.0,
            "motion_detected": False,
        }
        resp = self.client.post("/api/sensor-data", json=payload)
        self.assertEqual(resp.status_code, 200)

        self.assertTrue(is_esp32_connected())

        status_resp = self.client.get("/api/status")
        status_data = status_resp.get_json()
        self.assertTrue(status_data["sensor_connected"])
        self.assertEqual(status_data["status"], "connected")

        latest_resp = self.client.get("/api/latest")
        latest_data = latest_resp.get_json()
        self.assertTrue(latest_data["sensor_connected"])
        self.assertEqual(latest_data["device_id"], "aloe-pot-01")
        self.assertEqual(latest_data["soil_moisture_pct"], 25.0)

    def test_reading_expires_after_timeout(self):
        payload = {
            "device_id": "aloe-pot-01",
            "soil_moisture_pct": 30.0,
            "light_pct": 45.0,
            "temperature_c": 21.0,
            "humidity_pct": 40.0,
            "motion_detected": False,
        }
        process_sensor_reading(payload)
        self.assertTrue(is_esp32_connected())

        with app.reading_lock:
            app.latest_reading_time = datetime.now(timezone.utc) - timedelta(seconds=ESP32_TIMEOUT_SECONDS + 10)

        self.assertFalse(is_esp32_connected())

        resp = self.client.get("/api/latest")
        data = resp.get_json()
        self.assertFalse(data["sensor_connected"])
        self.assertEqual(data["status"], "disconnected")

    def test_reject_demo_and_fake_payloads(self):
        demo_payload = {
            "device_id": "demo-device",
            "soil_moisture_pct": 50.0,
            "light_pct": 50.0,
            "temperature_c": 20.0,
            "humidity_pct": 50.0,
            "motion_detected": False,
        }
        resp = self.client.post("/api/sensor-data", json=demo_payload)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("real connected sensor", resp.get_json()["message"])


if __name__ == "__main__":
    unittest.main()
