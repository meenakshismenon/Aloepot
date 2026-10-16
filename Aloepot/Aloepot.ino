#include <Arduino.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <HTTPClient.h>
#include <U8g2lib.h>
#include <WiFi.h>

// Hardware I2C constructor for SSD1306 OLED (uses default SCL=22, SDA=21)
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/U8X8_PIN_NONE);

// --- PIN DEFINITIONS ---
const int MOISTURE_PIN = 34; // Soil Moisture Sensor
const int LIGHT_PIN = 35;    // LDR Light Sensor
const int PIR_PIN = 23;      // Motion Sensor
const int DHT_PIN = 4;       // DHT Temp/Humidity Sensor (Left Side: GPIO 4)

#define DHTTYPE DHT11
DHT dht(DHT_PIN, DHTTYPE);

// --- CALIBRATION DATA ---
// ⚠️ Resistive sensor: LOW raw = dry air, HIGH raw = wet/water
//   DRY_RAW_SOIL = raw ADC value when sensor is in DRY AIR  (expect ~400–800)
//   WET_RAW_SOIL = raw ADC value when sensor is fully in WATER (expect
//   ~2500–3500)
// Check [DEBUG] lines in Serial Monitor to find your exact values, then update
// below.
const int DRY_RAW_SOIL = 600;  // Sensor in dry air  <-- tune this
const int WET_RAW_SOIL = 3000; // Sensor in water    <-- tune this
const int DRY_THRESHOLD = 10;  // Aloe Vera: Water ONLY when <= 10%

const int DARK_RAW_LIGHT = 4095;  // Darkness
const int BRIGHT_RAW_LIGHT = 200; // Bright Indirect Sunlight
const int LIGHT_THRESHOLD = 35;   // Aloe Vera needs >= 35% light

// --- TIMING & SLEEP ---
const unsigned long DISPLAY_TIMEOUT_MS =
    30000; // Screen sleeps after 30s of no motion
unsigned long lastMotionTime = 0;
bool isDisplayOn = false;
bool animationFrame = false;

unsigned long modeSwitchTimer = 0;
int displayState = 0; // 0 = Greeting, 1 = Soil, 2 = Light, 3 = Temp/Humidity

// --- WIFI + BACKEND CONFIG ---
const char *WIFI_SSID = "meeeeen";
const char *WIFI_PASSWORD = "1234567890";
const char *SERVER_URL =
    "http://192.167.9.13:5000/api/sensor-data"; // Current host IP
const char *DEVICE_ID = "aloe-pot-01";

const unsigned long SEND_INTERVAL_MS = 5000; // send to backend every 5s
unsigned long lastSendTime = 0;
const unsigned long MOTION_POST_INTERVAL_MS = 5000;
unsigned long lastMotionPostTime = 0;
bool hasMotionPost = false;
bool motionGreetingActive = false;
bool previousMotionState = false;

// Forward function declarations
bool sendSensorData(int moisturePercent, int lightPercent, float tempC,
                    float hum, bool motionDetected);
void drawAloeVera(bool isHappy, bool isGreeting);

// --- SMOOTHING FILTER ---
int getSmoothedRead(int pin) {
  long sum = 0;
  for (int i = 0; i < 10; i++) {
    sum += analogRead(pin);
    delay(2);
  }
  return sum / 10;
}

void connectWiFi() {
  WiFi.disconnect(true); // Clear old Wi-Fi configuration
  delay(500);
  WiFi.mode(WIFI_STA);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");

  unsigned long startAttempt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttempt < 15000) {
    delay(500);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected! IP: " + WiFi.localIP().toString());
  } else {
    Serial.println("\nWiFi connect failed - will retry in the background.");
  }
}

// --- SENSOR CONNECTION FLAGS ---
// Set to true only for sensors physically wired to the ESP32.
// Unconnected sensors will report 0 instead of garbage values.
const bool MOISTURE_SENSOR_CONNECTED = true; // <-- set true when wired
const bool LIGHT_SENSOR_CONNECTED = true;    // <-- LDR connected to pin 35
const bool PIR_SENSOR_CONNECTED = true;      // <-- PIR connected to GPIO 23
const bool DHT_SENSOR_CONNECTED = true;      // <-- set true when wired

void setup() {
  Serial.begin(115200);
  delay(1000);

  analogSetAttenuation(ADC_11db);
  analogReadResolution(12);

  pinMode(MOISTURE_PIN, INPUT);
  pinMode(LIGHT_PIN, INPUT);
  pinMode(PIR_PIN, INPUT);

  dht.begin();

  u8g2.begin();
  u8g2.setPowerSave(0);
  isDisplayOn = true;
  lastMotionTime = millis();

  // Show a loading screen so it doesn't look broken during the 15-second WiFi
  // wait
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tf);
  u8g2.setCursor(10, 30);
  u8g2.print("Connecting to:");
  u8g2.setCursor(10, 45);
  u8g2.print(WIFI_SSID);
  u8g2.sendBuffer();

  connectWiFi();

  // Show WiFi result screen for 2 seconds
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tf);
  if (WiFi.status() == WL_CONNECTED) {
    u8g2.setCursor(18, 20);
    u8g2.print("WiFi Connected!");
    u8g2.setCursor(10, 35);
    u8g2.print("IP:");
    u8g2.setCursor(28, 35);
    u8g2.print(WiFi.localIP().toString().c_str());
    u8g2.setCursor(20, 50);
    u8g2.print("Aloe Pot Ready!");
  } else {
    u8g2.setCursor(18, 25);
    u8g2.print("WiFi Failed :(");
    u8g2.setCursor(5, 40);
    u8g2.print("Check SSID/Password");
  }
  u8g2.sendBuffer();
  delay(2000);

  // Send an immediate reading on startup
  int moisturePercent = 0;
  if (MOISTURE_SENSOR_CONNECTED) {
    int rawSoil = getSmoothedRead(MOISTURE_PIN);
    // Resistive sensor: low raw = dry (0%), high raw = wet (100%)
    moisturePercent =
        constrain(map(rawSoil, DRY_RAW_SOIL, WET_RAW_SOIL, 0, 100), 0, 100);
  }
  int lightPercent = 0;
  if (LIGHT_SENSOR_CONNECTED) {
    int rawLight = getSmoothedRead(LIGHT_PIN);
    lightPercent = constrain(
        map(rawLight, DARK_RAW_LIGHT, BRIGHT_RAW_LIGHT, 0, 100), 0, 100);
  }
  float tempC = NAN;
  float hum = NAN;
  if (DHT_SENSOR_CONNECTED) {
    tempC = dht.readTemperature();
    hum = dht.readHumidity();
  }

  sendSensorData(moisturePercent, lightPercent, tempC, hum, false);
}
// Draw cute potted Aloe Vera plant
void drawAloeVera(bool isHappy, bool isGreeting) {
  // Pot
  u8g2.drawFrame(54, 26, 20, 14);
  u8g2.drawBox(52, 23, 24, 4);

  // Leaves
  u8g2.drawLine(64, 23, 64, 4);
  u8g2.drawLine(63, 23, 64, 4);
  u8g2.drawLine(60, 23, 50, 8);
  u8g2.drawLine(60, 23, 52, 10);
  u8g2.drawLine(68, 23, 78, 8);
  u8g2.drawLine(68, 23, 76, 10);
  u8g2.drawLine(62, 23, 57, 14);
  u8g2.drawLine(66, 23, 71, 14);

  // Face
  if (isGreeting) {
    u8g2.drawLine(58, 30, 60, 27);
    u8g2.drawLine(60, 27, 62, 30);
    u8g2.drawLine(66, 30, 68, 27);
    u8g2.drawLine(68, 27, 70, 30);
    u8g2.drawDisc(64, 33, 2);
  } else if (isHappy) {
    if (animationFrame) {
      u8g2.drawLine(58, 30, 60, 28);
      u8g2.drawLine(60, 28, 62, 30);
      u8g2.drawLine(66, 30, 68, 28);
      u8g2.drawLine(68, 28, 70, 30);
    } else {
      u8g2.drawLine(58, 30, 60, 28);
      u8g2.drawLine(60, 28, 62, 30);
      u8g2.drawBox(66, 29, 4, 1);
    }
    u8g2.drawPixel(64, 33);
  } else {
    u8g2.drawLine(58, 28, 62, 32);
    u8g2.drawLine(62, 28, 58, 32);
    u8g2.drawLine(66, 28, 70, 32);
    u8g2.drawLine(70, 32, 66, 35);
    u8g2.drawLine(62, 35, 66, 35);
  }
}

// Emits structured telemetry over Serial (USB) and sends via HTTP POST if Wi-Fi
// connected
bool sendSensorData(int moisturePercent, int lightPercent, float tempC,
                    float hum, bool motionDetected) {
  // If DHT is not connected or reading failed, substitute 0 instead of blocking
  // the upload
  float safeTemp = isnan(tempC) ? 0.0 : tempC;
  float safeHum = isnan(hum) ? 0.0 : hum;

  if (!DHT_SENSOR_CONNECTED) {
    Serial.println("DHT not connected - sending 0 for temp/humidity");
  } else if (isnan(tempC) || isnan(hum)) {
    Serial.println("DHT reading invalid - sending 0 for temp/humidity");
  }

  // JsonDocument works across both ArduinoJson v6 and v7
  JsonDocument doc;
  doc["device_id"] = DEVICE_ID;
  doc["temperature_c"] = safeTemp;
  doc["humidity_pct"] = safeHum;
  doc["soil_moisture_pct"] = moisturePercent;
  doc["light_pct"] = lightPercent;
  doc["motion_detected"] = motionDetected;

  String payload;
  serializeJson(doc, payload);

  // 1. Always output JSON line over USB Serial for direct PC connection
  Serial.println(payload);

  // 2. If Wi-Fi is connected, also POST to HTTP backend
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(5000);

  int httpCode = http.POST(payload);
  bool ok = false;
  if (httpCode > 0) {
    Serial.println("Backend response (" + String(httpCode) +
                   "): " + http.getString());
    ok = (httpCode == 200);
  } else {
    Serial.println("POST failed: " + http.errorToString(httpCode));
  }
  http.end();
  return ok;
}

void loop() {
  // Only read PIR if sensor is physically wired; otherwise motion is always
  // false.
  bool motionDetected = PIR_SENSOR_CONNECTED && (digitalRead(PIR_PIN) == HIGH);
  bool newMotionDetected = motionDetected && !previousMotionState;
  previousMotionState = motionDetected;

  // DEBUG: show raw PIR pin level — should say HIGH when motion, LOW when still
  Serial.print("[PIR] pin 23 = ");
  Serial.print(digitalRead(PIR_PIN) == HIGH ? "HIGH (motion!)"
                                            : "LOW (no motion)");
  Serial.print("  |  motionDetected = ");
  Serial.println(motionDetected ? "YES" : "NO");

  if (motionDetected) {
    lastMotionTime = millis();
  }

  if (newMotionDetected) {
    if (!isDisplayOn) {
      u8g2.setPowerSave(0); // Wake up OLED
      isDisplayOn = true;
    }
    displayState = 0;
    modeSwitchTimer = millis();
    motionGreetingActive = true;
  }

  // Auto-sleep OLED only when PIR is connected (motion can wake it).
  // If PIR is not connected, keep the screen on permanently.
  if (PIR_SENSOR_CONNECTED && isDisplayOn &&
      (millis() - lastMotionTime > DISPLAY_TIMEOUT_MS)) {
    u8g2.setPowerSave(1);
    isDisplayOn = false;
    motionGreetingActive = false;
  }

  // Read only connected sensors; unconnected ones stay at 0.
  int moisturePercent = 0;
  if (MOISTURE_SENSOR_CONNECTED) {
    int rawSoil = getSmoothedRead(MOISTURE_PIN);
    // DEBUG: check these values to calibrate DRY_RAW_SOIL / WET_RAW_SOIL
    Serial.print("[DEBUG] Raw soil ADC: ");
    Serial.println(rawSoil);
    // Resistive sensor: low raw = dry (0%), high raw = wet (100%)
    moisturePercent =
        constrain(map(rawSoil, DRY_RAW_SOIL, WET_RAW_SOIL, 0, 100), 0, 100);
  }

  int lightPercent = 0;
  if (LIGHT_SENSOR_CONNECTED) {
    int rawLight = getSmoothedRead(LIGHT_PIN);
    lightPercent = constrain(
        map(rawLight, DARK_RAW_LIGHT, BRIGHT_RAW_LIGHT, 0, 100), 0, 100);
  }

  float tempC = NAN;
  float hum = NAN;
  if (DHT_SENSOR_CONNECTED) {
    tempC = dht.readTemperature();
    hum = dht.readHumidity();
  }

  if (motionDetected && (!hasMotionPost || millis() - lastMotionPostTime >=
                                               MOTION_POST_INTERVAL_MS)) {
    lastMotionPostTime = millis();
    hasMotionPost = true;
    sendSensorData(moisturePercent, lightPercent, tempC, hum, true);
  } else if (!motionDetected) {
    hasMotionPost = false;
  }

  // Periodic send to backend
  if (millis() - lastSendTime >= SEND_INTERVAL_MS) {
    lastSendTime = millis();
    sendSensorData(moisturePercent, lightPercent, tempC, hum, motionDetected);
  }

  bool soilNeedsWater = (moisturePercent <= DRY_THRESHOLD);
  bool plantNeedsLight = (lightPercent < LIGHT_THRESHOLD);
  bool isHappy = (!soilNeedsWater && !plantNeedsLight);

  if (isDisplayOn) {
    // Rotate modes every 3 seconds (0: Greeting -> 1: Soil -> 2: Light -> 3:
    // Temp)
    if (millis() - modeSwitchTimer > 3000) {
      displayState = (displayState + 1) % 4;
      if (displayState == 0)
        displayState = 1;
      modeSwitchTimer = millis();
    }

    animationFrame = !animationFrame;
    u8g2.clearBuffer();

    drawAloeVera(isHappy, (displayState == 0));
    u8g2.setFont(u8g2_font_6x10_tf);

    if (displayState == 0) {
      u8g2.setCursor(8, 48);
      if (motionGreetingActive) {
        u8g2.print("HEY WASSUP!! ~");
        u8g2.setCursor(8, 58);
        u8g2.print("I SEE YOU!! :D");
      } else {
        u8g2.print("HELLO FRIEND~");
      }
    } else if (displayState == 1) {
      u8g2.setCursor(0, 46);
      u8g2.print("Soil: ");
      u8g2.print(moisturePercent);
      u8g2.print("%");

      u8g2.setCursor(0, 54);
      if (soilNeedsWater) {
        u8g2.print("DRY~ TIME TO WATER");
      } else {
        u8g2.print("MOISTURE IS JUST RIGHT");
      }
    } else if (displayState == 2) {
      u8g2.setCursor(0, 46);
      u8g2.print("Room Light: ");
      u8g2.print(lightPercent);
      u8g2.print("%");

      u8g2.setCursor(0, 54);
      if (lightPercent < 20) {
        u8g2.print("ROOM IS DARK");
      } else if (plantNeedsLight) {
        u8g2.print("ROOM IS A BIT DIM");
      } else if (lightPercent < 70) {
        u8g2.print("ROOM LIGHT IS GOOD");
      } else {
        u8g2.print("ROOM IS BRIGHT & COZY");
      }
    } else {
      u8g2.setCursor(0, 46);
      if (isnan(tempC) || isnan(hum)) {
        u8g2.print("Temp: Sensor Error");
      } else {
        u8g2.print("Temp: ");
        u8g2.print((int)tempC);
        u8g2.print("C  Hum: ");
        u8g2.print((int)hum);
        u8g2.print("%");

        u8g2.setCursor(0, 54);
        if (tempC < 13) {
          u8g2.print("BRRR~ TOO COLD!");
        } else if (tempC > 27) {
          u8g2.print("TOO WARM FOR ME~");
        } else {
          u8g2.print("COMFY TEMPERATURE~");
        }
      }
    }

    u8g2.sendBuffer();
  }

  // Send to the backend on its own timer, independent of the display state.
  if (millis() - lastSendTime >= SEND_INTERVAL_MS) {
    lastSendTime = millis();
    sendSensorData(moisturePercent, lightPercent, tempC, hum, false);
  }

  delay(400);
}
