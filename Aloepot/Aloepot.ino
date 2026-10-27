#include <Arduino.h>
#include <ArduinoJson.h>
#include <DHT.h>
#include <HTTPClient.h>
#include <U8g2lib.h>
#include <WiFi.h>

// Hardware I2C constructor for SSD1306 OLED (default SCL=22, SDA=21)
U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/U8X8_PIN_NONE);

// --- PIN DEFINITIONS ---
const int MOISTURE_PIN = 34; // Soil Moisture Sensor (Analog)
const int LIGHT_PIN = 35;    // LDR Light Sensor (Analog)
const int PIR_PIN = 23;      // Motion Sensor (Digital)
const int DHT_PIN = 4;       // DHT11 Temp/Humidity Sensor (GPIO 4)

#define DHTTYPE DHT11
DHT dht(DHT_PIN, DHTTYPE);

// --- SENSOR CALIBRATION DATA ---
// Capacitive Soil Moisture: Dry air ~3300, Saturated water ~1400
const int DRY_RAW_SOIL = 3300;  // Dry air / dry soil -> 0%
const int WET_RAW_SOIL = 1400;  // Water / wet soil -> 100%
const int DRY_THRESHOLD = 15;   // Aloe Vera: Water when <= 15%

// LDR Light Sensor: Darkness ~3900, Indoor ambient ~1400-2400, Bright direct ~400
const int DARK_RAW_LIGHT = 3900;  // Total Darkness -> 0%
const int BRIGHT_RAW_LIGHT = 400; // Bright Direct Light -> 100%
const int LIGHT_THRESHOLD = 30;   // Aloe Vera needs >= 30% light

// --- TIMING & DISPLAY ---
unsigned long lastMotionTime = 0;
bool isDisplayOn = true; // Keep OLED screen ALWAYS on so it never goes black

// Screen mode rotation: 6.5s per screen (relaxed, easy to read)
const unsigned long SCREEN_ROTATE_INTERVAL_MS = 6500;
unsigned long modeSwitchTimer = 0;
int displayState = 0; // 0 = Greeting, 1 = Soil, 2 = Light, 3 = Temp/Humidity, 4 = Weather/Rain

// Motion greeting lock (ensures greeting stays visible for 4s without constant re-triggering)
unsigned long greetingLockUntil = 0;
unsigned long lastOledGreetingTime = 0;
const unsigned long OLED_GREETING_COOLDOWN_MS = 600000; // 10 minutes (600,000 ms) between routine greetings
bool motionGreetingActive = false;
bool previousMotionState = false;

// Display rendering refresh timer (smooth 30 FPS rendering)
unsigned long lastDisplayRefresh = 0;
const unsigned long DISPLAY_REFRESH_INTERVAL_MS = 100;

// Weather state received from backend
bool isRaining = false;

// Filtered / smoothed sensor state (Exponential Moving Average)
float smoothedSoilPercent = -1.0;
float smoothedLightPercent = -1.0;

// Person recognition state received from backend
int personStatus = 0; // 0 = None (generic motion), 1 = Meenakshi, 2 = Stranger
int previousPersonStatus = 0;
String personName = "";
String greetingLine1 = "HEY WASSUP!! ~";
String greetingLine2 = "I SEE YOU!! :D";
unsigned long lastPersonDetectedTime = 0;
const unsigned long PERSON_ACTIVE_TIMEOUT_MS = 30000; // Person greeting active for 30s
unsigned long lastStrangerGreetingTime = 0;
bool strangerGreetedOnce = false; // Only trigger stranger alert 1 time

// --- WIFI + BACKEND CONFIG ---
const char *WIFI_SSID = "meeeeen";
const char *WIFI_PASSWORD = "1234567890";
const char *SERVER_URL = "http://10.214.189.5:5000/api/sensor-data"; // Flask Backend IP
const char *DEVICE_ID = "aloe-pot-01";

const unsigned long SEND_INTERVAL_MS = 5000; // Send periodic data every 5s
unsigned long lastSendTime = 0;
const unsigned long MOTION_POST_INTERVAL_MS = 4000; // Cooldown between immediate motion posts
unsigned long lastMotionPostTime = 0;

// Sensor connection flags
const bool MOISTURE_SENSOR_CONNECTED = true;
const bool LIGHT_SENSOR_CONNECTED = true;
const bool PIR_SENSOR_CONNECTED = true;
const bool DHT_SENSOR_CONNECTED = true;

// --- DHT11 CALIBRATION & TIMING ---
// DHT11 needs at least 2 seconds between reads to prevent sensor self-heating
const unsigned long DHT_SAMPLE_INTERVAL_MS = 2500;
unsigned long lastDHTSampleTime = 0;
float cachedTempC = 25.0;
float cachedHum = 50.0;
// Compensates for ESP32 Wi-Fi module & linear regulator PCB heat radiated to DHT11:
const float TEMP_CALIBRATION_OFFSET_C = -4.5; // Compensates for ESP32 self-heating
const float TEMP_TOO_HOT_C = 31.0;            // Aloe vera is a succulent; thrives up to 31 C (88 F)
const float TEMP_TOO_COLD_C = 13.0;           // Cold alert threshold

// Forward function declarations
void handlePersonStatusUpdate(int s, unsigned long now, String l1 = "", String l2 = "");
bool sendSensorData(int moisturePercent, int lightPercent, float tempC, float hum, bool motionDetected);
void drawAloeVera(bool isHappy, bool isGreeting, int personMode = 0, bool isBlinking = false);

// --- SMOOTHING FILTER (32-sample oversampling with min/max outlier rejection) ---
int getSmoothedRead(int pin) {
  long sum = 0;
  int minVal = 4096;
  int maxVal = 0;
  for (int i = 0; i < 32; i++) {
    int val = analogRead(pin);
    if (val < minVal) minVal = val;
    if (val > maxVal) maxVal = val;
    sum += val;
    delayMicroseconds(150);
  }
  // Discard the single lowest and highest noise spikes
  return (sum - minVal - maxVal) / 30;
}

void connectWiFi() {
  WiFi.disconnect(true);
  delay(300);
  WiFi.mode(WIFI_STA);

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to WiFi");

  unsigned long startAttempt = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startAttempt < 12000) {
    delay(400);
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected! IP: " + WiFi.localIP().toString());
  } else {
    Serial.println("\nWiFi connection failed - continuing offline, will retry.");
  }
}

void setup() {
  Serial.begin(115200);
  delay(600);

  analogSetAttenuation(ADC_11db);
  analogReadResolution(12);

  pinMode(MOISTURE_PIN, INPUT);
  pinMode(LIGHT_PIN, INPUT);
  // Using INPUT_PULLDOWN stabilizes the PIR output on ESP32 GPIO 23
  pinMode(PIR_PIN, INPUT_PULLDOWN);

  dht.begin();

  u8g2.begin();
  u8g2.setPowerSave(0);
  isDisplayOn = true;
  lastMotionTime = millis();

  // Startup splash screen
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tf);
  u8g2.setCursor(10, 28);
  u8g2.print("AloePulse");
  u8g2.setCursor(10, 44);
  u8g2.print("WiFi: ");
  u8g2.print(WIFI_SSID);
  u8g2.sendBuffer();

  connectWiFi();

  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_6x10_tf);
  if (WiFi.status() == WL_CONNECTED) {
    u8g2.setCursor(16, 20);
    u8g2.print("WiFi Connected!");
    u8g2.setCursor(10, 36);
    u8g2.print("IP: ");
    u8g2.print(WiFi.localIP().toString().c_str());
    u8g2.setCursor(12, 52);
    u8g2.print("AloePulse Online :)");
  } else {
    u8g2.setCursor(16, 25);
    u8g2.print("WiFi Offline :(");
    u8g2.setCursor(5, 42);
    u8g2.print("Running in Standalone");
  }
  u8g2.sendBuffer();
  delay(1500);

  // Initial sensor read & transmission
  int initMoist = 0;
  if (MOISTURE_SENSOR_CONNECTED) {
    int rawSoil = getSmoothedRead(MOISTURE_PIN);
    initMoist = constrain(map(rawSoil, DRY_RAW_SOIL, WET_RAW_SOIL, 0, 100), 0, 100);
    smoothedSoilPercent = initMoist;
  }
  int initLight = 0;
  if (LIGHT_SENSOR_CONNECTED) {
    int rawLight = getSmoothedRead(LIGHT_PIN);
    initLight = constrain(map(rawLight, DARK_RAW_LIGHT, BRIGHT_RAW_LIGHT, 0, 100), 0, 100);
    smoothedLightPercent = initLight;
  }
  float initTemp = DHT_SENSOR_CONNECTED ? dht.readTemperature() : 0.0;
  float initHum = DHT_SENSOR_CONNECTED ? dht.readHumidity() : 0.0;

  sendSensorData(initMoist, initLight, initTemp, initHum, false);
}

// Draw cute potted Aloe Vera plant with natural blinking expressions
void drawAloeVera(bool isHappy, bool isGreeting, int personMode, bool isBlinking) {
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
    if (personMode == 1) {
      // Meenakshi Recognized! Cute loving winking face + heart
      u8g2.drawLine(58, 30, 60, 27);
      u8g2.drawLine(60, 27, 62, 30);
      u8g2.drawBox(66, 28, 4, 2); // Winking eye
      u8g2.drawDisc(64, 33, 2);   // Happy open smile
      // Little heart above pot
      u8g2.drawPixel(75, 20);
      u8g2.drawPixel(77, 20);
      u8g2.drawPixel(76, 21);
    } else if (personMode == 2) {
      // Stranger / Unknown visitor! Surprised/inquisitive wide round eyes + small mouth
      u8g2.drawCircle(59, 29, 2);
      u8g2.drawCircle(68, 29, 2);
      u8g2.drawCircle(64, 34, 1);
    } else {
      // Generic motion greeting: Happy squinting eyes & open mouth
      u8g2.drawLine(58, 30, 60, 27);
      u8g2.drawLine(60, 27, 62, 30);
      u8g2.drawLine(66, 30, 68, 27);
      u8g2.drawLine(68, 27, 70, 30);
      u8g2.drawDisc(64, 33, 2);
    }
  } else if (isHappy) {
    if (isBlinking) {
      // Soft, natural blink
      u8g2.drawLine(58, 30, 60, 28);
      u8g2.drawLine(60, 28, 62, 30);
      u8g2.drawBox(66, 29, 4, 1);
    } else {
      // Wide open cheerful eyes
      u8g2.drawLine(58, 30, 60, 28);
      u8g2.drawLine(60, 28, 62, 30);
      u8g2.drawLine(66, 30, 68, 28);
      u8g2.drawLine(68, 28, 70, 30);
    }
    u8g2.drawPixel(64, 33);
  } else {
    // Sad/Stressed face
    u8g2.drawLine(58, 28, 62, 32);
    u8g2.drawLine(62, 28, 58, 32);
    u8g2.drawLine(66, 28, 70, 32);
    u8g2.drawLine(70, 32, 66, 35);
    u8g2.drawLine(62, 35, 66, 35);
  }
}

// Handles person status updates from AI Face Tracker (HTTP or USB Serial)
// Respects the 10-minute cooldown for owner greeting and immediate alerts for strangers
void handlePersonStatusUpdate(int s, unsigned long now, String l1, String l2) {
  if (s <= 0) {
    if (personStatus > 0 && (now - lastPersonDetectedTime > PERSON_ACTIVE_TIMEOUT_MS)) {
      personStatus = 0;
      previousPersonStatus = 0;
    }
    return;
  }

  bool isNewDetection = (s != previousPersonStatus);
  personStatus = s;
  lastPersonDetectedTime = now;
  if (l1.length() > 0) greetingLine1 = l1;
  if (l2.length() > 0) greetingLine2 = l2;

  bool triggerGreeting = false;
  if (s == 2) {
    // Stranger: alert ONLY 1 TIME for now so it does not irritate
    if (!strangerGreetedOnce) {
      triggerGreeting = true;
      strangerGreetedOnce = true;
      lastStrangerGreetingTime = now;
      Serial.println("[AI] Stranger detected! Alerting 'Hey stranger' (only 1 time).");
    }
  } else if (s == 1) {
    // Owner Meenakshi: ONLY trigger greeting on fresh arrival (or after 10-minute cooldown)
    if (isNewDetection && (now - lastOledGreetingTime >= OLED_GREETING_COOLDOWN_MS || lastOledGreetingTime == 0)) {
      triggerGreeting = true;
      lastOledGreetingTime = now;
      Serial.println("[AI] Owner Recognized: Meenakshi! Showing greeting (10-min cooldown started).");
    }
  }

  if (triggerGreeting) {
    if (!isDisplayOn) {
      u8g2.setPowerSave(0);
      isDisplayOn = true;
    }
    displayState = 0; // Mode 0 = Greeting screen
    greetingLockUntil = now + 4000; // Hold for 4 seconds then allow normal sensor rotation
  }
  previousPersonStatus = s;
}

// Emits structured telemetry over Serial (USB) and sends via HTTP POST with fast timeout
bool sendSensorData(int moisturePercent, int lightPercent, float tempC, float hum, bool motionDetected) {
  float safeTemp = isnan(tempC) ? 0.0 : tempC;
  float safeHum = isnan(hum) ? 0.0 : hum;

  JsonDocument doc;
  doc["device_id"] = DEVICE_ID;
  doc["temperature_c"] = safeTemp;
  doc["humidity_pct"] = safeHum;
  doc["soil_moisture_pct"] = moisturePercent;
  doc["light_pct"] = lightPercent;
  doc["motion_detected"] = motionDetected;

  String payload;
  serializeJson(doc, payload);

  // 1. Output line over USB Serial
  Serial.println(payload);

  // 2. If Wi-Fi connected, POST to HTTP backend
  if (WiFi.status() != WL_CONNECTED) {
    return false;
  }

  HTTPClient http;
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  http.setTimeout(1200); // 1.2s timeout ensures network hiccups NEVER freeze the loop

  int httpCode = http.POST(payload);
  bool ok = false;
  if (httpCode == 200) {
    String respStr = http.getString();
    Serial.println("Backend response: " + respStr);

    // Parse backend response to check real-time outdoor rain alerts and person recognition
    JsonDocument resDoc;
    DeserializationError err = deserializeJson(resDoc, respStr);
    if (!err) {
      if (resDoc.containsKey("is_raining")) {
        isRaining = resDoc["is_raining"].as<bool>();
      }
      if (resDoc.containsKey("person_status")) {
        int s = resDoc["person_status"].as<int>();
        String l1 = resDoc.containsKey("greeting_line1") ? resDoc["greeting_line1"].as<String>() : "";
        String l2 = resDoc.containsKey("greeting_line2") ? resDoc["greeting_line2"].as<String>() : "";
        handlePersonStatusUpdate(s, millis(), l1, l2);
      }
    }
    ok = true;
  } else if (httpCode > 0) {
    Serial.println("Backend returned code: " + String(httpCode));
  } else {
    Serial.println("POST failed: " + http.errorToString(httpCode));
  }
  http.end();
  return ok;
}

void loop() {
  unsigned long now = millis();

  // --- 0. CHECK USB SERIAL FOR LIVE COMMANDS/PERSON STATUS FROM PYTHON BACKEND ---
  if (Serial.available()) {
    String serLine = Serial.readStringUntil('\n');
    serLine.trim();
    if (serLine.startsWith("{") && serLine.endsWith("}")) {
      JsonDocument serDoc;
      if (!deserializeJson(serDoc, serLine)) {
        if (serDoc.containsKey("is_raining")) {
          isRaining = serDoc["is_raining"].as<bool>();
        }
        if (serDoc.containsKey("person_status")) {
          int s = serDoc["person_status"].as<int>();
          String l1 = serDoc.containsKey("line1") ? serDoc["line1"].as<String>() : "";
          String l2 = serDoc.containsKey("line2") ? serDoc["line2"].as<String>() : "";
          handlePersonStatusUpdate(s, now, l1, l2);
        }
      }
    }
  }

  // --- 1. RELIABLE PIR MOTION DETECTION ---
  // Read pin with digitalRead; INPUT_PULLDOWN ensures no floating line noise
  bool rawPin = (digitalRead(PIR_PIN) == HIGH);
  bool motionDetected = PIR_SENSOR_CONNECTED && rawPin;
  bool newMotionTrigger = motionDetected && !previousMotionState;
  previousMotionState = motionDetected;

  if (motionDetected) {
    lastMotionTime = now;

    // Instantly wake up the OLED display if it was sleeping
    bool justWokeUp = !isDisplayOn;
    if (justWokeUp) {
      u8g2.setPowerSave(0);
      isDisplayOn = true;
    }

    // Trigger greeting screen if:
    // 1) OLED just woke up from sleep (new arrival into room), OR
    // 2) At least 10 minutes (600s) have passed since the last greeting
    bool cooldownPassed = (now - lastOledGreetingTime >= OLED_GREETING_COOLDOWN_MS || lastOledGreetingTime == 0);

    if (justWokeUp || (newMotionTrigger && cooldownPassed)) {
      displayState = 0; // Mode 0 = Greeting
      greetingLockUntil = now + 4000; // Hold greeting for 4 seconds
      lastOledGreetingTime = now;
      motionGreetingActive = true;
      Serial.println("[PIR] Arrival detected! Showing greeting.");
    }
  }

  // Keep OLED active and rendering continuously (no auto-sleep)
  isDisplayOn = true;

  // --- 2. SENSOR READINGS (EMA Filtered) ---
  int moisturePercent = 0;
  if (MOISTURE_SENSOR_CONNECTED) {
    int rawSoil = getSmoothedRead(MOISTURE_PIN);
    float currentMoist = constrain(map(rawSoil, DRY_RAW_SOIL, WET_RAW_SOIL, 0, 100), 0, 100);
    if (smoothedSoilPercent < 0) {
      smoothedSoilPercent = currentMoist;
    } else {
      // Exponential Moving Average (85% previous, 15% new reading)
      smoothedSoilPercent = (smoothedSoilPercent * 0.85) + (currentMoist * 0.15);
    }
    moisturePercent = round(smoothedSoilPercent);
  }

  int lightPercent = 0;
  if (LIGHT_SENSOR_CONNECTED) {
    int rawLight = getSmoothedRead(LIGHT_PIN);
    float currentLight = constrain(map(rawLight, DARK_RAW_LIGHT, BRIGHT_RAW_LIGHT, 0, 100), 0, 100);
    if (smoothedLightPercent < 0) {
      smoothedLightPercent = currentLight;
    } else {
      smoothedLightPercent = (smoothedLightPercent * 0.85) + (currentLight * 0.15);
    }
    lightPercent = round(smoothedLightPercent);
  }

  // Throttle DHT11 sampling to prevent internal sensor self-heating
  if (DHT_SENSOR_CONNECTED && (now - lastDHTSampleTime >= DHT_SAMPLE_INTERVAL_MS || lastDHTSampleTime == 0)) {
    lastDHTSampleTime = now;
    float rawT = dht.readTemperature();
    float rawH = dht.readHumidity();
    if (!isnan(rawT)) {
      cachedTempC = rawT + TEMP_CALIBRATION_OFFSET_C;
    }
    if (!isnan(rawH)) {
      cachedHum = rawH;
    }
  }
  float tempC = cachedTempC;
  float hum = cachedHum;

  // --- 3. TELEMETRY & MOTION REPORTING ---
  // Immediate post on new motion (throttled to once every 4 seconds)
  if (newMotionTrigger && (now - lastMotionPostTime >= MOTION_POST_INTERVAL_MS)) {
    lastMotionPostTime = now;
    sendSensorData(moisturePercent, lightPercent, tempC, hum, true);
  }

  // Periodic telemetry to backend every 5 seconds
  if (now - lastSendTime >= SEND_INTERVAL_MS) {
    lastSendTime = now;
    sendSensorData(moisturePercent, lightPercent, tempC, hum, motionDetected);
  }

  // --- 4. OLED DISPLAY RENDERING (NON-BLOCKING) ---
  if (isDisplayOn && (now - lastDisplayRefresh >= DISPLAY_REFRESH_INTERVAL_MS)) {
    lastDisplayRefresh = now;

    // Natural eye blink: gentle 220ms blink every 4 seconds
    bool isBlinking = ((now % 4000) < 220);

    bool soilNeedsWater = (moisturePercent <= DRY_THRESHOLD);
    bool plantNeedsLight = (lightPercent < LIGHT_THRESHOLD);
    bool isHappy = (!soilNeedsWater && !plantNeedsLight);

    // If locked into greeting by recent motion, keep displayState = 0
    if (now < greetingLockUntil) {
      displayState = 0;
    } else {
      // Rotate through modes every 6.5 seconds (relaxed, easy to read)
      if (now - modeSwitchTimer > SCREEN_ROTATE_INTERVAL_MS) {
        modeSwitchTimer = now;
        displayState = (displayState + 1) % 5;
        // Skip greeting state during idle rotation (mode 0 is reserved for motion)
        if (displayState == 0) {
          displayState = 1;
        }
      }
    }

    // Check if person recognition has timed out
    if (personStatus > 0 && (now - lastPersonDetectedTime > PERSON_ACTIVE_TIMEOUT_MS)) {
      personStatus = 0;
    }

    u8g2.clearBuffer();
    drawAloeVera(isHappy, (displayState == 0), (displayState == 0 ? personStatus : 0), isBlinking);
    u8g2.setFont(u8g2_font_6x10_tf);

    if (displayState == 0) {
      if (personStatus == 1) {
        // Owner Meenakshi Recognized!
        u8g2.setCursor(6, 48);
        u8g2.print("HI MEENAKSHI! :D");
        u8g2.setCursor(6, 58);
        if (isRaining) {
          u8g2.print("RAIN ALERT! STAY DRY");
        } else {
          u8g2.print("WELCOME HOME! <3");
        }
      } else if (personStatus == 2) {
        // Stranger / Unknown visitor!
        u8g2.setCursor(6, 48);
        u8g2.print("HEY STRANGER! ~");
        u8g2.setCursor(6, 58);
        u8g2.print("WTSUP! WHO'S THERE?");
      } else {
        // Generic PIR Motion Greeting & Rain Alert
        u8g2.setCursor(8, 48);
        u8g2.print("HEY WASSUP!! ~");
        u8g2.setCursor(8, 58);
        if (isRaining) {
          u8g2.print("IT'S RAINING! :O");
        } else {
          u8g2.print("I SEE YOU!! :D");
        }
      }
    } else if (displayState == 1) {
      // Soil Moisture Mode
      u8g2.setCursor(0, 46);
      u8g2.print("Soil: ");
      u8g2.print(moisturePercent);
      u8g2.print("%");

      u8g2.setCursor(0, 56);
      if (soilNeedsWater) {
        u8g2.print("DRY~ TIME TO WATER");
      } else {
        u8g2.print("MOISTURE IS JUST RIGHT");
      }
    } else if (displayState == 2) {
      // Room Light Mode
      u8g2.setCursor(0, 46);
      u8g2.print("Room Light: ");
      u8g2.print(lightPercent);
      u8g2.print("%");

      u8g2.setCursor(0, 56);
      if (lightPercent < 20) {
        u8g2.print("ROOM IS DARK");
      } else if (plantNeedsLight) {
        u8g2.print("ROOM IS A BIT DIM");
      } else if (lightPercent < 70) {
        u8g2.print("ROOM LIGHT IS GOOD");
      } else {
        u8g2.print("ROOM IS BRIGHT & COZY");
      }
    } else if (displayState == 3) {
      // Temperature & Humidity Mode
      u8g2.setCursor(0, 46);
      if (isnan(tempC) || isnan(hum)) {
        u8g2.print("Temp: Sensor Error");
      } else {
        u8g2.print("Temp: ");
        u8g2.print((int)tempC);
        u8g2.print("C  Hum: ");
        u8g2.print((int)hum);
        u8g2.print("%");

        u8g2.setCursor(0, 56);
        if (tempC < TEMP_TOO_COLD_C) {
          u8g2.print("BRRR~ TOO COLD!");
        } else if (tempC > TEMP_TOO_HOT_C) {
          u8g2.print("TOO WARM FOR ME~");
        } else {
          u8g2.print("COMFY TEMPERATURE~");
        }
      }
    } else {
      // Outdoor Weather & Rain Status Mode
      u8g2.setCursor(0, 46);
      u8g2.print("Outdoor Weather:");

      u8g2.setCursor(0, 56);
      if (isRaining) {
        u8g2.print("RAIN ALERT! STAY DRY");
      } else {
        u8g2.print("OUTDOOR SKIES CLEAR");
      }
    }

    u8g2.sendBuffer();
  }

  // Small 10ms yield to allow ESP32 background tasks and WiFi stack to process smoothly
  delay(10);
}
