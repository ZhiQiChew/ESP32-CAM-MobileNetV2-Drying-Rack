#include <Arduino.h>
#include "esp_camera.h"
#include <WiFi.h>
#include <WiFiClient.h>
#include <WiFiUdp.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <WiFiManager.h>

// AI-Thinker ESP32-CAM pins
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y2_GPIO_NUM 5
#define Y3_GPIO_NUM 18
#define Y4_GPIO_NUM 19
#define Y5_GPIO_NUM 21
#define Y6_GPIO_NUM 36
#define Y7_GPIO_NUM 39
#define Y8_GPIO_NUM 34
#define Y9_GPIO_NUM 35
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

constexpr uint32_t CAPTURE_INTERVAL_MS = 30000;
constexpr uint16_t DISCOVERY_PORT = 4210;
constexpr uint16_t DISCOVERY_REPLY_PORT = 4211;
constexpr uint32_t DISCOVERY_TIMEOUT_MS = 5000;
constexpr uint32_t HTTP_TIMEOUT_MS = 15000;
const char *DISCOVERY_MESSAGE = "DISCOVER_DRYING_RACK_SERVER";
const char *DISCOVERY_PREFIX = "DRYING_RACK_SERVER:";

IPAddress serverIP;
uint16_t serverPort = 8080;
bool serverAvailable = false;

struct CameraSettings {
  int exposure_ctrl = 1, aec_value = 400, gain_ctrl = 1, agc_gain = 0;
  int gainceiling = 4, brightness = 0, contrast = 1, saturation = 0;
  int whitebal = 1, awb_gain = 1, wb_mode = 0, special_effect = 0;
  int ae_level = 0;
} camSettings;

void initCamera();
bool connectOrProvisionWiFi();
bool discoverServer();
bool fetchConfig();
void applyCameraSettings();
void warmUpSensor(uint8_t frames);
String captureAndUpload();
void pollDeviceInstructions();
bool uploadTestCapture(const String &requestId);
void handleRackCommand(const String &command);
void enterCompensatedDeepSleep(uint64_t cycleStartedUs);

void setup() {
  const uint64_t cycleStartedUs = esp_timer_get_time();
  Serial.begin(115200);
  Serial.setDebugOutput(true);
  Serial.println("\n--- ESP32-CAM wake cycle ---");
  initCamera();
  if (!connectOrProvisionWiFi()) {
    Serial.println("Wi-Fi setup timed out. Restarting setup portal.");
    delay(2000);
    ESP.restart();
  }
  serverAvailable = discoverServer();
  if (serverAvailable) {
    fetchConfig();
    applyCameraSettings();
  }
  warmUpSensor(5);
  if (serverAvailable) {
    // A test request stays queued on the server until a wake cycle handles it.
    pollDeviceInstructions();
    handleRackCommand(captureAndUpload());
  } else {
    Serial.println("Server not found during this wake cycle.");
    handleRackCommand("hold");
  }
  enterCompensatedDeepSleep(cycleStartedUs);
}

void loop() {
  // setup() always ends in deep sleep.
}

void enterCompensatedDeepSleep(uint64_t cycleStartedUs) {
  esp_camera_deinit();
  WiFi.disconnect(true);
  WiFi.mode(WIFI_OFF);
  delay(50);

  const uint64_t intervalUs = (uint64_t)CAPTURE_INTERVAL_MS * 1000ULL;
  const uint64_t elapsedUs = esp_timer_get_time() - cycleStartedUs;
  // When a slow connection exceeds the interval, rest for at least one second
  // instead of immediately rebooting and generating extra heat.
  const uint64_t sleepUs = elapsedUs < intervalUs
      ? intervalUs - elapsedUs
      : 1000000ULL;
  Serial.printf("Cycle %.2f s; deep sleeping %.2f s.\n",
                elapsedUs / 1000000.0, sleepUs / 1000000.0);
  Serial.flush();
  esp_sleep_enable_timer_wakeup(sleepUs);
  esp_deep_sleep_start();
}

void pollDeviceInstructions() {
  HTTPClient http;
  String url = "http://" + serverIP.toString() + ":" + String(serverPort) +
               "/device/instructions";
  http.begin(url);
  http.setTimeout(1000);
  int code = http.GET();
  if (code == HTTP_CODE_OK) {
    StaticJsonDocument<192> doc;
    if (!deserializeJson(doc, http.getString())) {
      String action = doc["action"] | "none";
      String requestId = doc["request_id"] | "";
      http.end();
      if (action == "capture_test" && requestId.length()) {
        uploadTestCapture(requestId);
      }
      return;
    }
  }
  http.end();
}

bool uploadTestCapture(const String &requestId) {
  camera_fb_t *frame = esp_camera_fb_get();
  if (!frame) return false;
  WiFiClient client;
  client.setTimeout(HTTP_TIMEOUT_MS);
  if (!client.connect(serverIP, serverPort)) {
    esp_camera_fb_return(frame);
    serverAvailable = false;
    return false;
  }

  const String boundary = "----ESP32TestBoundary";
  const String startBody = "--" + boundary + "\r\n"
      "Content-Disposition: form-data; name=\"request_id\"\r\n\r\n" + requestId + "\r\n"
      "--" + boundary + "\r\n"
      "Content-Disposition: form-data; name=\"image\"; filename=\"crop-test.jpg\"\r\n"
      "Content-Type: image/jpeg\r\n\r\n";
  const String endBody = "\r\n--" + boundary + "--\r\n";
  const size_t contentLength = startBody.length() + frame->len + endBody.length();
  client.printf("POST /crop/device-upload HTTP/1.1\r\nHost: %s:%u\r\n",
                serverIP.toString().c_str(), serverPort);
  client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary.c_str());
  client.printf("Content-Length: %u\r\nConnection: close\r\n\r\n",
                (unsigned int)contentLength);
  client.print(startBody);
  size_t sent = 0;
  while (sent < frame->len) {
    size_t amount = client.write(frame->buf + sent,
                                 min((size_t)1024, frame->len - sent));
    if (!amount) break;
    sent += amount;
    yield();
  }
  client.print(endBody);
  esp_camera_fb_return(frame);

  uint32_t started = millis();
  while (!client.available() && client.connected() &&
         millis() - started < HTTP_TIMEOUT_MS) delay(5);
  String status = client.readStringUntil('\n');
  client.stop();
  bool uploaded = status.indexOf(" 200 ") >= 0;
  Serial.printf("Test capture upload: %s\n", uploaded ? "successful" : "failed");
  return uploaded;
}

bool connectOrProvisionWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFiManager manager;
  manager.setConfigPortalTimeout(180);
  manager.setConnectTimeout(20);
  // Tries saved credentials first. If unavailable, creates a captive portal
  // called DryingRack-Setup that lists nearby Wi-Fi networks.
  bool connected = manager.autoConnect("DryingRack-Setup");
  if (connected) {
    Serial.printf("Wi-Fi connected. ESP32 IP: %s\n",
                  WiFi.localIP().toString().c_str());
  }
  return connected;
}

bool discoverServer() {
  WiFiUDP udp;
  if (!udp.begin(DISCOVERY_REPLY_PORT)) return false;
  udp.beginPacket(IPAddress(255, 255, 255, 255), DISCOVERY_PORT);
  udp.print(DISCOVERY_MESSAGE);
  udp.endPacket();

  uint32_t started = millis();
  while (millis() - started < DISCOVERY_TIMEOUT_MS) {
    int packetSize = udp.parsePacket();
    if (packetSize > 0) {
      char response[64] = {0};
      int count = udp.read(response, sizeof(response) - 1);
      if (count > 0) response[count] = '\0';
      String reply(response);
      if (reply.startsWith(DISCOVERY_PREFIX)) {
        uint16_t port = reply.substring(strlen(DISCOVERY_PREFIX)).toInt();
        if (port > 0) serverPort = port;
        serverIP = udp.remoteIP();
        serverAvailable = true;
        Serial.printf("Server discovered: %s:%u\n",
                      serverIP.toString().c_str(), serverPort);
        udp.stop();
        return true;
      }
    }
    delay(10);
  }
  udp.stop();
  return false;
}

bool fetchConfig() {
  HTTPClient http;
  String url = "http://" + serverIP.toString() + ":" + String(serverPort) + "/config";
  http.begin(url);
  http.setTimeout(5000);
  int code = http.GET();
  if (code != HTTP_CODE_OK) {
    Serial.printf("Config request failed: HTTP %d\n", code);
    http.end();
    if (code < 0) serverAvailable = false;
    return false;
  }
  StaticJsonDocument<512> doc;
  DeserializationError error = deserializeJson(doc, http.getString());
  http.end();
  if (error) return false;
  camSettings.exposure_ctrl = doc["exposure_ctrl"] | camSettings.exposure_ctrl;
  camSettings.aec_value = doc["aec_value"] | camSettings.aec_value;
  camSettings.gain_ctrl = doc["gain_ctrl"] | camSettings.gain_ctrl;
  camSettings.agc_gain = doc["agc_gain"] | camSettings.agc_gain;
  camSettings.gainceiling = doc["gainceiling"] | camSettings.gainceiling;
  camSettings.brightness = doc["brightness"] | camSettings.brightness;
  camSettings.contrast = doc["contrast"] | camSettings.contrast;
  camSettings.saturation = doc["saturation"] | camSettings.saturation;
  camSettings.whitebal = doc["whitebal"] | camSettings.whitebal;
  camSettings.awb_gain = doc["awb_gain"] | camSettings.awb_gain;
  camSettings.wb_mode = doc["wb_mode"] | camSettings.wb_mode;
  camSettings.special_effect = doc["special_effect"] | camSettings.special_effect;
  camSettings.ae_level = doc["ae_level"] | camSettings.ae_level;
  return true;
}

void applyCameraSettings() {
  sensor_t *sensor = esp_camera_sensor_get();
  if (!sensor) return;
  sensor->set_exposure_ctrl(sensor, camSettings.exposure_ctrl);
  sensor->set_gain_ctrl(sensor, camSettings.gain_ctrl);
  sensor->set_aec2(sensor, 1);
  sensor->set_gainceiling(sensor, (gainceiling_t)camSettings.gainceiling);
  sensor->set_brightness(sensor, camSettings.brightness);
  sensor->set_contrast(sensor, camSettings.contrast);
  sensor->set_saturation(sensor, camSettings.saturation);
  sensor->set_whitebal(sensor, camSettings.whitebal);
  sensor->set_awb_gain(sensor, camSettings.awb_gain);
  sensor->set_wb_mode(sensor, camSettings.wb_mode);
  sensor->set_special_effect(sensor, camSettings.special_effect);
  sensor->set_ae_level(sensor, camSettings.ae_level);
  if (!camSettings.exposure_ctrl) sensor->set_aec_value(sensor, camSettings.aec_value);
  if (!camSettings.gain_ctrl) sensor->set_agc_gain(sensor, camSettings.agc_gain);
}

void warmUpSensor(uint8_t frames) {
  for (uint8_t i = 0; i < frames; ++i) {
    camera_fb_t *frame = esp_camera_fb_get();
    if (frame) esp_camera_fb_return(frame);
    delay(100);
  }
}

String captureAndUpload() {
  camera_fb_t *frame = esp_camera_fb_get();
  if (!frame) return "hold";
  WiFiClient client;
  client.setTimeout(HTTP_TIMEOUT_MS);
  if (!client.connect(serverIP, serverPort)) {
    esp_camera_fb_return(frame);
    serverAvailable = false;
    return "hold";
  }

  const String boundary = "----ESP32RackBoundary";
  const String startBody = "--" + boundary + "\r\n"
      "Content-Disposition: form-data; name=\"image\"; filename=\"weather.jpg\"\r\n"
      "Content-Type: image/jpeg\r\n\r\n";
  const String endBody = "\r\n--" + boundary + "--\r\n";
  const size_t contentLength = startBody.length() + frame->len + endBody.length();
  client.printf("POST /upload HTTP/1.1\r\nHost: %s:%u\r\n",
                serverIP.toString().c_str(), serverPort);
  client.printf("Content-Type: multipart/form-data; boundary=%s\r\n", boundary.c_str());
  client.printf("Content-Length: %u\r\nConnection: close\r\n\r\n",
                (unsigned int)contentLength);
  client.print(startBody);
  size_t sent = 0;
  while (sent < frame->len) {
    size_t amount = client.write(frame->buf + sent,
                                 min((size_t)1024, frame->len - sent));
    if (amount == 0) break;
    sent += amount;
    yield();
  }
  client.print(endBody);
  esp_camera_fb_return(frame);

  uint32_t started = millis();
  while (!client.available() && client.connected() &&
         millis() - started < HTTP_TIMEOUT_MS) delay(5);
  String statusLine = client.readStringUntil('\n');
  bool ok = statusLine.indexOf(" 200 ") >= 0;
  while (client.connected() || client.available()) {
    String line = client.readStringUntil('\n');
    if (line == "\r" || line.length() == 0) break;
  }
  String command = client.readString();
  client.stop();
  command.trim();
  if (!ok || (command != "extend" && command != "retract" && command != "hold")) {
    Serial.printf("Invalid response: status='%s', body='%s'\n",
                  statusLine.c_str(), command.c_str());
    return "hold";
  }
  return command;
}

void handleRackCommand(const String &command) {
  // Attach ESP-NOW/motor transmission here after its peer MAC address and
  // packet format are defined. Until then, all unknown/failure cases are hold.
  Serial.printf("Rack command: %s\n", command.c_str());
}

void initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0; config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM; config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM; config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM; config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM; config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM; config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM; config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM; config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM; config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000; config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = psramFound() ? FRAMESIZE_SVGA : FRAMESIZE_VGA;
  config.jpeg_quality = 10; config.fb_count = psramFound() ? 2 : 1;
  config.grab_mode = CAMERA_GRAB_LATEST;
  config.fb_location = psramFound() ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  esp_err_t error = esp_camera_init(&config);
  if (error != ESP_OK) {
    Serial.printf("Camera initialization failed: 0x%x\n", error);
    delay(2000);
    ESP.restart();
  }
  sensor_t *sensor = esp_camera_sensor_get();
  sensor->set_vflip(sensor, 1);
  sensor->set_hmirror(sensor, 1);
}
