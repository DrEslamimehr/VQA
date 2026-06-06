#include <Arduino.h>

// Clean-room ESP32 node skeleton for the paper's wearable perception layer.
// Replace this stub with a generated TFLite Micro model array when deploying.

static const int ECG_PIN = 34;
static const int PPG_PIN = 35;
static const int WINDOW = 256;
static const int EMBED_DIM = 6;

static int16_t ecg_window[WINDOW];
static int16_t ppg_window[WINDOW];
static int cursor = 0;

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
}

static void emit_embedding() {
  float embedding[EMBED_DIM];
  long ecg_sum = 0;
  long ppg_sum = 0;
  for (int i = 0; i < WINDOW; ++i) {
    ecg_sum += ecg_window[i];
    ppg_sum += ppg_window[i];
  }
  const float ecg_mean = ecg_sum / float(WINDOW);
  const float ppg_mean = ppg_sum / float(WINDOW);
  for (int j = 0; j < EMBED_DIM; ++j) {
    embedding[j] = (j % 2 == 0 ? ecg_mean : ppg_mean) / 4096.0f;
  }

  Serial.print("QATM_EMBED");
  for (int j = 0; j < EMBED_DIM; ++j) {
    Serial.print(',');
    Serial.print(embedding[j], 6);
  }
  Serial.println();
}

void loop() {
  ecg_window[cursor] = analogRead(ECG_PIN);
  ppg_window[cursor] = analogRead(PPG_PIN);
  cursor = (cursor + 1) % WINDOW;
  if (cursor == 0) {
    emit_embedding();
  }
  delayMicroseconds(1000000 / 700);
}

