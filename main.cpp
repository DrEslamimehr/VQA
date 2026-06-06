// =============================================================================
// QA-TM wearable node firmware (ESP32 / TFLite Micro).
//
// Implements the on-device half of the agentic Observe-Plan-Act-Reflect loop
// (Section 4.1) for a single BAN sensor node (e.g. ECG or PPG):
//
//   Observe : sample the biosignal, run the quantized INT8 1D-CNN feature
//             extractor (TFLite Micro) to produce the compact embedding q_t.
//   Transmit: send q_t to the edge gateway over a Kyber-512-secured BLE link.
//   Act     : receive the action a_t (from the gateway's VQC policy) and
//             actuate locally (alert / adjust sampling rate). A locally-cached
//             Edge-NN policy provides graceful-degradation fallback when the
//             gateway is unreachable (Section 7).
//
// Hardware target (Table 1):
//   ESP32-WROOM-32, 240 MHz, 520 KB SRAM, BLE 5.0 (2 Mbps).
//   Feature model: INT8 1D-CNN, ~45 KB, 3 conv layers.
//
// This is a reference firmware skeleton. The model flatbuffer
// (feature_cnn_int8.tflite -> model_data.h) is produced by
// scripts/export_tflite_micro.py. Build with Arduino IDE v2.3 or PlatformIO.
// =============================================================================
#include <Arduino.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_data.h"   // const unsigned char g_feature_cnn_int8[]; (45 KB)
#include "qatm_config.h"  // sampling rates, qubits, BLE UUIDs, fallback weights

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena. 45 KB model + activations fit comfortably in 520 KB SRAM.
constexpr int kArenaSize = 60 * 1024;
alignas(16) uint8_t tensor_arena[kArenaSize];

// Rolling biosignal buffer (one 5 s window at the configured sampling rate).
float signal_window[QATM_WINDOW_SAMPLES];
int   write_idx = 0;
}  // namespace

// --- BLE / secure-channel hooks (see qatm_ble.cpp) --------------------------
extern bool     ble_gateway_available();
extern void     ble_send_embedding(const int8_t* q, size_t n);   // Kyber-secured
extern int      ble_recv_action(uint32_t timeout_ms);            // returns a_t or -1
extern int8_t   edge_nn_fallback(const int8_t* q, size_t n);     // cached policy

void setup() {
  tflite::InitializeTarget();
  Serial.begin(115200);

  model = tflite::GetModel(g_feature_cnn_int8);
  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddFullyConnected();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddQuantize();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("[QA-TM node] ready. Feature model: INT8 1D-CNN (~45 KB).");
}

// Observe -> feature embedding q_t (INT8), shape = QATM_FEATURE_DIM (= Nq = 6).
static void extract_features(int8_t* q_out) {
  for (int i = 0; i < QATM_WINDOW_SAMPLES; ++i) {
    // INT8 quantization of the normalized signal sample.
    input->data.int8[i] = static_cast<int8_t>(
        signal_window[i] / input->params.scale + input->params.zero_point);
  }
  interpreter->Invoke();
  for (int j = 0; j < QATM_FEATURE_DIM; ++j) q_out[j] = output->data.int8[j];
}

void loop() {
  // 1. OBSERVE: sample one window of the biosignal.
  for (write_idx = 0; write_idx < QATM_WINDOW_SAMPLES; ++write_idx) {
    signal_window[write_idx] = analogRead(QATM_ADC_PIN) / 4095.0f;
    delayMicroseconds(1000000UL / QATM_SAMPLE_RATE_HZ);
  }

  int8_t q[QATM_FEATURE_DIM];
  extract_features(q);

  // 2. PLAN: offload to the gateway VQC policy over the secure BLE channel,
  //    with graceful-degradation fallback to the cached Edge-NN policy.
  int action;
  if (ble_gateway_available()) {
    ble_send_embedding(q, QATM_FEATURE_DIM);
    action = ble_recv_action(/*timeout_ms=*/QATM_GATEWAY_TIMEOUT_MS);
    if (action < 0) action = edge_nn_fallback(q, QATM_FEATURE_DIM);
  } else {
    action = edge_nn_fallback(q, QATM_FEATURE_DIM);  // F1 degrades 0.97 -> 0.82
  }

  // 3. ACT: actuate locally.
  if (action == QATM_ACTION_ANOMALY) {
    digitalWrite(QATM_ALERT_PIN, HIGH);
    Serial.println("[QA-TM node] anomaly detected -> alert triggered");
  } else {
    digitalWrite(QATM_ALERT_PIN, LOW);
  }
}
