// QA-TM ESP32 node configuration (mirrors Table 1 of the paper).
#ifndef QATM_CONFIG_H_
#define QATM_CONFIG_H_

// --- Sampling (Table 1). Set QATM_SAMPLE_RATE_HZ per node modality. ---------
//   ECG node : 700 Hz   |   PPG node : 64 Hz
#ifndef QATM_SAMPLE_RATE_HZ
#define QATM_SAMPLE_RATE_HZ 700
#endif
#define QATM_WINDOW_SECONDS 5
#define QATM_WINDOW_SAMPLES (QATM_SAMPLE_RATE_HZ * QATM_WINDOW_SECONDS)

// --- Feature embedding dimension == Nq (compressed state vector). -----------
#define QATM_FEATURE_DIM 6

// --- Actions (Dec-POMDP local action space A_i). ----------------------------
#define QATM_ACTION_NORMAL  0
#define QATM_ACTION_ANOMALY 1

// --- Pins. ------------------------------------------------------------------
#define QATM_ADC_PIN   34
#define QATM_ALERT_PIN 2

// --- Gateway link. ----------------------------------------------------------
#define QATM_GATEWAY_TIMEOUT_MS 80   // sub-100 ms end-to-end budget (Sec. 6.6)

// --- BLE 5.0 service/characteristic UUIDs (Kyber-512-secured payloads). -----
#define QATM_BLE_SERVICE_UUID "5a1b0001-0000-1000-8000-00805f9b34fb"
#define QATM_BLE_STATE_UUID   "5a1b0002-0000-1000-8000-00805f9b34fb"
#define QATM_BLE_ACTION_UUID  "5a1b0003-0000-1000-8000-00805f9b34fb"

#endif  // QATM_CONFIG_H_
