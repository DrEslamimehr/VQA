// =============================================================================
// QA-TM wearable node: BLE transport + Kyber-512-secured channel + cached
// Edge-NN graceful-degradation policy.
//
// This file provides the four hooks declared `extern` in main.cpp:
//
//   bool   ble_gateway_available();
//   void   ble_send_embedding(const int8_t* q, size_t n);   // Kyber-secured
//   int    ble_recv_action(uint32_t timeout_ms);            // a_t or -1
//   int8_t edge_nn_fallback(const int8_t* q, size_t n);     // cached policy
//
// The BLE half follows the BLE 5.0 (2 Mbps) link in Table 1. The node acts as
// a GATT *client*: it connects to the edge gateway (a Raspberry-Pi-class hub),
// writes the INT8 state embedding q_t to the STATE characteristic, and reads
// the integer action a_t back from the ACTION characteristic. All payloads are
// wrapped by a Kyber-512 (ML-KEM-512) key-encapsulation handshake so the BAN
// link is post-quantum-secure (Section 6 / Section 7 of the paper).
//
// On the gateway side the decapsulated embedding is fed to the distributed
// tensor-network memory + VQC policy (the Python reference implementation in
// qatm/). On this node, when the gateway is unreachable, a small cached
// Edge-NN policy provides the graceful-degradation fallback whose F1 drops from
// 0.97 to 0.82 (Table 2, "Edge-NN baseline").
//
// The Kyber-512 primitive is provided here as a compile-time stub so the
// firmware builds without pulling a PQC library into the Arduino toolchain.
// To enable real post-quantum security, link against pq-crystals/kyber
// (reference C) or wolfSSL's ML-KEM and replace the marked stub bodies.
// =============================================================================
#include <Arduino.h>
#include <string.h>

#include "qatm_config.h"

#if defined(ARDUINO) && !defined(QATM_NO_BLE)
#include <BLEDevice.h>
#include <BLEClient.h>
#include <BLEUtils.h>
#endif

// -----------------------------------------------------------------------------
// Kyber-512 (ML-KEM-512) secured channel.
//
// Sizes are the FIPS-203 ML-KEM-512 parameters:
//   public key      : 800 bytes
//   secret key      : 1632 bytes
//   ciphertext      : 768 bytes
//   shared secret   : 32 bytes
//
// The shared secret seeds an AES-256-GCM stream that wraps every BLE payload.
// The bodies below are stubs (see header note). Replace with pq-crystals/kyber.
// -----------------------------------------------------------------------------
namespace kyber512 {
constexpr size_t kPubKeyBytes = 800;
constexpr size_t kSecKeyBytes = 1632;
constexpr size_t kCipherBytes = 768;
constexpr size_t kSharedBytes = 32;

static uint8_t g_shared_secret[kSharedBytes];
static bool    g_handshake_done = false;

// crypto_kem_keypair() — STUB. Replace with pq-crystals/kyber reference.
void keypair(uint8_t pk[kPubKeyBytes], uint8_t sk[kSecKeyBytes]) {
  for (size_t i = 0; i < kPubKeyBytes; ++i) pk[i] = static_cast<uint8_t>(esp_random());
  for (size_t i = 0; i < kSecKeyBytes; ++i) sk[i] = static_cast<uint8_t>(esp_random());
}

// crypto_kem_enc() — STUB. Encapsulate against the gateway public key.
void encapsulate(const uint8_t pk[kPubKeyBytes],
                 uint8_t ct[kCipherBytes],
                 uint8_t ss[kSharedBytes]) {
  (void)pk;
  for (size_t i = 0; i < kCipherBytes; ++i) ct[i] = static_cast<uint8_t>(esp_random());
  for (size_t i = 0; i < kSharedBytes; ++i) ss[i] = static_cast<uint8_t>(esp_random());
}

// Perform the KEM handshake with the gateway and cache the shared secret.
bool establish_session(const uint8_t gateway_pk[kPubKeyBytes]) {
  uint8_t ct[kCipherBytes];
  encapsulate(gateway_pk, ct, g_shared_secret);
  // (Real impl: write `ct` to the gateway so it can decapsulate.)
  g_handshake_done = true;
  return true;
}

// AES-256-GCM-style XOR keystream wrap (placeholder symmetric layer).
void seal(uint8_t* buf, size_t n) {
  if (!g_handshake_done) return;
  for (size_t i = 0; i < n; ++i) buf[i] ^= g_shared_secret[i % kSharedBytes];
}
void open(uint8_t* buf, size_t n) { seal(buf, n); }  // XOR is its own inverse
}  // namespace kyber512

// -----------------------------------------------------------------------------
// BLE GATT client state.
// -----------------------------------------------------------------------------
namespace {
#if defined(ARDUINO) && !defined(QATM_NO_BLE)
BLEClient*                  g_client      = nullptr;
BLERemoteCharacteristic*    g_state_char  = nullptr;
BLERemoteCharacteristic*    g_action_char = nullptr;
#endif
bool    g_connected = false;
int     g_last_action = -1;

// Cached Edge-NN fallback policy (Table 2 baseline). A tiny 2-layer MLP over the
// 6-D INT8 embedding, with INT8 weights baked in at export time. The weights
// below are placeholders; scripts/export_tflite_micro.py can emit a header that
// overrides QATM_EDGE_NN_W*/B* for the trained fallback.
#ifndef QATM_EDGE_NN_W1
// 6 inputs -> 4 hidden
const int8_t kEdgeW1[4][QATM_FEATURE_DIM] = {
  { 12, -8,  5,  3, -2,  7},
  { -6, 11, -4,  9,  1, -3},
  {  4,  2, 10, -7,  6, -5},
  { -9,  3, -1,  8, -4, 12},
};
const int8_t kEdgeB1[4] = { 1, -1, 2, 0 };
// 4 hidden -> 2 logits (normal / anomaly)
const int8_t kEdgeW2[2][4] = {
  {  9, -7,  5, -3 },
  { -8, 10, -4,  6 },
};
const int8_t kEdgeB2[2] = { 0, 1 };
#endif
}  // namespace

// -----------------------------------------------------------------------------
// Public hooks (declared extern in main.cpp).
// -----------------------------------------------------------------------------

// Returns true when a Kyber-secured GATT session to the gateway is live.
bool ble_gateway_available() {
#if defined(ARDUINO) && !defined(QATM_NO_BLE)
  if (g_connected && g_client != nullptr && g_client->isConnected())
    return kyber512::g_handshake_done;

  // (Re)connect: scan for the gateway advertising QATM_BLE_SERVICE_UUID, then
  // run the Kyber-512 handshake. Connection management is intentionally minimal
  // in this reference skeleton.
  BLEDevice::init("qatm-node");
  g_client = BLEDevice::createClient();
  // ... scan + connect to the gateway by service UUID (omitted) ...
  // On a real connect, fetch the gateway public key and establish the session:
  //   uint8_t gw_pk[kyber512::kPubKeyBytes]; /* read from a PK characteristic */
  //   kyber512::establish_session(gw_pk);
  return false;  // skeleton: report unavailable until wired to hardware
#else
  return false;  // host build: no BLE -> always fall back to Edge-NN
#endif
}

// Encrypt + write the INT8 state embedding q_t to the gateway STATE char.
void ble_send_embedding(const int8_t* q, size_t n) {
  uint8_t buf[QATM_FEATURE_DIM];
  memcpy(buf, q, n);
  kyber512::seal(buf, n);  // post-quantum-secured payload
#if defined(ARDUINO) && !defined(QATM_NO_BLE)
  if (g_state_char != nullptr)
    g_state_char->writeValue(buf, n, /*response=*/false);
#else
  (void)buf;
#endif
}

// Read the integer action a_t from the gateway ACTION char (or -1 on timeout).
int ble_recv_action(uint32_t timeout_ms) {
#if defined(ARDUINO) && !defined(QATM_NO_BLE)
  uint32_t t0 = millis();
  while (millis() - t0 < timeout_ms) {
    if (g_action_char != nullptr && g_action_char->canRead()) {
      std::string v = g_action_char->readValue();
      if (!v.empty()) {
        uint8_t a = static_cast<uint8_t>(v[0]);
        kyber512::open(&a, 1);
        g_last_action = static_cast<int>(a);
        return g_last_action;
      }
    }
    delay(2);
  }
  return -1;  // timed out -> caller uses edge_nn_fallback()
#else
  (void)timeout_ms;
  return -1;
#endif
}

// Cached Edge-NN policy: 6->4->2 INT8 MLP, argmax over the 2 action logits.
// Provides graceful degradation when the gateway is unreachable (F1 0.97->0.82).
int8_t edge_nn_fallback(const int8_t* q, size_t n) {
  const size_t in_dim = (n < QATM_FEATURE_DIM) ? n : QATM_FEATURE_DIM;
  int32_t hidden[4];
  for (int h = 0; h < 4; ++h) {
    int32_t acc = static_cast<int32_t>(kEdgeB1[h]) << 4;
    for (size_t i = 0; i < in_dim; ++i)
      acc += static_cast<int32_t>(kEdgeW1[h][i]) * static_cast<int32_t>(q[i]);
    hidden[h] = acc > 0 ? acc : 0;  // ReLU
  }
  int32_t logits[2];
  for (int o = 0; o < 2; ++o) {
    int32_t acc = static_cast<int32_t>(kEdgeB2[o]) << 8;
    for (int h = 0; h < 4; ++h)
      acc += static_cast<int32_t>(kEdgeW2[o][h]) * hidden[h];
    logits[o] = acc;
  }
  return (logits[QATM_ACTION_ANOMALY] > logits[QATM_ACTION_NORMAL])
             ? QATM_ACTION_ANOMALY
             : QATM_ACTION_NORMAL;
}
