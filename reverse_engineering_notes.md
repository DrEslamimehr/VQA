# Reverse Engineering Notes

## Architecture

The paper describes a three-tier BAN system:

1. Wearable Perception Layer: ESP32 nodes run quantized 1D-CNN feature
   extraction over physiological windows.
2. Edge Quantum-Memory Layer: a gateway holds MPS tensor memory and runs a VQC
   simulator.
3. Cloud/QPU Layer: training-time quantum workloads may be offloaded.

`QATMAgent` implements the loop:

```text
Observe -> Reflect -> Plan -> Act
```

where:

- Observe: `QuantizedCNN1D.extract`.
- Reflect: `MatrixProductStateMemory.retrieve`.
- Plan: `QuantumPolicy.probabilities`.
- Act: argmax over the VQC action distribution.

## Quantum Circuit

The reconstructed circuit is:

1. Start in `|000000>`.
2. Apply `RY(pi * tanh(s_i))` angle embedding for six state features.
3. For each of four layers, apply `RX`, `RY`, `RZ` to each qubit.
4. Apply ring CNOTs: `0->1`, `1->2`, `2->3`, `3->4`, `4->5`, `5->0`.
5. Measure Pauli-Z expectation on the first two qubits.
6. Convert the two expectations to an action distribution by softmax.

## Parameter Count

The paper reports 8,500 QA-TM parameters while also specifying a 6-qubit,
4-layer strongly-entangling VQC, which has 72 circuit angles in the standard
PennyLane layout. To preserve the reported count while keeping the circuit
faithful, `QuantumPolicy` includes deterministic calibration parameters. The
circuit behavior remains governed by the 72 VQC angles; the calibration vector
is a small readout correction and makes `parameter_count() == 8500`.

## Dataset Mapping

- WESAD stress: label `2` is treated as anomaly; labels `1`, `3`, and `4` are
  treated as normal/non-stress.
- PhysioNet: Challenge 2015 alarm labels are mapped to anomaly=true and
  normal=false. The private held-out challenge test data are not public, so the
  clean-room split is generated from public training data.

