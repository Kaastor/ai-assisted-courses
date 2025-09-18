# Lab 6 — Packaging and Deployment Readiness

The final lab converts a trained model into an inference-friendly artefact. Students practise:

- Scripting or tracing a PyTorch module to TorchScript.
- Validating tensor shapes and dtypes before batching a request.
- Measuring latency on CPU with configurable warmup/measurement loops.
- Comparing precision and throughput trade-offs (e.g., float32 vs. quantized copies).
- Writing lightweight regression tests that load the exported module and confirm numerical parity with the source model.

Starter utilities in `package.py` expose building blocks for these tasks. Acceptance tests ensure the export + load round-trip preserves predictions and that the latency benchmark runs quickly on synthetic data.
