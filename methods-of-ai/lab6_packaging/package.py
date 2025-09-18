"""Utilities for exporting and benchmarking PyTorch models."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import torch
from torch import nn


@dataclass
class PackagingConfig:
    """Configuration for latency benchmarking."""

    warmup_iters: int = 3
    benchmark_iters: int = 10
    device: str = "cpu"


@dataclass
class PackagingReport:
    """Latency summary produced by ``benchmark_model``."""

    mean_latency_ms: float
    p95_latency_ms: float
    throughput_samples_s: float


def validate_batch(tensor: torch.Tensor, expected_shape: Tuple[int, ...], expected_dtype: torch.dtype) -> torch.Tensor:
    """Ensure a tensor matches the expected batch shape and dtype."""

    if tensor.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, received {tuple(tensor.shape)}")
    if tensor.dtype != expected_dtype:
        raise ValueError(f"Expected dtype {expected_dtype}, received {tensor.dtype}")
    return tensor


def export_to_torchscript(
    model: nn.Module,
    sample_input: torch.Tensor,
    export_path: Path,
    strict: bool = True,
) -> Path:
    """Trace a model with ``sample_input`` and persist it as TorchScript."""

    model.eval()
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        scripted = torch.jit.trace(model, sample_input, strict=strict)
    scripted.save(str(export_path))
    return export_path


def load_torchscript(export_path: Path) -> torch.jit.ScriptModule:
    """Load a previously exported TorchScript module."""

    return torch.jit.load(str(export_path))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():  # pragma: no cover - CUDA not used in tests
        torch.cuda.synchronize()


def benchmark_model(
    model: nn.Module,
    input_builder: Callable[[], torch.Tensor],
    config: PackagingConfig = PackagingConfig(),
) -> PackagingReport:
    """Benchmark latency for ``model`` using inputs from ``input_builder``."""

    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    # Warmup passes
    for _ in range(max(0, config.warmup_iters)):
        with torch.no_grad():
            _ = model(input_builder().to(device))
    _synchronize(device)

    latencies: list[float] = []
    batch_sizes: list[int] = []
    for _ in range(max(1, config.benchmark_iters)):
        inputs = input_builder().to(device)
        batch_sizes.append(int(inputs.size(0) if inputs.dim() > 0 else 1))
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(inputs)
        _synchronize(device)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    mean_latency = statistics.mean(latencies)
    sorted_latencies = sorted(latencies)
    if len(sorted_latencies) > 1:
        index = max(0, int(round(0.95 * len(sorted_latencies))) - 1)
        p95_latency = sorted_latencies[min(index, len(sorted_latencies) - 1)]
    else:
        p95_latency = sorted_latencies[0]
    mean_batch = statistics.mean(batch_sizes)
    throughput = mean_batch / mean_latency if mean_latency > 0 else float("inf")
    return PackagingReport(
        mean_latency_ms=float(mean_latency * 1000.0),
        p95_latency_ms=float(p95_latency * 1000.0),
        throughput_samples_s=float(throughput),
    )


def compare_models(
    reference: nn.Module,
    candidate: nn.Module,
    inputs: torch.Tensor,
    atol: float = 1e-4,
) -> float:
    """Return the maximum absolute difference between two models' outputs."""

    reference.eval()
    candidate.eval()
    with torch.no_grad():
        ref_out = reference(inputs)
        cand_out = candidate(inputs)
    max_diff = (ref_out - cand_out).abs().max().item()
    if max_diff > atol:
        raise AssertionError(f"Model outputs diverged by {max_diff} (allowed {atol})")
    return max_diff
