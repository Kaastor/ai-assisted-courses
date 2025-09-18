import torch

from lab6_packaging.package import (
    PackagingConfig,
    benchmark_model,
    compare_models,
    export_to_torchscript,
    load_torchscript,
    validate_batch,
)


def test_validate_batch_raises_on_shape_mismatch() -> None:
    tensor = torch.zeros(4, 3)
    validate_batch(tensor, (4, 3), torch.float32)
    try:
        validate_batch(tensor, (2, 3), torch.float32)
    except ValueError:
        pass
    else:  # pragma: no cover - should not happen
        raise AssertionError("Expected ValueError for mismatched shape")


def test_export_and_load_round_trip(tmp_path) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
    sample = torch.randn(1, 4)
    export_path = export_to_torchscript(model, sample, tmp_path / "model.pt")
    scripted = load_torchscript(export_path)
    diff = compare_models(model, scripted, sample)
    assert diff <= 1e-4


def test_benchmark_model_produces_positive_throughput() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2))
    builder = lambda: torch.randn(8, 4)
    report = benchmark_model(model, builder, PackagingConfig(warmup_iters=1, benchmark_iters=2))
    assert report.mean_latency_ms > 0.0
    assert report.throughput_samples_s > 0.0
