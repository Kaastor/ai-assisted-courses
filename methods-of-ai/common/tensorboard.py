"""Utilities for safely creating TensorBoard writers in tests."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class _WriterProtocol(Protocol):
    def add_scalar(self, tag: str, scalar_value, global_step: int | None = None) -> None: ...
    def add_scalars(self, main_tag: str, tag_scalar_dict, global_step: int | None = None) -> None: ...
    def close(self) -> None: ...


try:  # pragma: no cover - exercised only when tensorboard is available
    from torch.utils.tensorboard import SummaryWriter as _TorchSummaryWriter

    class _SummaryWriterWrapper(_TorchSummaryWriter):
        pass

except Exception:  # pragma: no cover - fallback for environments without proper protobuf

    class _SummaryWriterWrapper:  # type: ignore[override]
        def __init__(self, log_dir: str) -> None:
            self.log_dir = log_dir

        def add_scalar(self, *args, **kwargs) -> None:
            return None

        def add_scalars(self, *args, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None


def create_summary_writer(log_dir: Optional[Path]) -> Optional[_WriterProtocol]:
    """Return a TensorBoard writer or ``None`` if logging disabled."""

    if log_dir is None:
        return None
    log_dir.mkdir(parents=True, exist_ok=True)
    return _SummaryWriterWrapper(str(log_dir))
