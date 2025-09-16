"""Methods of Deep Learning course package."""

from importlib import resources
from pathlib import Path

__all__ = ["get_project_root"]


def get_project_root() -> Path:
    """Return repository root assuming package is installed in editable mode."""
    return Path(resources.files(__package__)).resolve().parent.parent
