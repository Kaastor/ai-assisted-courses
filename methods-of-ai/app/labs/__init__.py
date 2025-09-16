"""Lab registry for the Methods of Deep Learning course."""

from __future__ import annotations

from importlib import import_module
from typing import Iterable

from .base import LabSpecification, ensure_all_specs

__all__ = ["iter_lab_specs", "LAB_MODULES"]

LAB_MODULES: tuple[str, ...] = (
    "app.labs.lab01_tabular_classification_with_an_mlp_uci_adult.spec",
    "app.labs.lab02_vision_classification_with_a_tiny_cnn_fashion_mnist.spec",
    "app.labs.lab03_text_classification_with_embeddingbag_sms_spam.spec",
    "app.labs.lab04_time_series_forecasting_with_gru_electricity_load.spec",
    "app.labs.lab05_recommendations_with_neural_matrix_factorization_movielens.spec",
    "app.labs.lab06_generative_modeling_with_a_variational_autoencoder_vae_on_fashion_mnist.spec",
)


def iter_lab_specs() -> Iterable[LabSpecification]:
    """Yield all registered lab specifications, ensuring READMEs exist."""

    specs = []
    for dotted in LAB_MODULES:
        module = import_module(dotted)
        spec = getattr(module, "LAB_SPEC")
        if not isinstance(spec, LabSpecification):  # pragma: no cover - defensive guard
            raise TypeError(f"{dotted}.LAB_SPEC must be a LabSpecification instance")
        specs.append(spec)
    return ensure_all_specs(specs)
