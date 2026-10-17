"""TODO: carga del .joblib e inferencia sobre características preparadas."""

# Declara DEFAULT_MODEL_PATH, load_wine_quality_model() e infer_wine_quality().
# Comprueba las características del artefacto antes de llamar al clasificador.
from pathlib import Path
from typing import Any

import joblib

from model_inference.preprocess import FEATURE_NAMES, WineFeatures

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "wine_quality_classifier.joblib"
)


def load_wine_quality_model(
    path: Path = DEFAULT_MODEL_PATH,
) -> dict[str, Any]:
    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError("El artefacto debe ser un diccionario.")

    estimator = artifact.get("estimator")
    if not callable(getattr(estimator, "predict", None)) or not callable(
        getattr(estimator, "predict_proba", None)
    ):
        raise ValueError("El artefacto no contiene un clasificador compatible.")

    feature_names = artifact.get("feature_names")
    if tuple(feature_names or ()) != FEATURE_NAMES:
        raise ValueError("El campo feature_names no coincide con el orden esperado.")

    model_version = artifact.get("model_version")
    if not isinstance(model_version, str) or not model_version.strip():
        raise ValueError("El artefacto debe incluir model_version.")

    return artifact


def infer_wine_quality(
    model: dict[str, Any],
    features: WineFeatures,
) -> tuple[str, float]:
    vector = [features.as_vector()]
    estimator = model["estimator"]

    quality_band = estimator.predict(vector)[0]
    probabilities = estimator.predict_proba(vector)[0]
    confidence = max(float(probability) for probability in probabilities)

    return str(quality_band), confidence
