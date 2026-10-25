"""Puntos de extensión del taller de serialización."""

from __future__ import annotations

import json
import joblib

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from model_packaging.contracts import (
    QualityBand,
    WineQualityPrediction,
    WineQualityRequest,
)

from model_packaging.preprocess import (
    FEATURE_NAMES,
    PREPROCESSING_VERSION,
    preprocess_wine_request,
)

ARTIFACT_SCHEMA_VERSION = "wine-quality-bundle-v1"
DEFAULT_BUNDLE_PATH = Path("models/wine_quality_bundle")
MANIFEST_FILENAME = "manifest.json"
MODEL_FILENAME = "model.joblib"
OUTPUT_LABELS: tuple[QualityBand, ...] = (
    "needs_review",
    "acceptable",
    "excellent",
)


class WineQualityEstimator(Protocol):
    """Interfaz mínima que debe cumplir el estimador cargado."""

    def predict(self, features: list[list[float]]) -> Sequence[str]:
        """Devuelve una etiqueta por fila."""

    def predict_proba(self, features: list[list[float]]) -> Sequence[Sequence[float]]:
        """Devuelve probabilidades por fila."""


class ArtifactManifest(BaseModel):
    """Metadatos y contrato de compatibilidad del modelo."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["wine-quality-bundle-v1"]
    model_version: str = Field(min_length=1)
    preprocessing_version: str
    feature_names: tuple[str, ...]
    output_labels: tuple[QualityBand, ...]
    estimator_type: str = Field(min_length=1)


@dataclass(frozen=True)
class LoadedModelBundle:
    """Bundle cargado; no modificar esta interfaz pública."""

    estimator: WineQualityEstimator
    manifest: ArtifactManifest


def create_manifest(
    estimator: WineQualityEstimator, model_version: str
) -> ArtifactManifest:
    """TODO: devuelve un manifiesto compatible con el contrato."""

    return ArtifactManifest(
            schema_version=ARTIFACT_SCHEMA_VERSION,
            model_version=model_version,
            preprocessing_version=PREPROCESSING_VERSION,
            feature_names=FEATURE_NAMES,
            output_labels=OUTPUT_LABELS,
            estimator_type=type(estimator).__name__,
        )


def save_model_bundle(
    bundle_path: Path,
    estimator: WineQualityEstimator,
    manifest: ArtifactManifest | None = None,
) -> ArtifactManifest:
    """TODO: escribe manifest.json y model.joblib."""

    #Comprobar que el estimador tiene los métodos necesarios
    if not callable(getattr(estimator, "predict", None)):
        raise ValueError("El estimador no tiene predict()")

    if not callable(getattr(estimator, "predict_proba", None)):
        raise ValueError("El estimador no tiene predict_proba()")
    if manifest is None:
        manifest = create_manifest(estimator, "wine-quality-rf-v1")

    #Crear carpeta
    bundle_path.mkdir(parents=True, exist_ok=True)

    #Guardar los metadatos como JSON
    with open(bundle_path / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=4, ensure_ascii=False)
    joblib.dump(estimator, bundle_path / MODEL_FILENAME)

    return manifest


def load_model_bundle(bundle_path: Path) -> LoadedModelBundle:
    """Valida el manifiesto antes de cargar el estimador."""

    manifest_path = bundle_path / MANIFEST_FILENAME
    model_path = bundle_path / MODEL_FILENAME

    if not manifest_path.is_file():
        raise FileNotFoundError("No existe manifest.json")
    if not model_path.is_file():
        raise FileNotFoundError("No existe model.joblib")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_json = json.load(f)
    manifest = ArtifactManifest.model_validate(manifest_json)

    if manifest.schema_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Versión del esquema incompatible")
    if manifest.preprocessing_version != PREPROCESSING_VERSION:
        raise ValueError("Versión del preprocesamiento incompatible")
    if manifest.feature_names != FEATURE_NAMES:
        raise ValueError("Características incompatibles o desordenadas")
    if manifest.output_labels != OUTPUT_LABELS:
        raise ValueError("Etiquetas de salida incompatibles")

    estimator = joblib.load(model_path)

    if not callable(getattr(estimator, "predict", None)):
        raise ValueError("El estimador no tiene predict()")
    if not callable(getattr(estimator, "predict_proba", None)):
        raise ValueError("El estimador no tiene predict_proba()")
    if type(estimator).__name__ != manifest.estimator_type:
        raise ValueError("Tipo de estimador incompatible")
    return LoadedModelBundle(
        estimator=estimator,
        manifest=manifest,
    )


def infer_wine_quality(
    bundle: LoadedModelBundle,
    request: WineQualityRequest,
) -> WineQualityPrediction:
    """TODO: preprocesa, invoca el estimador y valida la respuesta."""
    
    features = preprocess_wine_request(request).as_vector()

    predictions = bundle.estimator.predict([features])
    probabilities = bundle.estimator.predict_proba([features])

    if len(predictions) != 1 or len(probabilities) != 1:
        raise ValueError("El modelo debe devolver una predicción por fila")

    quality_band = str(predictions[0])

    if quality_band not in OUTPUT_LABELS:
        raise ValueError("El modelo ha devuelto una etiqueta no permitida")

    classes = list(bundle.estimator.classes_)

    if quality_band not in classes:
        raise ValueError("La etiqueta predicha no existe en las clases del modelo")
    class_index = classes.index(quality_band)
    row_probabilities = probabilities[0]

    if len(row_probabilities) != len(classes):
        raise ValueError("Número de probabilidades incompatible")
    confidence = float(row_probabilities[class_index])

    return WineQualityPrediction(
        quality_band=quality_band,
        confidence=confidence,
        model_version=bundle.manifest.model_version,
        preprocessing_version=bundle.manifest.preprocessing_version,
    )
