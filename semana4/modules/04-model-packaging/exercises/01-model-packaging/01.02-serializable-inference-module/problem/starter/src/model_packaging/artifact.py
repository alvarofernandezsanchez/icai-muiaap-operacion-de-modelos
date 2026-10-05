"""Puntos de extensión del taller de serialización."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from envs.entorno_muiaap.Lib import json
import joblib
from pydantic import BaseModel, ConfigDict, Field
from src.model_packaging.preprocess import (PREPROCESSING_VERSION, FEATURE_NAMES)

from model_packaging.contracts import (
    QualityBand,
    WineQualityPrediction,
    WineQualityRequest,
)

import joblib

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


#el decorador permite ejecutar una funcion dentro de otra funcion de python
#con @overload, permite sobrecargar una funcion con otra

class ArtifactManifest(BaseModel):#pydantic; si no se pasase el basemodel seria clase de python normal
    """TODO: declara y valida los metadatos del bundle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str
    model_version: str = Field(min_length=1)
    preprocessing_version: str
    feature_names: tuple[str, ...]
    output_labels: tuple[QualityBand, ...]
    estimator_type: str = Field(min_length=1)

    @field_validator("schema_version") #cuando se haga un objeto de artifactmanifest, se va a activar y va a validar toda la logica
    @classmethod
    def validate_schema_version(cls, value: str) -> str:
        """
        Impide crear el manifest si la version es invalida
        """
        if value != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(
                f"Invalid schema version: {value}. Expected: {ARTIFACT_SCHEMA_VERSION}"
            )
        return value

@dataclass(frozen=True)
class LoadedModelBundle:
    """Bundle cargado; no modificar esta interfaz pública."""

    estimator: WineQualityEstimator #tiene que recibir este
    manifest: ArtifactManifest #y este


def create_manifest( #funcion que, en base a unos parametros que recibe, DEVUELVE un artifact manifest
    estimator: WineQualityEstimator, model_version: str
) -> ArtifactManifest: #ver lo que la clase ArtifactManifest lleva, para saber lo que tiene que recibir.
    """TODO: devuelve un manifiesto compatible con el contrato."""

#En

    return ArtifactManifest( #todo esto es lo del manifest del 30 de septiembre 
        schema_version=ARTIFACT_SCHEMA_VERSION, #estan en mayus, pq son variables ya definidas previamente.
        model_version=model_version, #model version se tiene que pasar cuando 
        preprocessing_version=PREPROCESSING_VERSION, #esta variable esta definida en preprocess.py
        feature_names=FEATURE_NAMES, #esta variable esta definida en preprocess.py
        output_labels=OUTPUT_LABELS, #esta variable esta definida en preprocess.py
        estimator_type=type(estimator).__name__, #que hace el __name__? devuelve el nombre de la clase del objeto que se le pasa. En este caso, el nombre de la clase del estimador que se le pasa a la funcion create_manifest
    )


def save_model_bundle(
    bundle_path: Path,
    estimator: WineQualityEstimator,
    manifest: ArtifactManifest | None = None,
) -> ArtifactManifest:
    """TODO: escribe manifest.json y model.joblib de forma segura."""

    if manifest is not None:
        #manifest_json = json.load(ArtifactManifest) ->esto se quita porque es redundante, y me va a dar error

        with open(Path(DEFAULT_BUNDLE_PATH) / "manifest.json", "w") as m: #con el with open, le paso un path, y le estoy diciendo que en mi pc entre en mi carpeta
            json.dump(manifest_json, m, indent=4, ensure_ascii=False) #busque ese path + archivo, y en este caso "w" que es escribir en el
    else:   #y aqui se vuelca todo a un json
        raise ValueError("Manifest is None. Please provide a valid manifest.")

def load_model_bundle(bundle_path: Path) -> LoadedModelBundle:
    """TODO: valida el manifiesto antes de cargar el estimador."""

    raise NotImplementedError("Implementa load_model_bundle().")


def infer_wine_quality(
    bundle: LoadedModelBundle,
    request: WineQualityRequest,
) -> WineQualityPrediction:
    """TODO: preprocesa, invoca el estimador y valida la respuesta."""

    raise NotImplementedError("Implementa infer_wine_quality().")
