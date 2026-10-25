"""Punto de extensión del CLI de inferencia sobre un bundle."""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

from model_packaging.artifact import (
    DEFAULT_BUNDLE_PATH,
    load_model_bundle,
    infer_wine_quality,
)
from model_packaging.contracts import WineQualityRequest
from model_packaging.preprocess import FEATURE_NAMES


def predict_file(
    input_path: Path,
    output_path: Path,
    bundle_path: Path,
) -> int:
    """TODO: valida todas las filas y escribe el CSV solo al final."""

    bundle = load_model_bundle(bundle_path)
    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected_columns = ["sample_id", *FEATURE_NAMES]
        if reader.fieldnames is None or set(reader.fieldnames) != set(expected_columns):
            raise ValueError("Las columnas del CSV no coinciden con las esperadas")
        rows = list(reader)

    results = []

    for row in rows:
        sample_id = row["sample_id"]
        wine_data = {
            feature: row[feature]
            for feature in FEATURE_NAMES
        }
        request = WineQualityRequest.model_validate(wine_data)
        prediction = infer_wine_quality(bundle, request)
        results.append({
            "sample_id": sample_id,
            **prediction.model_dump(),
        })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "sample_id",
        "quality_band",
        "confidence",
        "model_version",
        "preprocessing_version",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(results)
    return len(results)


def parse_args() -> argparse.Namespace:
    """Declara la interfaz del comando público."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE_PATH, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(
        predict_file(
            input_path=arguments.input,
            output_path=arguments.output,
            bundle_path=arguments.bundle,
        )
    )
