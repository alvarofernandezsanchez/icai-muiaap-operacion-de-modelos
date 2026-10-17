"""TODO: script CLI que encadena contratos, preprocesado e inferencia."""

# Implementa el comando:
# python -m model_inference.predict_file --input <csv> --output <csv>
# No dejes un archivo de salida parcial si alguna fila es inválida.
import argparse
import csv
import sys
from pathlib import Path

from model_inference.contracts import (
    WineQualityPrediction,
    WineQualityRequest,
)
from model_inference.inference import (
    DEFAULT_MODEL_PATH,
    infer_wine_quality,
    load_wine_quality_model,
)
from model_inference.preprocess import (
    PREPROCESSING_VERSION,
    preprocess_wine_request,
)

OUTPUT_FIELDS = (
    "sample_id",
    "quality_band",
    "confidence",
    "model_version",
    "preprocessing_version",
)


def predict_file(
    input_path: Path,
    output_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> None:
    model = load_wine_quality_model(model_path)
    predictions = []

    with input_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)

        if not reader.fieldnames or "sample_id" not in reader.fieldnames:
            raise ValueError("El CSV debe incluir la columna sample_id.")

        for row_number, row in enumerate(reader, start=2):
            sample_id = row.pop("sample_id", None)

            if sample_id is None or not sample_id.strip():
                raise ValueError(f"Fila {row_number}: sample_id está vacío.")

            try:
                request = WineQualityRequest.model_validate(row)
                features = preprocess_wine_request(request)
                quality_band, confidence = infer_wine_quality(model, features)

                prediction = WineQualityPrediction(
                    quality_band=quality_band,
                    confidence=confidence,
                    model_version=model["model_version"],
                    preprocessing_version=PREPROCESSING_VERSION,
                )
            except Exception as error:
                raise ValueError(
                    f"Fila {row_number} (sample_id={sample_id}): {error}"
                ) from error

            predictions.append({"sample_id": sample_id, **prediction.model_dump()})

    # Solo crea la salida cuando todas las filas han pasado la validación.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(predictions)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()

    try:
        predict_file(args.input, args.output, args.model)
    except Exception as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
