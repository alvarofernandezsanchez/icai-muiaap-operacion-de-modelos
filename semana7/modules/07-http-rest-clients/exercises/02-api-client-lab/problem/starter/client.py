"""Starter del cliente HTTP. Completa los TODO sin modificar la API."""

from __future__ import annotations

from typing import Any

# El TODO de la práctica usará este cliente para realizar la petición HTTP.
import requests  # noqa: F401


class PredictionClientError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        self.status_code = status_code
        super().__init__(message)


def request_prediction(
    payload: dict[str, Any],
    *,
    base_url: str = "http://127.0.0.1:8000",
    timeout: float = 5.0,
    session: Any | None = None,
) -> dict[str, Any]:
    """Envía una predicción y devuelve un objeto JSON de respuesta."""
    # TODO 3: selecciona `session` o `requests` y envía POST con `json=`,
    # `Accept` y `timeout`. Comprueba los argumentos con test_client.py.
    cliente = session if session is not None else requests

    url = f"{base_url.rstrip('/')}/v1/predictions"

    try:
        response = cliente.post(
            url,
            json=payload,
            headers={"Accept": "application/json"},
            timeout=timeout,
        )
    # TODO 4: llama a `raise_for_status()` y conserva el status de HTTPError.
        response.raise_for_status()

    except requests.exceptions.HTTPError as error:
        status = (
            error.response.status_code
            if error.response is not None
            else None
        )
        raise PredictionClientError(
            "Error HTTP al solicitar la predicción",
            status_code=status,
        ) from error
    
    # TODO 5: traduce Timeout y ConnectionError a PredictionClientError.
    except requests.exceptions.Timeout as error:
        raise PredictionClientError(
            "Tiempo de espera agotado"
        ) from error

    except requests.exceptions.ConnectionError as error:
        raise PredictionClientError(
            "No se pudo conectar con la API"
        ) from error
    # TODO 6: convierte la respuesta con `.json()` y exige un diccionario.
    try:
        resultado = response.json()
    except ValueError as error:
        raise PredictionClientError(
            "La respuesta no contiene un JSON válido"
        ) from error

    if not isinstance(resultado, dict):
        raise PredictionClientError(
            "La respuesta JSON debe ser un objeto"
        )

    return resultado
