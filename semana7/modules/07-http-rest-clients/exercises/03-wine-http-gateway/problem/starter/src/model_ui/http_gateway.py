"""Cliente HTTP que adapta la API Wine al contrato de inferencia de la UI."""

from __future__ import annotations

from collections.abc import Mapping
from threading import local
from typing import Any

import requests
from pydantic import ValidationError  # noqa: F401 - pista para el TODO S7.4

from model_ui.contracts import PredictionPayload
from model_ui.errors import (  # noqa: F401 - pistas para el TODO S7.2
    ArtifactUnavailableError,
    BackendInferenceError,
    InferenceTimeoutError,
    InputContractError,
)


class HttpInferenceGateway:
    """Implementa la frontera de inferencia mediante una llamada HTTP REST."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 5.0,
        session: Any | None = None,
    ) -> None:
        normalized = base_url.strip().rstrip("/")
        if not normalized:
            raise ValueError("base_url no puede estar vacia")
        if timeout <= 0:
            raise ValueError("timeout debe ser positivo")

        self._prediction_url = f"{normalized}/v1/predictions"
        self._timeout = float(timeout)
        self._session = session if session is not None else requests.Session()
        self._request_context = local()
        self.last_request_id: str | None = None

    @property
    def last_request_id(self) -> str | None:
        """Devuelve el identificador de la ultima llamada en el contexto actual."""

        return getattr(self._request_context, "request_id", None)

    @last_request_id.setter
    def last_request_id(self, value: str | None) -> None:
        self._request_context.request_id = value

    def predict(self, values: Mapping[str, object]) -> PredictionPayload:
        """Envia la muestra por HTTP y devuelve el contrato interno de S6."""

        # TODO S7.1: envia POST a self._prediction_url con json={"features": ...},
        # Accept application/json y self._timeout. Compruebalo con el test de llamada.
        self.last_request_id = None

        try:
            response = self._session.post(
                self._prediction_url,
                json={"features": dict(values)},
                headers={"Accept": "application/json"},
                timeout=self._timeout,
            )

        # TODO S7.2: traduce Timeout, ConnectionError y RequestException a los
        # errores de dominio que ya entiende PredictionController.
        except requests.Timeout as error:
            raise InferenceTimeoutError(
                "La inferencia ha superado el tiempo permitido"
            ) from error
        except requests.ConnectionError as error:
            raise ArtifactUnavailableError(
                "No se pudo conectar con el backend"
            ) from error
        except requests.RequestException as error:
            raise BackendInferenceError(
                "No se pudo completar la solicitud HTTP"
            ) from error

        # TODO S7.3: lee un objeto JSON, conserva request_id y diferencia 422/503
        # del resto de errores HTTP.
        self.last_request_id = self._header_request_id(response)
        status = response.status_code

        if status == 422:
            raise InputContractError("La entrada no cumple el contrato")
        if status == 503:
            raise ArtifactUnavailableError("El backend no esta disponible")
        if not 200 <= status < 300:
            raise BackendInferenceError(f"Error HTTP {status}")

        body = self._read_object(response)
        self.last_request_id = self._request_id(response, body)

        # TODO S7.4: valida body["prediction"] como PredictionPayload; un dict sin
        # validar no satisface el contrato de salida.
        try:
            return PredictionPayload.model_validate(body["prediction"])
        except (KeyError, TypeError, ValueError, ValidationError) as error:
            raise BackendInferenceError(
                "La respuesta del backend no cumple el contrato"
            ) from error

    @staticmethod
    def _read_object(response: Any) -> dict[str, Any]:
        """Lee una respuesta JSON que debe representar un objeto."""

        # TODO S7.3: usa response.json(), traduce ValueError y rechaza listas u
        # otros valores que no sean un objeto JSON.
        try:
            body = response.json()
        except ValueError as error:
            raise BackendInferenceError(
                "El backend ha devuelto un JSON no valido"
            ) from error

        if not isinstance(body, dict):
            raise BackendInferenceError("La respuesta JSON debe ser un objeto")

        return body

    @staticmethod
    def _request_id(response: Any, body: dict[str, Any]) -> str | None:
        """Obtiene request_id del cuerpo y usa la cabecera como alternativa."""

        # TODO S7.3: prioriza body["request_id"] y usa X-Request-Id como fallback.
        request_id = body.get("request_id")
        if isinstance(request_id, str) and request_id:
            return request_id
        return HttpInferenceGateway._header_request_id(response)

    @staticmethod
    def _header_request_id(response: Any) -> str | None:
        headers = getattr(response, "headers", {})
        return headers.get("X-Request-Id")
