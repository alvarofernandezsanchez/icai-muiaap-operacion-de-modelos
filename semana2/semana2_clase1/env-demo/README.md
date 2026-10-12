# env-demo

Proyecto para ejecutar una misma base de código en los entornos DEV, PRE y PRO.

## Instalación

```bash
uv sync

## Para ejecución

APP_ENV=dev uv run python -m env_demo.main
APP_ENV=pre uv run python -m env_demo.main
APP_ENV=pro uv run python -m env_demo.main

## Comprobación

uv run pytest
uv run ruff check .