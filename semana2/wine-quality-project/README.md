# Semana 2 — Clase 2

- **Clase 2 — [Assignment 2.2](../assignments/semana02_clase02_assignment.pdf):** reorganizar el caso Wine de S1, comprobarlo y trabajar con una rama, `push` y pull request contra `main` del fork de la pareja.

1. En primer lugar he creado una rama llamada "feature/s2-wine-project" sobre la que voy a realizar el ejercicio de esta semana.

2. uv init --package --vcs none --name wine-quality semana2/wine-quality-project , con este comando creamos la carpeta semana2/wine-quality-project , y con el comando cd, nos metemos en dicha carpeta. Una vez dentro, ejecutamos 2 comandos para importar las dependencias pandas y scikit-learn.

3. Copiamos archivos en nuevas carpetas para trabajar sobre ellas:

    mkdir -p data/raw tests
    cp ../starter/WineQT.csv data/raw/WineQT.csv
    cp ../starter/train.py src/wine_quality/train.py
    cp ../starter/test_train.py tests/test_train.py

4. Sincronizamos el entorno virtual "uv sync --locked", y ejecutamos cada modulo

    uv run --frozen python -m wine_quality.train
    uv run --frozen pytest
    uv run --frozen ruff check .

5. Por ultimo, hacemos un push a github, y listo!