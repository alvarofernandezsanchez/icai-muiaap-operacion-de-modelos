import joblib

from app.config import MODEL_PATH
from app.schemas.prestamo import PrediccionSalida
from app.schemas.prestamo import PrestamoEntrada, PrediccionSalida

# Orden en el que el modelo espera las variables
VARIABLES = [
    "edad",
    "ingresos_mensuales",
    "importe_prestamo",
    "plazo_meses",
    "antiguedad_laboral_anios",
    "impagos_previos",
]

# TODO: cargar el modelo joblib una sola vez, al importar el modulo
modelo = joblib.load(MODEL_PATH)
# TODO: funcion decidir(probabilidad) -> "aprobar", "revisar" o "rechazar" (umbrales en el enunciado)
def decidir(probabilidad: float) -> str:
    if probabilidad < 0.2:
        return "aprobar"
    elif probabilidad < 0.5:
        return "revisar"
    else:
        return "rechazar"
# TODO: funcion predecir(entrada) -> PrediccionSalida, usando VARIABLES para ordenar los valores
def predecir(entrada: PrestamoEntrada) -> PrediccionSalida:
    datos = entrada.model_dump()

    valores = [datos[variable] for variable in VARIABLES]

    probabilidad = float(modelo.predict_proba([valores])[0][1])

    return PrediccionSalida(
        probabilidad_impago=probabilidad,
        decision=decidir(probabilidad)
    )