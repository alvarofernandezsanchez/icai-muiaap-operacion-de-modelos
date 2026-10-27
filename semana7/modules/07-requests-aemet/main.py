import os
import requests

API_KEY = os.getenv("AEMET_API_KEY")

if not API_KEY:
    raise ValueError("No se ha configurado AEMET_API_KEY")

url = "https://opendata.aemet.es/opendata/api/valores/climatologicos/inventarioestaciones/todasestaciones"

params = {
    "api_key": API_KEY
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

print("Código HTTP:", response.status_code)

datos_url = response.json()["datos"]

response_datos = requests.get(datos_url, timeout=30)
response_datos.raise_for_status()

estaciones = response_datos.json()

idema = None

for estacion in estaciones:
    if estacion["nombre"] == "MADRID, CIUDAD UNIVERSITARIA":
        idema = estacion["indicativo"]
        break

if idema is None:
    raise ValueError("No se ha encontrado la estación")

print("Indicativo encontrado:", idema)

fecha_inicio = "2019-10-01T00:00:00UTC"
fecha_fin = "2019-10-30T23:59:59UTC"

url_clima = (
    "https://opendata.aemet.es/opendata/api/"
    "valores/climatologicos/diarios/datos/"
    f"fechaini/{fecha_inicio}/"
    f"fechafin/{fecha_fin}/"
    f"estacion/{idema}"
)

respuesta_clima = requests.get(
    url_clima,
    params={"api_key": API_KEY},
    timeout=30
)

respuesta_clima.raise_for_status()

print("Estado de la petición:", respuesta_clima.status_code)
url_datos_clima = respuesta_clima.json()["datos"]

respuesta_datos_clima = requests.get(
    url_datos_clima,
    timeout=30
)

respuesta_datos_clima.raise_for_status()
climatologia = respuesta_datos_clima.json()

for dia in climatologia:
    print(dia)

import json

with open("climatologia_madrid_2019.json", "w", encoding="utf-8") as archivo:
    json.dump(climatologia, archivo, indent=4, ensure_ascii=False)

print("Datos guardados correctamente")