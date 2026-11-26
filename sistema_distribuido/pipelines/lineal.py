"""Regresión lineal (línea base): LinearRegression, Ridge o Lasso de scikit-learn."""
import pickle
import time

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipelines.datos import cargar_tabular, metricas_regresion


def ejecutar(config, semilla, dir_datos, progreso=None, checkpoint=None):
    t0 = time.time()
    datos = cargar_tabular(dir_datos, config["objetivo"])
    X, y = datos["train"]
    if config["tipo"] == "ridge":
        reg = Ridge(alpha=config["alpha"], random_state=semilla)
    elif config["tipo"] == "lasso":
        reg = Lasso(alpha=config["alpha"], random_state=semilla, max_iter=10000)
    else:
        reg = LinearRegression()
    pasos = [StandardScaler(), reg] if config["normalizar"] else [reg]
    modelo = make_pipeline(*pasos)
    log_y = config["log_objetivo"]
    modelo.fit(X, np.log1p(y) if log_y else y)

    def predecir(Xp):
        p = modelo.predict(Xp)
        return np.expm1(p) if log_y else p

    metricas = {parte: metricas_regresion(datos[parte][1], predecir(datos[parte][0]))
                for parte in ("train", "val", "test")}
    metricas["tiempo_entrenamiento"] = round(time.time() - t0, 3)
    if progreso:
        progreso(1, metricas["train"]["rmse"], None)
    return {"metricas": metricas, "modelo": pickle.dumps(modelo), "formato": "pickle"}
