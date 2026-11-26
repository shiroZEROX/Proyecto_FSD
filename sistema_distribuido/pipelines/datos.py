"""Carga de los datos ya preparados (objetos.csv, recortes.npy) y partición por escena.

Formato esperado en DIR_DATOS (lo producen datos/generar_sinteticos.py y datos/preparar_nuscenes.py):
  version.txt         una línea con la versión de los datos (entra en el job_id)
  particion.json      {"train": [escenas], "val": [escenas], "test": [escenas]}
  objetos.csv         escena, categoria, clase, ancho, largo, alto, distancia, visibilidad,
                      num_lidar, num_radar, velocidad
  recortes.npy        uint8 de forma (N, 64, 64, 3)
  recortes.csv        indice, escena, clase   (una fila por recorte, en el mismo orden)
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np

from sistema.comun import CLASES

COLUMNAS_NUMERICAS = ["ancho", "largo", "alto", "distancia", "visibilidad"]
PARTES = ("train", "val", "test")


def leer_version(dir_datos: str) -> str:
    with open(os.path.join(dir_datos, "version.txt")) as f:
        return f.read().strip()


def leer_particion(dir_datos: str) -> dict[str, set[str]]:
    with open(os.path.join(dir_datos, "particion.json")) as f:
        p = json.load(f)
    escenas = {k: set(p[k]) for k in PARTES}
    if escenas["train"] & escenas["val"] or escenas["train"] & escenas["test"] or escenas["val"] & escenas["test"]:
        raise ValueError("particion.json: una escena aparece en más de una partición")
    return escenas


def _leer_csv(ruta):
    with open(ruta, newline="") as f:
        return list(csv.DictReader(f))


def cargar_tabular(dir_datos: str, objetivo: str):
    """Devuelve {parte: (X, y)} con X = numéricas + one-hot de la clase agrupada."""
    particion = leer_particion(dir_datos)
    filas = _leer_csv(os.path.join(dir_datos, "objetos.csv"))
    salida = {}
    for parte in PARTES:
        sel = [r for r in filas if r["escena"] in particion[parte]]
        if not sel:
            raise ValueError(f"la partición '{parte}' no tiene objetos")
        num = np.array([[float(r[c]) for c in COLUMNAS_NUMERICAS] for r in sel], dtype=np.float64)
        onehot = np.array([[1.0 if r["clase"] == c else 0.0 for c in CLASES] for r in sel], dtype=np.float64)
        X = np.hstack([num, onehot])
        y = np.array([float(r[objetivo]) for r in sel], dtype=np.float64)
        salida[parte] = (X, y)
    return salida


def cargar_imagenes(dir_datos: str):
    """Devuelve {parte: (imagenes uint8 (n,64,64,3), etiquetas int)}."""
    particion = leer_particion(dir_datos)
    imgs = np.load(os.path.join(dir_datos, "recortes.npy"), mmap_mode="r")
    meta = _leer_csv(os.path.join(dir_datos, "recortes.csv"))
    if len(meta) != imgs.shape[0]:
        raise ValueError("recortes.csv y recortes.npy tienen distinto número de filas")
    indice_clase = {c: i for i, c in enumerate(CLASES)}
    salida = {}
    for parte in PARTES:
        idx = [int(r["indice"]) for r in meta if r["escena"] in particion[parte]]
        etiquetas = np.array([indice_clase[meta[i]["clase"]] for i in idx], dtype=np.int64)
        salida[parte] = (np.asarray(imgs[idx]), etiquetas)
    return salida


def metricas_regresion(y_real, y_pred) -> dict:
    y_real = np.asarray(y_real, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((y_real - y_pred) ** 2)))
    ss_res = float(np.sum((y_real - y_pred) ** 2))
    ss_tot = float(np.sum((y_real - y_real.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"rmse": round(rmse, 6), "r2": round(r2, 6), "n": int(len(y_real))}


def metricas_clasificacion(y_real, y_pred) -> dict:
    k = len(CLASES)
    matriz = np.zeros((k, k), dtype=int)
    for r, p in zip(y_real, y_pred):
        matriz[int(r), int(p)] += 1
    exactitud = float(np.trace(matriz) / max(1, matriz.sum()))
    return {"exactitud": round(exactitud, 6), "matriz_confusion": matriz.tolist(),
            "clases": list(CLASES), "n": int(len(y_real))}
