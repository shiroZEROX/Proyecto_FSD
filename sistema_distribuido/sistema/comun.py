"""Piezas compartidas por coordinador, nodos y cliente.

No importa torch ni scikit-learn: el coordinador y el cliente no los necesitan.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import random
import shutil
import tempfile
import time

import grpc

MODELOS = ("lineal", "mlp", "cnn")
SEMILLA_DEFECTO = 42
CLASES = ("vehiculo", "peaton", "otro")  # agrupación de categorías (SUPOSICIÓN, ver README)
OBJETIVOS = ("num_lidar", "num_radar", "velocidad")

# Peso relativo de cada modelo: se usa para ordenar la cola (pesados primero).
PESO_MODELO = {"cnn": 3, "mlp": 2, "lineal": 1}

MAX_MENSAJE = 64 * 1024 * 1024
OPCIONES_GRPC = [
    ("grpc.max_send_message_length", MAX_MENSAJE),
    ("grpc.max_receive_message_length", MAX_MENSAJE),
]


class ConfigInvalida(ValueError):
    """Error de configuración: no tiene caso reintentarlo."""


# ---------------------------------------------------------------- esquemas
# Cada hiperparámetro: (tipo, valor por defecto, validador opcional).
# Normalizar tipos importa para la idempotencia: alpha=1 y alpha=1.0 deben dar el mismo job_id.

def _pos(x):
    return x > 0


ESQUEMAS = {
    "lineal": {
        "tipo": (str, "ridge", lambda v: v in ("lineal", "ridge", "lasso")),
        "alpha": (float, 1.0, lambda v: v >= 0),
        "normalizar": (bool, True, None),
        "objetivo": (str, "num_lidar", lambda v: v in OBJETIVOS),
        "log_objetivo": (bool, False, None),
    },
    "mlp": {
        "capas": (list, [64, 32], lambda v: len(v) > 0 and all(isinstance(n, int) and n > 0 for n in v)),
        "lr": (float, 1e-3, _pos),
        "epocas": (int, 30, _pos),
        "batch": (int, 64, _pos),
        "objetivo": (str, "num_lidar", lambda v: v in OBJETIVOS),
        "log_objetivo": (bool, False, None),
    },
    "cnn": {
        "filtros": (int, 16, _pos),
        "capas": (int, 2, lambda v: 1 <= v <= 4),
        "lr": (float, 1e-3, _pos),
        "epocas": (int, 10, _pos),
        "batch": (int, 64, _pos),
        "tam_imagen": (int, 32, lambda v: 8 <= v <= 64),
    },
}


def _convertir(nombre, tipo, valor):
    if tipo is bool:
        if not isinstance(valor, bool):
            raise ConfigInvalida(f"{nombre} debe ser true/false")
        return valor
    if tipo is float:
        if isinstance(valor, bool) or not isinstance(valor, (int, float)):
            raise ConfigInvalida(f"{nombre} debe ser numérico")
        return float(valor)
    if tipo is int:
        if isinstance(valor, bool) or not isinstance(valor, (int, float)) or int(valor) != valor:
            raise ConfigInvalida(f"{nombre} debe ser entero")
        return int(valor)
    if tipo is list:
        if not isinstance(valor, list):
            raise ConfigInvalida(f"{nombre} debe ser una lista")
        return [int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and int(v) == v else v for v in valor]
    if not isinstance(valor, tipo):
        raise ConfigInvalida(f"{nombre} debe ser {tipo.__name__}")
    return valor


def normalizar_config(modelo: str, config: dict) -> dict:
    """Valida y completa la configuración. Lanza ConfigInvalida."""
    if modelo not in ESQUEMAS:
        raise ConfigInvalida(f"modelo '{modelo}' no soportado; use uno de {', '.join(MODELOS)}")
    if not isinstance(config, dict):
        raise ConfigInvalida("la configuración debe ser un objeto JSON")
    esquema = ESQUEMAS[modelo]
    desconocidas = set(config) - set(esquema)
    if desconocidas:
        raise ConfigInvalida(f"hiperparámetros desconocidos para {modelo}: {sorted(desconocidas)}")
    salida = {}
    for nombre, (tipo, defecto, valida) in esquema.items():
        valor = _convertir(nombre, tipo, config.get(nombre, defecto))
        if valida and not valida(valor):
            raise ConfigInvalida(f"valor inválido para {nombre}: {valor!r}")
        salida[nombre] = valor
    return salida


def expandir_grid(modelo: str, base: dict, grid: dict | None) -> list[dict]:
    """Producto cartesiano de las listas del grid sobre la configuración base."""
    if not grid:
        return [normalizar_config(modelo, base)]
    if not isinstance(grid, dict) or not all(isinstance(v, list) and v for v in grid.values()):
        raise ConfigInvalida("grid debe ser un objeto {hiperparámetro: [valores...]}")
    claves = sorted(grid)
    configs = []
    for combinacion in itertools.product(*(grid[c] for c in claves)):
        cfg = dict(base)
        cfg.update(zip(claves, combinacion))
        configs.append(normalizar_config(modelo, cfg))
    return configs


def json_canonico(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def calcular_job_id(modelo: str, config: dict, version_datos: str, semilla: int) -> str:
    """job_id determinista: mismo modelo + config + datos + semilla => mismo id."""
    contenido = json_canonico({"modelo": modelo, "config": config,
                               "version_datos": version_datos, "semilla": int(semilla)})
    return hashlib.sha256(contenido.encode()).hexdigest()[:16]


def calcular_peticion_id(job_ids: list[str]) -> str:
    return "p-" + hashlib.sha256(",".join(sorted(job_ids)).encode()).hexdigest()[:12]


# ---------------------------------------------------------------- escritura atómica

def escribir_atomico(ruta: str, datos: bytes | str) -> None:
    """Escribe en un temporal del mismo directorio y renombra (os.replace es atómico en POSIX)."""
    directorio = os.path.dirname(os.path.abspath(ruta))
    os.makedirs(directorio, exist_ok=True)
    modo = "wb" if isinstance(datos, (bytes, bytearray)) else "w"
    fd, tmp = tempfile.mkstemp(dir=directorio, prefix=".tmp-")
    try:
        with os.fdopen(fd, modo) as f:
            f.write(datos)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, ruta)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def escribir_directorio_atomico(destino: str, archivos: dict[str, bytes | str]) -> bool:
    """Escribe varios archivos en un directorio temporal y lo renombra al final.

    Si el destino ya existe (otro intento del mismo job_id ya escribió) no se toca y
    se devuelve False: el primer resultado completo es el que vale.
    """
    padre = os.path.dirname(os.path.abspath(destino))
    os.makedirs(padre, exist_ok=True)
    if os.path.isdir(destino):
        return False
    tmp = tempfile.mkdtemp(dir=padre, prefix=".tmp-")
    try:
        for nombre, datos in archivos.items():
            modo = "wb" if isinstance(datos, (bytes, bytearray)) else "w"
            with open(os.path.join(tmp, nombre), modo) as f:
                f.write(datos)
                f.flush()
                os.fsync(f.fileno())
        try:
            os.rename(tmp, destino)  # falla si destino ya existe y no está vacío
        except OSError:
            shutil.rmtree(tmp, ignore_errors=True)
            return False
        return True
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


# ---------------------------------------------------------------- retry con backoff

CODIGOS_REINTENTABLES = {
    grpc.StatusCode.UNAVAILABLE,
    grpc.StatusCode.DEADLINE_EXCEEDED,
    grpc.StatusCode.RESOURCE_EXHAUSTED,
    grpc.StatusCode.ABORTED,
}


def esperar_backoff(intento: int, base: float = 0.5, maximo: float = 10.0) -> float:
    """Backoff exponencial con 'full jitter': espera aleatoria en [0, min(maximo, base*2^intento)]."""
    espera = random.uniform(0, min(maximo, base * (2 ** intento)))
    time.sleep(espera)
    return espera


def llamar(metodo, peticion, plazo: float = 5.0, intentos: int = 5, base: float = 0.5,
           maximo: float = 10.0, registro=None):
    """Llama un RPC con plazo por llamada y reintentos solo para fallas pasajeras.

    Es seguro reintentar porque todas las operaciones del coordinador son idempotentes.
    """
    for intento in range(intentos):
        try:
            return metodo(peticion, timeout=plazo)
        except grpc.RpcError as e:
            if e.code() not in CODIGOS_REINTENTABLES or intento == intentos - 1:
                raise
            espera = esperar_backoff(intento, base, maximo)
            if registro:
                registro(f"reintento {intento + 1}/{intentos - 1} tras {e.code().name}; esperé {espera:.2f}s")


def entero_env(nombre: str, defecto: int) -> int:
    return int(os.environ.get(nombre, defecto))


def real_env(nombre: str, defecto: float) -> float:
    return float(os.environ.get(nombre, defecto))
