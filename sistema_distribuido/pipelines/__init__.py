"""Plantillas de pipeline. Todas tienen la misma interfaz:

    ejecutar(config, semilla, dir_datos, progreso=None, checkpoint=None) -> dict
        config      dict ya normalizado por sistema.comun.normalizar_config
        semilla     int; con la misma semilla se obtienen las mismas métricas
        dir_datos   carpeta con objetos.csv / recortes.npy (ver pipelines/datos.py)
        progreso    callback(epoca, perdida, checkpoint_bytes|None); puede lanzar Cancelado
        checkpoint  bytes de un checkpoint previo del mismo job_id para continuar desde ahí
    devuelve {"metricas": {...}, "modelo": bytes, "formato": "pickle"|"torch"}
"""
import importlib


class Cancelado(Exception):
    """El coordinador pidió abandonar el trabajo (fue reasignado o ya terminó en otro nodo)."""


def ejecutar(modelo, config, semilla, dir_datos, progreso=None, checkpoint=None):
    modulo = importlib.import_module(f"pipelines.{modelo}")
    return modulo.ejecutar(config, semilla, dir_datos, progreso=progreso, checkpoint=checkpoint)
