"""Nodo worker: consumidor competidor con heartbeat, bulkhead por modelo y checkpoints.

Ejecutar:  python -m sistema.worker
Variables de entorno:
  COORDINADOR=localhost:50051   WORKER_ID=<hostname>   DIR_DATOS=./datos/generados
  DIR_TRABAJO=./trabajo_nodo    SLOTS=lineal=1,mlp=1,cnn=1   MAX_TRABAJOS=1
  HEARTBEAT_SEG=3   RETARDO_POR_EPOCA=0 (simula nodo lento)   HILOS_TORCH=2
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import traceback

import grpc

from sistema import comun
from sistema import entrenamiento_pb2 as pb
from sistema import entrenamiento_pb2_grpc as pbg

log = logging.getLogger("worker")
ARCHIVO_LISTO = os.environ.get("ARCHIVO_LISTO", "/tmp/nodo_listo")


def leer_slots(texto: str) -> dict[str, int]:
    slots = {}
    for parte in texto.split(","):
        modelo, _, n = parte.partition("=")
        if modelo.strip():
            slots[modelo.strip()] = int(n or 1)
    return slots


class Nodo:
    def __init__(self):
        self.id = os.environ.get("WORKER_ID") or socket.gethostname()
        self.dir_datos = os.environ.get("DIR_DATOS", "./datos/generados")
        self.dir_trabajo = os.path.join(os.environ.get("DIR_TRABAJO", "./trabajo_nodo"), self.id)
        self.slots = leer_slots(os.environ.get("SLOTS", "lineal=1,mlp=1,cnn=1"))
        self.max_trabajos = comun.entero_env("MAX_TRABAJOS", 1)
        self.intervalo = comun.entero_env("HEARTBEAT_SEG", 3)
        self.canal = grpc.insecure_channel(os.environ.get("COORDINADOR", "localhost:50051"),
                                           options=comun.OPCIONES_GRPC)
        self.stub = pbg.CoordinadorStub(self.canal)
        self.candado = threading.Lock()
        self.activos: dict[str, dict] = {}        # job_id -> {"asignacion", "cancelar": Event, "modelo"}
        self.alto = threading.Event()
        self.version_datos, self.listo = self._revisar_datos()

    def _revisar_datos(self):
        """Health check del nodo: está listo solo si tiene los datos y las bibliotecas."""
        try:
            from pipelines.datos import leer_particion, leer_version
            version = leer_version(self.dir_datos)
            leer_particion(self.dir_datos)
            for archivo in ("objetos.csv", "recortes.npy", "recortes.csv"):
                if not os.path.exists(os.path.join(self.dir_datos, archivo)):
                    raise FileNotFoundError(archivo)
            import sklearn  # noqa: F401
            import torch  # noqa: F401
            log.info("datos '%s' en %s; nodo listo", version, self.dir_datos)
            return version, True
        except Exception as e:  # el nodo sigue mandando heartbeat, pero como "no listo"
            log.error("nodo NO listo: %s", e)
            return "", False

    def registro(self, texto):
        log.warning(texto)

    # ------------------------------------------------------------ heartbeat
    def latir(self):
        while not self.alto.is_set():
            with self.candado:
                activos = [pb.TrabajoActivo(job_id=j, asignacion=a["asignacion"]) for j, a in self.activos.items()]
            try:
                r = comun.llamar(self.stub.Heartbeat,
                                 pb.Latido(worker_id=self.id, listo=self.listo, version_datos=self.version_datos,
                                           activos=activos), plazo=2.0, intentos=2)
                if self.listo:
                    open(ARCHIVO_LISTO, "w").close()
                with self.candado:
                    for job_id in r.cancelar:
                        if job_id in self.activos and not self.activos[job_id]["cancelar"].is_set():
                            log.warning("[%s] el coordinador pide cancelarlo", job_id)
                            self.activos[job_id]["cancelar"].set()
            except grpc.RpcError as e:
                log.warning("heartbeat falló: %s", e.code().name)
            self.alto.wait(self.intervalo)

    # ------------------------------------------------------------ bulkhead
    def modelos_libres(self) -> list[str]:
        with self.candado:
            if len(self.activos) >= self.max_trabajos:
                return []
            en_uso = {}
            for a in self.activos.values():
                en_uso[a["modelo"]] = en_uso.get(a["modelo"], 0) + 1
            return [m for m, n in self.slots.items() if en_uso.get(m, 0) < n]

    # ------------------------------------------------------------ ejecución
    def ejecutar(self, asig: pb.Asignacion):
        from pipelines import Cancelado, ejecutar
        job_id, cancelar = asig.job_id, self.activos[asig.job_id]["cancelar"]
        dir_resultado = os.path.join(self.dir_trabajo, "resultados", job_id)
        t0 = time.time()

        def progreso(epoca, perdida, ckpt):
            if cancelar.is_set():
                raise Cancelado()
            # El checkpoint se guarda localmente (atómico) y se manda al coordinador para
            # que otro nodo pueda continuar si este se cae.
            if ckpt:
                comun.escribir_atomico(os.path.join(self.dir_trabajo, "checkpoints", job_id + ".ckpt"), ckpt)
            try:
                r = comun.llamar(self.stub.ReportarAvance,
                                 pb.Avance(worker_id=self.id, job_id=job_id, asignacion=asig.asignacion,
                                           epoca=epoca, perdida=perdida, checkpoint=ckpt or b""),
                                 plazo=10.0, intentos=3, registro=self.registro)
                if r.cancelar:
                    raise Cancelado()
            except grpc.RpcError as e:
                log.warning("[%s] no pude reportar avance (%s); sigo entrenando", job_id, e.code().name)
            log.info("[%s] época %d pérdida %.5f", job_id, epoca, perdida)

        try:
            ckpt = asig.checkpoint or None
            log.info("[%s] empiezo %s %s%s", job_id, asig.modelo, asig.config_json,
                     " (desde checkpoint)" if ckpt else "")
            salida = ejecutar(asig.modelo, json.loads(asig.config_json), asig.semilla, self.dir_datos,
                              progreso=progreso, checkpoint=ckpt)
            duracion = time.time() - t0
            salida["metricas"].update(nodo=self.id, job_id=job_id, semilla=asig.semilla,
                                      version_datos=asig.version_datos)
            metricas_json = json.dumps(salida["metricas"], ensure_ascii=False)
            # Escritura atómica local: directorio temporal + rename.
            comun.escribir_directorio_atomico(dir_resultado, {"modelo.bin": salida["modelo"],
                                                             "metricas.json": metricas_json})
            peticion = pb.ResultadoTrabajo(worker_id=self.id, job_id=job_id, asignacion=asig.asignacion, ok=True,
                                           metricas_json=metricas_json, modelo=salida["modelo"],
                                           formato_modelo=salida["formato"], duracion=duracion)
        except Cancelado:
            log.warning("[%s] abandonado: fue reasignado o ya lo terminó otro nodo", job_id)
            peticion = None
        except comun.ConfigInvalida as e:
            peticion = pb.ResultadoTrabajo(worker_id=self.id, job_id=job_id, asignacion=asig.asignacion, ok=False,
                                           error=str(e), reintentable=False)
        except Exception as e:
            log.error("[%s] error:\n%s", job_id, traceback.format_exc())
            peticion = pb.ResultadoTrabajo(worker_id=self.id, job_id=job_id, asignacion=asig.asignacion, ok=False,
                                           error=f"{type(e).__name__}: {e}", reintentable=True)
        finally:
            with self.candado:
                self.activos.pop(job_id, None)
        if peticion is not None:
            self.entregar(peticion)

    def entregar(self, peticion: pb.ResultadoTrabajo):
        """Entrega con reintentos. Es seguro repetirla: el coordinador guarda un solo resultado por job_id."""
        intento = 0
        while not self.alto.is_set():
            try:
                r = comun.llamar(self.stub.ReportarResultado, peticion, plazo=30.0, intentos=5,
                                 registro=self.registro)
                log.info("[%s] resultado entregado: %s", peticion.job_id, r.motivo)
                return
            except grpc.RpcError as e:
                intento += 1
                log.warning("[%s] no pude entregar el resultado (%s); lo guardo y reintento",
                            peticion.job_id, e.code().name)
                comun.esperar_backoff(min(intento, 5), base=1.0, maximo=30.0)

    # ------------------------------------------------------------ bucle principal
    def correr(self):
        threading.Thread(target=self.latir, daemon=True).start()
        intento = 0
        while not self.alto.is_set():
            libres = self.modelos_libres() if self.listo else []
            if not libres:
                self.alto.wait(0.5)
                continue
            try:
                asig = comun.llamar(self.stub.PedirTrabajo,
                                    pb.PeticionTrabajo(worker_id=self.id, modelos=libres,
                                                       version_datos=self.version_datos),
                                    plazo=10.0, intentos=3, registro=self.registro)
                intento = 0
            except grpc.RpcError as e:
                intento += 1
                log.warning("coordinador no disponible (%s)", e.code().name)
                comun.esperar_backoff(min(intento, 5), base=1.0, maximo=15.0)
                continue
            if not asig.hay_trabajo:
                continue
            with self.candado:
                self.activos[asig.job_id] = {"asignacion": asig.asignacion, "modelo": asig.modelo,
                                             "cancelar": threading.Event()}
            threading.Thread(target=self.ejecutar, args=(asig,), daemon=True).start()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if os.path.exists(ARCHIVO_LISTO):
        os.unlink(ARCHIVO_LISTO)
    nodo = Nodo()
    log.info("nodo %s; slots %s; máximo %d trabajos a la vez", nodo.id, nodo.slots, nodo.max_trabajos)
    try:
        nodo.correr()
    except KeyboardInterrupt:
        nodo.alto.set()


if __name__ == "__main__":
    main()
