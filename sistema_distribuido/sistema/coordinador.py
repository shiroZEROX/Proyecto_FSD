"""Coordinador: cola de trabajos, estado, heartbeat, timeouts, reasignación e idempotencia.

Ejecutar:  python -m sistema.coordinador
Variables de entorno principales (ver README):
  PUERTO=50051  DIR_ESTADO=./estado  VERSION_DATOS=sinteticos-v1
  HEARTBEAT_SEG=3  FALTAS_MAX=3  MAX_INTENTOS=4
  TIMEOUT_LINEAL=120  TIMEOUT_MLP=900  TIMEOUT_CNN=1800
  ORDEN_COLA=pesados_primero|fifo
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from concurrent import futures

import grpc

from sistema import comun
from sistema import entrenamiento_pb2 as pb
from sistema import entrenamiento_pb2_grpc as pbg

log = logging.getLogger("coordinador")

PENDIENTE, CORRIENDO, TERMINADO, FALLIDO = "PENDIENTE", "CORRIENDO", "TERMINADO", "FALLIDO"


class Estado:
    """Estado del coordinador. Todo cambio pasa por el candado y se persiste en disco."""

    def __init__(self, directorio: str):
        self.dir = directorio
        self.ruta = os.path.join(directorio, "estado.json")
        self.dir_resultados = os.path.join(directorio, "resultados")
        self.dir_checkpoints = os.path.join(directorio, "checkpoints")
        os.makedirs(self.dir_resultados, exist_ok=True)
        os.makedirs(self.dir_checkpoints, exist_ok=True)
        self.candado = threading.Lock()
        self.hay_trabajo = threading.Condition(self.candado)
        self.trabajos: dict[str, dict] = {}
        self.peticiones: dict[str, dict] = {}
        self.secuencia = 0
        self.workers: dict[str, dict] = {}  # en memoria: se reconstruye con los heartbeats
        self.iniciado = time.time()
        self._cargar()

    # -------- persistencia
    def _cargar(self):
        if not os.path.exists(self.ruta):
            log.info("sin estado previo; empiezo vacío")
            return
        with open(self.ruta) as f:
            datos = json.load(f)
        self.trabajos = datos.get("trabajos", {})
        self.peticiones = datos.get("peticiones", {})
        self.secuencia = datos.get("secuencia", 0)
        cuenta = {}
        for t in self.trabajos.values():
            cuenta[t["estado"]] = cuenta.get(t["estado"], 0) + 1
        log.info("estado recuperado de disco: %d trabajos %s", len(self.trabajos), cuenta)
        # Los trabajos CORRIENDO se conservan: si su nodo sigue vivo mandará heartbeat y
        # el resultado; si no, el monitor los devuelve a la cola tras el plazo de heartbeat.

    def guardar(self):
        """Llamar con el candado tomado. Escritura atómica: tmp + rename."""
        datos = {"trabajos": self.trabajos, "peticiones": self.peticiones, "secuencia": self.secuencia}
        comun.escribir_atomico(self.ruta, json.dumps(datos, indent=1, ensure_ascii=False))

    # -------- utilidades
    def evento(self, trabajo: dict, texto: str):
        trabajo.setdefault("historial", []).append([round(time.time(), 3), texto])
        log.info("[%s] %s", trabajo["job_id"], texto)

    def devolver_a_cola(self, t: dict, motivo: str, max_intentos: int, contar_reasignacion=True):
        """Quita el trabajo a su nodo. Si ya agotó los intentos queda FALLIDO."""
        anterior = t.get("worker")
        t["worker"] = None
        t["inicio"] = None
        if contar_reasignacion:
            t["reasignaciones"] = t.get("reasignaciones", 0) + 1
        if t["intentos"] >= max_intentos:
            t["estado"] = FALLIDO
            t["error"] = f"agotó {max_intentos} intentos; último motivo: {motivo}"
            self.evento(t, f"FALLIDO ({motivo}, nodo {anterior})")
        else:
            t["estado"] = PENDIENTE
            # retry con backoff a nivel de trabajo: el siguiente intento espera más
            espera = min(30.0, 1.0 * (2 ** (t["intentos"] - 1)))
            t["no_antes_de"] = time.time() + espera * (0.5 + 0.5 * random.random())
            self.evento(t, f"reasignado ({motivo}, nodo {anterior}); vuelve a la cola")
            self.hay_trabajo.notify_all()


class ServicioCoordinador(pbg.CoordinadorServicer):
    def __init__(self, estado: Estado):
        self.e = estado
        self.version_defecto = os.environ.get("VERSION_DATOS", "sinteticos-v1")
        self.heartbeat_seg = comun.entero_env("HEARTBEAT_SEG", 3)
        self.faltas_max = comun.entero_env("FALTAS_MAX", 3)
        self.max_intentos = comun.entero_env("MAX_INTENTOS", 4)
        self.plazos = {
            "lineal": comun.real_env("TIMEOUT_LINEAL", 120),
            "mlp": comun.real_env("TIMEOUT_MLP", 900),
            "cnn": comun.real_env("TIMEOUT_CNN", 1800),
        }
        self.orden = os.environ.get("ORDEN_COLA", "pesados_primero")
        self.plazo_espera_trabajo = comun.real_env("ESPERA_LARGA_SEG", 2.0)

    # ================================================================ usuario
    def Enviar(self, req, ctx):
        modelo = req.modelo.strip().lower()
        if modelo not in comun.MODELOS:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT,
                      f"modelo '{req.modelo}' no soportado; use lineal, mlp o cnn")
        try:
            base = json.loads(req.config_json) if req.config_json else {}
            grid = json.loads(req.grid_json) if req.grid_json else None
            configs = comun.expandir_grid(modelo, base, grid)
        except (comun.ConfigInvalida, json.JSONDecodeError) as err:
            ctx.abort(grpc.StatusCode.INVALID_ARGUMENT, str(err))
        semilla = req.semilla or comun.SEMILLA_DEFECTO
        version = req.version_datos or self.version_defecto

        respuesta = []
        with self.e.candado:
            for cfg in configs:
                job_id = comun.calcular_job_id(modelo, cfg, version, semilla)
                t = self.e.trabajos.get(job_id)
                if t is not None and t["estado"] != FALLIDO:
                    # Envío repetido: no se ejecuta otra vez, se devuelve lo que ya existe.
                    respuesta.append(pb.TrabajoEnviado(job_id=job_id, nuevo=False, estado=t["estado"]))
                    continue
                if t is not None:  # FALLIDO: reenviarlo es una petición explícita de reintento
                    t.update(estado=PENDIENTE, intentos=0, error="", no_antes_de=0)
                    self.e.evento(t, "reenviado por el usuario tras FALLIDO")
                else:
                    self.e.secuencia += 1
                    t = {
                        "job_id": job_id, "modelo": modelo, "config": cfg, "semilla": semilla,
                        "version_datos": version, "estado": PENDIENTE, "intentos": 0,
                        "reasignaciones": 0, "asignacion": 0, "worker": None, "inicio": None,
                        "no_antes_de": 0, "creado": time.time(), "secuencia": self.e.secuencia,
                        "metricas": None, "error": "", "epoca": 0, "perdida": 0.0, "duracion": 0.0,
                    }
                    self.e.trabajos[job_id] = t
                    self.e.evento(t, f"creado ({modelo} {comun.json_canonico(cfg)})")
                respuesta.append(pb.TrabajoEnviado(job_id=job_id, nuevo=True, estado=t["estado"]))
            ids = [r.job_id for r in respuesta]
            pid = comun.calcular_peticion_id(ids)
            self.e.peticiones.setdefault(pid, {"job_ids": ids, "modelo": modelo, "creado": time.time(),
                                               "grid": bool(grid)})
            self.e.guardar()
            self.e.hay_trabajo.notify_all()
        return pb.RespuestaEnvio(peticion_id=pid, trabajos=respuesta)

    def _info(self, t: dict) -> pb.InfoTrabajo:
        return pb.InfoTrabajo(
            job_id=t["job_id"], modelo=t["modelo"], config_json=comun.json_canonico(t["config"]),
            semilla=t["semilla"], estado=t["estado"], worker=t.get("worker") or t.get("worker_resultado") or "",
            intentos=t["intentos"], reasignaciones=t.get("reasignaciones", 0),
            metricas_json=json.dumps(t["metricas"]) if t.get("metricas") else "",
            error=t.get("error", ""), epoca=t.get("epoca", 0), perdida=t.get("perdida", 0.0),
            duracion=t.get("duracion", 0.0))

    def Estado(self, req, ctx):
        with self.e.candado:
            if req.job_id:
                ids = [req.job_id]
            elif req.peticion_id:
                if req.peticion_id not in self.e.peticiones:
                    ctx.abort(grpc.StatusCode.NOT_FOUND, "petición desconocida")
                ids = self.e.peticiones[req.peticion_id]["job_ids"]
            else:
                ids = sorted(self.e.trabajos, key=lambda j: self.e.trabajos[j]["secuencia"])
            trabajos = [self.e.trabajos[j] for j in ids if j in self.e.trabajos]
            if req.job_id and not trabajos:
                ctx.abort(grpc.StatusCode.NOT_FOUND, "job_id desconocido")
            resp = pb.RespuestaEstado(trabajos=[self._info(t) for t in trabajos])
            for t in trabajos:
                if t["estado"] == PENDIENTE: resp.pendientes += 1
                elif t["estado"] == CORRIENDO: resp.corriendo += 1
                elif t["estado"] == TERMINADO: resp.terminados += 1
                else: resp.fallidos += 1
            mejor, criterio = self._mejor(trabajos) if req.peticion_id else ("", "")
            if mejor:
                resp.mejor_job_id, resp.criterio = mejor, criterio
            return resp

    @staticmethod
    def _mejor(trabajos):
        """Mejor configuración del grid, elegida con la métrica de VALIDACIÓN (no la de prueba)."""
        hechos = [t for t in trabajos if t["estado"] == TERMINADO and t.get("metricas")]
        if not hechos or len(hechos) < len(trabajos) or len({t["modelo"] for t in hechos}) != 1:
            return "", ""
        if hechos[0]["modelo"] == "cnn":
            t = max(hechos, key=lambda t: t["metricas"]["val"]["exactitud"])
            return t["job_id"], "max exactitud en validación"
        t = min(hechos, key=lambda t: t["metricas"]["val"]["rmse"])
        return t["job_id"], "min RMSE en validación"

    def Resultado(self, req, ctx):
        with self.e.candado:
            t = self.e.trabajos.get(req.job_id)
            if t is None:
                ctx.abort(grpc.StatusCode.NOT_FOUND, "job_id desconocido")
            if t["estado"] != TERMINADO:
                return pb.RespuestaResultado(job_id=req.job_id, estado=t["estado"])
            directorio = os.path.join(self.e.dir_resultados, req.job_id)
            formato = t.get("formato_modelo", "")
        with open(os.path.join(directorio, "modelo.bin"), "rb") as f:
            modelo = f.read()
        with open(os.path.join(directorio, "metricas.json")) as f:
            metricas = f.read()
        return pb.RespuestaResultado(job_id=req.job_id, estado=TERMINADO, metricas_json=metricas,
                                     modelo=modelo, formato_modelo=formato)

    def Nodos(self, req, ctx):
        with self.e.candado:
            nodos = []
            for wid, w in sorted(self.e.workers.items()):
                activos = [j for j, t in self.e.trabajos.items() if t["estado"] == CORRIENDO and t["worker"] == wid]
                nodos.append(pb.InfoNodo(worker_id=wid, vivo=w["vivo"], listo=w["listo"], trabajos=activos,
                                         ultimo_latido=w["ultimo"], version_datos=w["version_datos"],
                                         terminados=w.get("terminados", 0)))
        return pb.RespuestaNodos(nodos=nodos)

    def Ping(self, req, ctx):
        return pb.Pong(estado="ok", iniciado_en=self.e.iniciado)

    # ================================================================ nodos
    def _latido(self, worker_id, listo, version):
        w = self.e.workers.get(worker_id)
        if w is None:
            w = self.e.workers[worker_id] = {"terminados": 0}
            log.info("nodo nuevo: %s", worker_id)
        elif not w["vivo"]:
            log.info("nodo %s volvió a mandar heartbeat", worker_id)
        w.update(ultimo=time.time(), listo=listo, version_datos=version, vivo=True)
        return w

    def Heartbeat(self, req, ctx):
        with self.e.candado:
            self._latido(req.worker_id, req.listo, req.version_datos)
            cancelar = []
            for activo in req.activos:
                t = self.e.trabajos.get(activo.job_id)
                if (t is None or t["estado"] != CORRIENDO or t["worker"] != req.worker_id
                        or t["asignacion"] != activo.asignacion):
                    cancelar.append(activo.job_id)
        return pb.RespuestaLatido(cancelar=cancelar, intervalo_seg=self.heartbeat_seg)

    def _siguiente(self, modelos, version):
        ahora = time.time()
        candidatos = [t for t in self.e.trabajos.values()
                      if t["estado"] == PENDIENTE and t["modelo"] in modelos
                      and t["version_datos"] == version and t.get("no_antes_de", 0) <= ahora]
        if not candidatos:
            return None
        if self.orden == "pesados_primero":
            # Los trabajos largos primero (heurística LPT): reduce el tiempo total del conjunto.
            return min(candidatos, key=lambda t: (-comun.PESO_MODELO[t["modelo"]], t["secuencia"]))
        return min(candidatos, key=lambda t: t["secuencia"])

    def PedirTrabajo(self, req, ctx):
        fin = time.time() + self.plazo_espera_trabajo  # espera larga: evita sondear sin parar
        with self.e.candado:
            w = self._latido(req.worker_id, self.e.workers.get(req.worker_id, {}).get("listo", True),
                             req.version_datos)
            if not w["listo"]:
                return pb.Asignacion(hay_trabajo=False)
            while True:
                t = self._siguiente(set(req.modelos), req.version_datos)
                restante = fin - time.time()
                if t is not None or restante <= 0:
                    break
                self.e.hay_trabajo.wait(timeout=min(restante, 0.5))
            if t is None:
                return pb.Asignacion(hay_trabajo=False)
            t["estado"] = CORRIENDO
            t["worker"] = req.worker_id
            t["asignacion"] += 1
            t["intentos"] += 1
            t["inicio"] = time.time()
            self.e.evento(t, f"asignado a {req.worker_id} (intento {t['intentos']}, asignación {t['asignacion']})")
            self.e.guardar()
            ruta_ckpt = os.path.join(self.e.dir_checkpoints, t["job_id"] + ".ckpt")
            ckpt = b""
            if os.path.exists(ruta_ckpt):
                with open(ruta_ckpt, "rb") as f:
                    ckpt = f.read()
            return pb.Asignacion(hay_trabajo=True, job_id=t["job_id"], modelo=t["modelo"],
                                 config_json=comun.json_canonico(t["config"]), semilla=t["semilla"],
                                 version_datos=t["version_datos"], asignacion=t["asignacion"],
                                 plazo_seg=self.plazos[t["modelo"]], checkpoint=ckpt)

    def _vigente(self, t, worker_id, asignacion):
        return t is not None and t["estado"] == CORRIENDO and t["worker"] == worker_id and t["asignacion"] == asignacion

    def ReportarAvance(self, req, ctx):
        with self.e.candado:
            self._latido(req.worker_id, True, self.e.workers.get(req.worker_id, {}).get("version_datos", ""))
            t = self.e.trabajos.get(req.job_id)
            if not self._vigente(t, req.worker_id, req.asignacion):
                return pb.RespuestaAvance(cancelar=True)
            t["epoca"], t["perdida"] = req.epoca, req.perdida
            if req.checkpoint:
                comun.escribir_atomico(os.path.join(self.e.dir_checkpoints, req.job_id + ".ckpt"), req.checkpoint)
            self.e.guardar()
        return pb.RespuestaAvance(cancelar=False)

    def ReportarResultado(self, req, ctx):
        with self.e.candado:
            t = self.e.trabajos.get(req.job_id)
            if t is None:
                return pb.AcuseResultado(aceptado=False, motivo="desconocido")
            if t["estado"] == TERMINADO:
                # Resultado único por job_id: un segundo resultado (nodo lento ya reemplazado,
                # reintento de envío) se descarta.
                self.e.evento(t, f"resultado DUPLICADO de {req.worker_id} (asignación {req.asignacion}) descartado")
                self.e.guardar()
                return pb.AcuseResultado(aceptado=False, motivo="duplicado")
            vigente = self._vigente(t, req.worker_id, req.asignacion)
            if not req.ok:
                if not vigente:
                    return pb.AcuseResultado(aceptado=False, motivo="asignacion_vieja")
                t["error"] = req.error
                if req.reintentable:
                    self.e.devolver_a_cola(t, f"error en nodo: {req.error[:200]}", self.max_intentos,
                                           contar_reasignacion=False)
                else:
                    t["estado"], t["worker"] = FALLIDO, None
                    self.e.evento(t, f"FALLIDO sin reintento: {req.error[:200]}")
                self.e.guardar()
                return pb.AcuseResultado(aceptado=True, motivo="falla_registrada")
            # Éxito. Con semillas fijas el resultado es el mismo venga de la asignación
            # vigente o de una vieja, así que se acepta el primero que llegue completo.
            directorio = os.path.join(self.e.dir_resultados, req.job_id)
            comun.escribir_directorio_atomico(directorio, {
                "modelo.bin": req.modelo, "metricas.json": req.metricas_json})
            t.update(estado=TERMINADO, metricas=json.loads(req.metricas_json), worker_resultado=req.worker_id,
                     worker=None, duracion=req.duracion, formato_modelo=req.formato_modelo,
                     terminado=time.time(), error="")
            self.e.evento(t, f"TERMINADO por {req.worker_id} en {req.duracion:.1f}s"
                             + ("" if vigente else " (de una asignación anterior)"))
            w = self.e.workers.get(req.worker_id)
            if w:
                w["terminados"] = w.get("terminados", 0) + 1
            ruta_ckpt = os.path.join(self.e.dir_checkpoints, req.job_id + ".ckpt")
            if os.path.exists(ruta_ckpt):
                os.unlink(ruta_ckpt)
            self.e.guardar()
        return pb.AcuseResultado(aceptado=True, motivo="aceptado")

    # ================================================================ monitor
    def monitor(self, alto: threading.Event):
        """Detecta nodos caídos (heartbeat) y trabajos vencidos (timeout por modelo)."""
        limite = self.heartbeat_seg * self.faltas_max
        while not alto.wait(1.0):
            ahora = time.time()
            with self.e.candado:
                cambio = False
                for wid, w in self.e.workers.items():
                    if w["vivo"] and ahora - w["ultimo"] > limite:
                        w["vivo"] = False
                        log.warning("nodo %s CAÍDO: %d heartbeats sin llegar", wid, self.faltas_max)
                for t in self.e.trabajos.values():
                    if t["estado"] != CORRIENDO:
                        continue
                    w = self.e.workers.get(t["worker"])
                    if w is None:
                        # Nodo que no ha hablado desde que arrancó este coordinador.
                        if ahora - self.e.iniciado > limite:
                            self.e.devolver_a_cola(t, "nodo desconocido tras reinicio del coordinador",
                                                   self.max_intentos)
                            cambio = True
                    elif not w["vivo"]:
                        self.e.devolver_a_cola(t, "nodo caído (sin heartbeat)", self.max_intentos)
                        cambio = True
                    elif t["inicio"] and ahora - t["inicio"] > self.plazos[t["modelo"]]:
                        self.e.devolver_a_cola(t, f"timeout de {self.plazos[t['modelo']]:.0f}s ({t['modelo']})",
                                               self.max_intentos)
                        cambio = True
                if cambio:
                    self.e.guardar()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    estado = Estado(os.environ.get("DIR_ESTADO", "./estado"))
    servicio = ServicioCoordinador(estado)
    servidor = grpc.server(futures.ThreadPoolExecutor(max_workers=64), options=comun.OPCIONES_GRPC)
    pbg.add_CoordinadorServicer_to_server(servicio, servidor)
    puerto = os.environ.get("PUERTO", "50051")
    servidor.add_insecure_port(f"[::]:{puerto}")
    servidor.start()
    alto = threading.Event()
    threading.Thread(target=servicio.monitor, args=(alto,), daemon=True).start()
    log.info("coordinador escuchando en :%s (orden de cola: %s, plazos: %s)", puerto, servicio.orden, servicio.plazos)
    try:
        servidor.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        alto.set()
        servidor.stop(2)


if __name__ == "__main__":
    main()
