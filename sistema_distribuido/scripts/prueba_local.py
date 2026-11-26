"""Prueba de punta a punta SIN Docker: coordinador y nodos como procesos locales.

  python scripts/prueba_local.py            (requiere datos/generados; ver README)

Verifica: rechazo de modelo inválido, envío repetido, caída de un nodo a media ejecución
(heartbeat + reasignación + continuar desde checkpoint), reinicio del coordinador con
estado en disco, nodo lento (timeout), resultado duplicado descartado, y que las métricas
del sistema son idénticas a las de una ejecución directa.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import grpc  # noqa: E402

from pipelines import ejecutar  # noqa: E402
from sistema import comun  # noqa: E402
from sistema import entrenamiento_pb2 as pb  # noqa: E402
from sistema import entrenamiento_pb2_grpc as pbg  # noqa: E402

DATOS = os.path.join(RAIZ, "datos", "generados")
PUERTO = "50077"
procesos = {}
tmp = tempfile.mkdtemp(prefix="prueba_sd_")
fallas = []
ESPERADOS = 0


def ok(cond, texto):
    print(("  OK   " if cond else "  FALLA ") + texto, flush=True)
    if not cond:
        fallas.append(texto)


def lanzar(nombre, modulo, **env):
    e = dict(os.environ, PYTHONPATH=RAIZ, PUERTO=PUERTO, COORDINADOR=f"localhost:{PUERTO}",
             DIR_ESTADO=os.path.join(tmp, "estado"), DIR_DATOS=DATOS, DIR_TRABAJO=os.path.join(tmp, "nodos"),
             HEARTBEAT_SEG="1", FALTAS_MAX="3", HILOS_TORCH="1", ARCHIVO_LISTO=os.path.join(tmp, f"listo_{nombre}"),
             WORKER_ID=nombre, **env)
    log = open(os.path.join(tmp, f"{nombre}.log"), "a")
    procesos[nombre] = subprocess.Popen([sys.executable, "-m", modulo], cwd=RAIZ, env=e, stdout=log, stderr=log)


def matar(nombre, sig=signal.SIGKILL):
    p = procesos.pop(nombre)
    p.send_signal(sig)
    p.wait()


def estado(stub, **kw):
    return stub.Estado(pb.PeticionEstado(**kw), timeout=5)


def esperar(cond, plazo, cada=0.3):
    fin = time.time() + plazo
    while time.time() < fin:
        r = cond()
        if r:
            return r
        time.sleep(cada)
    return None


def main():
    if not os.path.exists(os.path.join(DATOS, "version.txt")):  # sin datos los nodos quedan "no listos"
        print("0) no hay datos en datos/generados: genero los sintéticos")
        subprocess.run([sys.executable, os.path.join(RAIZ, "datos", "generar_sinteticos.py"), "--salida", DATOS], check=True)
    stub = pbg.CoordinadorStub(grpc.insecure_channel(f"localhost:{PUERTO}", options=comun.OPCIONES_GRPC))
    coord_env = dict(TIMEOUT_LINEAL="60", TIMEOUT_MLP="60", TIMEOUT_CNN="120", ESPERA_LARGA_SEG="1")
    lanzar("coordinador", "sistema.coordinador", **coord_env)
    esperar(lambda: _ping(stub), 15)

    print("1) modelo no soportado")
    try:
        stub.Enviar(pb.PeticionEnvio(modelo="xgboost"), timeout=5)
        ok(False, "debió rechazarse")
    except grpc.RpcError as e:
        ok(e.code() == grpc.StatusCode.INVALID_ARGUMENT, f"rechazado con {e.code().name}")
    try:
        stub.Enviar(pb.PeticionEnvio(modelo="mlp", config_json='{"neuronas": 3}'), timeout=5)
        ok(False, "hiperparámetro desconocido debió rechazarse")
    except grpc.RpcError as e:
        ok(e.code() == grpc.StatusCode.INVALID_ARGUMENT, f"hiperparámetro desconocido -> {e.code().name}")

    print("2) envío de trabajos (grid) y envío repetido")
    envios = [
        ("lineal", {}, {"tipo": ["ridge", "lasso"], "alpha": [0.1, 1]}),
        ("mlp", {"epocas": 8}, {"capas": [[32], [64, 32]]}),
        ("cnn", {"epocas": 8, "filtros": 8}, {"tam_imagen": [32, 48]}),
    ]
    pids = []
    global ESPERADOS
    for modelo, cfg, grid in envios:
        r = stub.Enviar(pb.PeticionEnvio(modelo=modelo, config_json=json.dumps(cfg), grid_json=json.dumps(grid)),
                        timeout=5)
        pids.append(r.peticion_id)
        ESPERADOS += len(r.trabajos)
        ok(all(t.nuevo for t in r.trabajos), f"{modelo}: {len(r.trabajos)} trabajos nuevos")
    r2 = stub.Enviar(pb.PeticionEnvio(modelo="lineal", config_json="{}",
                                      grid_json=json.dumps({"alpha": [1.0, 0.1], "tipo": ["lasso", "ridge"]})),
                     timeout=5)
    ok(r2.peticion_id == pids[0] and not any(t.nuevo for t in r2.trabajos),
       "el mismo grid (otro orden, 1 vs 1.0) da los mismos job_id y no crea trabajos")

    print("3) dos nodos; mato uno a media CNN")
    lanzar("nodoA", "sistema.worker", RETARDO_POR_EPOCA="0.6")
    lanzar("nodoB", "sistema.worker", RETARDO_POR_EPOCA="0.6")

    def cnn_avanzada():
        for t in estado(stub, peticion_id=pids[2]).trabajos:
            if t.estado == "CORRIENDO" and t.epoca >= 3 and t.worker == "nodoA":
                return t
            if t.estado == "CORRIENDO" and t.epoca >= 3 and t.worker == "nodoB":
                return t
    victima = esperar(cnn_avanzada, 120)
    ok(victima is not None, f"CNN {victima.job_id if victima else '?'} corriendo en {victima.worker if victima else '?'}")
    if victima is None:
        return
    nodo_caido = victima.worker
    matar(nodo_caido)
    print(f"     maté {nodo_caido} (SIGKILL) en la época {victima.epoca}")

    def reasignado():
        t = estado(stub, job_id=victima.job_id).trabajos[0]
        return t if t.reasignaciones >= 1 else None
    t = esperar(reasignado, 20)
    ok(t is not None, "el coordinador detectó la caída por heartbeat y devolvió el trabajo a la cola")
    nodos = {n.worker_id: n for n in stub.Nodos(pb.Vacio(), timeout=5).nodos}
    ok(not nodos[nodo_caido].vivo, f"{nodo_caido} aparece CAÍDO")

    print("4) reinicio el coordinador a media ejecución")
    matar("coordinador")
    time.sleep(2)
    lanzar("coordinador", "sistema.coordinador", **coord_env)
    esperar(lambda: _ping(stub), 15)
    total = sum(len(estado(stub, peticion_id=p).trabajos) for p in pids)
    ok(total == ESPERADOS, f"el coordinador recuperó los {total} trabajos de disco")
    lanzar("nodoC", "sistema.worker")

    finales = esperar(lambda: (lambda rs: rs if all(r.pendientes + r.corriendo == 0 for r in rs) else None)(
        [estado(stub, peticion_id=p) for p in pids]), 600, cada=1)
    ok(finales is not None, "todos los trabajos llegaron a un estado final")
    if finales is None:
        return
    ok(all(r.terminados == len(r.trabajos) for r in finales), "todos TERMINADOS, ninguno FALLIDO")
    ok(all(r.mejor_job_id for r in finales), "cada grid reporta su mejor configuración: "
       + ", ".join(r.mejor_job_id for r in finales))
    tv = estado(stub, job_id=victima.job_id).trabajos[0]
    mv = json.loads(tv.metricas_json)
    ok(mv["continuado_desde_epoca"] > 0,
       f"la CNN reasignada siguió desde el checkpoint de la época {mv['continuado_desde_epoca']} en {tv.worker}")

    print("5) un solo resultado por job_id")
    dirs = os.listdir(os.path.join(tmp, "estado", "resultados"))
    ok(len([d for d in dirs if not d.startswith(".")]) == ESPERADOS and not any(d.startswith(".tmp") for d in dirs),
       f"{ESPERADOS} directorios de resultado (uno por job_id), sin temporales a medias")
    acuse = stub.ReportarResultado(pb.ResultadoTrabajo(worker_id="nodo-lento", job_id=victima.job_id, asignacion=1,
                                                       ok=True, metricas_json='{"falso": 1}', modelo=b"x"),
                                   timeout=5)
    ok(not acuse.aceptado and acuse.motivo == "duplicado", f"resultado tardío de un nodo reemplazado: {acuse.motivo}")
    ok(json.loads(estado(stub, job_id=victima.job_id).trabajos[0].metricas_json) == mv, "el resultado guardado no cambió")

    print("6) métricas del sistema == ejecución directa (misma semilla)")
    iguales = 0
    trabajos = [t for r in finales for t in r.trabajos]
    for t in trabajos:
        directo = ejecutar(t.modelo, json.loads(t.config_json), t.semilla, DATOS)["metricas"]
        m = json.loads(estado(stub, job_id=t.job_id).trabajos[0].metricas_json)
        iguales += all(directo[p] == m[p] for p in ("val", "test"))
    ok(iguales == len(trabajos), f"{iguales}/{len(trabajos)} trabajos con métricas idénticas")

    print("7) nodo lento: vence el plazo y el trabajo se reasigna")
    for n in list(procesos):
        if n != "coordinador":
            matar(n)
    matar("coordinador")
    shutil.rmtree(os.path.join(tmp, "estado"))
    lanzar("coordinador", "sistema.coordinador", TIMEOUT_MLP="5", ESPERA_LARGA_SEG="1")
    esperar(lambda: _ping(stub), 15)
    lanzar("nodoLento", "sistema.worker", RETARDO_POR_EPOCA="2", SLOTS="mlp=1")
    r = stub.Enviar(pb.PeticionEnvio(modelo="mlp", config_json=json.dumps({"epocas": 6})), timeout=5)
    jid = r.trabajos[0].job_id
    t = esperar(lambda: (lambda t: t if t.reasignaciones >= 1 else None)(estado(stub, job_id=jid).trabajos[0]), 30)
    ok(t is not None, "timeout de 5 s del MLP: el trabajo volvió a la cola")
    ruta_log = os.path.join(tmp, "nodoLento.log")
    ok(esperar(lambda: "abandonado" in open(ruta_log).read(), 15) is not None,
       "el nodo lento recibió la orden de cancelar y abandonó el trabajo")
    matar("nodoLento")
    lanzar("nodoRapido", "sistema.worker", SLOTS="mlp=1")
    t = esperar(lambda: (lambda t: t if t.estado in ("TERMINADO", "FALLIDO") else None)(
        estado(stub, job_id=jid).trabajos[0]), 120)
    ok(t is not None and t.estado == "TERMINADO" and t.worker == "nodoRapido",
       f"lo terminó otro nodo: {t.estado if t else '?'} en {t.worker if t else '?'}")


def _ping(stub):
    try:
        return stub.Ping(pb.Vacio(), timeout=1)
    except grpc.RpcError:
        return None


if __name__ == "__main__":
    t0 = time.time()
    try:
        main()
    finally:
        for n in list(procesos):
            matar(n, signal.SIGTERM)
    print(f"\nlogs en {tmp}  ({time.time() - t0:.0f}s)")
    print("RESULTADO:", "TODO OK" if not fallas else f"{len(fallas)} FALLAS: {fallas}")
    sys.exit(1 if fallas else 0)
