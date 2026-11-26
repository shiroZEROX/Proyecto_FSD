"""Cliente de línea de comandos.

Ejemplos:
  python -m sistema.cliente enviar lineal --config '{"tipo": "ridge", "alpha": 1.0}'
  python -m sistema.cliente enviar cnn --config '{"epocas": 5}' --grid '{"filtros": [8, 16]}'
  python -m sistema.cliente referencia experimentos/referencia.json --esperar
  python -m sistema.cliente estado [--peticion P | --job J]
  python -m sistema.cliente esperar P1 P2 ...
  python -m sistema.cliente resultado JOB_ID --guardar modelo.bin
  python -m sistema.cliente nodos
  python -m sistema.cliente esperar-corriendo --modelo cnn --epoca 2
  python -m sistema.cliente exportar resultados.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import grpc

from sistema import comun
from sistema import entrenamiento_pb2 as pb
from sistema import entrenamiento_pb2_grpc as pbg


def conectar(direccion):
    canal = grpc.insecure_channel(direccion, options=comun.OPCIONES_GRPC)
    return pbg.CoordinadorStub(canal)


def enviar(stub, modelo, config, grid=None, semilla=0, version=""):
    r = comun.llamar(stub.Enviar, pb.PeticionEnvio(
        modelo=modelo, config_json=json.dumps(config or {}), grid_json=json.dumps(grid) if grid else "",
        semilla=semilla, version_datos=version), plazo=10.0)
    nuevos = sum(t.nuevo for t in r.trabajos)
    print(f"petición {r.peticion_id}: {len(r.trabajos)} trabajos ({nuevos} nuevos, "
          f"{len(r.trabajos) - nuevos} ya existían)")
    for t in r.trabajos:
        print(f"  {t.job_id}  {'nuevo' if t.nuevo else 'EXISTENTE'}  {t.estado}")
    return r


def resumen(stub, peticion_id="", job_id=""):
    return comun.llamar(stub.Estado, pb.PeticionEstado(peticion_id=peticion_id, job_id=job_id), plazo=10.0)


def imprimir_estado(r):
    print(f"pendientes={r.pendientes} corriendo={r.corriendo} terminados={r.terminados} fallidos={r.fallidos}")
    for t in r.trabajos:
        m = json.loads(t.metricas_json) if t.metricas_json else {}
        val = m.get("val", {})
        metrica = (f"val_exactitud={val['exactitud']:.4f}" if "exactitud" in val
                   else f"val_rmse={val['rmse']:.4f} val_r2={val['r2']:.4f}" if "rmse" in val else "")
        avance = f"época {t.epoca} pérdida {t.perdida:.4f}" if t.estado == "CORRIENDO" and t.epoca else ""
        print(f"  {t.job_id} {t.modelo:6s} {t.estado:10s} nodo={t.worker or '-':14s} intentos={t.intentos} "
              f"reasig={t.reasignaciones} {metrica}{avance} {t.config_json}")
        if t.error:
            print(f"      error: {t.error}")
    if r.mejor_job_id:
        print(f"MEJOR: {r.mejor_job_id} ({r.criterio})")


def esperar(stub, peticiones, cada=2.0, mostrar=True):
    while True:
        try:
            rs = [resumen(stub, peticion_id=p) for p in peticiones]
        except grpc.RpcError as e:
            if e.code() not in comun.CODIGOS_REINTENTABLES:
                raise
            print(f"[{time.strftime('%H:%M:%S')}] coordinador no disponible ({e.code().name}); sigo esperando",
                  flush=True)
            time.sleep(cada)
            continue
        faltan = sum(r.pendientes + r.corriendo for r in rs)
        if mostrar:
            hechos = sum(r.terminados for r in rs)
            fallidos = sum(r.fallidos for r in rs)
            print(f"[{time.strftime('%H:%M:%S')}] terminados={hechos} fallidos={fallidos} faltan={faltan}", flush=True)
        if faltan == 0:
            return rs
        time.sleep(cada)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cliente del servidor distribuido de entrenamiento")
    ap.add_argument("--coordinador", default=os.environ.get("COORDINADOR", "localhost:50051"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("enviar")
    s.add_argument("modelo")
    s.add_argument("--config", default="{}")
    s.add_argument("--grid", default="")
    s.add_argument("--semilla", type=int, default=0)
    s.add_argument("--version-datos", default="")
    s.add_argument("--esperar", action="store_true")

    s = sub.add_parser("referencia", help="envía el conjunto de referencia (34 trabajos)")
    s.add_argument("archivo", nargs="?", default="experimentos/referencia.json")
    s.add_argument("--esperar", action="store_true")

    s = sub.add_parser("estado")
    s.add_argument("--peticion", default="")
    s.add_argument("--job", default="")

    s = sub.add_parser("esperar")
    s.add_argument("peticiones", nargs="+")

    s = sub.add_parser("resultado")
    s.add_argument("job_id")
    s.add_argument("--guardar", default="")

    sub.add_parser("nodos")

    s = sub.add_parser("esperar-corriendo", help="espera a que un trabajo de ese modelo llegue a cierta época "
                                                   "e imprime su job_id y su nodo (para la demo de fallas)")
    s.add_argument("--modelo", default="cnn")
    s.add_argument("--epoca", type=int, default=2)
    s.add_argument("--plazo", type=float, default=600)

    s = sub.add_parser("exportar", help="CSV con las métricas de todos los trabajos terminados")
    s.add_argument("salida")

    a = ap.parse_args(argv)
    stub = conectar(a.coordinador)
    try:
        if a.cmd == "enviar":
            r = enviar(stub, a.modelo, json.loads(a.config), json.loads(a.grid) if a.grid else None,
                       a.semilla, a.version_datos)
            if a.esperar:
                imprimir_estado(esperar(stub, [r.peticion_id])[0])
        elif a.cmd == "referencia":
            with open(a.archivo) as f:
                ref = json.load(f)
            pids = []
            for bloque in ref["peticiones"]:
                r = enviar(stub, bloque["modelo"], bloque.get("config", {}), bloque.get("grid"),
                           ref.get("semilla", 0), ref.get("version_datos", ""))
                pids.append(r.peticion_id)
            print("peticiones:", " ".join(pids))
            if a.esperar:
                t0 = time.time()
                for r in esperar(stub, pids):
                    imprimir_estado(r)
                print(f"conjunto terminado en {time.time() - t0:.1f}s desde el envío")
        elif a.cmd == "estado":
            imprimir_estado(resumen(stub, a.peticion, a.job))
        elif a.cmd == "esperar":
            for r in esperar(stub, a.peticiones):
                imprimir_estado(r)
        elif a.cmd == "resultado":
            r = comun.llamar(stub.Resultado, pb.PeticionResultado(job_id=a.job_id), plazo=30.0)
            print(f"estado: {r.estado}")
            if r.metricas_json:
                print(json.dumps(json.loads(r.metricas_json), indent=2, ensure_ascii=False))
            if a.guardar and r.modelo:
                comun.escribir_atomico(a.guardar, r.modelo)
                print(f"modelo ({r.formato_modelo}, {len(r.modelo)} bytes) guardado en {a.guardar}")
        elif a.cmd == "nodos":
            r = comun.llamar(stub.Nodos, pb.Vacio(), plazo=5.0)
            for n in r.nodos:
                print(f"{n.worker_id:16s} {'VIVO ' if n.vivo else 'CAÍDO'} {'listo' if n.listo else 'no-listo'} "
                      f"datos={n.version_datos} terminados={n.terminados} "
                      f"hace={time.time() - n.ultimo_latido:.1f}s trabajos={list(n.trabajos)}")
        elif a.cmd == "esperar-corriendo":
            fin = time.time() + a.plazo
            while time.time() < fin:
                for t in resumen(stub).trabajos:
                    if t.modelo == a.modelo and t.estado == "CORRIENDO" and t.epoca >= a.epoca:
                        print(t.job_id, t.worker)
                        return
                time.sleep(1)
            print("ningún trabajo llegó a esa época", file=sys.stderr)
            sys.exit(1)
        elif a.cmd == "exportar":
            r = resumen(stub)
            with open(a.salida, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["job_id", "modelo", "config", "semilla", "estado", "nodo", "intentos", "reasignaciones",
                            "duracion", "val_rmse", "val_r2", "test_rmse", "test_r2", "val_exactitud",
                            "test_exactitud"])
                for t in r.trabajos:
                    m = json.loads(t.metricas_json) if t.metricas_json else {}
                    v, p = m.get("val", {}), m.get("test", {})
                    w.writerow([t.job_id, t.modelo, t.config_json, t.semilla, t.estado, t.worker, t.intentos,
                                t.reasignaciones, round(t.duracion, 3), v.get("rmse"), v.get("r2"), p.get("rmse"),
                                p.get("r2"), v.get("exactitud"), p.get("exactitud")])
            print(f"{len(r.trabajos)} trabajos exportados a {a.salida}")
    except grpc.RpcError as e:
        print(f"error {e.code().name}: {e.details()}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
