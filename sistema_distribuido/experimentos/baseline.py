"""T_base: corre el conjunto de referencia SIN el sistema (un proceso, en serie).

Sirve para dos cosas del enunciado:
  1. medir T_base para la fórmula de reducción de tiempo;
  2. ser la "ejecución de referencia" contra la que se comparan las métricas del sistema.

  python experimentos/baseline.py --datos datos/generados --salida experimentos/baseline.csv
Usa los mismos job_id que el coordinador, así comparar.py puede emparejar trabajo por trabajo.
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines import ejecutar  # noqa: E402
from pipelines.datos import leer_version  # noqa: E402
from sistema import comun  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--referencia", default="experimentos/referencia.json")
    ap.add_argument("--datos", default=os.environ.get("DIR_DATOS", "datos/generados"))
    ap.add_argument("--salida", default="experimentos/baseline.csv")
    ap.add_argument("--solo", default="", help="limitar a un modelo (lineal|mlp|cnn)")
    a = ap.parse_args()
    with open(a.referencia) as f:
        ref = json.load(f)
    version = ref.get("version_datos") or leer_version(a.datos)
    semilla = ref.get("semilla") or comun.SEMILLA_DEFECTO
    trabajos = []
    for bloque in ref["peticiones"]:
        if a.solo and bloque["modelo"] != a.solo:
            continue
        for cfg in comun.expandir_grid(bloque["modelo"], bloque.get("config", {}), bloque.get("grid")):
            trabajos.append((bloque["modelo"], cfg))
    print(f"{len(trabajos)} trabajos en serie, datos {version}")
    t0 = time.time()
    with open(a.salida, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["job_id", "modelo", "config", "duracion", "val_rmse", "val_r2", "test_rmse", "test_r2",
                    "val_exactitud", "test_exactitud"])
        for i, (modelo, cfg) in enumerate(trabajos, 1):
            job_id = comun.calcular_job_id(modelo, cfg, version, semilla)
            ti = time.time()
            m = ejecutar(modelo, cfg, semilla, a.datos)["metricas"]
            v, p = m["val"], m["test"]
            w.writerow([job_id, modelo, comun.json_canonico(cfg), round(time.time() - ti, 3), v.get("rmse"),
                        v.get("r2"), p.get("rmse"), p.get("r2"), v.get("exactitud"), p.get("exactitud")])
            f.flush()
            print(f"[{i}/{len(trabajos)}] {job_id} {modelo} {time.time() - ti:.1f}s", flush=True)
    print(f"T_base = {time.time() - t0:.1f} s  -> {a.salida}")


if __name__ == "__main__":
    main()
