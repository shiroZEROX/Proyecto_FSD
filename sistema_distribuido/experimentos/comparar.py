"""Compara las métricas del sistema distribuido contra la ejecución de referencia (baseline.csv).

  python -m sistema.cliente exportar experimentos/sistema.csv
  python experimentos/comparar.py experimentos/baseline.csv experimentos/sistema.csv --tol 1e-6
Empareja por job_id (determinista). Sale con código 1 si alguna métrica difiere más que la tolerancia.
"""
import argparse
import csv
import sys

METRICAS = ["val_rmse", "val_r2", "test_rmse", "test_r2", "val_exactitud", "test_exactitud"]


def leer(ruta):
    with open(ruta, newline="") as f:
        return {r["job_id"]: r for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("referencia")
    ap.add_argument("sistema")
    ap.add_argument("--tol", type=float, default=1e-6, help="diferencia absoluta máxima permitida")
    a = ap.parse_args()
    ref, sis = leer(a.referencia), leer(a.sistema)
    malos, faltan, revisados = 0, 0, 0
    for job_id, r in ref.items():
        s = sis.get(job_id)
        if s is None or s.get("estado", "TERMINADO") != "TERMINADO":
            faltan += 1
            print(f"FALTA   {job_id} {r['modelo']} {r['config']}")
            continue
        for m in METRICAS:
            if r[m] in ("", "None") or s[m] in ("", "None"):
                continue
            revisados += 1
            d = abs(float(r[m]) - float(s[m]))
            if d > a.tol:
                malos += 1
                print(f"DIFIERE {job_id} {r['modelo']} {m}: ref={r[m]} sistema={s[m]} (|d|={d:.3g})")
    print(f"{len(ref)} trabajos, {revisados} métricas revisadas, {malos} fuera de tolerancia {a.tol}, {faltan} faltantes")
    sys.exit(1 if malos or faltan else 0)


if __name__ == "__main__":
    main()
