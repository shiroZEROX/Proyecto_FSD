"""Genera datos SINTÉTICOS con el mismo formato que la preparación de nuScenes.

Sirven para probar el sistema de punta a punta mientras no estén los datos reales.
Las métricas que salgan con estos datos NO valen para el reporte.

  python datos/generar_sinteticos.py --salida datos/generados [--objetos 18000] [--recortes 4000]

Imita nuScenes mini: 10 escenas, partición por escena (6 train / 2 val / 2 test).
"""
import argparse
import csv
import json
import os

import numpy as np

CATEGORIAS = {  # categoría estilo nuScenes -> (clase agrupada, ancho, largo, alto medios)
    "vehicle.car": ("vehiculo", 1.9, 4.6, 1.7),
    "vehicle.truck": ("vehiculo", 2.5, 7.0, 2.9),
    "vehicle.bus.rigid": ("vehiculo", 2.9, 11.0, 3.5),
    "human.pedestrian.adult": ("peaton", 0.7, 0.7, 1.75),
    "human.pedestrian.child": ("peaton", 0.5, 0.5, 1.2),
    "movable_object.barrier": ("otro", 2.5, 0.5, 1.0),
    "movable_object.trafficcone": ("otro", 0.4, 0.4, 0.9),
    "vehicle.bicycle": ("otro", 0.6, 1.7, 1.3),
}
ESCENAS = [f"scene-{n:04d}" for n in (61, 103, 553, 655, 757, 796, 916, 1077, 1094, 1100)]
PARTICION = {"train": ESCENAS[:6], "val": ESCENAS[6:8], "test": ESCENAS[8:]}


def objetos(rng, n):
    nombres = list(CATEGORIAS)
    pesos = np.array([0.40, 0.08, 0.03, 0.22, 0.02, 0.12, 0.10, 0.03])
    filas = []
    for _ in range(n):
        cat = nombres[rng.choice(len(nombres), p=pesos)]
        clase, a, l, h = CATEGORIAS[cat]
        ancho, largo, alto = (max(0.2, v * rng.normal(1, 0.1)) for v in (a, l, h))
        distancia = float(rng.uniform(2, 60))
        visibilidad = int(rng.choice([1, 2, 3, 4], p=[0.15, 0.15, 0.2, 0.5]))
        # Puntos de lidar ~ área aparente / distancia^2, atenuado por la visibilidad.
        esperado = 2500 * (ancho * alto + 0.3 * largo * alto) / distancia ** 2 * (visibilidad / 4)
        num_lidar = int(rng.poisson(max(esperado, 0.05)))
        num_radar = int(rng.poisson(max(0.02, 30 * ancho * largo / distancia ** 1.5) if clase == "vehiculo" else 0.3))
        velocidad = float(abs(rng.normal(6 if clase == "vehiculo" else 1.2, 3 if clase == "vehiculo" else 0.6))
                          * (rng.random() > 0.4))
        filas.append({"escena": ESCENAS[rng.integers(len(ESCENAS))], "categoria": cat, "clase": clase,
                      "ancho": round(ancho, 3), "largo": round(largo, 3), "alto": round(alto, 3),
                      "distancia": round(distancia, 3), "visibilidad": visibilidad, "num_lidar": num_lidar,
                      "num_radar": num_radar, "velocidad": round(velocidad, 3)})
    return filas


def recorte(rng, clase, tam=64):
    """Figura simple con ruido: vehículo = rectángulo ancho, peatón = rectángulo alto, otro = disco."""
    img = rng.normal(110, 35, (tam, tam, 3))
    color = rng.uniform(20, 235, 3)
    cy, cx = rng.integers(22, 42, 2)
    yy, xx = np.mgrid[:tam, :tam]
    if clase == "vehiculo":
        h, w = rng.integers(10, 18), rng.integers(22, 30)
        m = (abs(yy - cy) < h // 2) & (abs(xx - cx) < w // 2)
    elif clase == "peaton":
        h, w = rng.integers(24, 34), rng.integers(6, 11)
        m = (abs(yy - cy) < h // 2) & (abs(xx - cx) < w // 2)
    else:
        r = rng.integers(7, 14)
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    img[m] = color + rng.normal(0, 25, (m.sum(), 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salida", default="datos/generados")
    ap.add_argument("--objetos", type=int, default=18000)
    ap.add_argument("--recortes", type=int, default=4000)
    ap.add_argument("--semilla", type=int, default=2026)
    ap.add_argument("--version", default="sinteticos-v1")
    a = ap.parse_args()
    rng = np.random.default_rng(a.semilla)
    os.makedirs(a.salida, exist_ok=True)

    filas = objetos(rng, a.objetos)
    with open(os.path.join(a.salida, "objetos.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)

    clases = ["vehiculo", "peaton", "otro"]
    meta, imgs = [], np.zeros((a.recortes, 64, 64, 3), dtype=np.uint8)
    for i in range(a.recortes):
        clase = clases[rng.choice(3, p=[0.45, 0.30, 0.25])]
        imgs[i] = recorte(rng, clase)
        meta.append({"indice": i, "escena": ESCENAS[rng.integers(len(ESCENAS))], "clase": clase})
    np.save(os.path.join(a.salida, "recortes.npy"), imgs)
    with open(os.path.join(a.salida, "recortes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["indice", "escena", "clase"])
        w.writeheader()
        w.writerows(meta)

    with open(os.path.join(a.salida, "particion.json"), "w") as f:
        json.dump(PARTICION, f, indent=1)
    with open(os.path.join(a.salida, "version.txt"), "w") as f:
        f.write(a.version + "\n")
    print(f"{a.objetos} objetos y {a.recortes} recortes sintéticos en {a.salida} (versión {a.version})")


if __name__ == "__main__":
    main()
