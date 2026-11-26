"""Preparación de nuScenes (una sola vez, antes de medir).

Produce exactamente el mismo formato que datos/generar_sinteticos.py:
  objetos.csv, recortes.npy, recortes.csv, particion.json, version.txt

Estado de prueba: corrido de punta a punta contra un dataset FALSO con el esquema de
v1.0-mini (scripts/nuscenes_falso.py) y la salida la leen bien los tres pipelines. Todavía NO se
ha corrido contra el nuScenes mini real del profesor.

Instalación (solo donde se preparan los datos, NO en la imagen del nodo). nuscenes-devkit 1.1.11
pide matplotlib<3.6, que no instala en Python 3.11; por eso va con --no-deps:
  pip install --no-deps nuscenes-devkit==1.1.11
  pip install -r requirements-datos.txt
O con Docker:  docker compose --profile datos run --rm preparar   (ver README)

Uso:
  python datos/preparar_nuscenes.py --dataroot /ruta/nuscenes --salida datos/nuscenes_mini

Decisiones abiertas (las fija el profesor, ver README). Cada una tiene la opción del esqueleto y la
del repo del equipo (scripts/preparar_datos.py de Karol), para que se pueda cambiar sin reescribir:
  --agrupacion esqueleto  bicicleta y moto van a "otro"            (defecto)
               equipo     todo vehicle.* es "vehiculo"
  --particion  nombres    escenas ordenadas por nombre, 6/2/2       (defecto)
               equipo     permutación con semilla de los tokens, 60/20/20 (misma que el repo del equipo)
               archivo.json  {train:[...], val:[...], test:[...]} con nombres de escena
  --camaras    las 6 (defecto) o solo CAM_FRONT como el repo del equipo
  --visibilidad completa  la caja entera dentro de la imagen       (defecto)
                parcial   recorta la parte visible, como el repo del equipo
"""
import argparse
import csv
import json
import os

import numpy as np

AGRUPACIONES = {
    # SUPOSICIÓN del esqueleto: las dos ruedas no se parecen a un auto en un recorte de 64x64.
    "esqueleto": {"vehicle.bicycle": "otro", "vehicle.motorcycle": "otro",
                  "vehicle.": "vehiculo", "human.pedestrian.": "peaton"},
    # Repo del equipo: category_group() de scripts/preparar_datos.py.
    "equipo": {"vehicle.": "vehiculo", "human.pedestrian": "peaton"},
}

CAMARAS = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def agrupar(categoria: str, tabla: dict) -> str:
    for prefijo in sorted(tabla, key=len, reverse=True):
        if categoria.startswith(prefijo):
            return tabla[prefijo]
    return "otro"


def particion_equipo(nusc, semilla: int) -> dict:
    """Reproduce la partición del repo del equipo (60/20/20 sobre tokens permutados) y la da en nombres."""
    tokens = sorted(s["token"] for s in nusc.scene)
    rng = np.random.default_rng(semilla)
    revueltos = [tokens[i] for i in rng.permutation(len(tokens))]
    n = len(revueltos)
    n_train = max(1, int(n * .6))
    n_val = max(1, int(n * .2)) if n >= 3 else 0
    p = {"train": revueltos[:n_train], "val": revueltos[n_train:n_train + n_val], "test": revueltos[n_train + n_val:]}
    if n >= 3 and not p["test"]:
        p["test"], p["val"] = p["val"][-1:], p["val"][:-1]
    nombre = {s["token"]: s["name"] for s in nusc.scene}
    return {k: sorted(nombre[t] for t in v) for k, v in p.items()}


def esquinas_en_imagen(caja, K, ancho, alto, parcial):
    """Rectángulo (x0, y0, x1, y1) de la caja en la imagen, o None si no se ve."""
    from nuscenes.utils.geometry_utils import view_points
    esquinas3d = caja.corners()
    delante = esquinas3d[2, :] > 0.1
    if parcial:
        if not delante.any():
            return None
        esquinas3d = esquinas3d[:, delante]
    elif not delante.all():
        return None
    pts = view_points(esquinas3d, K, normalize=True)[:2]
    x0, y0 = np.floor(pts.min(1)).astype(int)
    x1, y1 = np.ceil(pts.max(1)).astype(int)
    if parcial:
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(ancho, x1), min(alto, y1)
    elif x0 < 0 or y0 < 0 or x1 > ancho or y1 > alto:
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-mini")
    ap.add_argument("--salida", default="datos/nuscenes_mini")
    ap.add_argument("--agrupacion", default="esqueleto", choices=sorted(AGRUPACIONES))
    ap.add_argument("--particion", default="nombres", help="nombres | equipo | ruta a un JSON")
    ap.add_argument("--semilla", type=int, default=42, help="solo para --particion equipo")
    ap.add_argument("--camaras", default=",".join(CAMARAS))
    ap.add_argument("--visibilidad", default="completa", choices=["completa", "parcial"])
    ap.add_argument("--tam", type=int, default=64)
    ap.add_argument("--min_px", type=int, default=16, help="descarta recortes más chicos que esto")
    a = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.geometry_utils import BoxVisibility
    from PIL import Image

    nusc = NuScenes(version=a.version, dataroot=a.dataroot, verbose=True)
    os.makedirs(a.salida, exist_ok=True)
    tabla = AGRUPACIONES[a.agrupacion]
    camaras = [c.strip() for c in a.camaras.split(",") if c.strip()]
    parcial = a.visibilidad == "parcial"

    escenas = sorted(s["name"] for s in nusc.scene)
    if a.particion == "nombres":
        particion = {"train": escenas[:6], "val": escenas[6:8], "test": escenas[8:]}
    elif a.particion == "equipo":
        particion = particion_equipo(nusc, a.semilla)
    else:
        with open(a.particion) as f:
            particion = json.load(f)
    faltan = set(escenas) - set(sum((particion[k] for k in ("train", "val", "test")), []))
    if faltan:
        print(f"AVISO: escenas fuera de la partición (sus objetos no se usan): {sorted(faltan)}")

    filas, recortes, meta = [], [], []
    for escena in sorted(nusc.scene, key=lambda s: s["name"]):   # orden fijo => salida reproducible
        token = escena["first_sample_token"]
        while token:
            muestra = nusc.get("sample", token)
            lidar = nusc.get("sample_data", muestra["data"]["LIDAR_TOP"])
            ego = nusc.get("ego_pose", lidar["ego_pose_token"])
            # Cajas de todas las cámaras de la muestra, una sola vez por muestra.
            por_camara = {}
            for cam in camaras:
                sd = nusc.get("sample_data", muestra["data"][cam])
                ruta, cajas, K = nusc.get_sample_data(sd["token"], box_vis_level=BoxVisibility.ANY)
                por_camara[cam] = (ruta, {c.token: c for c in cajas}, K, sd["width"], sd["height"])
            imagenes = {}
            for ann_token in muestra["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                clase = agrupar(ann["category_name"], tabla)
                ancho, largo, alto = ann["size"]
                distancia = float(np.linalg.norm(np.array(ann["translation"][:2]) - np.array(ego["translation"][:2])))
                vel = nusc.box_velocity(ann_token)[:2]
                velocidad = 0.0 if np.any(np.isnan(vel)) else float(np.linalg.norm(vel))
                filas.append({"escena": escena["name"], "categoria": ann["category_name"], "clase": clase,
                              "ancho": ancho, "largo": largo, "alto": alto, "distancia": round(distancia, 3),
                              "visibilidad": int(nusc.get("visibility", ann["visibility_token"])["token"]),
                              "num_lidar": ann["num_lidar_pts"], "num_radar": ann["num_radar_pts"],
                              "velocidad": round(velocidad, 3)})
                # Recorte: la primera cámara (en el orden de --camaras) donde la caja se ve.
                for cam in camaras:
                    ruta, cajas, K, w, h = por_camara[cam]
                    if ann_token not in cajas:
                        continue
                    rect = esquinas_en_imagen(cajas[ann_token], K, w, h, parcial)
                    if rect is None:
                        continue
                    x0, y0, x1, y1 = rect
                    if min(x1 - x0, y1 - y0) < a.min_px:
                        break
                    if ruta not in imagenes:
                        imagenes[ruta] = Image.open(ruta).convert("RGB")
                    img = imagenes[ruta].crop((x0, y0, x1, y1)).resize((a.tam, a.tam), Image.BILINEAR)
                    meta.append({"indice": len(recortes), "escena": escena["name"], "clase": clase})
                    recortes.append(np.asarray(img, dtype=np.uint8))
                    break
            token = muestra["next"]
        print(f"{escena['name']}: {len(filas)} objetos y {len(recortes)} recortes acumulados", flush=True)

    with open(os.path.join(a.salida, "objetos.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    np.save(os.path.join(a.salida, "recortes.npy"), np.stack(recortes))
    with open(os.path.join(a.salida, "recortes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["indice", "escena", "clase"])
        w.writeheader()
        w.writerows(meta)
    with open(os.path.join(a.salida, "particion.json"), "w") as f:
        json.dump(particion, f, indent=1)
    # La versión entra en el job_id: si cambia una decisión de preparación, cambian los job_id.
    variante = f"{a.agrupacion}-{os.path.basename(a.particion).removesuffix('.json')}-{a.visibilidad}-{len(camaras)}cam-{a.tam}px"
    with open(os.path.join(a.salida, "version.txt"), "w") as f:
        f.write(f"nuscenes-{a.version}-prep2-{variante}\n")
    clases = {}
    for m in meta:
        clases[m["clase"]] = clases.get(m["clase"], 0) + 1
    print(f"{len(filas)} objetos y {len(recortes)} recortes en {a.salida}; recortes por clase: {clases}")


if __name__ == "__main__":
    main()
