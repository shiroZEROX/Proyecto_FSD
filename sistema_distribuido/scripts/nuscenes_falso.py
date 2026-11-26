"""Genera un nuScenes FALSO con el esquema de v1.0-mini para probar datos/preparar_nuscenes.py sin los datos reales.

Cada objeto se pinta en las cámaras como un disco de color según su clase (rojo vehículo, verde peatón,
azul otro), así se puede revisar que cada recorte cae sobre su objeto.
  python scripts/nuscenes_falso.py /tmp/nuscenes_falso   (requiere nuscenes-devkit, ver requirements-datos.txt)
"""
import json, os, sys, uuid
import numpy as np
from PIL import Image
from pyquaternion import Quaternion

root = sys.argv[1]; ver = "v1.0-mini"
rng = np.random.default_rng(0)
tok = lambda: uuid.UUID(int=int(rng.integers(0, 2**63)) << 64 | int(rng.integers(0, 2**63))).hex
T = {k: [] for k in ["category","attribute","visibility","instance","sensor","calibrated_sensor","ego_pose","log","scene","sample","sample_data","sample_annotation","map"]}
cats = ["vehicle.car","vehicle.truck","vehicle.bicycle","human.pedestrian.adult","movable_object.barrier","movable_object.trafficcone"]
cat_tok = {}
COL = {"vehicle.car":(255,0,0),"vehicle.truck":(255,0,0),"vehicle.bicycle":(0,0,255),"human.pedestrian.adult":(0,255,0),"movable_object.barrier":(0,0,255),"movable_object.trafficcone":(0,0,255)}
for c in cats:
    t = tok(); cat_tok[c] = t; T["category"].append({"token": t, "name": c, "description": c})
for i, lv in enumerate(["v0-40","v40-60","v60-80","v80-100"], 1):
    T["visibility"].append({"token": str(i), "level": lv, "description": lv})
K = [[1266.4, 0, 816.3], [0, 1266.4, 491.5], [0, 0, 1]]
R_cam = np.array([[0,0,1],[-1,0,0],[0,-1,0]], float)  # columnas: ejes de la cámara en el marco del ego
cams = {"CAM_FRONT":0,"CAM_FRONT_LEFT":55,"CAM_FRONT_RIGHT":-55,"CAM_BACK":180,"CAM_BACK_LEFT":110,"CAM_BACK_RIGHT":-110}
cs = {}
for ch, yaw in list(cams.items()) + [("LIDAR_TOP", None)]:
    st = tok(); T["sensor"].append({"token": st, "channel": ch, "modality": "lidar" if yaw is None else "camera"})
    ct = tok(); cs[ch] = ct
    if yaw is None:
        T["calibrated_sensor"].append({"token": ct, "sensor_token": st, "translation": [0.9,0,1.8], "rotation": [1,0,0,0], "camera_intrinsic": []})
    else:
        R = Quaternion(axis=[0,0,1], degrees=yaw).rotation_matrix @ R_cam
        T["calibrated_sensor"].append({"token": ct, "sensor_token": st, "translation": [1.5,0,1.5], "rotation": list(Quaternion(matrix=R).elements), "camera_intrinsic": K})
logt = tok(); T["log"].append({"token": logt, "logfile": "fake", "vehicle": "n015", "date_captured": "2018-07-24", "location": "singapore-onenorth"})
os.makedirs(os.path.join(root, "maps"), exist_ok=True)
Image.new("L", (10, 10)).save(os.path.join(root, "maps", "fake.png"))
T["map"].append({"token": tok(), "log_tokens": [logt], "category": "semantic_prior", "filename": "maps/fake.png"})
for d in ["samples/" + c for c in list(cams) + ["LIDAR_TOP"]]:
    os.makedirs(os.path.join(root, d), exist_ok=True)
ts = 1532402927647951
names = [f"scene-{n:04d}" for n in (61, 103, 553, 655, 757, 796, 916, 1077, 1094, 1100)]
for name in names:
    sc = tok(); n_s = 3
    s_toks = [tok() for _ in range(n_s)]
    ego_xy = np.array(rng.uniform(0, 500, 2))
    # objetos de la escena: posición relativa al ego, ángulo en grados
    objs = []
    for _ in range(int(rng.integers(4, 8))):
        c = cats[int(rng.integers(len(cats)))]
        ang = float(rng.choice(list(cams.values())) + rng.uniform(-15, 15)); dist = float(rng.uniform(6, 30))
        size = {"vehicle.car":[1.9,4.6,1.7],"vehicle.truck":[2.5,7,2.9],"vehicle.bicycle":[0.6,1.7,1.3],"human.pedestrian.adult":[0.7,0.7,1.75],"movable_object.barrier":[2.5,0.5,1],"movable_object.trafficcone":[0.4,0.4,0.9]}[c]
        it = tok(); objs.append((c, ang, dist, size, it, []))
        T["instance"].append({"token": it, "category_token": cat_tok[c], "nbr_annotations": n_s, "first_annotation_token": "", "last_annotation_token": ""})
    for i, st in enumerate(s_toks):
        t = ts + i * 500000
        T["sample"].append({"token": st, "timestamp": t, "prev": s_toks[i-1] if i else "", "next": s_toks[i+1] if i < n_s-1 else "", "scene_token": sc})
        pos = [float(ego_xy[0] + 2*i), float(ego_xy[1]), 0.0]
        objpos = []
        for (c, ang, dist, size, it, anns) in objs:
            a = np.deg2rad(ang)
            objpos.append((c, np.array([pos[0] + dist*np.cos(a) + 0.5*i, pos[1] + dist*np.sin(a), size[2]/2]), size))
        for ch in list(cams) + ["LIDAR_TOP"]:
            ep = tok(); T["ego_pose"].append({"token": ep, "timestamp": t, "rotation": [1,0,0,0], "translation": pos})
            fn = f"samples/{ch}/{name}_{i}_{ch}." + ("pcd.bin" if ch == "LIDAR_TOP" else "jpg")
            if ch != "LIDAR_TOP":
                img = np.full((900, 1600, 3), 128, np.uint8)
                Rc = Quaternion(axis=[0,0,1], degrees=cams[ch]).rotation_matrix @ R_cam
                yy, xx = np.mgrid[:900, :1600]
                for (c, w, size) in sorted(objpos, key=lambda o: -np.linalg.norm(o[1]-np.array(pos))):
                    pc = Rc.T @ (w - np.array(pos) - np.array([1.5, 0, 1.5]))
                    if pc[2] < 1: continue
                    u, v = 1266.4*pc[0]/pc[2] + 816.3, 1266.4*pc[1]/pc[2] + 491.5
                    r = max(3, 1266.4*min(size)/pc[2]/2)
                    img[(xx-u)**2 + (yy-v)**2 < r*r] = COL[c]
                Image.fromarray(img).save(os.path.join(root, fn), quality=95)
            else:
                open(os.path.join(root, fn), "wb").close()
            T["sample_data"].append({"token": tok(), "sample_token": st, "ego_pose_token": ep, "calibrated_sensor_token": cs[ch], "timestamp": t, "fileformat": "jpg" if ch != "LIDAR_TOP" else "pcd", "is_key_frame": True, "height": 900 if ch != "LIDAR_TOP" else 0, "width": 1600 if ch != "LIDAR_TOP" else 0, "filename": fn, "prev": "", "next": ""})
        for (c, ang, dist, size, it, anns) in objs:
            a = np.deg2rad(ang); at = tok(); anns.append(at)  # misma posición que objpos
            T["sample_annotation"].append({"token": at, "sample_token": st, "instance_token": it, "visibility_token": str(int(rng.integers(1, 5))), "attribute_tokens": [],
                "translation": [pos[0] + dist*np.cos(a) + 0.5*i, pos[1] + dist*np.sin(a), size[2]/2], "size": size, "rotation": list(Quaternion(axis=[0,0,1], angle=float(rng.uniform(0, 3))).elements),
                "prev": "", "next": "", "num_lidar_pts": int(rng.integers(0, 400)), "num_radar_pts": int(rng.integers(0, 5))})
    for (c, ang, dist, size, it, anns) in objs:
        idx = {r["token"]: r for r in T["sample_annotation"]}
        for j, at in enumerate(anns):
            idx[at]["prev"] = anns[j-1] if j else ""; idx[at]["next"] = anns[j+1] if j < len(anns)-1 else ""
        inst = next(r for r in T["instance"] if r["token"] == it); inst["first_annotation_token"] = anns[0]; inst["last_annotation_token"] = anns[-1]
    T["scene"].append({"token": sc, "log_token": logt, "nbr_samples": n_s, "first_sample_token": s_toks[0], "last_sample_token": s_toks[-1], "name": name, "description": "fake"})
    ts += 20_000_000
os.makedirs(os.path.join(root, ver), exist_ok=True)
for k, v in T.items():
    json.dump(v, open(os.path.join(root, ver, k + ".json"), "w"))
print("anotaciones:", len(T["sample_annotation"]))
