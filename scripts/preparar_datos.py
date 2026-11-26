#!/usr/bin/env python3
"""Prepare nuScenes mini annotations and camera crops for the training pipelines."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points


def category_group(name: str) -> str:
    if name.startswith("vehicle."):
        return "vehiculo"
    if name.startswith("human.pedestrian"):
        return "peaton"
    return "otros"


def prepare(dataset: Path, output: Path, version: str, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    nusc = NuScenes(version=version, dataroot=str(dataset), verbose=True)
    scenes = sorted(scene["token"] for scene in nusc.scene)
    rng = np.random.default_rng(seed)
    shuffled = [scenes[i] for i in rng.permutation(len(scenes))]
    # A fixed 60/20/20 scene split, with at least one scene in each partition where possible.
    n = len(shuffled)
    n_train = max(1, int(n * .6))
    n_val = max(1, int(n * .2)) if n >= 3 else 0
    split = {"train": shuffled[:n_train], "val": shuffled[n_train:n_train+n_val],
             "test": shuffled[n_train+n_val:]}
    if n >= 3 and not split["test"]:
        split["test"] = split["val"][-1:]
        split["val"] = split["val"][:-1]
    (output / "particion.json").write_text(json.dumps({"seed": seed, "version": version,
        "scenes": split}, ensure_ascii=False, indent=2), encoding="utf-8")

    fields = ["category", "ancho", "largo", "alto", "distancia", "visibilidad",
              "num_lidar_pts", "num_radar_pts", "velocidad", "scene_token"]
    rows, crops = [], []
    for sample in nusc.sample:
        scene_token = nusc.get("scene", sample["scene_token"])["token"]
        cam = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
        ego = np.asarray(nusc.get("ego_pose", cam["ego_pose_token"])["translation"], dtype=np.float64)
        image_path, boxes, intrinsic = nusc.get_sample_data(cam["token"])
        image = Image.open(image_path).convert("RGB")
        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            category = ann["category_name"]
            center = np.asarray(ann["translation"], dtype=np.float64)
            speed_vec = nusc.box_velocity(ann_token)
            speed = float(np.linalg.norm(speed_vec)) if np.all(np.isfinite(speed_vec)) else 0.0
            rows.append({"category": category, "ancho": ann["size"][0], "largo": ann["size"][1],
                "alto": ann["size"][2], "distancia": float(np.linalg.norm(center - ego)),
                "visibilidad": int(ann["visibility_token"]), "num_lidar_pts": int(ann.get("num_lidar_pts", 0)),
                "num_radar_pts": int(ann.get("num_radar_pts", 0)), "velocidad": speed,
                "scene_token": scene_token})
            box = next((b for b in boxes if b.token == ann_token), None)
            if box is None:
                continue
            pts = view_points(box.corners(), np.asarray(intrinsic), normalize=True)
            visible = box.corners()[2, :] > 0.1
            if not np.any(visible):
                continue
            x0, y0 = np.floor(pts[:2, visible].min(axis=1)).astype(int)
            x1, y1 = np.ceil(pts[:2, visible].max(axis=1)).astype(int)
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image.width, x1), min(image.height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            crop = image.crop((x0, y0, x1, y1)).resize((64, 64), Image.Resampling.BILINEAR)
            crops.append((np.asarray(crop, dtype=np.uint8), category_group(category), scene_token))
    with (output / "objetos.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    recortes = np.empty(len(crops), dtype=[("image", "u1", (64, 64, 3)), ("label", "U12"), ("scene_token", "U32")])
    for i, item in enumerate(crops):
        recortes[i] = item
    np.save(output / "recortes.npy", recortes, allow_pickle=False)
    print(f"Objetos: {len(rows)}; recortes: {len(crops)}; escenas: {n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/preparados"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare(args.dataroot, args.out, args.version, args.seed)


if __name__ == "__main__":
    main()
