from __future__ import annotations

import csv
import json
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np


def atomic_pickle(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def atomic_json(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def load_objects(data_dir):
    path = Path(data_dir)
    with (path / "particion.json").open(encoding="utf-8") as f: partition = json.load(f)
    with (path / "objetos.csv").open(encoding="utf-8", newline="") as f: rows = list(csv.DictReader(f))
    return rows, partition


def load_crops(data_dir):
    path = Path(data_dir)
    partition = json.loads((path / "particion.json").read_text(encoding="utf-8"))
    return np.load(path / "recortes.npy", allow_pickle=False), partition


def scene_mask(scenes, selected):
    return np.isin(np.asarray(scenes), np.asarray(selected))


def job_id(config):
    import hashlib
    packed = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(packed).hexdigest()[:24]
