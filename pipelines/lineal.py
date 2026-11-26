from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ._common import atomic_json, atomic_pickle, load_objects, scene_mask


def run(config: dict, data_dir, out_dir) -> dict:
    rows, part = load_objects(data_dir)
    scenes = np.asarray([r["scene_token"] for r in rows])
    tr, va, te = (scene_mask(scenes, part["scenes"][k]) for k in ("train", "val", "test"))
    features = ["distancia", "ancho", "largo", "alto", "visibilidad", "category"]
    X = np.asarray([[r[k] for k in features] for r in rows], dtype=object)
    y = np.asarray([float(r["num_lidar_pts"]) for r in rows], dtype=np.float64)
    numeric = [0, 1, 2, 3, 4]
    prep = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [5])])
    kind = config.get("method", config.get("model", "ridge")).lower()
    alpha = float(config.get("alpha", 0.1))
    estimator = Ridge(alpha=alpha) if kind == "ridge" else Lasso(alpha=alpha, max_iter=10000, random_state=int(config.get("seed", 42)))
    model = make_pipeline(prep, estimator)
    model.fit(X[tr], y[tr])
    pred = model.predict(X[te])
    metrics = {"rmse": float(np.sqrt(mean_squared_error(y[te], pred))), "r2": float(r2_score(y[te], pred)),
               "n_test": int(te.sum()), "method": kind, "alpha": alpha}
    out = Path(out_dir); atomic_pickle(model, out / "model.pkl"); atomic_json(metrics, out / "metrics.json")
    return metrics
