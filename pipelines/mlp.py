from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ._common import atomic_json, atomic_pickle, job_id, load_objects, scene_mask


def run(config: dict, data_dir, out_dir) -> dict:
    rows, part = load_objects(data_dir)
    scenes = np.asarray([r["scene_token"] for r in rows])
    tr, _, te = (scene_mask(scenes, part["scenes"][k]) for k in ("train", "val", "test"))
    cols = ["distancia", "ancho", "largo", "alto", "visibilidad", "category"]
    X = np.asarray([[r[k] for k in cols] for r in rows], dtype=object)
    y = np.asarray([float(r["num_lidar_pts"]) for r in rows], dtype=np.float64)
    prep = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), [0, 1, 2, 3, 4]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), [5])])
    Xtr = prep.fit_transform(X[tr]); Xte = prep.transform(X[te])
    if hasattr(Xtr, "toarray"): Xtr, Xte = Xtr.toarray(), Xte.toarray()
    seed = int(config.get("seed", 42)); epochs = int(config.get("epochs", 100))
    key = job_id({"config": config, "partition": part})
    out = Path(out_dir); checkpoints = out / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    ckpt = checkpoints / f"{key}.pkl"
    model = None; done = 0
    if config.get("resume", True) and ckpt.exists():
        with ckpt.open("rb") as f:
            state = __import__("pickle").load(f)
        model, done = state["model"], state["epoch"]
    if model is None:
        model = MLPRegressor(hidden_layer_sizes=tuple(config.get("hidden_layers", (64, 32))),
            learning_rate_init=float(config.get("learning_rate", .001)), max_iter=1, warm_start=True,
            random_state=seed, shuffle=True, tol=0.0, n_iter_no_change=epochs + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(done, epochs):
            model.fit(Xtr, y[tr])
            atomic_pickle({"model": model, "epoch": epoch + 1}, ckpt)
    pred = model.predict(Xte)
    metrics = {"rmse": float(np.sqrt(mean_squared_error(y[te], pred))), "r2": float(r2_score(y[te], pred)),
               "n_test": int(te.sum()), "epochs": epochs}
    atomic_pickle({"preprocessor": prep, "model": model}, out / "model.pkl")
    atomic_json(metrics, out / "metrics.json")
    return metrics
