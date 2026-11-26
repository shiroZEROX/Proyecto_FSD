from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ._common import atomic_json, job_id, load_crops, scene_mask


class SmallCNN(nn.Module):
    def __init__(self, filters: int):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, filters, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(filters, filters * 2, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(filters * 2 * 16, 3))

    def forward(self, x): return self.layers(x)


def run(config: dict, data_dir, out_dir) -> dict:
    torch.set_num_threads(int(config.get("num_threads", 1)))
    seed = int(config.get("seed", 42)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    records, part = load_crops(data_dir)
    scenes = records["scene_token"]
    tr, _, te = (scene_mask(scenes, part["scenes"][k]) for k in ("train", "val", "test"))
    labels = {"vehiculo": 0, "peaton": 1, "otros": 2}
    image_size = int(config.get("image_size", 64)); filters = int(config.get("filters", 16))
    images = torch.from_numpy(records["image"].copy()).permute(0, 3, 1, 2).float().div_(255)
    if image_size != 64:
        images = nn.functional.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
    y = torch.tensor([labels[str(x)] for x in records["label"]], dtype=torch.long)
    train_ds = TensorDataset(images[tr], y[tr])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_ds, batch_size=int(config.get("batch_size", 64)), shuffle=True,
                        num_workers=0, generator=generator)
    model = SmallCNN(filters); optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("learning_rate", .001)))
    loss_fn = nn.CrossEntropyLoss()
    out = Path(out_dir); checkpoints = out / "checkpoints"; checkpoints.mkdir(parents=True, exist_ok=True)
    key = job_id({"config": config, "partition": part}); ckpt = checkpoints / f"{key}.pt"
    done = 0; accumulated_seconds = 0.0
    if config.get("resume", True) and ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        done = state["epoch"]
        accumulated_seconds = float(state.get("training_seconds", 0.0))
        torch.set_rng_state(state["torch_rng"]); np.random.set_state(state["numpy_rng"]); random.setstate(state["python_rng"])
        if "loader_rng" in state: generator.set_state(state["loader_rng"])
    epochs = int(config.get("epochs", 100)); min_seconds = float(config.get("min_training_seconds", 60)); model.train()
    started = time.perf_counter()
    epoch = done
    while epoch < epochs or accumulated_seconds + time.perf_counter() - started < min_seconds:
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb); loss.backward(); optimizer.step()
        epoch += 1
        total_elapsed = accumulated_seconds + time.perf_counter() - started
        temp = ckpt.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "training_seconds": total_elapsed,
                    "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                    "loader_rng": generator.get_state()}, temp)
        temp.replace(ckpt)
    elapsed = accumulated_seconds + time.perf_counter() - started
    model.eval()
    with torch.no_grad():
        pred = model(images[te]).argmax(dim=1).numpy(); truth = y[te].numpy()
    matrix = confusion_matrix(truth, pred, labels=[0, 1, 2])
    metrics = {"accuracy": float(accuracy_score(truth, pred)), "confusion_matrix": matrix.tolist(),
        "classes": ["vehiculo", "peaton", "otros"], "n_test": int(te.sum()), "epochs": epoch,
        "training_seconds": elapsed}
    temp_model = out / "model.tmp"
    torch.save({"model": model.state_dict(), "filters": filters, "image_size": image_size, "classes": labels}, temp_model)
    temp_model.replace(out / "model.pt")
    atomic_json(metrics, out / "metrics.json")
    return metrics
