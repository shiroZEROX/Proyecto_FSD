#!/usr/bin/env python3
"""Run the fixed 34-job reference grid sequentially in one process."""
from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipelines._common import atomic_json, job_id


def reference_jobs():
    seed = 42
    for method, alpha in itertools.product(("ridge", "lasso"), (.01, .1, 1., 10.)):
        yield "lineal", {"method": method, "alpha": alpha, "seed": seed}
    for layers, rate, epochs in itertools.product(((64, 32), (128, 64), (256, 128)), (.001, .0005, .01), (50, 100)):
        yield "mlp", {"hidden_layers": layers, "learning_rate": rate, "epochs": epochs, "seed": seed}
    for filters, rate, size in itertools.product((16, 32), (.001, .0005), (32, 64)):
        yield "cnn", {"filters": filters, "learning_rate": rate, "image_size": size,
                      "epochs": 100, "min_training_seconds": 60, "seed": seed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/preparados"))
    parser.add_argument("--out", type=Path, default=Path("resultados/referencia"))
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    jobs = list(reference_jobs())
    if len(jobs) != 34: raise RuntimeError(f"Expected 34 reference jobs, got {len(jobs)}")
    results = []; started = time.perf_counter()
    for index, (name, config) in enumerate(jobs, 1):
        jid = job_id({"pipeline": name, "config": config})
        job_dir = args.out / jid
        pipeline = importlib.import_module(f"pipelines.{name}")
        t0 = time.perf_counter()
        metrics = pipeline.run(config, args.data, job_dir)
        elapsed = time.perf_counter() - t0
        results.append({"job_id": jid, "pipeline": name, "config": config,
                        "metrics": metrics, "wall_seconds": elapsed})
        print(f"[{index:02d}/34] {name} {jid}: {elapsed:.2f} s")
    total = time.perf_counter() - started
    report = {"jobs": results, "job_count": len(results), "T_base_seconds": total,
              "measurement": "single process, sequential, CPU", "seed": 42}
    atomic_json(report, args.out / "reporte_referencia.json")
    print(f"T_base = {total:.3f} s; reporte: {args.out / 'reporte_referencia.json'}")


if __name__ == "__main__": main()
