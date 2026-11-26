"""MLP (perceptrón multicapa) en PyTorch, CPU. Misma tarea que la regresión lineal."""
import time

import numpy as np
import torch
from torch import nn

from pipelines import _torch_comun as tc
from pipelines.datos import cargar_tabular, metricas_regresion


def construir(n_entradas, capas):
    bloques, anterior = [], n_entradas
    for n in capas:
        bloques += [nn.Linear(anterior, n), nn.ReLU()]
        anterior = n
    bloques.append(nn.Linear(anterior, 1))
    return nn.Sequential(*bloques)


def ejecutar(config, semilla, dir_datos, progreso=None, checkpoint=None):
    t0 = time.time()
    tc.preparar_determinismo(semilla)
    datos = cargar_tabular(dir_datos, config["objetivo"])
    log_y = config["log_objetivo"]
    Xtr, ytr = datos["train"]
    ytr_t = np.log1p(ytr) if log_y else ytr
    # Normalización con estadísticas SOLO de entrenamiento.
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8
    my, sy = ytr_t.mean(), ytr_t.std() + 1e-8

    def a_tensor(X):
        return torch.tensor((X - mx) / sx, dtype=torch.float32)

    X = a_tensor(Xtr)
    y = torch.tensor((ytr_t - my) / sy, dtype=torch.float32).unsqueeze(1)
    modelo = construir(X.shape[1], config["capas"])
    optim = torch.optim.Adam(modelo.parameters(), lr=config["lr"])
    historial, inicio = tc.entrenar_epocas(modelo, optim, nn.MSELoss(), X, y, config, semilla, progreso, checkpoint)

    modelo.eval()

    def predecir(Xp):
        with torch.no_grad():
            p = modelo(a_tensor(Xp)).squeeze(1).double().numpy() * sy + my
        return np.expm1(p) if log_y else p

    metricas = {parte: metricas_regresion(datos[parte][1], predecir(datos[parte][0]))
                for parte in ("train", "val", "test")}
    metricas.update(tiempo_entrenamiento=round(time.time() - t0, 3), perdidas=[round(v, 6) for v in historial],
                    continuado_desde_epoca=inicio)
    paquete = tc.serializar({"modelo": modelo.state_dict(), "config": config,
                             "normalizacion": {"mx": mx, "sx": sx, "my": my, "sy": sy}})
    return {"metricas": metricas, "modelo": paquete, "formato": "torch"}
