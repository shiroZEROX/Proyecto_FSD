"""Utilidades de PyTorch compartidas por mlp.py y cnn.py: determinismo y checkpoints."""
import io
import os

import torch


def preparar_determinismo(semilla: int):
    torch.manual_seed(semilla)
    torch.use_deterministic_algorithms(True)
    # Mismo número de hilos en todos los nodos: cambiarlo puede mover las métricas en el
    # último decimal porque cambia el orden de las sumas.
    torch.set_num_threads(int(os.environ.get("HILOS_TORCH", "2")))


def generador_epoca(semilla: int, epoca: int) -> torch.Generator:
    """El orden de los lotes de cada época depende solo de (semilla, época).

    Así, continuar desde un checkpoint da exactamente lo mismo que no haberse detenido.
    """
    g = torch.Generator()
    g.manual_seed(semilla * 100003 + epoca)
    return g


def serializar(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def deserializar(datos: bytes):
    return torch.load(io.BytesIO(datos), map_location="cpu", weights_only=False)


def entrenar_epocas(modelo, optim, perdida_fn, X, y, config, semilla, progreso, checkpoint, extra_ckpt=None):
    """Bucle de entrenamiento común con checkpoint por época. Devuelve el historial de pérdidas."""
    inicio, historial = 0, []
    if checkpoint:
        estado = deserializar(checkpoint)
        modelo.load_state_dict(estado["modelo"])
        optim.load_state_dict(estado["optim"])
        inicio, historial = estado["epoca"], estado["historial"]
    n = X.shape[0]
    retardo = float(os.environ.get("RETARDO_POR_EPOCA", "0"))  # solo para simular un nodo lento
    for epoca in range(inicio, config["epocas"]):
        modelo.train()
        orden = torch.randperm(n, generator=generador_epoca(semilla, epoca))
        total = 0.0
        for i in range(0, n, config["batch"]):
            idx = orden[i:i + config["batch"]]
            optim.zero_grad()
            loss = perdida_fn(modelo(X[idx]), y[idx])
            loss.backward()
            optim.step()
            total += loss.item() * len(idx)
        historial.append(total / n)
        if retardo:
            import time
            time.sleep(retardo)
        if progreso:
            ckpt = serializar({"epoca": epoca + 1, "modelo": modelo.state_dict(),
                               "optim": optim.state_dict(), "historial": historial, **(extra_ckpt or {})})
            progreso(epoca + 1, historial[-1], ckpt)
    return historial, inicio
