"""CNN en PyTorch, CPU: clasifica el recorte de un objeto en su categoría agrupada."""
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from pipelines import _torch_comun as tc
from pipelines.datos import cargar_imagenes, metricas_clasificacion
from sistema.comun import CLASES


def construir(filtros, capas, n_clases):
    bloques, entrada = [], 3
    for i in range(capas):
        salida = filtros * (2 ** i)
        bloques += [nn.Conv2d(entrada, salida, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]
        entrada = salida
    return nn.Sequential(*bloques, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(entrada, n_clases))


def preparar(imgs: np.ndarray, tam: int) -> torch.Tensor:
    x = torch.tensor(np.ascontiguousarray(imgs), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    if x.shape[-1] != tam:
        x = F.interpolate(x, size=(tam, tam), mode="bilinear", align_corners=False)
    return x


def ejecutar(config, semilla, dir_datos, progreso=None, checkpoint=None):
    t0 = time.time()
    tc.preparar_determinismo(semilla)
    datos = cargar_imagenes(dir_datos)
    tam = config["tam_imagen"]
    X = preparar(datos["train"][0], tam)
    y = torch.tensor(datos["train"][1])
    modelo = construir(config["filtros"], config["capas"], len(CLASES))
    optim = torch.optim.Adam(modelo.parameters(), lr=config["lr"])
    historial, inicio = tc.entrenar_epocas(modelo, optim, nn.CrossEntropyLoss(), X, y, config, semilla,
                                           progreso, checkpoint)
    modelo.eval()
    metricas = {}
    for parte in ("train", "val", "test"):
        imgs, etiquetas = datos[parte]
        with torch.no_grad():
            pred = torch.cat([modelo(preparar(imgs[i:i + 256], tam)).argmax(1)
                              for i in range(0, len(imgs), 256)]).numpy()
        metricas[parte] = metricas_clasificacion(etiquetas, pred)
    metricas.update(tiempo_entrenamiento=round(time.time() - t0, 3), perdidas=[round(v, 6) for v in historial],
                    continuado_desde_epoca=inicio)
    return {"metricas": metricas, "modelo": tc.serializar({"modelo": modelo.state_dict(), "config": config}),
            "formato": "torch"}
