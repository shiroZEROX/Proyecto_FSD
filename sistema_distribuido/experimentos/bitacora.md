# Bitácora de experimentos

Cada experimento se envía por el sistema distribuido (`python -m sistema.cliente enviar ...`), no a mano.
Los job_id salen de la respuesta del cliente o de `python -m sistema.cliente estado --peticion <id>`.

> Las filas de abajo son PLANTILLAS. No hay resultados todavía: se llenan con los datos reales de nuScenes.

## REF — Conjunto de referencia (34 trabajos)

| Campo | Valor |
|---|---|
| ID | REF · peticiones: _(pegar ids)_ |
| Hipótesis | Ridge/Lasso como línea base; el MLP debería bajar el RMSE al capturar la relación no lineal distancia–puntos de lidar (~1/d²). |
| Modelo y datos | lineal, mlp, cnn · nuScenes _(versión)_ · partición por escena _(archivo)_ · variables: distancia, ancho, largo, alto, visibilidad, clase |
| Configuración | `experimentos/referencia.json` · semilla 42 |
| Resultados | _(val/test, tiempo, nodo; mejor job_id de cada grid)_ |
| Conclusión | _(pendiente)_ |

## E1 — Transformar el objetivo con log1p

| Campo | Valor |
|---|---|
| ID | E1 · job_id: _(pendiente)_ |
| Hipótesis | num_lidar tiene cola larga; entrenar sobre log1p(y) debería mejorar R² de la regresión lineal porque log(1/d²) es lineal en log(d). |
| Modelo y datos | lineal y mlp, `{"log_objetivo": true}` |
| Configuración | _(pendiente)_ · semilla 42 |
| Resultados | _(pendiente)_ |
| Conclusión | _(pendiente)_ |

## E2 — Otra variable a predecir (velocidad)

| Campo | Valor |
|---|---|
| ID | E2 · job_id: _(pendiente)_ |
| Hipótesis | _(pendiente)_ |
| Modelo y datos | lineal y mlp, `{"objetivo": "velocidad"}` |
| Configuración | _(pendiente)_ |
| Resultados | _(pendiente)_ |
| Conclusión | _(pendiente)_ |

## E3 — Efecto de normalizar en la regresión lineal

| Campo | Valor |
|---|---|
| ID | E3 · job_id: _(pendiente)_ |
| Hipótesis | Sin normalizar, Lasso penaliza distinto a variables en metros que a la visibilidad 1–4; esperamos peor RMSE con alpha alto. |
| Modelo y datos | lineal, `{"normalizar": false}` vs `true` |
| Configuración | _(pendiente)_ |
| Resultados | _(pendiente)_ |
| Conclusión | _(pendiente)_ |
