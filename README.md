# Pipelines de entrenamiento sobre nuScenes mini

Este repositorio contiene la preparación local de datos y los pipelines CPU de regresión, MLP y CNN. Cada pipeline expone run(config: dict, data_dir, out_dir) -> dict, evalúa sobre la partición de prueba por escena y escribe modelo y métricas en el directorio del trabajo. scripts/ejecutar_referencia.py corre en serie las 34 configuraciones y mide T_base.

## Entorno

Usa Python 3.11.9 y las dependencias fijadas en requirements.txt. Instala PyTorch para CPU y el resto de dependencias:

~~~bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
python -m pip install -r requirements.txt
~~~

## 1. Preparar datos (una vez)

Descarga/ubica v1.0-mini de nuScenes y el directorio samples, sweeps y v1.0-mini bajo el mismo dataroot. La carpeta no se incluye en este repositorio.

~~~bash
python scripts/preparar_datos.py --dataroot D:/datasets/nuScenes --version v1.0-mini --out data/preparados --seed 42
~~~

Genera objetos.csv, recortes.npy y particion.json. La partición se calcula con semilla 42 y separa las escenas completas en train/val/test. Los recortes guardan imagen RGB de 64x64, etiqueta agrupada (vehiculo, peaton, otros) y scene_token.

## 2. Ejecutar un pipeline

Todos usan run(config, data_dir, out_dir). Ejemplos desde la raíz:

~~~bash
python -c "from pipelines.lineal import run; print(run({'method':'ridge','alpha':0.1,'seed':42}, 'data/preparados', 'resultados/demo-lineal'))"
python -c "from pipelines.mlp import run; print(run({'hidden_layers':[64,32],'learning_rate':0.001,'epochs':50,'seed':42}, 'data/preparados', 'resultados/demo-mlp'))"
python -c "from pipelines.cnn import run; print(run({'filters':16,'learning_rate':0.001,'image_size':32,'epochs':100,'min_training_seconds':60,'seed':42}, 'data/preparados', 'resultados/demo-cnn'))"
~~~

La regresión y el MLP predicen num_lidar_pts con distancia, dimensiones, categoría y visibilidad; devuelven RMSE y R². La CNN devuelve exactitud y matriz de confusión. La CNN CPU continúa entrenando hasta cubrir 100 épocas y al menos 60 segundos de entrenamiento por configuración de referencia. MLP y CNN guardan checkpoints por job_id, y se pueden reanudar con resume: true (predeterminado). Todas las semillas predeterminadas son 42. Los modelos finales se guardan mediante archivo temporal y renombrado atómico.

## 3. Medir la referencia de 34 trabajos

~~~bash
python scripts/ejecutar_referencia.py --data data/preparados --out resultados/referencia
~~~

El proceso ejecuta 8 configuraciones Ridge/Lasso, 18 MLP y 8 CNN de forma secuencial, en un solo proceso y CPU. Emite T_base_seconds, tiempo por trabajo, configuración y métricas en resultados/referencia/reporte_referencia.json. El tiempo depende del hardware. Este proyecto no incluye nuScenes ni resultados precalculados: el reporte se crea al correrlo localmente con datos preparados y no se deben presentar cifras antes de esa corrida.

## Diseño de archivos

- scripts/preparar_datos.py: extracción de anotaciones, recortes y partición.
- pipelines/lineal.py, pipelines/mlp.py, pipelines/cnn.py: entrenamiento y evaluación.
- pipelines/_common.py: lectura y escritura atómica compartida.
- scripts/ejecutar_referencia.py: grilla fija y cronómetro de referencia.

La implementación de coordinador, nodos, gRPC y Docker corresponde al resto del equipo, según el alcance indicado para esta parte.
