#!/usr/bin/env bash
# Demo de tolerancia a fallos: mata (docker kill) el nodo que corre una CNN a media ejecución
# y muestra cómo el coordinador lo detecta por heartbeat y reasigna el trabajo, que sigue
# desde su checkpoint en otro nodo.
#   Requisitos: sistema arriba con >= 2 nodos (docker compose up -d --build).
#   Uso: bash scripts/demo_falla.sh [epoca_en_la_que_matar=2]
set -euo pipefail
cd "$(dirname "$0")/.."
EPOCA="${1:-2}"
cli() { docker compose run --rm -T cliente "$@"; }

echo "== Nodos antes de la falla"
cli nodos

echo "== Envío una CNN (si ya existe, el coordinador devuelve la existente)"
cli enviar cnn --config '{"epocas": 12, "filtros": 16, "capas": 3, "tam_imagen": 64}' | tee /tmp/demo_envio.txt
JOB=$(awk '/^  [0-9a-f]{16}/{print $1; exit}' /tmp/demo_envio.txt)

echo "== Espero a que la CNN llegue a la época $EPOCA"
read -r JOB_CORRIENDO NODO < <(cli esperar-corriendo --modelo cnn --epoca "$EPOCA")
echo "   trabajo $JOB_CORRIENDO corriendo en el nodo $NODO"

# El WORKER_ID por defecto es el hostname del contenedor = su ID corto de Docker.
CONTENEDOR=$(docker ps --filter "label=com.docker.compose.service=worker" --format '{{.ID}} {{.Names}}' \
             | awk -v n="$NODO" 'index($1, n) == 1 || $2 == n {print $2}')
echo "== docker kill $CONTENEDOR   (falla inyectada a media ejecución)"
docker kill "$CONTENEDOR" >/dev/null

echo "== Durante ~15 s: el coordinador deja de recibir heartbeats y reasigna"
for _ in $(seq 1 8); do
  sleep 2
  cli estado --job "$JOB_CORRIENDO" | sed -n 2p
done
cli nodos

echo "== Espero a que termine en otro nodo"
until cli estado --job "$JOB_CORRIENDO" | grep -qE "TERMINADO|FALLIDO"; do sleep 3; done
cli estado --job "$JOB_CORRIENDO"
cli resultado "$JOB_CORRIENDO" | grep -E '"(nodo|continuado_desde_epoca|exactitud)"' | head -5

echo "== Un solo resultado por job_id en el disco del coordinador:"
docker compose exec -T coordinador sh -c "ls /estado/resultados | grep -c $JOB_CORRIENDO"
echo "== Para volver a levantar el nodo: docker compose up -d --scale worker=\${NODOS:-2}"
