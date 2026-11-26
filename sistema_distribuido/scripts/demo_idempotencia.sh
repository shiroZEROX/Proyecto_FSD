#!/usr/bin/env bash
# Demo de idempotencia: el mismo trabajo enviado dos veces se ejecuta una sola vez.
#   Uso: bash scripts/demo_idempotencia.sh
set -euo pipefail
cd "$(dirname "$0")/.."
cli() { docker compose run --rm -T cliente "$@"; }

echo "== Primer envío"
cli enviar lineal --config '{"tipo": "ridge", "alpha": 0.5}'
echo "== Segundo envío idéntico (alpha escrito como 0.50 y claves en otro orden): mismo job_id, no se re-ejecuta"
cli enviar lineal --config '{"alpha": 0.50, "tipo": "ridge"}'
echo "== Grid repetido: mismos job_id, 0 nuevos"
cli enviar lineal --grid '{"alpha": [0.01, 0.1]}'
cli enviar lineal --grid '{"alpha": [0.1, 0.01]}'
sleep 5
echo "== Estado (cada job_id aparece una vez, con un solo resultado)"
cli estado | head -20
echo "== Resultados guardados en el coordinador (debe ser igual al número de terminados de arriba):"
docker compose exec -T coordinador sh -c 'ls /estado/resultados | grep -vc "^\."'
