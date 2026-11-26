#!/usr/bin/env bash
# Mide T_distribuido del conjunto de referencia con 1, 2 y 4 nodos, REPS veces cada uno.
# Cuenta el arranque de contenedores (desde 'up' hasta el último trabajo terminado).
# Antes: correr experimentos/baseline.py en el mismo hardware para obtener T_base.
#   Uso: REPS=3 bash scripts/medir.sh [1 2 4]
set -euo pipefail
cd "$(dirname "$0")/.."
REPS="${REPS:-3}"
NODOS_LISTA=("${@:-1 2 4}")
SALIDA=experimentos/mediciones.csv
[ -f "$SALIDA" ] || echo "nodos,repeticion,segundos" > "$SALIDA"
docker compose build
for N in ${NODOS_LISTA[@]}; do
  for R in $(seq 1 "$REPS"); do
    docker compose down -v --remove-orphans >/dev/null 2>&1 || true   # estado limpio: si no, todo sale "ya existía"
    T0=$(date +%s.%N)
    NODOS=$N docker compose up -d --scale worker="$N" --wait coordinador worker
    docker compose run --rm -T cliente referencia experimentos/referencia.json --esperar > "experimentos/medicion_${N}n_${R}.log"
    T1=$(date +%s.%N)
    S=$(echo "$T1 - $T0" | bc)
    echo "$N,$R,$S" | tee -a "$SALIDA"
    docker compose run --rm -T cliente exportar "experimentos/sistema_${N}n_${R}.csv" >/dev/null
  done
done
docker compose down -v
echo "listo: $SALIDA  (reducción % = (T_base - T_dist) / T_base * 100)"
