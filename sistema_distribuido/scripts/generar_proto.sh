#!/usr/bin/env bash
# Regenera sistema/entrenamiento_pb2*.py a partir de proto/sistema/entrenamiento.proto
set -euo pipefail
cd "$(dirname "$0")/.."
python -m grpc_tools.protoc -I proto --python_out=. --grpc_python_out=. proto/sistema/entrenamiento.proto
echo "listo: sistema/entrenamiento_pb2.py y sistema/entrenamiento_pb2_grpc.py"
