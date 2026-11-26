"""Health check del coordinador para Docker: sale con 0 si responde al Ping."""
import os
import sys

import grpc

from sistema import entrenamiento_pb2 as pb
from sistema import entrenamiento_pb2_grpc as pbg

try:
    canal = grpc.insecure_channel(f"localhost:{os.environ.get('PUERTO', '50051')}")
    pbg.CoordinadorStub(canal).Ping(pb.Vacio(), timeout=2)
except grpc.RpcError:
    sys.exit(1)
