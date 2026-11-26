#!/usr/bin/env python3
"""Menú para ejecutar el proyecto sin recordar los comandos.

  python menu.py          menú interactivo
  python menu.py 4        ejecuta directo la opción 4 (pide lo que falte)

Solo usa la biblioteca estándar de Python 3 y llama a Docker Compose. Funciona en Linux, macOS y
Windows. Las demos y la medición son scripts de bash: en Windows hace falta Git Bash o WSL.
Cada opción imprime el comando que corre, para poder copiarlo al README o a la demo.
"""
import os
import shlex
import shutil
import subprocess
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
ENV = os.path.join(RAIZ, ".env")


# ------------------------------------------------------------ utilidades

def correr(cmd, env_extra=None, revisar=False):
    """Muestra y ejecuta un comando desde la raíz del proyecto. Devuelve el código de salida."""
    print("\n$ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    env = dict(os.environ, **(env_extra or {}))
    try:
        r = subprocess.run(cmd, cwd=RAIZ, env=env)
    except FileNotFoundError:
        print(f"No encontré '{cmd[0]}'. ¿Está instalado y en el PATH?")
        return 127
    except KeyboardInterrupt:
        print("\n(interrumpido)")
        return 130
    if r.returncode != 0:
        print(f"(terminó con código {r.returncode})")
        if revisar:
            raise SystemExit(r.returncode)
    return r.returncode


def compose(*args, **kw):
    return correr(["docker", "compose", *args], **kw)


def cliente(*args):
    return compose("run", "--rm", "cliente", *args)


def bash(script, *args, env_extra=None):
    if not shutil.which("bash"):
        print("Esta opción usa bash. En Windows ábrela desde Git Bash o WSL.")
        return 1
    return correr(["bash", script, *args], env_extra=env_extra)


def preguntar(texto, defecto=""):
    try:
        r = input(f"{texto}" + (f" [{defecto}]" if defecto else "") + ": ").strip()
    except EOFError:
        r = ""
    return r or defecto


def confirmar(texto):
    return preguntar(f"{texto} (s/n)", "n").lower() in ("s", "si", "sí", "y")


def leer_env():
    valores = {}
    if os.path.exists(ENV):
        with open(ENV) as f:
            for linea in f:
                if "=" in linea and not linea.lstrip().startswith("#"):
                    k, v = linea.rstrip("\n").split("=", 1)
                    valores[k.strip()] = v.strip()
    return valores


def escribir_env(**nuevos):
    """Guarda DATOS / VERSION_DATOS / NODOS en .env: así todos los comandos de compose usan lo mismo."""
    valores = leer_env()
    valores.update({k: str(v) for k, v in nuevos.items()})
    with open(ENV, "w") as f:
        f.write("# Lo escribe menu.py. Docker Compose lo lee solo.\n")
        for k, v in valores.items():
            f.write(f"{k}={v}\n")
    print("   .env => " + ", ".join(f"{k}={v}" for k, v in valores.items()))


def usar_datos(carpeta_rel):
    """Apunta el sistema a una carpeta de datos preparados (ruta relativa a la raíz)."""
    version_txt = os.path.join(RAIZ, carpeta_rel, "version.txt")
    if not os.path.exists(version_txt):
        print(f"No hay {carpeta_rel}/version.txt: esa carpeta no tiene datos preparados.")
        return False
    with open(version_txt) as f:
        version = f.read().strip()
    escribir_env(DATOS="./" + carpeta_rel.replace(os.sep, "/"), VERSION_DATOS=version)
    return True


def nodos_actuales():
    return leer_env().get("NODOS", "2")


# ------------------------------------------------------------ opciones

def op_sinteticos():
    """Genera datos sintéticos (para probar sin nuScenes)."""
    if compose("--profile", "datos", "run", "--rm", "--entrypoint", "python", "preparar",
               "datos/generar_sinteticos.py", "--salida", "/salida/generados") == 0:
        usar_datos(os.path.join("datos", "generados"))


def op_preparar():
    """Prepara nuScenes mini (una sola vez)."""
    print("La carpeta debe tener v1.0-mini/, samples/, sweeps/ y maps/.")
    ruta = preguntar("Ruta a nuScenes en esta computadora", leer_env().get("NUSCENES", "./nuscenes"))
    if not os.path.isdir(os.path.join(ruta, "v1.0-mini")):
        print(f"No encuentro {os.path.join(ruta, 'v1.0-mini')}.")
        return
    agrupacion = preguntar("Agrupación (esqueleto | equipo)", "esqueleto")
    particion = preguntar("Partición (nombres | equipo | ruta a JSON dentro de datos/)", "nombres")
    escribir_env(NUSCENES=os.path.abspath(ruta))
    extra = ["--agrupacion", agrupacion, "--particion", particion]
    if compose("--profile", "datos", "run", "--rm", "preparar", "--dataroot", "/nuscenes",
               "--version", "v1.0-mini", "--salida", "/salida/nuscenes_mini", *extra) == 0:
        usar_datos(os.path.join("datos", "nuscenes_mini"))


def op_falso():
    """Prueba la preparación con un nuScenes falso."""
    if compose("--profile", "datos", "run", "--rm", "--entrypoint", "python", "preparar",
               "scripts/nuscenes_falso.py", "/salida/nuscenes_falso") == 0:
        compose("--profile", "datos", "run", "--rm", "preparar", "--dataroot", "/salida/nuscenes_falso",
                "--salida", "/salida/prueba_prep")
        print("Salida en datos/prueba_prep (no se usa para el sistema).")


def op_levantar():
    """Levanta el sistema con 1, 2 o 4 nodos."""
    env = leer_env()
    if "VERSION_DATOS" not in env:
        print(f"Primero elige datos (opción {num(op_sinteticos)} o {num(op_preparar)}): sin VERSION_DATOS en .env los nodos no reciben trabajos.")
        return
    n = preguntar("¿Cuántos nodos? (1, 2, 4...)", nodos_actuales())
    if not n.isdigit() or int(n) < 1:
        print("Número de nodos inválido.")
        return
    escribir_env(NODOS=n)
    compose("up", "-d", "--build", "--wait", "--scale", f"worker={n}", "coordinador", "worker")
    cliente("nodos")


def op_estado():
    """Ver nodos y estado de los trabajos."""
    cliente("nodos")
    cliente("estado")


def op_referencia():
    """Envía el conjunto de referencia (34 trabajos)."""
    esperar = confirmar("¿Esperar a que terminen?")
    cliente("referencia", "experimentos/referencia.json", *(["--esperar"] if esperar else []))


def op_enviar():
    """Envía un trabajo o un grid."""
    modelo = preguntar("Modelo (lineal | mlp | cnn)", "lineal")
    config = preguntar("Configuración JSON", "{}")
    grid = preguntar("Grid JSON (vacío = sin grid)", "")
    args = ["enviar", modelo, "--config", config]
    if grid:
        args += ["--grid", grid]
    if confirmar("¿Esperar a que termine?"):
        args.append("--esperar")
    cliente(*args)


def op_resultado():
    """Ver el resultado de un job_id."""
    job = preguntar("job_id")
    if job:
        cliente("resultado", job)


def op_demo_falla():
    """Demo: mata un nodo a media CNN y se reasigna."""
    epoca = preguntar("¿En qué época matar el nodo?", "2")
    bash("scripts/demo_falla.sh", epoca)


def op_demo_idempotencia():
    """Demo: envío repetido = un solo resultado."""
    bash("scripts/demo_idempotencia.sh")


def op_nodo_lento():
    """Demo: nodo lento, vence el plazo y se reasigna."""
    print("Reinicia el sistema con nodos lentos (5 s por época) y plazo de 20 s para el MLP.")
    if confirmar("¿Continuar?"):
        compose("up", "-d", "--wait", "coordinador", "worker",
                env_extra={"RETARDO_POR_EPOCA": "5", "TIMEOUT_MLP": "20"})
        cliente("enviar", "mlp", "--config", '{"epocas": 10}')
        print(f"Mira la reasignación con la opción {num(op_estado)} o con: docker compose logs -f coordinador")
        print(f"Para volver a la normalidad, levanta otra vez con la opción {num(op_levantar)}.")


def op_tbase():
    """Mide T_base (sin el sistema, un proceso)."""
    reps = preguntar("Repeticiones", "3")
    for i in range(int(reps)):
        print(f"\n== T_base, repetición {i + 1}/{reps}")
        compose("run", "--rm", "--no-deps", "-v", f"{os.path.join(RAIZ, 'experimentos')}:/app/experimentos",
                "worker", "python", "experimentos/baseline.py", "--datos", "/datos",
                "--salida", f"experimentos/baseline_{i + 1}.csv")
    print(f"Usa experimentos/baseline_1.csv como referencia de métricas (opción {num(op_comparar)}).")


def op_medir():
    """Mide T_distribuido con 1, 2 y 4 nodos."""
    reps = preguntar("Repeticiones por número de nodos", "3")
    nodos = preguntar("Números de nodos", "1 2 4")
    print("Ojo: cada medición hace 'docker compose down -v' (borra la cola y los resultados).")
    if confirmar("¿Continuar?"):
        bash("scripts/medir.sh", *nodos.split(), env_extra={"REPS": reps})


def op_comparar():
    """Compara métricas del sistema contra la referencia."""
    ref = preguntar("CSV de referencia", "experimentos/baseline_1.csv")
    if not os.path.exists(os.path.join(RAIZ, ref)):
        print(f"No existe {ref}. Corre antes la opción {num(op_tbase)} (T_base).")
        return
    tol = preguntar("Tolerancia", "1e-6")
    cliente("exportar", "experimentos/sistema_actual.csv")
    compose("run", "--rm", "--no-deps", "--entrypoint", "python", "cliente",
            "experimentos/comparar.py", ref, "experimentos/sistema_actual.csv", "--tol", tol)


def op_dashboard():
    """Abre el dashboard (punto extra)."""
    servicios = subprocess.run(["docker", "compose", "config", "--services"], cwd=RAIZ,
                               capture_output=True, text=True).stdout.split()
    if "dashboard" not in servicios:
        print("Todavía no hay servicio 'dashboard' en docker-compose.yml (es el punto extra pendiente).")
        print("Cuando exista, esta opción lo levanta con: docker compose up -d dashboard")
        return
    compose("up", "-d", "dashboard")
    puerto = preguntar("Puerto del dashboard", "8080")
    url = f"http://localhost:{puerto}"
    print(f"Abre {url}")
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


def op_logs():
    """Ver logs del coordinador (Ctrl+C para salir)."""
    compose("logs", "-f", "--tail", "50", "coordinador", "worker")


def op_apagar():
    """Apagar el sistema."""
    if confirmar("¿Borrar también la cola y los resultados (down -v)?"):
        compose("down", "-v")
    else:
        compose("down")


def op_prueba_local():
    """Prueba automática sin Docker (necesita Python con requirements-dev.txt y torch)."""
    correr([sys.executable, "scripts/prueba_local.py"])


OPCIONES = [
    ("Datos", [op_sinteticos, op_preparar, op_falso]),
    ("Sistema", [op_levantar, op_estado, op_referencia, op_enviar, op_resultado]),
    ("Demostraciones", [op_demo_falla, op_demo_idempotencia, op_nodo_lento]),
    ("Medición y métricas", [op_tbase, op_medir, op_comparar]),
    ("Otros", [op_dashboard, op_logs, op_apagar, op_prueba_local]),
]
LISTA = [f for _, grupo in OPCIONES for f in grupo]


def num(funcion):
    return LISTA.index(funcion) + 1


def mostrar():
    env = leer_env()
    print("\n=== Servidor distribuido de entrenamiento ===")
    print(f"datos: {env.get('DATOS', '(sin elegir)')}   versión: {env.get('VERSION_DATOS', '-')}   "
          f"nodos: {env.get('NODOS', '2')}")
    i = 1
    for titulo, grupo in OPCIONES:
        print(f"\n {titulo}")
        for f in grupo:
            print(f"  {i:2d}. {f.__doc__.strip().rstrip('.')}")
            i += 1
    print("\n   0. Salir")


def ejecutar(eleccion):
    if not eleccion.isdigit() or not 0 < int(eleccion) <= len(LISTA):
        print("Opción inválida.")
        return
    LISTA[int(eleccion) - 1]()


def main():
    if not shutil.which("docker"):
        print("AVISO: no encuentro 'docker'. Casi todas las opciones lo necesitan.")
    if len(sys.argv) > 1:
        ejecutar(sys.argv[1])
        return
    while True:
        mostrar()
        try:
            eleccion = input("\nOpción: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if eleccion in ("0", "q", "salir"):
            return
        ejecutar(eleccion)


if __name__ == "__main__":
    main()
