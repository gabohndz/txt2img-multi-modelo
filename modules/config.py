"""
config.py — Resolucion de rutas de ComfyUI y de modelos.

Orden de prioridad:
    1. Variable de entorno COMFYUI_PATH (gana siempre que este definida)
    2. config.json en la raiz del repositorio
    3. Deteccion automatica en ubicaciones habituales

Si no se encuentra ninguna instalacion de ComfyUI, las rutas quedan apuntando
a un candidato por defecto y las funciones de listado devuelven listas vacias
(la interfaz arranca igual, pero no muestra modelos). Ver README, seccion
"Configuracion".
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

RAIZ_REPO = Path(__file__).resolve().parent.parent
CONFIG_JSON = RAIZ_REPO / "config.json"

# Claves validas de config.json y sus valores por defecto.
# Cadena vacia = "derivar automaticamente".
POR_DEFECTO = {
    "comfyui_path": "",
    "comfyui_api": "http://localhost:8188",
    "checkpoints_dir": "",
    "loras_dir": "",
    "output_dir": "",
}

# Ubicaciones donde suele instalarse ComfyUI si el usuario no configura nada.
CANDIDATOS = [
    Path("C:/ComfyUI"),
    Path("C:/AI/ComfyUI"),
    Path.home() / "ComfyUI",
    Path.home() / "Documents" / "ComfyUI",
    Path.home() / "Desktop" / "ComfyUI",
    Path.home() / "Desktop" / "Data" / "Packages" / "ComfyUI",
    Path("/opt/ComfyUI"),
    Path("/workspace/ComfyUI"),
    Path("/content/ComfyUI"),
    RAIZ_REPO / "ComfyUI",
]


def _leer_config() -> dict:
    """Lee config.json (si existe) sobre los valores por defecto."""
    cfg = dict(POR_DEFECTO)
    if not CONFIG_JSON.exists():
        return cfg

    try:
        datos = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"[config] config.json tiene un JSON invalido: {e}")

    if not isinstance(datos, dict):
        sys.exit("[config] config.json debe contener un objeto JSON { ... }")

    desconocidas = set(datos) - set(POR_DEFECTO)
    if desconocidas:
        print(f"[config] Aviso: claves ignoradas en config.json: {sorted(desconocidas)}")

    cfg.update({k: v for k, v in datos.items() if k in POR_DEFECTO})
    return cfg


def _resolver_comfyui(cfg: dict) -> Optional[Path]:
    """Devuelve la carpeta de ComfyUI, o None si no se encuentra."""
    if os.environ.get("COMFYUI_PATH"):
        return Path(os.environ["COMFYUI_PATH"])

    if cfg["comfyui_path"]:
        return Path(cfg["comfyui_path"])

    for candidato in CANDIDATOS:
        if (candidato / "main.py").exists():
            return candidato

    return None


def _resolver_ruta(valor: str, derivada: Path) -> Path:
    """Ruta absoluta configurada, o <comfyui_path>/<derivada>, o <repo>/<derivada>."""
    if valor:
        return Path(valor)
    if COMFYUI_PATH is not None:
        return COMFYUI_PATH / derivada
    return RAIZ_REPO / derivada


CONFIG = _leer_config()
COMFYUI_PATH = _resolver_comfyui(CONFIG)

COMFY_API = CONFIG["comfyui_api"]
COMFY_OUTPUT = _resolver_ruta(CONFIG["output_dir"], Path("output"))
RUTA_CHECKPOINTS = _resolver_ruta(CONFIG["checkpoints_dir"], Path("models") / "checkpoints")
RUTA_LORAS = _resolver_ruta(CONFIG["loras_dir"], Path("models") / "loras")


def resumen() -> str:
    """Estado legible de la configuracion, util para logs y depuracion."""
    if os.environ.get("COMFYUI_PATH"):
        origen = "variable de entorno COMFYUI_PATH"
    elif CONFIG["comfyui_path"]:
        origen = "config.json"
    else:
        origen = "autodeteccion"

    return "\n".join([
        f"config.json:  {'leido' if CONFIG_JSON.exists() else 'no existe (valores por defecto)'}",
        f"ComfyUI:      {COMFYUI_PATH if COMFYUI_PATH else 'NO ENCONTRADO'}  [{origen}]",
        f"API:          {COMFY_API}",
        f"Checkpoints:  {RUTA_CHECKPOINTS}",
        f"LoRAs:        {RUTA_LORAS}",
        f"Output:       {COMFY_OUTPUT}",
    ])
