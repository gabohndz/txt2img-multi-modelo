#!/usr/bin/env bash
# Arranca la interfaz en Linux / macOS. Requiere ComfyUI ya corriendo (ver README).
set -e
cd "$(dirname "$0")"

if [ -x "venv/bin/python" ]; then
    PY="venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python3"
fi
echo "[1/2] Python: $PY"

echo "[2/2] Comprobando ComfyUI en http://localhost:8188 ..."
if curl -s -o /dev/null http://localhost:8188/system_stats; then
    echo "      ComfyUI detectado."
else
    echo "      AVISO: ComfyUI no responde. Arrancalo con: python main.py --listen"
fi

echo
echo "Iniciando interfaz en http://localhost:7862 (Ctrl+C para detenerla)"
exec "$PY" interfaz_txt2img_multi.py
