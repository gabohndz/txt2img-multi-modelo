@echo off
setlocal
title Generador de imagenes - Multi-Modelo
cd /d "%~dp0"
color 0A

:: Flags con los que se arranca ComfyUI.
:: Si tienes poca VRAM, cambia esta linea por:  set "COMFY_FLAGS=--listen --lowvram"
set "COMFY_FLAGS=--listen"

echo ============================================
echo   Generador de imagenes - Multi-Modelo
echo ============================================
echo.

:: ── [1/4] Python ─────────────────────────────────────────────────
where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] No se encontro "python" en el PATH.
    echo         Instala Python 3.10 o superior desde python.org y marca
    echo         la casilla "Add python.exe to PATH" durante la instalacion.
    echo.
    pause
    exit /b 1
)

if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
    echo [1/4] Usando el entorno virtual .\venv
) else (
    set "PY=python"
    echo [1/4] Sin entorno virtual: se usara el Python del sistema.
    echo       Recomendado: python -m venv venv
    echo                    venv\Scripts\pip install -r requirements.txt
)

:: ── [2/4] Localizar ComfyUI (config.json + autodeteccion) ────────
set "COMFY="
for /f "usebackq delims=" %%p in (`"%PY%" -c "import sys;sys.path.insert(0,'modules');from config import COMFYUI_PATH;print(COMFYUI_PATH or '')" 2^>nul`) do set "COMFY=%%p"

if "%COMFY%"=="" (
    echo [2/4] [AVISO] No se encontro ComfyUI.
    echo       Edita "comfyui_path" en config.json con la ruta de tu ComfyUI.
    echo       La interfaz se abrira igual, pero la lista de modelos estara vacia.
    goto interfaz
)
echo [2/4] ComfyUI en: %COMFY%

:: ── [3/4] Arrancar ComfyUI si no esta corriendo ──────────────────
curl -s -o nul http://localhost:8188/system_stats
if not errorlevel 1 (
    echo [3/4] ComfyUI ya estaba corriendo.
    goto interfaz
)

set "COMFY_PY=%COMFY%\venv\Scripts\python.exe"
if not exist "%COMFY_PY%" set "COMFY_PY=%COMFY%\python_embeded\python.exe"
if not exist "%COMFY_PY%" set "COMFY_PY=python"

echo [3/4] Arrancando ComfyUI en segundo plano...
start "ComfyUI" /MIN /D "%COMFY%" "%COMFY_PY%" main.py %COMFY_FLAGS%

set /a intentos=0
:esperar_comfy
timeout /t 3 /nobreak >nul
curl -s -o nul http://localhost:8188/system_stats
if not errorlevel 1 goto comfy_listo
set /a intentos+=1
if %intentos% GEQ 40 (
    echo       [AVISO] ComfyUI no responde tras 2 minutos. Continuo igualmente.
    goto interfaz
)
goto esperar_comfy

:comfy_listo
echo       ComfyUI listo.

:: ── [4/4] Interfaz ───────────────────────────────────────────────
:interfaz
echo.
echo [4/4] Iniciando interfaz en http://localhost:7862 ...
echo       Cierra esta ventana para detenerla.
echo.
"%PY%" interfaz_txt2img_multi.py

echo.
echo Interfaz detenida. Codigo de salida: %ERRORLEVEL%
pause
