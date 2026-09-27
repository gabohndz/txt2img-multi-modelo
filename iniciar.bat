@echo off
title Generador de imagenes - Multi-Modelo
cd /d "%~dp0"
color 0A

echo ============================================
echo   Generador de imagenes - Multi-Modelo
echo ============================================
echo.

:: [1/3] Python disponible
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
    echo [1/3] Usando el entorno virtual .\venv
) else (
    set "PY=python"
    echo [1/3] Sin entorno virtual: se usara el Python del sistema.
    echo       Recomendado: python -m venv venv
    echo                    venv\Scripts\pip install -r requirements.txt
)

:: [2/3] ComfyUI responde en la API
echo [2/3] Comprobando ComfyUI en http://localhost:8188 ...
curl -s -o nul http://localhost:8188/system_stats
if errorlevel 1 (
    echo       [AVISO] ComfyUI no responde.
    echo       Arrancalo en OTRA ventana ^(ver README, seccion "Uso"^).
    echo       La interfaz se abrira igual, pero la lista de modelos estara vacia.
) else (
    echo       ComfyUI detectado.
)

:: [3/3] Interfaz
echo [3/3] Iniciando interfaz en http://localhost:7862 ...
echo       Cierra esta ventana para detenerla.
echo.
"%PY%" interfaz_txt2img_multi.py

echo.
echo Interfaz detenida. Codigo de salida: %ERRORLEVEL%
pause
