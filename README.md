# Generador de Imágenes Multi-Modelo (txt2img)

Interfaz web local (Gradio) para **generar imágenes a partir de texto** eligiendo
entre varios modelos de Stable Diffusion (SDXL y SD 1.5) y LoRAs, sin tocar
nodos ni workflows: ComfyUI hace el trabajo pesado por detrás y esta interfaz
le manda las órdenes por su API.

- Modelo concreto que quieras: Pony Diffusion V6 XL, DreamShaper XL, Juggernaut XL,
  Nova Comic XL, Anima Pencil XL, Realistic Vision... o **cualquier otro `.safetensors`
  que pongas en tu carpeta de checkpoints**.
- LoRAs opcionales con fuerza ajustable (p. ej. CleanLineArt para dibujos de líneas limpias).
- 6 resoluciones predefinidas, pasos, CFG, semilla, prompt negativo.

Idioma del proyecto: español. Pensado para uso local en un PC con GPU (NVIDIA
recomendado).

> **Lo mínimo para empezar:** Python + ComfyUI + **un solo** modelo.
> Son unos **12 GB de disco** y **6–8 GB de VRAM**. No hace falta descargar los
> 6 modelos ni tener una GPU potente: ver [Requisitos](#3-requisitos).

---

## Índice

1. [Cómo funciona (arquitectura)](#1-cómo-funciona-arquitectura)
2. [Estructura del repositorio](#2-estructura-del-repositorio)
3. [Requisitos](#3-requisitos)
4. [Instalación paso a paso](#4-instalación-paso-a-paso)
5. [Configuración (`config.json`)](#5-configuración-configjson)
6. [Uso](#6-uso)
7. [La interfaz, control por control](#7-la-interfaz-control-por-control)
8. [Modelos soportados y parámetros](#8-modelos-soportados-y-parámetros)
9. [LoRAs](#9-loras)
10. [Prompts de ejemplo](#10-prompts-de-ejemplo)
11. [Uso como librería de Python](#11-uso-como-librería-de-python)
12. [Solución de problemas](#12-solución-de-problemas)
13. [Limitaciones y lo que falta por hacer](#13-limitaciones-y-lo-que-falta-por-hacer)
14. [Licencia](#14-licencia)

---

## 1. Cómo funciona (arquitectura)

```
┌──────────────────────────┐        HTTP (JSON)        ┌────────────────────────┐
│  interfaz_txt2img_multi  │  POST /prompt ──────────► │                        │
│  (Gradio, :7862)         │  GET  /history/<id> ────► │   ComfyUI (:8188)      │
│                          │  GET  /system_stats ────► │   + tu GPU             │
│  Sliders, dropdowns,     │                           │                        │
│  preview de la imagen    │  ◄── PNG en disco ─────── │  escribe en output/    │
└──────────┬───────────────┘                           └────────────────────────┘
           │ importa
           ▼
┌──────────────────────────┐
│ modules/                 │
│  comfyui_generador_multi │  construye el grafo (workflow) con GraphBuilder
│  comfyui_backend         │  habla con la API de ComfyUI y localiza el PNG
│  config                  │  resuelve las rutas desde config.json
└──────────────────────────┘
```

Flujo exacto de una generación:

1. La interfaz llama a `generar_txt2img_con_checkpoint()`.
2. Esa función construye el workflow como diccionario JSON con `GraphBuilder`:
   `CheckpointLoaderSimple` → (`LoraLoader` si hay LoRA) → 2 × `CLIPTextEncode`
   (positivo y negativo) → `EmptyLatentImage` → `KSampler` → `VAEDecode` → `SaveImage`.
3. Se envía a ComfyUI con `POST http://localhost:8188/prompt`.
4. Se hace *polling* a `GET /history/<prompt_id>` hasta que el trabajo termina
   (o salta el timeout, 600 s por defecto).
5. ComfyUI ha guardado el PNG en **su propia** carpeta `output/`
   (nombre `multi<xxxx>_00001_.png`). El backend lo localiza ahí usando la ruta
   configurada en `config.json`.
6. La interfaz lo carga con Pillow y lo muestra en el panel derecho.

**Punto clave:** ComfyUI y esta interfaz deben ejecutarse en la **misma máquina**,
porque el PNG se lee del disco. Ver [limitaciones](#13-limitaciones-y-lo-que-falta-por-hacer).

---

## 2. Estructura del repositorio

```
generador-imagenes-multi/
├── interfaz_txt2img_multi.py    # Interfaz Gradio (punto de entrada). Puerto 7862.
├── iniciar.bat                  # Lanzador Windows (doble clic)
├── iniciar.sh                   # Lanzador Linux / macOS
├── config.json                  # Rutas de ComfyUI y ajustes. EDITA ESTO.
├── requirements.txt             # Dependencias de Python de este repo
├── LICENSE                      # MIT (solo el código; los modelos tienen la suya)
├── .gitignore
└── modules/
    ├── __init__.py
    ├── config.py                # Lee config.json + autodetección de ComfyUI
    ├── comfyui_backend.py       # Cliente de la API de ComfyUI + workflows base
    └── comfyui_generador_multi.py  # Catálogo de modelos + workflows con checkpoint/LoRA
```

No hay más. ComfyUI **no** está dentro de este repositorio: es un programa aparte
que se instala por separado (paso 4.2).

---

## 3. Requisitos

Antes de nada: **no necesitas descargar los 6 modelos ni tener una GPU potente.**
Con Python, ComfyUI y **un** modelo ya funciona. Aquí está todo lo que hay que
bajar y cuánta VRAM hace falta de verdad.

### 3.1 Checklist: todo lo que hay que descargar

| # | Qué | Peso | ¿Obligatorio? | Dónde va |
|---|---|---|---|---|
| 1 | **Python 3.10 o superior** | ~30 MB | Sí | Instalador del sistema |
| 2 | **ComfyUI** (el motor) | ~5 GB | Sí | Carpeta aparte, ej. `C:\ComfyUI` |
| 3 | **Un modelo SDXL** (ver 4.3) | ~6.5 GB | Sí, basta con uno | `ComfyUI/models/checkpoints/` |
| 4 | **Este repositorio** | < 1 MB | Sí | Donde quieras |
| 5 | Los otros 5 modelos del catálogo | ~6.5 GB cada uno | Opcional | `ComfyUI/models/checkpoints/` |
| 6 | LoRA CleanLineArt | ~18 MB | Opcional | `ComfyUI/models/loras/` |
| 7 | Git (solo para clonar) | ~50 MB | Opcional ¹ | — |

¹ La versión portable de ComfyUI para Windows no necesita Git.

| Escenario | Disco necesario | Qué descargas |
|---|---|---|
| **Mínimo para empezar** | **~12 GB** | Python + ComfyUI + 1 modelo + este repo |
| Cómodo | ~25 GB | 3 modelos + LoRA |
| Catálogo completo | ~45 GB | Los 6 modelos + LoRA |

**Recomendación:** empieza con **un solo modelo**, `DreamShaperXL.safetensors`
(versátil) o `ponyDiffusionV6XL_v6StartWithThisOne.safetensors` (anime). Los demás
se añaden después: basta con copiarlos en la carpeta y reiniciar la interfaz.

### 3.2 Hardware

| Qué | Mínimo | Recomendado | Notas |
|---|---|---|---|
| Sistema | Windows 10/11, Linux o macOS | Windows 11 | El `.bat` está probado en Windows; el `.sh` cubre Linux/macOS |
| **GPU** | NVIDIA con **6 GB VRAM** (**4 GB** con `--lowvram`, más lento) | NVIDIA con **8–12 GB VRAM** | Ver la tabla de VRAM en 3.3. AMD funciona en Linux (ROCm); en Windows es complicado |
| RAM del sistema | 8 GB | 16 GB | Con `--lowvram` / `--medvram` parte del modelo vive en la RAM: con 8 GB justos puede ir lento o fallar |
| Python | 3.10 | 3.10.6 | Probado en 3.10.6. Funciona en 3.11 y 3.12 |
| Disco libre | ~12 GB | 45 GB+ | Ver checklist en 3.1 |
| ComfyUI | última versión | — | Instalación aparte, ver 4.2 |
| Navegador | cualquiera moderno | — | La UI se abre en el navegador |

### 3.3 VRAM: cuánta necesitas de verdad

Los 6 modelos del catálogo son **SDXL**, salvo `Realistic_Vision_V5.1` (SD 1.5).
SDXL es mucho más exigente: subir la resolución puede duplicar la memoria necesaria.

**Modelos SDXL** (Pony, DreamShaper, Nova Comic, Anima Pencil, Juggernaut):

| Resolución | VRAM cómoda | Si tienes menos |
|---|---|---|
| 1024×1024 (cuadrada) | 8 GB+ | 6 GB → `--medvram` · 4 GB → `--lowvram` |
| 1216×832 / 832×1216 | 8 GB+ | 6 GB → `--medvram` · 4 GB → `--lowvram` |
| 896×1152 (retrato) | 8 GB+ | 6 GB → `--medvram` |
| 1536×640 (panorámica) | 8 GB+ | 6 GB → `--medvram` |
| 1408×896 (paisaje) | 10 GB+ | 8 GB → `--medvram` |

**Modelo SD 1.5** (Realistic Vision V5.1): con **4 GB** va sobrado, y hasta se puede
usar sin GPU en un tiempo aceptable.

Los *flags* se añaden al arrancar ComfyUI (ver sección 6):

- `--lowvram` — pasa casi todo el modelo a la RAM del sistema. Funciona con 4 GB,
  pero **cada imagen tarda bastante más** (a veces varios minutos).
- `--medvram` — punto medio, pensado para unos 6 GB.
- `--cpu` — sin GPU. No recomendado: más de 10 minutos por imagen.

**Velocidad orientativa** con SDXL a 1024×1024 y 25 pasos (varía mucho según GPU
y resolución):

| GPU | Tiempo por imagen |
|---|---|
| RTX 3060 12 GB / 4060 Ti 16 GB | ~15–30 s |
| RTX 3060 6 GB con `--medvram` | ~30–60 s |
| GTX 1650 4 GB con `--lowvram` | ~2–5 min |
| Solo CPU | 10 min o más |

Las resoluciones grandes (1408×896) tardan aproximadamente el doble que 1024×1024.

> **En resumen:** con **6 GB de VRAM y `--medvram`** puedes usar todos los modelos
> del catálogo a 1024×1024 sin problema. Con 8 GB o más, sin tocar nada.

### 3.4 Dependencias de Python

Ya están en `requirements.txt`; se instalan con un solo comando (paso 4.4):

- `gradio` — la interfaz web
- `Pillow` — leer y mostrar el PNG generado
- `numpy` — convertir la imagen para Gradio

> Nota: `comfyui_backend.py` usa solo la librería estándar (`urllib.request`), no
> necesita `requests`.

---

## 4. Instalación paso a paso

Los comandos son para Windows (PowerShell/CMD). Entre paréntesis, el equivalente
en Linux/macOS.

### 4.1 Instalar Python

Descarga Python 3.10 o superior desde <https://www.python.org/downloads/>.
**En Windows marca la casilla "Add python.exe to PATH"** durante la instalación.

Comprueba:

```bash
python --version      # Linux/macOS: python3 --version
```

### 4.2 Instalar ComfyUI

ComfyUI es el motor que genera las imágenes. Va aparte y ocupa unos 5 GB.

**Opción A — Git (todas las plataformas):**

```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
python -m venv venv
venv\Scripts\pip install -r requirements.txt      # Linux/macOS: venv/bin/pip install -r requirements.txt
```

**Opción B — Windows portable (sin Git ni Python propio):**

Descarga el `.7z` "ComfyUI Windows Portable" desde
<https://github.com/comfyanonymous/ComfyUI/releases>, descomprímelo y usa
`run_nvidia_gpu.bat` para arrancarlo. Anota la carpeta donde lo descomprimiste:
la necesitarás en `config.json`.

Ruta final esperada de la carpeta ComfyUI (apunta la tuya):

```
C:\ComfyUI\                ← ejemplo en Windows
/home/tuusuario/ComfyUI/   ← ejemplo en Linux
```

### 4.3 Descargar los modelos

Crea la carpeta de checkpoints si no existe:

```bash
mkdir C:\ComfyUI\models\checkpoints      # Linux/macOS: mkdir -p ~/ComfyUI/models/checkpoints
mkdir C:\ComfyUI\models\loras
```

Cada archivo `.safetensors` va **directamente** en `models\checkpoints\` (sin
subcarpetas). Tabla de modelos (el nombre de archivo es el que espera el catálogo
interno; ver la nota después de la tabla):

| Nombre del archivo | Modelo | Tipo | Peso | Enlace |
|---|---|---|---|---|
| `ponyDiffusionV6XL_v6StartWithThisOne.safetensors` | Pony Diffusion V6 XL | SDXL | ~7 GB | Busca **"Pony Diffusion V6 XL"** en <https://civitai.com> |
| `DreamShaperXL.safetensors` | DreamShaper XL | SDXL | ~6.5 GB | <https://huggingface.co/ACEGameAI/DreamShaperXL/resolve/main/DreamShaperXL.safetensors> |
| `novaComicXL_v20.safetensors` | Nova Comic XL v2.0 | SDXL | ~6.7 GB | <https://civitai.com/api/download/models/2431039> |
| `animaPencilXLClear_v2.safetensors` | Anima Pencil XL Clear v2 | SDXL | ~6.6 GB | <https://civitai.com/api/download/models/2778774> |
| `juggernautXL_v10.safetensors` | Juggernaut XL v10 | SDXL | ~6.5 GB | <https://huggingface.co/RunDiffusion/Juggernaut-XL-v10/resolve/main/Juggernaut-XL-v10.safetensors> |
| `Realistic_Vision_V5.1.safetensors` | Realistic Vision V5.1 | SD 1.5 | ~5 GB | Busca **"Realistic Vision V5.1"** en <https://civitai.com> |

LoRA (opcional), va en `models\loras\`:

| Nombre del archivo | LoRA | Base | Peso | Enlace |
|---|---|---|---|---|
| `CleanLineArt.safetensors` | CleanLineArt SDXL | SDXL | ~17.6 MB | <https://civitai.com/api/download/models/348186> |

**Notas importantes sobre las descargas:**

- Los enlaces de Civitai pueden pedir **iniciar sesión** (algunos modelos son
  "gated"). Si el enlace de descarga directa falla, entra en la página del modelo
  y descarga desde ahí.
- Hugging Face suele descargar con el nombre correcto. Civitai a veces guarda el
  archivo con otro nombre: **renómbralo** al de la tabla para que la interfaz
  muestre los tips y los parámetros recomendados.
- Si el nombre no coincide, **aun así funciona**: el modelo aparece en la lista y
  se puede usar, solo pierde los consejos automáticos (ver sección 8).
- No hace falta descargar todos. Con uno basta para empezar; se recomienda
  **DreamShaper XL** (versátil) o **Pony Diffusion V6 XL** (anime/estilizado).

### 4.4 Instalar este repositorio

```bash
git clone <URL-DE-ESTE-REPO>
cd generador-imagenes-multi
python -m venv venv
venv\Scripts\pip install -r requirements.txt      # Linux/macOS: venv/bin/pip install -r requirements.txt
```

### 4.5 Configurar las rutas

Abre `config.json` y pon la ruta de tu ComfyUI (siguiente sección). Si tu
ComfyUI está en una ubicación habitual, puede que la autodetección ya lo
encuentre y no tengas que tocar nada.

### 4.6 Comprobar la instalación

```bash
venv\Scripts\python -c "from modules.config import resumen; print(resumen())"
```

Debe imprimir la ruta de tu ComfyUI, la de checkpoints y cuántos modelos ve.
Si dice `NO ENCONTRADO` o la carpeta de checkpoints no es la tuya, corrige
`config.json`.

---

## 5. Configuración (`config.json`)

Todas las rutas del proyecto salen de aquí. **Este es el único archivo que
necesitas editar** para adaptarlo a tu PC.

```json
{
  "comfyui_path": "",
  "comfyui_api": "http://localhost:8188",
  "checkpoints_dir": "",
  "loras_dir": "",
  "output_dir": ""
}
```

| Clave | Qué es | Ejemplo |
|---|---|---|
| `comfyui_path` | Carpeta raíz de ComfyUI (donde está `main.py`) | `"C:/ComfyUI"` |
| `comfyui_api` | URL de la API de ComfyUI | `"http://localhost:8188"` |
| `checkpoints_dir` | Carpeta de modelos. Vacío = `<comfyui_path>/models/checkpoints` | `"D:/modelos/checkpoints"` |
| `loras_dir` | Carpeta de LoRAs. Vacío = `<comfyui_path>/models/loras` | `"D:/modelos/loras"` |
| `output_dir` | Carpeta `output` de ComfyUI. Vacío = `<comfyui_path>/output` | `""` |

**Reglas de resolución de rutas** (implementadas en `modules/config.py`), por orden:

1. Si existe la variable de entorno `COMFYUI_PATH`, esa gana sobre todo lo demás.
   Útil en Docker o entornos compartidos.
2. Si `config.json` tiene `comfyui_path`, se usa esa.
3. Si no, se prueba una lista de ubicaciones habituales buscando `main.py`:
   `C:/ComfyUI`, `C:/AI/ComfyUI`, `~/ComfyUI`, `~/Desktop/ComfyUI`,
   `~/Desktop/Data/Packages/ComfyUI`, `/opt/ComfyUI`, `/workspace/ComfyUI`,
   `./ComfyUI`, etc.
4. Si nada encaja, las rutas quedan apuntando a `<repo>/models/...` (no existen)
   y la lista de modelos sale vacía. **La interfaz arranca igual**, solo que sin
   modelos: es la señal de que falta configurar `comfyui_path`.

Usa `/` o `\\` en las rutas JSON de Windows. Las claves desconocidas se ignoran
con un aviso por consola. Si el JSON está mal formado, el programa se detiene y
te dice en qué línea.

---

## 6. Uso

### Paso 1 — Arrancar ComfyUI (en una ventana aparte)

Windows (instalación por Git):

```bash
cd C:\ComfyUI
venv\Scripts\python main.py --listen
```

Linux/macOS:

```bash
cd ~/ComfyUI
venv/bin/python main.py --listen
```

Windows portable: ejecuta `run_nvidia_gpu.bat` (ya arranca con la API activa).

- `--listen` hace que la API escuche en la red local (lo necesita esta interfaz).
- Añade `--lowvram` si tienes poca VRAM (< 6 GB) y da error de memoria.
- Espera a ver en la consola algo como `To see the GUI go to: http://0.0.0.0:8188`.
  **Deja esa ventana abierta** mientras generas imágenes.

### Paso 2 — Arrancar la interfaz (en otra ventana)

- **Windows:** doble clic en `iniciar.bat`
- **Linux/macOS:** `./iniciar.sh`
- **Manual, cualquier plataforma:**

```bash
python interfaz_txt2img_multi.py
```

La consola imprime los modelos detectados y abre automáticamente el navegador en
<http://localhost:7862>. Si no se abre, entra a mano.

### Paso 3 — Generar

Escribe el prompt → elige modelo → pulsa **✨ Generar imagen**. El PNG queda
guardado en `<ComfyUI>/output/` y se muestra en el panel derecho.

Para **detener** todo: `Ctrl+C` en la ventana de la interfaz y cierra también la
de ComfyUI.

---

## 7. La interfaz, control por control

### Columna izquierda

| Control | Tipo | Por defecto | Qué hace |
|---|---|---|---|
| **Modelo (Checkpoint)** | Desplegable | Primer modelo no-Pony (prefiere `DreamShaperXL.safetensors`) | Lista todos los `.safetensors`/`.ckpt` de `checkpoints_dir`. Al cambiarlo, actualiza los tips y **sobrescribe** los sliders de Pasos y CFG con los recomendados del modelo |
| **Tips del modelo** | Texto | — | Nombre, tipo (SDXL/SD 1.5), descripción, consejo de prompt y valores recomendados |
| **Usar LoRA** | Casilla | Desactivada | Activa/desactiva el LoRA por completo |
| **LoRA** | Desplegable | Primer LoRA disponible | LoRAs de `loras_dir`. Se **filtra por tipo**: con un modelo SDXL solo salen LoRAs SDXL; si no hay ninguna compatible, el desplegable se oculta |
| **Fuerza LoRA** | Deslizador | `0.8` | 0.0 – 1.5. Aplica a `strength_model` y `strength_clip`. Al elegir un LoRA se pone su valor recomendado |
| **Prompt** | Caja de texto (4 líneas) | vacío | Descripción de la imagen. **Obligatorio**: si está vacío, avisa y no genera |
| **Prompt Negativo** | Caja de texto (3 líneas) | ver abajo | Lo que NO quieres que aparezca |
| **Seed** | Número | `42` | Semilla. **`0` o negativo = aleatoria**. La misma semilla + mismos ajustes = misma imagen |
| **Pasos** | Deslizador | `25` | 5 – 60. Más pasos = más detalle y más tiempo |
| **CFG** | Deslizador | `6.0` | 1.0 – 20.0 (pasos de 0.5). Cuánto obedece el prompt. Muy alto = colores quemados; muy bajo = no sigue el prompt |
| **Resolución** | Desplegable | `Cuadrado 1024x1024` | Ver tabla abajo |
| **✨ Generar imagen** | Botón | — | Lanza la generación. Bloquea el botón mientras dura |
| **Estado** | Texto | `Listo para generar.` | Mensajes de progreso o el error concreto |

Resoluciones disponibles:

| Etiqueta | Píxeles | Cuándo usarla |
|---|---|---|
| `Cuadrado 1024x1024` | 1024×1024 | Uso general |
| `Retrato 896x1152` | 896×1152 | Vertical tipo móvil/impresión |
| `Paisaje 1408x896` | 1408×896 | Escenas amplias |
| `Horizontal 1216x832` | 1216×832 | Formato 3:2 |
| `Vertical 832x1216` | 832×1216 | Stories / Shorts / Reels (9:16) |
| `Panorámica 1536x640` | 1536×640 | Banners, fondos de cabecera |

Prompt negativo por defecto (bueno para casi todo):

```
blurry, low quality, worst quality, distorted, watermark, text, deformed, bad anatomy,
extra limbs, bad hands, bad face, cropped, ugly, monochrome, nsfw, lowres, bad proportions,
error, extra digits, fewer digits, signature, username, artist name, letters, words
```

### Columna derecha

- **Imagen generada**: previsualización del último resultado.
- **Acordeón "Estado de modelos instalados"**: qué modelos del catálogo tienes
  instalados, cuáles faltan y su enlace de descarga.

> La carpeta de modelos se lee **al arrancar**. Si descargas un modelo con la
> interfaz abierta, reiníciala para que aparezca.

---

## 8. Modelos soportados y parámetros

Estos son los modelos del catálogo interno (`MODEL_INFO` en
`modules/comfyui_generador_multi.py`). Valores de Pasos y CFG que la interfaz
aplica automáticamente al seleccionarlos:

| Archivo | Modelo | Tipo | Estilo | Pasos | CFG | Peso |
|---|---|---|---|---|---|---|
| `ponyDiffusionV6XL_v6StartWithThisOne.safetensors` | Pony Diffusion V6 XL | SDXL | Anime/estilizado, saturado, artístico | 30 | 7.5 | ~7 GB |
| `DreamShaperXL.safetensors` | DreamShaper XL | SDXL | Versátil, semirrealista a artístico | 25 | 6.0 | ~6.5 GB |
| `novaComicXL_v20.safetensors` | Nova Comic XL v2.0 | SDXL | Cómic americano / superhéroes | 25 | 5.0 | ~6.7 GB |
| `animaPencilXLClear_v2.safetensors` | Anima Pencil XL Clear v2 | SDXL | Ilustración de líneas limpias, lápiz | 15 | 3.5 | ~6.6 GB |
| `juggernautXL_v10.safetensors` | Juggernaut XL v10 | SDXL | Fotorrealista, retratos y escenas | 25 | 6.0 | ~6.5 GB |
| `Realistic_Vision_V5.1.safetensors` | Realistic Vision V5.1 | SD 1.5 | Fotorrealista, más rápido | 25 | 6.0 | ~5 GB |

**Cualquier otro modelo vale.** Si el archivo no está en el catálogo, la interfaz
lo lista igualmente y rellena los tips con valores genéricos (25 pasos, CFG 6,
tipo SDXL si el nombre contiene "xl", si no SD 1.5). El sampler y el scheduler
**no** son ajustables desde la interfaz: el workflow usa siempre `euler` + `simple`
con `denoise = 1.0`. Los valores de sampler/scheduler que se muestran en los tips
son informativos (ver [limitaciones](#13-limitaciones-y-lo-que-falta-por-hacer)).

---

## 9. LoRAs

Un LoRA es un archivo pequeño que modifica el estilo del modelo base sin
necesidad de otro checkpoint. Se ponen en `loras_dir`.

Catálogo conocido:

| Archivo | LoRA | Base | Fuerza | Estilo |
|---|---|---|---|---|
| `CleanLineArt.safetensors` | CleanLineArt SDXL | SDXL | 0.8 | Líneas limpias, colores planos, estilo "explicativo" |

Trigger words de CleanLineArt (van en el prompt):
`flat color, simple background, clean lineart, [color] background`

Uso típico: **DreamShaper XL + CleanLineArt** produce ilustraciones limpias tipo
infografía o dibujo explicativo.

Fuerza recomendada: `0.7 – 0.9`. Por encima de 1.0 el estilo se impone
demasiado y puede deformar la imagen.

---

## 10. Prompts de ejemplo

Cada modelo espera un tipo de prompt distinto. Los prompts van **en inglés**:
los modelos están entrenados así.

**Pony Diffusion V6 XL** — obligatorio empezar por los tags de calidad y rating:

```
score_9, score_8_up, score_7_up, source_anime, rating_safe,
a young explorer standing on a cliff at sunset, detailed background, vibrant colors
```

**DreamShaper XL** — lenguaje natural:

```
a cozy bookshop on a rainy street, warm light in the windows, digital art, highly detailed
```

**Nova Comic XL** — estilo cómic:

```
western_comics_(style), a superhero landing on a rooftop, dynamic pose, dramatic lighting, halftone shading
```

**Anima Pencil XL Clear** — CFG bajo (2–4), pocos pasos:

```
a peaceful mountain village in the morning, soft colors, clean lineart, simple background
```

**Juggernaut XL / Realistic Vision** — fotorrealismo:

```
portrait of an elderly fisherman, weathered face, natural window light, 50mm photo, sharp focus
```

Con LoRA CleanLineArt activo, añade al final:

```
flat color, simple background, clean lineart, white background
```

---

## 11. Uso como librería de Python

Puedes saltarte la interfaz y generar desde tu propio script:

```python
import sys
sys.path.insert(0, "modules")

from comfyui_generador_multi import generar_txt2img_con_checkpoint, listar_checkpoints

print(listar_checkpoints())   # ['DreamShaperXL.safetensors', ...]

ruta = generar_txt2img_con_checkpoint(
    prompt="a lighthouse in a storm, dramatic sky, digital art",
    negative_prompt="blurry, low quality, watermark",
    seed=42,                       # 0 = aleatoria
    checkpoint_name="DreamShaperXL.safetensors",
    ancho=1216, alto=832,
    pasos=25, cfg=6.0,
    lora_name=None,                # o "CleanLineArt.safetensors"
    lora_strength=0.8,
    timeout=600,
)
print(ruta)   # C:\ComfyUI\output\multi1a2b_00001_.png  (o None si falló)
```

Funciones públicas de `modules/comfyui_generador_multi.py`:

| Función | Para qué |
|---|---|
| `generar_txt2img_con_checkpoint(...)` | txt2img con checkpoint + LoRA opcional. Devuelve `Path` o `None` |
| `listar_checkpoints()` / `listar_loras()` | Lista los archivos disponibles en disco |
| `info_checkpoint(nombre)` / `info_lora(nombre)` | Ficha del modelo (tipo, pasos, cfg, tips...). Devuelve valores genéricos si no lo conoce |
| `generar_variaciones(...)` | Varias imágenes parecidas (pseudo-animación) |
| `generar_animacion_ipadapter(...)` | Animación a partir de una imagen de referencia |
| `generar_animacion_progresiva(...)` | Secuencia de interpolación entre dos prompts |
| `generar_animacion_controlnet(...)` | Animación guiada por pose (necesita ControlNet OpenPose XL) |

Las cuatro últimas **no están expuestas en la interfaz Gradio** y requieren
modelos extra (motion models, ControlNet). Están en el módulo por si quieres
llamarlas desde código.

`modules/comfyui_backend.py` ofrece además las primitivas de bajo nivel:
`generar_txt2img`, `generar_img2img`, `generar_img2animation`, `generar_txt2img_ipadapter`,
`generar_lote_img2img_encadenado`, y los helpers `_post`, `_get`, `_upload_image`,
`_esperar_resultado`, `_obtener_imagen_salida`, más el `GraphBuilder` para armar
workflows propios.

---

## 12. Solución de problemas

| Síntoma | Causa | Solución |
|---|---|---|
| El desplegable de modelos está **vacío** | ComfyUI no encontrado, o `checkpoints_dir` mal | Ejecuta `python -c "from modules.config import resumen; print(resumen())"` y corrige `config.json` |
| `ComfyUI no responde` en `iniciar.bat` | ComfyUI apagado | Arrancarlo con `--listen` en otra ventana (paso 1 de Uso) |
| `Error: ComfyUI no genero imagen` | ComfyUI caído a mitad, o timeout (600 s) | Mira la consola de ComfyUI: casi siempre es VRAM o un modelo corrupto |
| `Value not in list: ckpt_name` en la consola de ComfyUI | El nombre del archivo no está en `checkpoints/` | Revisa mayúsculas y que el archivo esté **directo** en la carpeta, no en una subcarpeta |
| `CUDA out of memory` | Poca VRAM para SDXL | Arranca ComfyUI con `--medvram` (6 GB) o `--lowvram` (4 GB). Baja a 1024×1024 o prueba un modelo SD 1.5. Ver 3.3 |
| La **primera** imagen tarda mucho más que las siguientes | El modelo se está cargando en VRAM/RAM | Normal: la primera carga puede tardar 1–2 min. Las siguientes son rápidas |
| Va lentísimo y tengo poca VRAM | Estás usando SDXL a resolución alta | Baja la resolución, usa `--lowvram` o cambia a `Realistic_Vision_V5.1` (SD 1.5, mucho más ligero) |
| Imagen completamente negra o ruido | Modelo incompatible o corrupto | Revisa el tamaño del archivo (un SDXL sano pesa ~6.5 GB). Vuelve a descargarlo |
| La imagen generada no aparece en la interfaz | El PNG se escribió pero `output_dir` apunta a otro sitio | Comprueba `output_dir` en `config.json`; debe ser la carpeta `output` de **tu** ComfyUI |
| El navegador se abre en blanco | Puerto 7862 ocupado por un proceso viejo | Cierra el proceso que use el 7862 y vuelve a arrancar |
| Un modelo nuevo no aparece en la lista | La carpeta se lee al arrancar | Reinicia la interfaz |
| Los tips/parámetros del modelo no salen | El archivo no tiene el nombre exacto del catálogo | Renómbralo (sección 4.3) o ignóralo: el modelo funciona igual |
| `config.json tiene un JSON invalido` | Coma o comilla de más | Valida el JSON; no admite comentarios ni comas finales |
| En Linux/macOS `./iniciar.sh` no arranca | Falta permiso de ejecución | `chmod +x iniciar.sh` |

---

## 13. Limitaciones y lo que falta por hacer

Lo que **no** está resuelto todavía, por orden de utilidad:

1. **No hay instalador ni descarga automática de modelos.** Toda la descarga es
   manual y cada usuario debe colocar los `.safetensors` a mano y renombrarlos si
   quiere los tips. Un script de descarga por URL (existe `descargar_modelos_redes.py`
   en el proyecto original, no incluido aquí) lo automatizaría.
2. **El sampler y el scheduler no se pueden elegir.** El workflow usa siempre
   `euler` + `simple`. El catálogo *muestra* el sampler recomendado por modelo
   (Anima Pencil pide `beta`) pero no lo aplica: falta pasar esos dos valores al
   `KSampler` y exponerlos en la interfaz.
3. **No se guarda un historial ni una galería.** La imagen solo se muestra; el
   archivo vive en la carpeta `output` de ComfyUI con un nombre aleatorio
   (`multi<xxxx>_00001_.png`). No hay copia organizada por fecha ni metadatos
   (prompt/semilla) guardados junto al PNG.
4. **No hay botón de cancelar.** Cerrar la interfaz no cancela el trabajo en
   ComfyUI: hay que esperar o reiniciar ComfyUI.
5. **Requiere ComfyUI en la misma máquina.** El PNG se lee del disco. Se
   solucionaría usando el endpoint `/view` de ComfyUI en lugar de la ruta local,
   lo que además permitiría usar ComfyUI remoto.
6. **Sin tests.** No hay ni una prueba automatizada; todo se ha validado a mano.
7. **Solo Windows está probado.** El `.sh` existe pero no se ha ejecutado en un
   entorno real de Linux/macOS.
8. **La lista de modelos se lee una sola vez al arrancar** (no se refresca en
   caliente) y **no hay botón de "recargar modelos"**.
9. **El filtro de LoRAs es por tipo genérico** (SDXL / SD 1.5), no por
   compatibilidad real: un LoRA de SDXL para un modelo concreto puede dar
   resultados pobres aunque la interfaz lo permita.
10. **Las animaciones no están en la interfaz.** El módulo soporta variaciones,
    interpolación y ControlNet, pero no hay UI ni están documentados los modelos
    extra que necesitan.
11. **`config.json` se versiona con rutas vacías** (autodetección). Si editas
    `comfyui_path` con tu ruta personal y haces commit, subirás tu ruta al repo.
12. **Sin control de upscaling, img2img ni inpainting** desde la interfaz.

---

## 14. Licencia

El **código** de este repositorio es **MIT** (ver `LICENSE`): puedes usarlo,
modificarlo y redistribuirlo libremente manteniendo el aviso de copyright.

Los **modelos** (Pony Diffusion, DreamShaper XL, Juggernaut XL, Nova Comic,
Anima Pencil, Realistic Vision, CleanLineArt...) **no** se incluyen en este
repositorio y tienen cada uno su propia licencia, que decide su autor en Civitai
o Hugging Face. Algunas permiten uso comercial y otras no. **Revisa la licencia
de cada modelo antes de usarlo, sobre todo comercialmente.**

ComfyUI es un proyecto independiente de Comfy Org, con su propia licencia
(GPL-3.0): <https://github.com/comfyanonymous/ComfyUI>.
