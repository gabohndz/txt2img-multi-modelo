"""
Interfaz txt2img Multi-Modelo — Genera imagenes seleccionando checkpoint + LoRA.

Permite elegir entre multiples modelos (Pony, DreamShaper, Nova Comic, Anima Pencil, etc.)
y activar LoRAs como CleanLineArt para cambiar el estilo.

Uso:
    python interfaz_txt2img_multi.py

Requiere ComfyUI corriendo (por defecto en http://localhost:8188) y abierto
con --listen. Las rutas se configuran en config.json (ver README).
"""

import sys
import random
import numpy as np
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "modules"))

from PIL import Image as PILImage
import gradio as gr

from modules.comfyui_generador_multi import (
    generar_txt2img_con_checkpoint,
    listar_checkpoints, listar_loras,
    info_checkpoint, info_lora,
    MODEL_INFO, LORA_INFO,
)
from modules.comfyui_backend import COMFY_API

# ─── Prompt negativo por defecto ─────────────────────────────────────

NEG_DEFECTO = (
    "blurry, low quality, worst quality, distorted, watermark, text, deformed, "
    "bad anatomy, extra limbs, bad hands, bad face, cropped, ugly, monochrome, "
    "nsfw, lowres, bad proportions, error, extra digits, fewer digits, "
    "signature, username, artist name, letters, words"
)

# ─── Resoluciones ─────────────────────────────────────────────────────

RESOLUCIONES = {
    "Cuadrado 1024x1024": (1024, 1024),
    "Retrato 896x1152":   (896, 1152),
    "Paisaje 1408x896":   (1408, 896),
    "Horizontal 1216x832": (1216, 832),
    "Vertical 832x1216":  (832, 1216),
    "Panoramica 1536x640": (1536, 640),
}

# ─── Logica de generacion ────────────────────────────────────────────


def generar(prompt, neg_prompt, seed, pasos, cfg, resolucion,
            checkpoint, usar_lora, lora, lora_fuerza, progress=None):
    if not prompt or prompt.strip() == "":
        return None, "Error: El prompt no puede estar vacio."

    if seed is None or seed <= 0:
        seed = random.randint(1, 999999999)

    ancho, alto = RESOLUCIONES.get(resolucion, (1024, 1024))

    if progress:
        progress(0.1, desc="Conectando con ComfyUI...")

    try:
        lora_name = lora if usar_lora else None
        lora_str = lora_fuerza if usar_lora else 0.0

        print(f"  Modelo: {checkpoint}")
        print(f"  LoRA: {lora_name or '(ninguno)'}")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Seed: {seed} | Pasos: {pasos} | CFG: {cfg} | {ancho}x{alto}")

        ruta = generar_txt2img_con_checkpoint(
            prompt=prompt.strip(),
            negative_prompt=neg_prompt.strip(),
            seed=seed,
            checkpoint_name=checkpoint,
            ancho=ancho,
            alto=alto,
            pasos=pasos,
            cfg=cfg,
            lora_name=lora_name,
            lora_strength=lora_str,
            timeout=600,
        )

        if ruta is None:
            return None, "Error: ComfyUI no genero imagen. Revisa que este corriendo."

        if progress:
            progress(1.0, desc="Listo!")
        print(f"  Imagen generada: {ruta}")

        img_array = np.asarray(PILImage.open(ruta))
        return img_array, None

    except Exception as e:
        return None, f"Error inesperado: {type(e).__name__}: {e}"


# ─── Actualizar info al cambiar modelo ───────────────────────────────


def al_cambiar_modelo(ckpt_name):
    """Actualiza tips y valores por defecto al cambiar de modelo."""
    info = info_checkpoint(ckpt_name)
    tips = (
        f"**{info['nombre']}** ({info['tipo']}) — {info['desc']}\n\n"
        f"💡 {info['tips']}\n\n"
        f"Valores recomendados: {info['pasos']} pasos, CFG {info['cfg']}, "
        f"sampler {info['sampler']}, scheduler {info['scheduler']}"
    )

    loras_disp = listar_loras()
    loras_filtradas = []
    for l in loras_disp:
        l_info = info_lora(l)
        if l_info["base"] == info["tipo"] or l_info["base"] == "?":
            loras_filtradas.append(l)

    return (
        gr.update(value=tips),
        gr.update(value=info["pasos"]),
        gr.update(value=info["cfg"]),
        gr.update(choices=loras_filtradas) if loras_filtradas else gr.update(choices=[], visible=False),
    )


def al_cambiar_lora(lora_name):
    """Muestra info del LoRA seleccionado."""
    if not lora_name:
        return gr.update(value=""), gr.update(value=0.8)
    info = info_lora(lora_name)
    txt = f"**{info['nombre']}** — {info['desc']}\nTrigger: {info['trigger']}"
    return gr.update(value=txt), gr.update(value=info["strength"])


# ─── UI ──────────────────────────────────────────────────────────────

CSS = """
    footer { display: none !important; }
    .header-text { text-align: center; margin-bottom: 0.5em; }
"""

# Detectar modelos disponibles al inicio
checkpoints_disponibles = listar_checkpoints()
loras_disponibles = listar_loras()

if not checkpoints_disponibles:
    print("⚠️  No se encontraron checkpoints en la carpeta de ComfyUI.")
    checkpoints_disponibles = list(MODEL_INFO.keys())

# Seleccionar default: el primero que NO sea Pony (preferimos DreamShaper si existe)
default_ckpt = checkpoints_disponibles[0] if checkpoints_disponibles else ""
for fav in ["DreamShaperXL.safetensors", "ponyDiffusionV6XL_v6StartWithThisOne.safetensors"]:
    if fav in checkpoints_disponibles:
        default_ckpt = fav
        break

default_lora = loras_disponibles[0] if loras_disponibles else None

# ─── Construir mensaje de estado de modelos ──────────────────────────

DESCARGAS_URLS = {
    "novaComicXL_v20.safetensors": "https://civitai.com/api/download/models/2431039",
    "animaPencilXLClear_v2.safetensors": "https://civitai.com/api/download/models/2778774",
    "CleanLineArt.safetensors": "https://civitai.com/api/download/models/348186",
}


def _construir_estado_modelos():
    instalados = set(listar_checkpoints())
    lineas = ["### 🎯 Checkpoints"]
    for ckpt, info in MODEL_INFO.items():
        ok = ckpt in instalados
        icono = "✅" if ok else "⬇️"
        nombre = f"{info['nombre']} ({info['tipo']}, {info['peso']})"
        if ok:
            lineas.append(f"- {icono} **{nombre}**")
        else:
            url = DESCARGAS_URLS.get(ckpt, "")
            link = f"[descargar]({url})" if url else "descarga manual"
            lineas.append(f"- {icono} {nombre} — {link}")
    loras_inst = set(listar_loras())
    if LORA_INFO:
        lineas.append("\n### 🎯 LoRAs")
        for lora_ckpt, l_info in LORA_INFO.items():
            ok = lora_ckpt in loras_inst
            icono = "✅" if ok else "⬇️"
            nombre = f"{l_info['nombre']} ({l_info['peso']})"
            if ok:
                lineas.append(f"- {icono} **{nombre}**")
            else:
                url = DESCARGAS_URLS.get(lora_ckpt, "")
                link = f"[descargar]({url})" if url else "descarga manual (Civitai bloquea auto-download)"
                lineas.append(f"- {icono} {nombre} — {link}")
    return "\n".join(lineas)


with gr.Blocks(
    title="txt2img — Multi-Modelo",
) as demo:

    gr.Markdown(
        "# 🎨 Generador txt2img — Multi-Modelo",
        elem_classes="header-text",
    )
    gr.Markdown(
        "Genera imagenes seleccionando el modelo SDXL y LoRA. "
        "Requiere ComfyUI corriendo en segundo plano.",
        elem_classes="header-text",
    )

    with gr.Accordion("Estado de modelos instalados", open=False):
        gr.Markdown(_construir_estado_modelos())

    with gr.Row():
        with gr.Column(scale=2):

            # ── Selector de modelo ──
            modelo = gr.Dropdown(
                label="Modelo (Checkpoint)",
                choices=checkpoints_disponibles,
                value=default_ckpt,
            )

            modelo_tips = gr.Markdown(
                value="Selecciona un modelo para ver sus tips.",
                elem_classes="header-text",
            )

            # ── LoRA ──
            with gr.Row():
                usar_lora = gr.Checkbox(
                    label="Usar LoRA",
                    value=False,
                )
                lora_selector = gr.Dropdown(
                    label="LoRA",
                    choices=loras_disponibles,
                    value=default_lora,
                    interactive=True,
                )
                lora_fuerza = gr.Slider(
                    label="Fuerza LoRA",
                    value=0.8,
                    minimum=0.0,
                    maximum=1.5,
                    step=0.05,
                )

            lora_tips = gr.Markdown(
                value="",
                elem_classes="header-text",
            )

            # ── Prompt ──
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe la imagen que quieres generar...",
                lines=4,
            )

            neg_prompt = gr.Textbox(
                label="Prompt Negativo",
                value=NEG_DEFECTO,
                lines=3,
            )

            # ── Parametros ──
            with gr.Row():
                seed = gr.Number(
                    label="Seed (0 = aleatorio)",
                    value=42,
                    minimum=None,
                    maximum=None,
                    precision=0,
                )
                pasos = gr.Slider(
                    label="Pasos",
                    value=25,
                    minimum=5,
                    maximum=60,
                    step=1,
                )
                cfg = gr.Slider(
                    label="CFG",
                    value=6.0,
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                )

            with gr.Row():
                resolucion = gr.Dropdown(
                    label="Resolucion",
                    choices=list(RESOLUCIONES.keys()),
                    value="Cuadrado 1024x1024",
                )

            generar_btn = gr.Button(
                "✨ Generar imagen",
                variant="primary",
                size="lg",
            )

            estado = gr.Textbox(
                label="Estado",
                value="Listo para generar.",
                interactive=False,
            )

        with gr.Column(scale=3):
            salida = gr.Image(
                label="Imagen generada",
                height=600,
            )

    # ── Eventos ────────────────────────────────────────────────────

    modelo.change(
        fn=al_cambiar_modelo,
        inputs=[modelo],
        outputs=[modelo_tips, pasos, cfg, lora_selector],
    )

    lora_selector.change(
        fn=al_cambiar_lora,
        inputs=[lora_selector],
        outputs=[lora_tips, lora_fuerza],
    )

    def on_generar(prompt, neg, seed, pasos, cfg, resolucion, modelo, usar_lora, lora, lora_fuerza,
                   progress=gr.Progress()):
        try:
            progress(0.05, desc="Iniciando...")
            print("  Generacion iniciada...")

            img, err = generar(prompt, neg, int(seed), int(pasos), cfg, resolucion,
                               modelo, usar_lora, lora, lora_fuerza, progress)
            if err:
                print(f"  Error: {err}")
                return gr.update(visible=False), gr.update(value=f"❌ {err}")

            print("  Generacion exitosa!")
            progress(1.0, desc="Completado!")

            nombre_modelo = info_checkpoint(modelo)["nombre"]
            msg = f"✅ Imagen generada con {nombre_modelo}"
            if usar_lora and lora:
                msg += f" + {info_lora(lora)['nombre']}"

            return (
                gr.update(value=img, visible=True),
                gr.update(value=msg),
            )

        except Exception as e:
            print(f"  Error critico: {e}")
            return (
                gr.update(visible=False),
                gr.update(value=f"❌ Error inesperado: {type(e).__name__}: {e}"),
            )

    generar_btn.click(
        fn=on_generar,
        inputs=[prompt, neg_prompt, seed, pasos, cfg, resolucion,
                modelo, usar_lora, lora_selector, lora_fuerza],
        outputs=[salida, estado],
        concurrency_limit=5,
        api_name="generar",
    )

    prompt.submit(
        fn=on_generar,
        inputs=[prompt, neg_prompt, seed, pasos, cfg, resolucion,
                modelo, usar_lora, lora_selector, lora_fuerza],
        outputs=[salida, estado],
        concurrency_limit=5,
    )

    # Mostrar tips iniciales
    if default_ckpt:
        info_inicial = info_checkpoint(default_ckpt)
        tips_inicial = (
            f"**{info_inicial['nombre']}** ({info_inicial['tipo']}) — {info_inicial['desc']}\n\n"
            f"💡 {info_inicial['tips']}\n\n"
            f"Valores recomendados: {info_inicial['pasos']} pasos, CFG {info_inicial['cfg']}, "
            f"sampler {info_inicial['sampler']}, scheduler {info_inicial['scheduler']}"
        )
        # Actualizar via JS en carga inicial
        demo.load(
            fn=lambda: tips_inicial,
            inputs=None,
            outputs=[modelo_tips],
        )

    gr.Markdown("""
    ---
    ### 💡 Tips por modelo

    | Modelo | Prompt recomendado |
    |--------|-------------------|
    | **Pony Diffusion** | `score_9, score_8_up, source_anime, rating_safe, ...` |
    | **DreamShaper** | Prompt natural: `a cat wearing a hat, digital art, clean lines` |
    | **Nova Comic** | `western_comics_(style), superhero, dynamic pose, ...` |
    | **Anima Pencil** | Prompt natural, CFG 2-4: `a peaceful village, soft colors` |

    **CleanLineArt LoRA** + DreamShaper = dibujos explicativos lineas limpias.
    """)

# ─── Entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    import webbrowser
    print("=" * 50)
    print("  Interfaz txt2img — Multi-Modelo")
    print("=" * 50)
    print(f"  Modelos disponibles: {len(checkpoints_disponibles)}")
    for ckpt in checkpoints_disponibles:
        info = info_checkpoint(ckpt)
        print(f"    - {info['nombre']} ({ckpt})")
    if loras_disponibles:
        print(f"  LoRAs disponibles: {len(loras_disponibles)}")
        for l in loras_disponibles:
            print(f"    - {l}")
    print(f"  Abriendo en http://localhost:7862")
    print(f"  ComfyUI debe estar en {COMFY_API}")
    print("=" * 50)
    webbrowser.open("http://localhost:7862")
    demo.queue(default_concurrency_limit=5).launch(
        server_name="localhost",
        server_port=7862,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
