"""
comfyui_generador_multi.py — Generacion txt2img con checkpoint y LoRA seleccionables.

Reutiliza helpers de API de comfyui_backend.py (no lo modifica).
Soporta cualquier checkpoint SDXL/SD1.5 y LoRA opcional.

Uso:
    from modules.comfyui_generador_multi import generar_txt2img_con_checkpoint, listar_checkpoints
    ruta = generar_txt2img_con_checkpoint(
        prompt="...", negative_prompt="...", seed=42,
        checkpoint_name="DreamShaperXL.safetensors",
        lora_name="cleanLineArt_sdxl.safetensors",
    )
"""

import uuid
import sys
from pathlib import Path
from typing import Optional

RAIZ_MODULES = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ_MODULES))

from comfyui_backend import (
    COMFY_OUTPUT, COMFY_API,
    _post, _upload_image,
    _esperar_resultado, _obtener_imagen_salida, _obtener_imagenes_salida,
    GraphBuilder,
)

# ─── Rutas de modelos (resueltas en modules/config.py) ────────────────

from config import RUTA_CHECKPOINTS, RUTA_LORAS  # noqa: E402


# ─── Informacion de modelos conocidos ──────────────────────────────────

MODEL_INFO = {
    "ponyDiffusionV6XL_v6StartWithThisOne.safetensors": {
        "nombre": "Pony Diffusion V6 XL",
        "desc": "Arte anime/estilizado, saturado, artistico",
        "tips": "Usa tags: score_9, score_8_up, source_anime, rating_safe, ...",
        "sampler": "euler",
        "scheduler": "simple",
        "pasos": 30,
        "cfg": 7.5,
        "peso": "~7 GB",
        "tipo": "SDXL",
    },
    "DreamShaperXL.safetensors": {
        "nombre": "DreamShaper XL",
        "desc": "Versatil, semi-realista a artistico",
        "tips": "Prompt natural en ingles. Funciona bien con CleanLineArt LoRA.",
        "sampler": "euler",
        "scheduler": "simple",
        "pasos": 25,
        "cfg": 6.0,
        "peso": "~6.5 GB",
        "tipo": "SDXL",
    },
    "novaComicXL_v20.safetensors": {
        "nombre": "Nova Comic XL v2.0",
        "desc": "Comic americano / superheroico (Marvel/DC)",
        "tips": "Usa 'western_comics_(style)' en el prompt. CFG 4-6.",
        "sampler": "euler",
        "scheduler": "simple",
        "pasos": 25,
        "cfg": 5.0,
        "peso": "~6.7 GB",
        "tipo": "SDXL",
    },
    "animaPencilXLClear_v2.safetensors": {
        "nombre": "Anima Pencil XL Clear v2",
        "desc": "Ilustracion lineas limpias, lapiz, color vivo",
        "tips": "CFG BAJO (2-4). Usa sampler Euler, scheduler Beta. ~15 pasos basta.",
        "sampler": "euler",
        "scheduler": "beta",
        "pasos": 15,
        "cfg": 3.5,
        "peso": "~6.6 GB",
        "tipo": "SDXL",
    },
    "juggernautXL_v10.safetensors": {
        "nombre": "Juggernaut XL v10",
        "desc": "Foto-realista, escenas, retratos",
        "tips": "Prompt natural. Ideal para fondos realistas.",
        "sampler": "euler",
        "scheduler": "simple",
        "pasos": 25,
        "cfg": 6.0,
        "peso": "~6.5 GB",
        "tipo": "SDXL",
    },
    "Realistic_Vision_V5.1.safetensors": {
        "nombre": "Realistic Vision V5.1",
        "desc": "Foto-realista (SD 1.5), mas rapido",
        "tips": "Prompt natural. Mas rapido por ser SD 1.5.",
        "sampler": "euler",
        "scheduler": "simple",
        "pasos": 25,
        "cfg": 6.0,
        "peso": "~5 GB",
        "tipo": "SD 1.5",
    },
}

# ─── LoRAs conocidos ───────────────────────────────────────────────────

LORA_INFO = {
    "CleanLineArt.safetensors": {
        "nombre": "CleanLineArt SDXL",
        "desc": "Lineas limpias, colores planos, estilo explicativo",
        "peso": "~17.6 MB",
        "base": "SDXL",
        "strength": 0.8,
        "trigger": "flat color, simple background, clean lineart, [color] background",
    },
}

CHECKPOINT_PATH = RUTA_CHECKPOINTS
LORAS_PATH = RUTA_LORAS


# ─── Funciones de descubrimiento ───────────────────────────────────────


def listar_checkpoints() -> list[str]:
    """Devuelve lista de archivos .safetensors/.ckpt en checkpoints."""
    if not CHECKPOINT_PATH.exists():
        return []
    return sorted([
        f.name for f in CHECKPOINT_PATH.iterdir()
        if f.suffix in (".safetensors", ".ckpt", ".pt")
    ])


def listar_loras() -> list[str]:
    """Devuelve lista de archivos .safetensors/.ckpt en loras."""
    if not LORAS_PATH.exists():
        return []
    return sorted([
        f.name for f in LORAS_PATH.iterdir()
        if f.suffix in (".safetensors", ".ckpt", ".pt")
    ])


def info_checkpoint(ckpt_name: str) -> dict:
    """Devuelve info del checkpoint si es conocido, o datos genericos."""
    if ckpt_name in MODEL_INFO:
        return dict(MODEL_INFO[ckpt_name])
    return {
        "nombre": ckpt_name.replace(".safetensors", "").replace("_", " ").title(),
        "desc": "Checkpoint generico",
        "tips": "Prompt natural en ingles.",
        "sampler": "euler",
        "scheduler": "simple",
        "pasos": 25,
        "cfg": 6.0,
        "peso": "?",
        "tipo": "SDXL" if "xl" in ckpt_name.lower() else "SD 1.5",
    }


def info_lora(lora_name: str) -> dict:
    """Devuelve info del LoRA si es conocido, o datos genericos."""
    if lora_name in LORA_INFO:
        return dict(LORA_INFO[lora_name])
    return {
        "nombre": lora_name.replace(".safetensors", "").replace("_", " ").title(),
        "desc": "LoRA generico",
        "peso": "?",
        "base": "?",
        "strength": 0.8,
        "trigger": "",
    }


# ─── Workflow builder ──────────────────────────────────────────────────


def _workflow_txt2img_multi(
    prompt: str,
    negative_prompt: str,
    seed: int,
    pasos: int,
    cfg: float,
    ancho: int,
    alto: int,
    client_id: str,
    checkpoint_name: str,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
) -> dict:
    """Workflow txt2img con checkpoint y LoRA seleccionables."""
    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple", ckpt_name=checkpoint_name)

    if lora_name:
        lora = g.node(
            "LoraLoader",
            model=ckpt.out(0),
            clip=ckpt.out(1),
            lora_name=lora_name,
            strength_model=lora_strength,
            strength_clip=lora_strength,
        )
        model_out = lora.out(0)
        clip_out = lora.out(1)
    else:
        model_out = ckpt.out(0)
        clip_out = ckpt.out(1)

    pos = g.node("CLIPTextEncode", text=prompt, clip=clip_out)
    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=clip_out)
    latent = g.node("EmptyLatentImage", width=ancho, height=alto, batch_size=1)
    sampler = g.node("KSampler",
                     seed=seed, steps=pasos, cfg=cfg,
                     sampler_name="euler", scheduler="simple",
                     denoise=1.0,
                     model=model_out, positive=pos.out(0),
                     negative=neg.out(0), latent_image=latent.out(0))
    decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
    g.node("SaveImage",
           filename_prefix=f"multi{client_id[:4]}",
           images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


# ─── Workflow builder: variaciones (pseudo-animacion) ─────────────────


def _workflow_variaciones(
    prompt: str,
    negative_prompt: str,
    seed: int,
    pasos: int,
    cfg: float,
    denoise: float,
    num_frames: int,
    nombre_imagen: str,
    client_id: str,
    checkpoint_name: str,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
) -> dict:
    """
    Workflow que genera N variaciones de una misma imagen en UNA sola llamada.

    Cada variacion usa el mismo prompt y la misma imagen base codificada,
    pero con seeds diferentes. El resultado son N frames que varian
    sutilmente (como un personaje que "respira" o "parpadea").

    La imagen base se codifica UNA VEZ (VAE Encode) y se reusa para
    todos los KSamplers, ahorrando VRAM y tiempo.
    """
    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple", ckpt_name=checkpoint_name)

    if lora_name:
        lora = g.node(
            "LoraLoader",
            model=ckpt.out(0), clip=ckpt.out(1),
            lora_name=lora_name,
            strength_model=lora_strength,
            strength_clip=lora_strength,
        )
        model_out = lora.out(0)
        clip_out = lora.out(1)
    else:
        model_out = ckpt.out(0)
        clip_out = ckpt.out(1)

    pos = g.node("CLIPTextEncode", text=prompt, clip=clip_out)
    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=clip_out)

    # Cargar y codificar imagen base UNA SOLA VEZ
    loader = g.node("LoadImage", image=nombre_imagen)
    encode = g.node("VAEEncode", pixels=loader.out(0), vae=ckpt.out(2))

    # N KSamplers paralelos desde el mismo latente codificado
    for i in range(num_frames):
        s = seed + i
        sampler = g.node("KSampler",
                         seed=s, steps=pasos, cfg=cfg,
                         sampler_name="euler", scheduler="simple",
                         denoise=denoise,
                         model=model_out,
                         positive=pos.out(0),
                         negative=neg.out(0),
                         latent_image=encode.out(0))
        decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
        g.node("SaveImage",
               filename_prefix=f"var{client_id[:4]}_{i:02d}",
               images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


# ─── Workflow builder: animacion por progresion de prompts ───────────


def _workflow_animacion_progresiva(
    prompts: list[str],
    negative_prompt: str,
    seed: int,
    pasos: int,
    cfg: float,
    denoise: float,
    nombre_imagen: str,
    client_id: str,
    checkpoint_name: str,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
) -> dict:
    """
    Workflow que genera N frames desde una imagen base, cada uno con su
    propio prompt de progresion. Todos los frames parten de la MISMA
    imagen base (no encadenados).

    Cada frame tiene:
      - Prompt diferente (describiendo el movimiento en esa etapa)
      - Seed diferente (seed + i)
      - Mismo denoise

    Esto produce movimiento DIRIGIDO y visible (ej: brazo que se levanta
    gradualmente), a diferencia de generar_variaciones donde solo cambia
    la seed y los cambios son aleatorios.

    Args:
        prompts: Lista de N prompts, uno por frame (cada uno describe
                 el movimiento en ese punto)
        negative_prompt: Prompt negativo comun
        seed: Seed base (cada frame suma i)
        pasos: Pasos por frame
        cfg: Escala CFG
        denoise: 0.35-0.5 para movimiento visible
        nombre_imagen: Nombre de la imagen ya subida a ComfyUI
        client_id: ID unico
        checkpoint_name, lora_name, lora_strength: modelo
    """
    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple", ckpt_name=checkpoint_name)

    if lora_name:
        lora = g.node(
            "LoraLoader",
            model=ckpt.out(0), clip=ckpt.out(1),
            lora_name=lora_name,
            strength_model=lora_strength,
            strength_clip=lora_strength,
        )
        model_out = lora.out(0)
        clip_out = lora.out(1)
    else:
        model_out = ckpt.out(0)
        clip_out = ckpt.out(1)

    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=clip_out)

    # Cargar y codificar imagen base
    loader = g.node("LoadImage", image=nombre_imagen)
    current_latent = g.node("VAEEncode", pixels=loader.out(0), vae=ckpt.out(2)).out(0)

    # N KSamplers ENCADENADOS: cada frame usa el latente del anterior
    # Esto produce movimiento REAL porque cada frame construye sobre el cambio anterior
    for i, frame_prompt in enumerate(prompts):
        s = seed + i
        pos = g.node("CLIPTextEncode",
                      text=frame_prompt, clip=clip_out)
        sampler = g.node("KSampler",
                         seed=s, steps=pasos, cfg=cfg,
                         sampler_name="euler", scheduler="simple",
                         denoise=denoise,
                         model=model_out,
                         positive=pos.out(0),
                         negative=neg.out(0),
                         latent_image=current_latent)
        current_latent = sampler.out(0)  # El output es el input del siguiente frame
        decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
        g.node("SaveImage",
               filename_prefix=f"chain{client_id[:4]}_{i:02d}",
               images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


# ─── Workflow: animacion txt2img + IP-Adapter ───────────────────────


def _workflow_animacion_ipadapter(
    prompts: list[str],
    negative_prompt: str,
    seed: int,
    pasos: int,
    cfg: float,
    nombre_imagen_ref: str,
    client_id: str,
    checkpoint_name: str,
    ip_weight: float = 0.20,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
) -> dict:
    """
    Workflow que genera N frames DESDE CERO (txt2img) usando IP-Adapter
    para mantener la consistencia del personaje.

    Cada frame es independiente (txt2img, denoise=1.0), con su propio
    prompt describiendo la pose. IP-Adapter se asegura de que el
    personaje (cara, ropa, estilo) se mantenga igual entre frames.

    A diferencia del enfoque img2img encadenado, aqui el modelo NO esta
    limitado por la estructura de una imagen base — puede generar
    CUALQUIER pose que describa el prompt.

    Args:
        prompts: Lista de N prompts, cada uno describe la pose en ese frame
        negative_prompt: Prompt negativo comun
        seed: Seed base (cada frame suma i)
        pasos: Pasos por frame
        cfg: Escala CFG
        nombre_imagen_ref: Nombre de la imagen ya subida (referencia IP-Adapter)
        client_id: ID unico
        checkpoint_name: Checkpoint a usar
        ip_weight: Peso del IP-Adapter (0.1-0.3 recomendado)
        lora_name, lora_strength: LoRA opcional
    """
    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple", ckpt_name=checkpoint_name)

    if lora_name:
        lora = g.node(
            "LoraLoader",
            model=ckpt.out(0), clip=ckpt.out(1),
            lora_name=lora_name,
            strength_model=lora_strength,
            strength_clip=lora_strength,
        )
        model_out = lora.out(0)
        clip_out = lora.out(1)
    else:
        model_out = ckpt.out(0)
        clip_out = ckpt.out(1)

    # IP-Adapter: carga la imagen de referencia UNA VEZ y la conecta a todos los frames
    ref_loader = g.node("LoadImage", image=nombre_imagen_ref)
    ip_unified = g.node("IPAdapterUnifiedLoader",
                         model=model_out,
                         preset="STANDARD (medium strength)")

    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=clip_out)

    # N frames independientes en paralelo, cada uno desde ruido puro
    for i, frame_prompt in enumerate(prompts):
        s = seed + i
        pos = g.node("CLIPTextEncode",
                      text=frame_prompt, clip=clip_out)

        ipa = g.node("IPAdapterAdvanced",
                      model=ip_unified.out(0),
                      ipadapter=ip_unified.out(1),
                      image=ref_loader.out(0),
                      weight=ip_weight,
                      weight_type="linear",
                      combine_embeds="concat",
                      start_at=0.0,
                      end_at=1.0,
                      embeds_scaling="K+mean(V) w/ C penalty")

        latent = g.node("EmptyLatentImage",
                         width=1024, height=1024, batch_size=1)

        sampler = g.node("KSampler",
                         seed=s, steps=pasos, cfg=cfg,
                         sampler_name="euler", scheduler="simple",
                         denoise=1.0,
                         model=ipa.out(0),
                         positive=pos.out(0),
                         negative=neg.out(0),
                         latent_image=latent.out(0))

        decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
        g.node("SaveImage",
               filename_prefix=f"ipa{client_id[:4]}_{i:02d}",
               images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


# ─── Workflow: animacion ControlNet + IP-Adapter ──────────────────────


def _workflow_animacion_controlnet(
    prompts: list[str],
    negative_prompt: str,
    seed: int,
    pasos: int,
    cfg: float,
    nombre_imagen_ref: str,
    nombre_imagen_base: str,
    nombres_poses: list[str],
    client_id: str,
    checkpoint_name: str,
    controlnet_name: str = "OpenPoseXL2.safetensors",
    ip_weight: float = 0.20,
    controlnet_strength: float = 1.0,
    denoise: float = 0.45,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
) -> dict:
    """
    Workflow que genera N frames con ControlNet (OpenPose) + IP-Adapter.

    Cada frame:
      1. Parte de la MISMA imagen base codificada (img2img, no ruido puro)
      2. ControlNet lo guia con el esqueleto de pose
      3. IP-Adapter mantiene consistencia del personaje
      4. Denoise bajo (~0.45) preserva la estructura del personaje

    Esto da movimiento PRECISO (ControlNet) sin que el personaje cambie
    drasticamente (img2img desde la misma base).
    """
    num_frames = len(prompts)
    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple", ckpt_name=checkpoint_name)

    if lora_name:
        lora = g.node(
            "LoraLoader",
            model=ckpt.out(0), clip=ckpt.out(1),
            lora_name=lora_name,
            strength_model=lora_strength,
            strength_clip=lora_strength,
        )
        model_out = lora.out(0)
        clip_out = lora.out(1)
    else:
        model_out = ckpt.out(0)
        clip_out = ckpt.out(1)

    # ── ControlNet: cargar modelo una vez ──
    controlnet = g.node("ControlNetLoader",
                         control_net_name=controlnet_name)

    # ── IP-Adapter: cargar imagen de referencia ──
    ref_loader = g.node("LoadImage", image=nombre_imagen_ref)
    ip_unified = g.node("IPAdapterUnifiedLoader",
                         model=model_out,
                         preset="STANDARD (medium strength)")

    # ── Imagen base: codificar UNA VEZ para todos los frames ──
    base_loader = g.node("LoadImage", image=nombre_imagen_base)
    base_latent = g.node("VAEEncode",
                          pixels=base_loader.out(0),
                          vae=ckpt.out(2)).out(0)

    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=clip_out)

    # ── N frames, cada uno con su pose ──
    for i, frame_prompt in enumerate(prompts):
        s = seed + i
        pos = g.node("CLIPTextEncode",
                      text=frame_prompt, clip=clip_out)

        # Cargar pose para este frame
        pose_loader = g.node("LoadImage", image=nombres_poses[i])

        # Aplicar ControlNet a la conditioning (guia la pose)
        pos_con_pose = g.node("ControlNetApply",
                               conditioning=pos.out(0),
                               control_net=controlnet.out(0),
                               image=pose_loader.out(0),
                               strength=controlnet_strength)

        # Aplicar IP-Adapter al modelo (consistencia del personaje)
        ipa = g.node("IPAdapterAdvanced",
                      model=ip_unified.out(0),
                      ipadapter=ip_unified.out(1),
                      image=ref_loader.out(0),
                      weight=ip_weight,
                      weight_type="linear",
                      combine_embeds="concat",
                      start_at=0.0,
                      end_at=1.0,
                      embeds_scaling="K+mean(V) w/ C penalty")

        # KSampler desde la imagen base codificada (img2img, denoise controlado)
        sampler = g.node("KSampler",
                         seed=s, steps=pasos, cfg=cfg,
                         sampler_name="euler", scheduler="simple",
                         denoise=denoise,
                         model=ipa.out(0),
                         positive=pos_con_pose.out(0),
                         negative=neg.out(0),
                         latent_image=base_latent)

        decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
        g.node("SaveImage",
               filename_prefix=f"cn{client_id[:4]}_{i:02d}",
               images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


# ─── API publica ───────────────────────────────────────────────────────


def generar_txt2img_con_checkpoint(
    prompt: str,
    negative_prompt: str,
    seed: int,
    checkpoint_name: str,
    ancho: int = 1024,
    alto: int = 1024,
    pasos: int = 25,
    cfg: float = 6.0,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
    api_url: str = COMFY_API,
    timeout: int = 600,
) -> Optional[Path]:
    """
    Genera una imagen txt2img usando cualquier checkpoint + LoRA opcional.

    Args:
        prompt: Descripcion de la imagen
        negative_prompt: Lo que NO queremos en la imagen
        seed: Semilla (0 = aleatorio)
        checkpoint_name: Nombre del archivo .safetensors en checkpoints/
        ancho, alto: Dimensiones de la imagen
        pasos: Pasos de muestreo
        cfg: Escala CFG
        lora_name: Nombre del archivo LoRA .safetensors en loras/ (opcional)
        lora_strength: Fuerza del LoRA (0.0 - 1.0)
        api_url: URL de ComfyUI
        timeout: Timeout en segundos

    Returns:
        Path al archivo generado o None si falla
    """
    if seed <= 0:
        import random
        seed = random.randint(1, 999999999)

    client_id = str(uuid.uuid4())
    data = _workflow_txt2img_multi(
        prompt, negative_prompt, seed, pasos, cfg, ancho, alto,
        client_id, checkpoint_name, lora_name, lora_strength,
    )

    print(f"  Enviando a ComfyUI (checkpoint={checkpoint_name}, seed={seed})...")
    if lora_name:
        print(f"    Con LoRA: {lora_name} (strength={lora_strength})")

    try:
        result = _post(api_url, "/prompt", data)
    except Exception as e:
        print(f"  ERROR al enviar: {e}")
        return None

    prompt_id = result.get("prompt_id", "")
    if not prompt_id:
        print(f"  ERROR: no se recibio prompt_id")
        if "error" in result:
            print(f"  Detalles: {result['error']}")
        return None

    print(f"    prompt_id: {prompt_id}")
    history = _esperar_resultado(api_url, prompt_id, timeout)
    if not history:
        return None
    return _obtener_imagen_salida(history, prefijo_log="    ")


def generar_variaciones(
    prompt: str,
    negative_prompt: str,
    seed: int,
    num_frames: int,
    denoise: float,
    imagen_ref: Path,
    checkpoint_name: str,
    pasos: int = 20,
    cfg: float = 6.0,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
    api_url: str = COMFY_API,
    timeout: int = 600,
) -> Optional[list[Path]]:
    """
    Genera N variaciones sutiles de una imagen usando img2img con seeds
    diferentes. Todas parten de la MISMA imagen base con el MISMO prompt,
    variando solo la seed y con denoise bajo.

    Esto crea una pseudo-animacion donde el personaje/elemento parece
    "cobrar vida" (parpadeos, respiracion, micro-movimientos).

    A diferencia de generar_lote_img2img_encadenado, cada frame es
    independiente (no se encadenan), lo que evita la deriva visual.

    Args:
        prompt: Mismo prompt que se uso para generar la imagen original
        negative_prompt: Prompt negativo
        seed: Seed base (cada frame suma i)
        num_frames: Cantidad de variaciones a generar
        denoise: 0.1-0.3 ideal (bajo = cambios sutiles)
        imagen_ref: Path a la imagen PNG generada original
        checkpoint_name: Checkpoint a usar
        pasos: Pasos por frame
        cfg: Escala CFG
        lora_name: LoRA opcional
        lora_strength: Fuerza del LoRA
        api_url: URL de ComfyUI
        timeout: Timeout total

    Returns:
        Lista de Paths a los frames ordenados, o None si falla
    """
    if seed <= 0:
        import random
        seed = random.randint(1, 999999999)

    if not imagen_ref or not imagen_ref.exists():
        print(f"  ERROR: imagen de referencia no existe: {imagen_ref}")
        return None

    client_id = str(uuid.uuid4())

    # 1. Subir imagen a ComfyUI
    print(f"  Subiendo imagen base: {imagen_ref.name}...")
    try:
        nombre_img = _upload_image(api_url, imagen_ref)
    except Exception as e:
        print(f"  ERROR al subir imagen: {e}")
        return None

    # 2. Construir workflow con N variaciones paralelas
    print(f"  Construyendo workflow con {num_frames} variaciones...")
    data = _workflow_variaciones(
        prompt, negative_prompt, seed, pasos, cfg, denoise,
        num_frames, nombre_img, client_id,
        checkpoint_name, lora_name, lora_strength,
    )

    # 3. Enviar a ComfyUI
    print(f"  Enviando a ComfyUI (checkpoint={checkpoint_name}, "
          f"seed={seed}, denoise={denoise}, frames={num_frames})...")
    if lora_name:
        print(f"    Con LoRA: {lora_name} (strength={lora_strength})")

    try:
        result = _post(api_url, "/prompt", data)
    except Exception as e:
        print(f"  ERROR al enviar: {e}")
        return None

    prompt_id = result.get("prompt_id", "")
    if not prompt_id:
        print(f"  ERROR: no se recibio prompt_id")
        if "error" in result:
            print(f"  Detalles: {result['error']}")
        return None

    # 4. Esperar resultado
    print(f"    prompt_id: {prompt_id}")
    history = _esperar_resultado(api_url, prompt_id, timeout)
    if not history:
        return None

    # 5. Extraer TODAS las imagenes generadas
    frames = _obtener_imagenes_salida(history, prefijo_log="    ")

    if not frames:
        print(f"  ERROR: no se generaron frames")
        return None

    # Ordenar por nombre (los SaveImage tienen prefijo con indice)
    frames.sort(key=lambda p: p.stem)

    if len(frames) < num_frames:
        print(f"  WARN: se pidieron {num_frames} frames, se obtuvieron {len(frames)}")

    print(f"  ✅ {len(frames)} variaciones generadas")
    return frames


def generar_animacion_ipadapter(
    prompts: list[str],
    negative_prompt: str,
    seed: int,
    imagen_ref: Path,
    checkpoint_name: str,
    pasos: int = 25,
    cfg: float = 6.0,
    ip_weight: float = 0.20,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
    api_url: str = COMFY_API,
    timeout: int = 600,
) -> Optional[list[Path]]:
    """
    Genera N frames DESDE CERO (txt2img) manteniendo el personaje
    consistente con IP-Adapter. Cada frame arranca de ruido puro
    con su propio prompt de pose. IP-Adapter mantiene cara, ropa y estilo.

    A diferencia de generar_animacion_progresiva (img2img encadenado),
    aqui el modelo NO esta limitado por una imagen base y puede
    generar CUALQUIER pose.
    """
    num_frames = len(prompts)
    if num_frames == 0:
        return None

    if seed <= 0:
        import random
        seed = random.randint(1, 999999999)

    if not imagen_ref or not imagen_ref.exists():
        print(f"  ERROR: imagen de referencia no existe: {imagen_ref}")
        return None

    client_id = str(uuid.uuid4())

    print(f"  Subiendo imagen referencia para IP-Adapter: {imagen_ref.name}...")
    try:
        nombre_ref = _upload_image(api_url, imagen_ref)
    except Exception as e:
        print(f"  ERROR al subir imagen: {e}")
        return None

    print(f"  Workflow: {num_frames} frames txt2img + IP-Adapter (weight={ip_weight})")
    for i, p in enumerate(prompts):
        print(f"    Frame {i}: {p[:80]}")
    data = _workflow_animacion_ipadapter(
        prompts, negative_prompt, seed, pasos, cfg,
        nombre_ref, client_id, checkpoint_name, ip_weight,
        lora_name, lora_strength,
    )

    print(f"  Enviando a ComfyUI (checkpoint={checkpoint_name}, seed={seed}, frames={num_frames})...")
    if lora_name:
        print(f"    Con LoRA: {lora_name} (strength={lora_strength})")
    try:
        result = _post(api_url, "/prompt", data)
    except Exception as e:
        print(f"  ERROR al enviar: {e}")
        return None

    prompt_id = result.get("prompt_id", "")
    if not prompt_id:
        print(f"  ERROR: no se recibio prompt_id")
        return None

    print(f"    prompt_id: {prompt_id}")
    history = _esperar_resultado(api_url, prompt_id, timeout)
    if not history:
        return None

    frames = _obtener_imagenes_salida(history, prefijo_log="    ")
    if not frames:
        print(f"  ERROR: no se generaron frames")
        return None

    frames.sort(key=lambda p: p.stem)
    print(f"  ✅ {len(frames)} frames generados con IP-Adapter")
    return frames


def generar_animacion_progresiva(
    prompts: list[str],
    negative_prompt: str,
    seed: int,
    denoise: float,
    imagen_ref: Path,
    checkpoint_name: str,
    pasos: int = 25,
    cfg: float = 6.0,
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
    api_url: str = COMFY_API,
    timeout: int = 600,
) -> Optional[list[Path]]:
    """
    Genera N frames desde una imagen base, cada uno con un prompt
    diferente que describe el movimiento PASO A PASO.

    A diferencia de generar_variaciones (mismo prompt, cambios aleatorios),
    aqui CADA FRAME tiene su propio prompt. Esto permite movimiento
    DIRIGIDO y VISIBLE: un brazo que se levanta, una cabeza que gira,
    una mano que senala, etc.

    Todos los frames parten de la MISMA imagen base (no encadenados),
    evitando la deriva visual.

    Args:
        prompts: Lista de strings. Cada uno describe el movimiento en
                 ese frame. Ej:
                   ["mano en la calculadora, mirando abajo",
                    "mano levantando la calculadora",
                    "mano con calculadora en alto, sonrisa",
                    "mano bajando la calculadora",
                    "mano de vuelta en la calculadora"]
                 Todos se concatenan al prompt base internamente.
        negative_prompt: Prompt negativo comun
        seed: Seed base (cada frame suma i)
        denoise: 0.35-0.5 para movimiento claro
        imagen_ref: Path a la imagen PNG base
        checkpoint_name: Checkpoint a usar
        pasos: Pasos por frame
        cfg: Escala CFG
        lora_name: LoRA opcional
        lora_strength: Fuerza del LoRA
        api_url: URL de ComfyUI
        timeout: Timeout total

    Returns:
        Lista de Paths a los frames, o None si falla
    """
    num_frames = len(prompts)
    if num_frames == 0:
        print(f"  ERROR: lista de prompts vacia")
        return None

    if seed <= 0:
        import random
        seed = random.randint(1, 999999999)

    if not imagen_ref or not imagen_ref.exists():
        print(f"  ERROR: imagen de referencia no existe: {imagen_ref}")
        return None

    client_id = str(uuid.uuid4())

    # 1. Subir imagen a ComfyUI
    print(f"  Subiendo imagen base: {imagen_ref.name}...")
    try:
        nombre_img = _upload_image(api_url, imagen_ref)
    except Exception as e:
        print(f"  ERROR al subir imagen: {e}")
        return None

    # 2. Construir workflow con N prompts progresivos
    print(f"  Construyendo workflow con {num_frames} frames progresivos...")
    for i, p in enumerate(prompts):
        print(f"    Frame {i}: {p[:70]}")
    data = _workflow_animacion_progresiva(
        prompts, negative_prompt, seed, pasos, cfg, denoise,
        nombre_img, client_id,
        checkpoint_name, lora_name, lora_strength,
    )

    # 3. Enviar a ComfyUI
    print(f"  Enviando a ComfyUI (checkpoint={checkpoint_name}, "
          f"seed={seed}, denoise={denoise}, frames={num_frames})...")
    if lora_name:
        print(f"    Con LoRA: {lora_name} (strength={lora_strength})")

    try:
        result = _post(api_url, "/prompt", data)
    except Exception as e:
        print(f"  ERROR al enviar: {e}")
        return None

    prompt_id = result.get("prompt_id", "")
    if not prompt_id:
        print(f"  ERROR: no se recibio prompt_id")
        if "error" in result:
            print(f"  Detalles: {result['error']}")
        return None

    # 4. Esperar resultado
    print(f"    prompt_id: {prompt_id}")
    history = _esperar_resultado(api_url, prompt_id, timeout)
    if not history:
        return None

    # 5. Extraer frames
    frames = _obtener_imagenes_salida(history, prefijo_log="    ")
    if not frames:
        print(f"  ERROR: no se generaron frames")
        return None

    frames.sort(key=lambda p: p.stem)

    if len(frames) < num_frames:
        print(f"  WARN: se pidieron {num_frames} frames, se obtuvieron {len(frames)}")

    print(f"  ✅ {len(frames)} frames progresivos generados")
    return frames


def generar_animacion_controlnet(
    prompts: list[str],
    negative_prompt: str,
    seed: int,
    poses: list[Path],
    imagen_base: Path,
    checkpoint_name: str,
    pasos: int = 25,
    cfg: float = 7.0,
    ip_weight: float = 0.30,
    controlnet_strength: float = 1.2,
    denoise: float = 0.60,
    controlnet_name: str = "OpenPoseXL2.safetensors",
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
    api_url: str = COMFY_API,
    timeout: int = 600,
) -> Optional[list[Path]]:
    """
    Genera N frames con ControlNet (OpenPose) + IP-Adapter + img2img.

    Cada frame parte de la MISMA imagen base (img2img, denoise~0.60).
    ControlNet guia la pose segun el esqueleto.
    IP-Adapter mantiene la consistencia del personaje.

    Args:
        prompts: Lista de N prompts, uno por frame
        negative_prompt: Prompt negativo
        seed: Seed base
        poses: Lista de Paths a las imagenes de esqueleto (OpenPose)
        imagen_base: Path a la imagen base (para img2img e IP-Adapter)
        checkpoint_name: Checkpoint a usar
        pasos: Pasos por frame
        cfg: Escala CFG
        ip_weight: Peso del IP-Adapter (0.20-0.40)
        controlnet_strength: Fuerza del ControlNet (1.0-2.0)
        denoise: 0.45-0.70 recomendado. Controla cuánto cambia respecto a la base
        controlnet_name: Nombre del archivo ControlNet
        lora_name, lora_strength: LoRA opcional
        api_url: URL de ComfyUI
        timeout: Timeout total

    Returns:
        Lista de Paths a los frames, o None si falla
    """
    num_frames = len(prompts)
    if num_frames == 0:
        print(f"  ERROR: lista de prompts vacia")
        return None

    if seed <= 0:
        import random
        seed = random.randint(1, 999999999)

    if not imagen_base or not imagen_base.exists():
        print(f"  ERROR: imagen base no existe: {imagen_base}")
        return None

    client_id = str(uuid.uuid4())

    # 1. Subir imagen base (sirve para IP-Adapter Y para img2img)
    print(f"  Subiendo imagen base: {imagen_base.name}...")
    try:
        nombre_base = _upload_image(api_url, imagen_base)
    except Exception as e:
        print(f"  ERROR al subir imagen base: {e}")
        return None

    # 2. Subir imagenes de pose
    nombres_poses = []
    print(f"  Subiendo {len(poses)} imagenes de pose...")
    for i, pose_path in enumerate(poses):
        if not pose_path or not pose_path.exists():
            print(f"  WARN: pose {i} no existe, saltando")
            continue
        try:
            nombre_pose = _upload_image(api_url, pose_path)
            nombres_poses.append(nombre_pose)
        except Exception as e:
            print(f"  ERROR al subir pose {i}: {e}")
            return None

    if len(nombres_poses) != num_frames:
        print(f"  ERROR: se subieron {len(nombres_poses)} poses de {num_frames}")
        while len(nombres_poses) < num_frames:
            nombres_poses.append(nombres_poses[-1] if nombres_poses else nombre_base)

    # 3. Construir workflow
    print(f"  Workflow: {num_frames} frames | img2img + ControlNet + IP-Adapter")
    print(f"    ControlNet: {controlnet_name} (strength={controlnet_strength})")
    print(f"    IP-Adapter: weight={ip_weight}")
    print(f"    Denoise: {denoise} (desde misma imagen base)")
    for i, p in enumerate(prompts):
        print(f"    Frame {i}: {p[:70]}")

    data = _workflow_animacion_controlnet(
        prompts, negative_prompt, seed, pasos, cfg,
        nombre_base, nombre_base, nombres_poses, client_id,
        checkpoint_name, controlnet_name, ip_weight, controlnet_strength,
        denoise, lora_name, lora_strength,
    )

    # 4. Enviar a ComfyUI
    print(f"  Enviando a ComfyUI (checkpoint={checkpoint_name}, seed={seed})...")
    if lora_name:
        print(f"    Con LoRA: {lora_name} (strength={lora_strength})")
    try:
        result = _post(api_url, "/prompt", data)
    except Exception as e:
        print(f"  ERROR al enviar: {e}")
        return None

    prompt_id = result.get("prompt_id", "")
    if not prompt_id:
        print(f"  ERROR: no se recibio prompt_id")
        if "error" in result:
            print(f"  Detalles: {result['error']}")
        return None

    # 5. Esperar resultado
    print(f"    prompt_id: {prompt_id}")
    history = _esperar_resultado(api_url, prompt_id, timeout)
    if not history:
        return None

    # 6. Extraer frames
    frames = _obtener_imagenes_salida(history, prefijo_log="    ")
    if not frames:
        print(f"  ERROR: no se generaron frames")
        return None

    frames.sort(key=lambda p: p.stem)
    print(f"  ✅ {len(frames)} frames generados con ControlNet + IP-Adapter")
    return frames


# ─── Workflow AnimateDiff (UN solo prompt — el que SÍ funciona) ──────


def _workflow_txt2anim(
    prompt: str,
    negative_prompt: str,
    seed: int,
    pasos: int,
    cfg: float,
    num_frames: int,
    client_id: str,
    checkpoint_name: str,
    motion_model_name: str = "mm_sdxl_v10_beta.safetensors",
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
) -> dict:
    """
    Workflow AnimateDiff con UN solo prompt para todos los frames.
    Es exactamente el test que mejor funciono.

    El motion module de AnimateDiff genera movimiento a partir de
    las variaciones de ruido entre frames del batch.
    """
    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple", ckpt_name=checkpoint_name)

    if lora_name:
        lora = g.node("LoraLoader",
            model=ckpt.out(0), clip=ckpt.out(1),
            lora_name=lora_name,
            strength_model=lora_strength,
            strength_clip=lora_strength,
        )
        model_out = lora.out(0)
        clip_out = lora.out(1)
    else:
        model_out = ckpt.out(0)
        clip_out = ckpt.out(1)

    # AnimateDiff Gen2 con context window para mas de 32 frames
    motion = g.node("ADE_LoadAnimateDiffModel",
        model_name=motion_model_name)
    m_models = g.node("ADE_ApplyAnimateDiffModelSimple",
        motion_model=motion.out(0))
    # Context window permite procesar mas frames del limite del motion module (32)
    context = g.node("ADE_LoopedUniformContextOptions",
        context_length=16, context_stride=1, context_overlap=4,
        closed_loop=False)
    model_w = g.node("ADE_UseEvolvedSampling",
        model=model_out,
        m_models=m_models.out(0),
        context_options=context.out(0),
        beta_schedule="autoselect")

    # UN solo prompt para todos los frames
    pos = g.node("CLIPTextEncode", text=prompt, clip=clip_out)
    neg_node = g.node("CLIPTextEncode", text=negative_prompt, clip=clip_out)

    # Batch latent
    latent = g.node("EmptyLatentImage", width=1024, height=1024, batch_size=num_frames)

    sampler = g.node("KSampler",
        seed=seed, steps=pasos, cfg=cfg,
        sampler_name="euler", scheduler="simple",
        denoise=1.0,
        model=model_w.out(0),
        positive=pos.out(0),
        negative=neg_node.out(0),
        latent_image=latent.out(0))

    decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
    g.node("SaveImage",
        filename_prefix=f"anim_{client_id[:4]}",
        images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


def generar_txt2anim(
    prompt: str,
    negative_prompt: str,
    seed: int,
    num_frames: int,
    checkpoint_name: str,
    pasos: int = 25,
    cfg: float = 6.0,
    motion_model: str = "mm_sdxl_v10_beta.safetensors",
    lora_name: Optional[str] = None,
    lora_strength: float = 0.8,
    api_url: str = COMFY_API,
    timeout: int = 900,
) -> Optional[list[Path]]:
    """
    Genera animacion con AnimateDiff usando UN SOLO prompt.
    Es exactamente la configuracion del test que mejor resultado dio.

    El motion module de AnimateDiff genera movimiento natural a partir
    de las variaciones de ruido entre los frames del batch.
    No necesita prompts progresivos, ControlNet, IP-Adapter ni img2img.

    Args:
        prompt: Un solo prompt describiendo la escena
        negative_prompt: Prompt negativo
        seed: Seed (misma para todos los frames)
        num_frames: Cantidad de frames a generar
        checkpoint_name: Checkpoint SDXL
        pasos: Pasos de muestreo
        cfg: Escala CFG (6.0 recomendado)
        motion_model: Nombre del motion module
        lora_name, lora_strength: LoRA opcional
        api_url: URL de ComfyUI
        timeout: Timeout en segundos

    Returns:
        Lista de Paths a los frames, o None si falla
    """
    if num_frames < 2:
        print("  ERROR: num_frames debe ser >= 2")
        return None

    if seed <= 0:
        import random
        seed = random.randint(1, 999999999)

    client_id = str(uuid.uuid4())[:8]

    print(f"  Generando {num_frames} frames con AnimateDiff...")
    print(f"  Checkpoint: {checkpoint_name}")
    print(f"  Prompt: {prompt[:100]}")
    print(f"  CFG: {cfg}, Seed: {seed}")

    data = _workflow_txt2anim(
        prompt, negative_prompt, seed, pasos, cfg, num_frames,
        client_id, checkpoint_name, motion_model,
        lora_name, lora_strength,
    )

    try:
        result = _post(api_url, "/prompt", data)
    except Exception as e:
        print(f"  ERROR al enviar: {e}")
        return None

    prompt_id = result.get("prompt_id", "")
    if not prompt_id:
        print(f"  ERROR: no se recibio prompt_id")
        if "error" in result:
            print(f"  Detalles: {result['error']}")
        return None

    print(f"    prompt_id: {prompt_id}")
    history = _esperar_resultado(api_url, prompt_id, timeout)
    if not history:
        return None

    frames = _obtener_imagenes_salida(history, prefijo_log="    ")
    if not frames:
        print(f"  ERROR: no se generaron frames")
        return None

    frames.sort(key=lambda p: p.stem)
    print(f"  {len(frames)} frames generados con AnimateDiff!")
    return frames
