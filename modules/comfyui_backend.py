"""
comfyui_backend.py — Generacion de imagenes usando ComfyUI como backend.

Requiere ComfyUI corriendo con --listen (ver README, seccion "Uso"):
  cd <carpeta_de_ComfyUI>
  "<venv>/Scripts/python.exe" main.py --listen

Usa la API de ComfyUI para ejecutar workflows de txt2img e img2img.
Los workflows se construyen con GraphBuilder (el mismo que usa ComfyUI internamente)
para garantizar compatibilidad total con el formato de la API.

Las rutas de ComfyUI se resuelven en modules/config.py (config.json).
"""

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional
import urllib.request
from io import BytesIO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COMFY_API, COMFY_OUTPUT  # noqa: E402


# ── Helpers de GraphBuilder (replica de comfy_execution.graph_utils) ──
# Usamos el mismo formato que ComfyUI internamente para evitar
# problemas de compatibilidad con la API.

class GraphBuilder:
    """Construye grafos en el formato que ComfyUI espera en /prompt.

    Cada builder genera IDs de nodo unicos (basados en uuid) para evitar
    colisiones de cache entre distintas generaciones.
    """

    def __init__(self):
        self.nodes = {}
        self._prefix = uuid.uuid4().hex[:6]

    def _next_id(self) -> str:
        return f"{self._prefix}_{len(self.nodes) + 1}"

    def node(self, class_type: str, **inputs) -> "GraphNode":
        nid = self._next_id()
        node = GraphNode(nid, class_type, inputs)
        self.nodes[nid] = node
        return node

    def finalize(self) -> dict:
        """Devuelve el dict en formato 'prompt' para la API de ComfyUI."""
        out = {}
        for nid, node in self.nodes.items():
            serialized = {"class_type": node.class_type, "inputs": {}}
            for k, v in node.inputs.items():
                if isinstance(v, GraphNodeOutput):
                    serialized["inputs"][k] = [v.node_id, v.index]
                else:
                    serialized["inputs"][k] = v
            out[nid] = serialized
        return out


class GraphNode:
    def __init__(self, nid: str, class_type: str, inputs: dict):
        self.id = nid
        self.class_type = class_type
        self.inputs = inputs

    def out(self, index: int = 0) -> "GraphNodeOutput":
        return GraphNodeOutput(self.id, index)


class GraphNodeOutput:
    def __init__(self, node_id: str, index: int):
        self.node_id = node_id
        self.index = index


# ── Construccion de workflows ─────────────────────────────────────────

def _workflow_txt2img(prompt: str, negative_prompt: str, seed: int,
                      pasos: int, cfg: float, ancho: int, alto: int,
                      client_id: str) -> dict:
    """Workflow txt2img completo via GraphBuilder."""
    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple",
                   ckpt_name="ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
    pos = g.node("CLIPTextEncode", text=prompt, clip=ckpt.out(1))
    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=ckpt.out(1))
    latent = g.node("EmptyLatentImage", width=ancho, height=alto, batch_size=1)
    sampler = g.node("KSampler",
                     seed=seed, steps=pasos, cfg=cfg,
                     sampler_name="euler", scheduler="simple",
                     denoise=1.0,
                     model=ckpt.out(0), positive=pos.out(0),
                     negative=neg.out(0), latent_image=latent.out(0))
    decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
    g.node("SaveImage",
           filename_prefix=f"comic{client_id[:4]}",
           images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


def _workflow_txt2img_ipadapter(prompt: str, negative_prompt: str, seed: int,
                                pasos: int, cfg: float, ancho: int, alto: int,
                                nombre_img_ref: str, client_id: str,
                                weight: float = 0.10) -> dict:
    """
    Workflow txt2img + IP-Adapter.
    Genera desde ruido puro (sin img2img) pero con IP-Adapter guiando la
    apariencia del personaje. El prompt controla fondo, pose y composicion.
    weight: fuerza del IP-Adapter (0.0-1.0).
    """
    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple",
                   ckpt_name="ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
    ref_loader = g.node("LoadImage", image=nombre_img_ref)
    unified = g.node("IPAdapterUnifiedLoader",
                     model=ckpt.out(0),
                     preset="STANDARD (medium strength)")
    ipa = g.node("IPAdapterAdvanced",
                 model=unified.out(0),
                 ipadapter=unified.out(1),
                 image=ref_loader.out(0),
                 weight=weight,
                 weight_type="linear",
                 combine_embeds="concat",
                 start_at=0.0,
                 end_at=1.0,
                 embeds_scaling="K+mean(V) w/ C penalty")
    pos = g.node("CLIPTextEncode", text=prompt, clip=ckpt.out(1))
    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=ckpt.out(1))
    latent = g.node("EmptyLatentImage", width=ancho, height=alto, batch_size=1)
    sampler = g.node("KSampler",
                     seed=seed, steps=pasos, cfg=cfg,
                     sampler_name="euler", scheduler="simple",
                     denoise=1.0,
                     model=ipa.out(0), positive=pos.out(0),
                     negative=neg.out(0), latent_image=latent.out(0))
    decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
    g.node("SaveImage",
           filename_prefix=f"comic{client_id[:4]}",
           images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


def _workflow_img2img(prompt: str, negative_prompt: str, seed: int,
                      pasos: int, cfg: float, denoise: float,
                      nombre_imagen: str, client_id: str) -> dict:
    """Workflow img2img completo via GraphBuilder."""
    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple",
                   ckpt_name="ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
    loader = g.node("LoadImage", image=nombre_imagen)
    pos = g.node("CLIPTextEncode", text=prompt, clip=ckpt.out(1))
    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=ckpt.out(1))
    encode = g.node("VAEEncode", pixels=loader.out(0), vae=ckpt.out(2))
    sampler = g.node("KSampler",
                     seed=seed, steps=pasos, cfg=cfg,
                     sampler_name="euler", scheduler="simple",
                     denoise=denoise,
                     model=ckpt.out(0), positive=pos.out(0),
                     negative=neg.out(0), latent_image=encode.out(0))
    decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
    g.node("SaveImage",
           filename_prefix=f"comic{client_id[:4]}",
           images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


def _workflow_img2img_ipadapter(prompt: str, negative_prompt: str, seed: int,
                                 pasos: int, cfg: float, denoise: float,
                                 nombre_img_base: str,
                                 nombre_img_ref: str,
                                 client_id: str) -> dict:
    """
    Workflow img2img + IP-Adapter para consistencia de personaje.

    Usa dos imagenes:
      - nombre_img_base: el recorte facial (img2img base, controla estructura)
      - nombre_img_ref:  la imagen completa del personaje (IP-Adapter, guia apariencia)
    """
    g = GraphBuilder()

    # 1. Cargar checkpoint
    ckpt = g.node("CheckpointLoaderSimple",
                   ckpt_name="ponyDiffusionV6XL_v6StartWithThisOne.safetensors")

    # 2. Cargar imagen base para img2img (face crop)
    loader = g.node("LoadImage", image=nombre_img_base)

    # 3. Cargar imagen de referencia para IP-Adapter (full body)
    ref_loader = g.node("LoadImage", image=nombre_img_ref)

    # 4. IP-Adapter Unified Loader (carga modelo + CLIP vision)
    #    Preset "STANDARD (medium strength)" para ip-adapter_sdxl_vit-h
    #    No usamos plus-face porque fuerza demasiado la cara y descuida el cuerpo
    unified = g.node("IPAdapterUnifiedLoader",
                     model=ckpt.out(0),
                     preset="STANDARD (medium strength)")

    # 5. IP-Adapter Advanced: minima guia, Pony controla casi todo
    #    weight bajo para que fondo y composicion vengan del prompt
    ipa = g.node("IPAdapterAdvanced",
                 model=unified.out(0),
                 ipadapter=unified.out(1),
                 image=ref_loader.out(0),
                 weight=0.10,
                 weight_type="linear",
                 combine_embeds="concat",
                 start_at=0.0,
                 end_at=1.0,
                 embeds_scaling="K+mean(V) w/ C penalty")

    # 6. CLIP encoders
    pos = g.node("CLIPTextEncode", text=prompt, clip=ckpt.out(1))
    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=ckpt.out(1))

    # 7. VAE Encode (face crop)
    encode = g.node("VAEEncode", pixels=loader.out(0), vae=ckpt.out(2))

    # 8. KSampler (modelo con IP-Adapter aplicado)
    sampler = g.node("KSampler",
                     seed=seed, steps=pasos, cfg=cfg,
                     sampler_name="euler", scheduler="simple",
                     denoise=denoise,
                     model=ipa.out(0),
                     positive=pos.out(0),
                     negative=neg.out(0),
                     latent_image=encode.out(0))

    # 9. Decode + Save
    decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
    g.node("SaveImage",
           filename_prefix=f"comic{client_id[:4]}",
           images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


# ── Workflows de dos pasadas (transformar + refinar) ─────────────────


def _workflow_img2img_twopass(prompt: str, negative_prompt: str, seed: int,
                               pasos: int, cfg: float, denoise: float,
                               nombre_imagen: str, client_id: str,
                               denoise2: float = 0.25) -> dict:
    """
    Workflow img2img de DOS pasadas encadenadas en el espacio latente.
    (Sin decodificar/re-codificar entre pasadas, preserva calidad.)

    - Pasada 1 (denoise alto): transforma escena, fondo, accion
    - Pasada 2 (denoise bajo): refina detalles, estabiliza la cara
    """
    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple",
                   ckpt_name="ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
    loader = g.node("LoadImage", image=nombre_imagen)
    pos = g.node("CLIPTextEncode", text=prompt, clip=ckpt.out(1))
    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=ckpt.out(1))

    # Primera pasada: transformar
    encode = g.node("VAEEncode", pixels=loader.out(0), vae=ckpt.out(2))
    s1 = g.node("KSampler",
                seed=seed, steps=pasos, cfg=cfg,
                sampler_name="euler", scheduler="simple",
                denoise=denoise,
                model=ckpt.out(0), positive=pos.out(0),
                negative=neg.out(0), latent_image=encode.out(0))

    # Segunda pasada: refinar (mismo seed, mismo prompt, denoise bajo)
    # Toma el LATENT de la primera pasada directamente
    s2 = g.node("KSampler",
                seed=seed, steps=pasos, cfg=cfg,
                sampler_name="euler", scheduler="simple",
                denoise=denoise2,
                model=ckpt.out(0), positive=pos.out(0),
                negative=neg.out(0), latent_image=s1.out(0))

    decoded = g.node("VAEDecode", samples=s2.out(0), vae=ckpt.out(2))
    g.node("SaveImage",
           filename_prefix=f"comic{client_id[:4]}",
           images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


def _workflow_img2img_ipadapter_twopass(prompt: str, negative_prompt: str,
                                         seed: int, pasos: int, cfg: float,
                                         denoise: float,
                                         nombre_img_base: str,
                                         nombre_img_ref: str,
                                         client_id: str,
                                         denoise2: float = 0.25) -> dict:
    """
    Workflow img2img + IP-Adapter con dos pasadas.
    La segunda pasada va contra el mismo modelo con IP-Adapter para
    refinar la cara del personaje.
    """
    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple",
                   ckpt_name="ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
    loader = g.node("LoadImage", image=nombre_img_base)
    ref_loader = g.node("LoadImage", image=nombre_img_ref)
    unified = g.node("IPAdapterUnifiedLoader",
                     model=ckpt.out(0),
                     preset="STANDARD (medium strength)")
    ipa = g.node("IPAdapterAdvanced",
                 model=unified.out(0), ipadapter=unified.out(1),
                 image=ref_loader.out(0),
                 weight=0.10, weight_type="linear",
                 combine_embeds="concat",
                 start_at=0.0, end_at=1.0,
                 embeds_scaling="K+mean(V) w/ C penalty")
    pos = g.node("CLIPTextEncode", text=prompt, clip=ckpt.out(1))
    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=ckpt.out(1))

    # Primera pasada
    encode = g.node("VAEEncode", pixels=loader.out(0), vae=ckpt.out(2))
    s1 = g.node("KSampler",
                seed=seed, steps=pasos, cfg=cfg,
                sampler_name="euler", scheduler="simple",
                denoise=denoise,
                model=ipa.out(0), positive=pos.out(0),
                negative=neg.out(0), latent_image=encode.out(0))

    # Segunda pasada (tambien con IP-Adapter para mantener la cara)
    s2 = g.node("KSampler",
                seed=seed, steps=pasos, cfg=cfg,
                sampler_name="euler", scheduler="simple",
                denoise=denoise2,
                model=ipa.out(0), positive=pos.out(0),
                negative=neg.out(0), latent_image=s1.out(0))

    decoded = g.node("VAEDecode", samples=s2.out(0), vae=ckpt.out(2))
    g.node("SaveImage",
           filename_prefix=f"comic{client_id[:4]}",
           images=decoded.out(0))

    return {"prompt": g.finalize(), "client_id": client_id}


# ── Helpers de API ───────────────────────────────────────────────────

def _post(api_url: str, endpoint: str, data: dict) -> dict:
    url = f"{api_url}{endpoint}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get(api_url: str, endpoint: str) -> dict:
    url = f"{api_url}{endpoint}"
    with urllib.request.urlopen(url) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _upload_image(api_url: str, ruta_local: Path) -> str:
    """Sube una imagen a ComfyUI y devuelve el nombre del archivo."""
    nombre = ruta_local.name
    print(f"    Subiendo {ruta_local}...")
    boundary = "----" + uuid.uuid4().hex
    body = BytesIO()

    def add_field(name, value, filename=None):
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"'.encode())
        if filename:
            body.write(f'; filename="{filename}"'.encode())
        body.write(b"\r\n\r\n")
        if isinstance(value, bytes):
            body.write(value)
        else:
            body.write(str(value).encode())
        body.write(b"\r\n")

    add_field("image", ruta_local.read_bytes(), filename=nombre)
    add_field("type", "input")
    add_field("overwrite", "true")
    body.write(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{api_url}/upload/image",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    nombre_real = result.get("name", nombre)
    print(f"    Upload OK -> {result}  (usando: {nombre_real})")
    return nombre_real


def _esperar_resultado(api_url: str, prompt_id: str, timeout: int = 600) -> Optional[dict]:
    """Espera a que ComfyUI termine y devuelve el history. None si error/timeout."""
    espera = 0
    while espera < timeout:
        time.sleep(2)
        espera += 2
        try:
            history = _get(api_url, f"/history/{prompt_id}")
            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})
                status_str = status.get("status_str", "success")
                print(f"    Completado (status={status_str}, espera={espera}s)")
                if status_str == "error":
                    msgs = entry.get("status", {}).get("messages", [])
                    for m in msgs:
                        print(f"      {m}")
                    return None
                return entry
        except urllib.error.HTTPError as e:
            if e.code != 404:
                print(f"    HTTP Error: {e.code}")
                return None
        except Exception as e:
            print(f"    Error: {e}")
            return None
        if espera % 10 == 0:
            print(f"    esperando... ({espera}s)")
    print(f"    Timeout tras {timeout}s")
    return None


def _obtener_imagen_salida(history_data: dict, prefijo_log: str = "") -> Optional[Path]:
    """Extrae la ruta de la imagen generada del history de ComfyUI."""
    outputs = history_data.get("outputs", {})
    print(f"{prefijo_log}Outputs en history: {list(outputs.keys())}")
    for node_id, node_data in outputs.items():
        imagenes = node_data.get("images", [])
        print(f"{prefijo_log}  Nodo {node_id}: {len(imagenes)} imagenes")
        for img in imagenes:
            print(f"{prefijo_log}    -> {img.get('filename')} (subfolder={img.get('subfolder','')})")
        if imagenes:
            nombre = imagenes[0]["filename"]
            ruta = COMFY_OUTPUT / nombre
            print(f"{prefijo_log}Usando: {ruta}")
            return ruta
    print(f"{prefijo_log}No se encontraron imagenes en: {list(outputs.keys())}")
    return None


# ── API publica ──────────────────────────────────────────────────────

def generar_txt2img(
    prompt: str,
    negative_prompt: str,
    seed: int,
    ancho: int = 1024,
    alto: int = 1024,
    pasos: int = 30,
    cfg: float = 7.5,
    api_url: str = COMFY_API,
    timeout: int = 600,
) -> Optional[Path]:
    """
    Genera una imagen desde cero (txt2img) via ComfyUI.
    Devuelve la ruta al archivo generado o None si falla.
    """
    client_id = str(uuid.uuid4())
    data = _workflow_txt2img(prompt, negative_prompt, seed, pasos, cfg,
                              ancho, alto, client_id)

    print(f"  Enviando txt2img a ComfyUI (seed={seed})...")
    try:
        result = _post(api_url, "/prompt", data)
    except Exception as e:
        print(f"  ERROR al enviar: {e}")
        return None

    prompt_id = result.get("prompt_id", "")
    if not prompt_id:
        print(f"  ERROR: no se recibio prompt_id. Respuesta: {result}")
        if "error" in result:
            print(f"  Detalles: {result['error']}")
        return None

    print(f"    prompt_id: {prompt_id}")
    history = _esperar_resultado(api_url, prompt_id, timeout)
    if not history:
        return None
    return _obtener_imagen_salida(history, prefijo_log="    ")


def generar_txt2img_ipadapter(
    prompt: str,
    negative_prompt: str,
    seed: int,
    imagen_ref: Path,
    ancho: int = 1024,
    alto: int = 1024,
    pasos: int = 30,
    cfg: float = 7.5,
    weight: float = 0.10,
    api_url: str = COMFY_API,
    timeout: int = 600,
) -> Optional[Path]:
    """
    Genera desde txt2img (ruido puro) + IP-Adapter para consistencia.
    El prompt controla fondo, pose y composicion completamente.
    """
    client_id = str(uuid.uuid4())

    print(f"  Subiendo referencia para IP-Adapter...")
    try:
        nombre_ref = _upload_image(api_url, imagen_ref)
    except Exception as e:
        print(f"  ERROR al subir referencia: {e}")
        return None

    data = _workflow_txt2img_ipadapter(
        prompt, negative_prompt, seed, pasos, cfg, ancho, alto,
        nombre_ref, client_id, weight,
    )
    print(f"  Enviando txt2img + IP-Adapter a ComfyUI (seed={seed})...")

    try:
        result = _post(api_url, "/prompt", data)
    except Exception as e:
        print(f"  ERROR al enviar: {e}")
        return None

    prompt_id = result.get("prompt_id", "")
    if not prompt_id:
        print(f"  ERROR: no se recibio prompt_id. Respuesta: {result}")
        if "error" in result:
            print(f"  Detalles: {result['error']}")
        return None

    print(f"    prompt_id: {prompt_id}")
    history = _esperar_resultado(api_url, prompt_id, timeout)
    if not history:
        return None
    return _obtener_imagen_salida(history, prefijo_log="    ")


def generar_lote_img_ipadapter(
    prompt: str,
    negative_prompt: str,
    seed: int,
    imagen_ref: Path,
    num_frames: int = 16,
    ancho: int = 1024,
    alto: int = 1024,
    pasos: int = 30,
    cfg: float = 7.5,
    weight: float = 0.10,
    api_url: str = COMFY_API,
    timeout: int = 600,
    progress=None,
) -> Optional[list[Path]]:
    """
    Genera N frames independientes usando txt2img + IP-Adapter
    con seeds secuenciales para crear una animacion.

    Cada frame se genera desde ruido puro con el mismo prompt y referencia.
    El IP-Adapter mantiene la consistencia del personaje entre frames.
    Seeds secuenciales aseguran variacion natural.

    Args:
        progress: callable(porcentaje, descripcion) opcional.
    """
    frames = []
    for i in range(num_frames):
        s = seed + i

        if progress:
            progress((i + 0.5) / num_frames, f"Frame {i+1}/{num_frames} generando...")

        print(f"  Frame {i+1}/{num_frames} (seed={s})...")
        ruta = generar_txt2img_ipadapter(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=s,
            imagen_ref=imagen_ref,
            ancho=ancho,
            alto=alto,
            pasos=pasos,
            cfg=cfg,
            weight=weight,
            api_url=api_url,
            timeout=timeout,
        )
        if ruta is None:
            print(f"  ERROR: frame {i+1} fallo (seed={s})")
            if i == 0:
                return None
            continue

        frames.append(ruta)

        if progress:
            progress((i + 1) / num_frames, f"Frame {i+1}/{num_frames} listo")

    if not frames:
        return None

    print(f"  Lote completado: {len(frames)}/{num_frames} frames")
    return frames


def generar_lote_img2img_encadenado(
    prompt: str,
    negative_prompt: str,
    seed: int,
    imagen_ref: Path,
    num_frames: int = 16,
    pasos: int = 30,
    cfg: float = 7.0,
    denoise: float = 0.35,
    weight: float = 0.10,
    api_url: str = COMFY_API,
    timeout: int = 600,
    ipadapter_ref: Optional[Path] = None,
    progresion: Optional[str] = None,
    progress=None,
) -> Optional[list[Path]]:
    """
    Genera animacion encadenando frames via img2img secuencial.

    Frame 1: parte de imagen_ref (img2img con denoise).
    Frame N: parte del frame N-1, creando deriva visual continua.

    progresion: texto multi-linea opcional. Cada linea describe el
      movimiento para ese frame y se concatena al prompt.
      Ej: "mirando al frente, quieta"
          "comenzando a girar la cabeza"
          "media vuelta, perfil izquierdo"
    """
    # Upload IP-Adapter reference once (reused for all frames)
    usar_ip = ipadapter_ref is not None
    nombre_ip_ref = None
    if usar_ip:
        print(f"  Subiendo referencia IP-Adapter...")
        nombre_ip_ref = _upload_image(api_url, ipadapter_ref)

    frames = []
    base_actual = imagen_ref  # changes each iteration

    # Parse progression lines (one per frame, optional)
    lineas_progresion = []
    if progresion and progresion.strip():
        lineas_progresion = [l.strip() for l in progresion.split('\n') if l.strip()]

    for i in range(num_frames):
        s = seed  # misma seed siempre

        if progress:
            progress((i + 0.5) / num_frames, f"Frame {i+1}/{num_frames}...")

        # Build per-frame prompt with progression
        frame_prompt = prompt
        if lineas_progresion and i < len(lineas_progresion):
            frame_prompt = f"{prompt}, {lineas_progresion[i]}"

        client_id = str(uuid.uuid4())

        # Upload current base image for this frame
        nombre_base = _upload_image(api_url, base_actual)

        # Build workflow with GraphBuilder
        g = GraphBuilder()

        ckpt = g.node("CheckpointLoaderSimple",
                       ckpt_name="ponyDiffusionV6XL_v6StartWithThisOne.safetensors")

        loader = g.node("LoadImage", image=nombre_base)

        if usar_ip:
            ref_loader = g.node("LoadImage", image=nombre_ip_ref)
            unified = g.node("IPAdapterUnifiedLoader",
                             model=ckpt.out(0),
                             preset="STANDARD (medium strength)")
            ipa = g.node("IPAdapterAdvanced",
                         model=unified.out(0),
                         ipadapter=unified.out(1),
                         image=ref_loader.out(0),
                         weight=weight,
                         weight_type="linear",
                         combine_embeds="concat",
                         start_at=0.0,
                         end_at=1.0,
                         embeds_scaling="K+mean(V) w/ C penalty")
            model = ipa.out(0)
        else:
            model = ckpt.out(0)

        pos = g.node("CLIPTextEncode", text=frame_prompt, clip=ckpt.out(1))
        neg = g.node("CLIPTextEncode", text=negative_prompt, clip=ckpt.out(1))
        encode = g.node("VAEEncode", pixels=loader.out(0), vae=ckpt.out(2))

        sampler = g.node("KSampler",
                         seed=s, steps=pasos, cfg=cfg,
                         sampler_name="euler", scheduler="simple",
                         denoise=denoise,
                         model=model,
                         positive=pos.out(0),
                         negative=neg.out(0),
                         latent_image=encode.out(0))

        decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
        g.node("SaveImage",
               filename_prefix=f"chain{client_id[:4]}",
               images=decoded.out(0))

        data = {"prompt": g.finalize(), "client_id": client_id}

        prog_info = f", prog: {lineas_progresion[i][:40]}" if lineas_progresion and i < len(lineas_progresion) else ""
        print(f"  Frame {i+1}/{num_frames} (seed={s}, denoise={denoise:.2f}{prog_info})...")
        try:
            result = _post(api_url, "/prompt", data)
        except Exception as e:
            print(f"  ERROR al enviar frame {i+1}: {e}")
            if i == 0:
                return None
            continue

        prompt_id = result.get("prompt_id", "")
        if not prompt_id:
            print(f"  ERROR: no se recibio prompt_id en frame {i+1}")
            if i == 0:
                return None
            continue

        history = _esperar_resultado(api_url, prompt_id, timeout)
        if not history:
            print(f"  ERROR: el frame {i+1} no se completo")
            if i == 0:
                return None
            continue

        ruta = _obtener_imagen_salida(history, prefijo_log="    ")
        if ruta is None:
            print(f"  ERROR: no se obtuvo imagen del frame {i+1}")
            if i == 0:
                return None
            continue

        frames.append(ruta)
        base_actual = ruta  # next frame chains from this one

        if progress:
            progress((i + 1) / num_frames, f"Frame {i+1}/{num_frames} listo")

    if not frames:
        return None

    print(f"  Animacion encadenada: {len(frames)}/{num_frames} frames")
    return frames


def generar_img2img(
    prompt: str,
    negative_prompt: str,
    seed: int,
    imagen_ref: Path,
    denoise: float = 0.60,
    pasos: int = 30,
    cfg: float = 7.5,
    api_url: str = COMFY_API,
    timeout: int = 600,
    ipadapter_ref: Optional[Path] = None,
    two_pass: bool = False,
    denoise2: float = 0.25,
) -> Optional[Path]:
    """
    Genera una imagen desde una referencia (img2img) via ComfyUI.

    Si se proporciona ipadapter_ref, usa IP-Adapter para mantener
    consistencia del personaje (la cara se guia por referencia adicional).

    Si two_pass=True, hace dos pasadas de KSampler encadenadas:
      - Primera: transforma escena (denoise)
      - Segunda: refina detalles/cara (denoise2)
    """
    client_id = str(uuid.uuid4())

    print(f"  Subiendo imagen base a ComfyUI...")
    try:
        nombre_base = _upload_image(api_url, imagen_ref)
    except Exception as e:
        print(f"  ERROR al subir imagen: {e}")
        return None

    usar_two_pass = two_pass and denoise2 > 0.0

    # Build workflow
    if ipadapter_ref:
        print(f"  Subiendo referencia para IP-Adapter...")
        try:
            nombre_ref = _upload_image(api_url, ipadapter_ref)
        except Exception as e:
            print(f"  ERROR al subir referencia IP-Adapter: {e}")
            return None

        if usar_two_pass:
            data = _workflow_img2img_ipadapter_twopass(
                prompt, negative_prompt, seed, pasos, cfg, denoise,
                nombre_base, nombre_ref, client_id, denoise2,
            )
        else:
            data = _workflow_img2img_ipadapter(
                prompt, negative_prompt, seed, pasos, cfg, denoise,
                nombre_base, nombre_ref, client_id,
            )
    else:
        if usar_two_pass:
            data = _workflow_img2img_twopass(
                prompt, negative_prompt, seed, pasos, cfg, denoise,
                nombre_base, client_id, denoise2,
            )
        else:
            data = _workflow_img2img(
                prompt, negative_prompt, seed, pasos, cfg, denoise,
                nombre_base, client_id,
            )

    print(f"  Enviando img2img a ComfyUI (seed={seed}, denoise={denoise})...")
    try:
        result = _post(api_url, "/prompt", data)
    except Exception as e:
        print(f"  ERROR al enviar: {e}")
        return None

    prompt_id = result.get("prompt_id", "")
    if not prompt_id:
        print(f"  ERROR: no se recibio prompt_id. Respuesta: {result}")
        if "error" in result:
            print(f"  Detalles: {result['error']}")
        return None

    print(f"    prompt_id: {prompt_id}")
    history = _esperar_resultado(api_url, prompt_id, timeout)
    if not history:
        return None
    return _obtener_imagen_salida(history, prefijo_log="    ")


def _obtener_imagenes_salida(history_data: dict, prefijo_log: str = "") -> list[Path]:
    """Extrae TODAS las imagenes generadas del history de ComfyUI."""
    outputs = history_data.get("outputs", {})
    rutas = []
    for node_id, node_data in outputs.items():
        imagenes = node_data.get("images", [])
        for img in imagenes:
            nombre = img.get("filename")
            ruta = COMFY_OUTPUT / nombre
            rutas.append(ruta)
    return rutas


def generar_img2animation(
    prompt: str,
    negative_prompt: str,
    seed: int,
    imagen_ref: Path,
    modo: str = "injection",
    ipadapter_ref: Optional[Path] = None,
    ipadapter_weight: float = 0.40,
    motion_model: str = "mm_sdxl_v10_beta.safetensors",
    denoise: float = 0.60,
    pasos: int = 25,
    cfg: float = 7.0,
    num_frames: int = 16,
    api_url: str = COMFY_API,
    timeout: int = 600,
) -> Optional[list[Path]]:
    """
    Genera una animacion desde una imagen usando AnimateDiff.

    Modos:
      "injection" (default) — Usa NoisedImageInjection para inyectar la
          imagen en el sampling + IP-Adapter + AnimateDiff. Cada frame
          arranca de ruido diferente (movimiento real) pero la imagen
          inyectada preserva la escena y el personaje.

      "txt2anim" — Arranca de ruido puro. AnimateDiff crea el movimiento
          e IP-Adapter guia la apariencia. La escena puede cambiar.

      "img2img" — VAE Encode + repetir latente. Conserva el fondo pero
          apenas hay movimiento. Requiere denoise < 1.0.

    Devuelve lista de rutas a los frames generados.
    """
    client_id = str(uuid.uuid4())

    # Subir imagen
    print(f"  Subiendo imagen a ComfyUI...")
    try:
        nombre_img = _upload_image(api_url, imagen_ref)
    except Exception as e:
        print(f"  ERROR al subir imagen: {e}")
        return None

    g = GraphBuilder()

    ckpt = g.node("CheckpointLoaderSimple",
                   ckpt_name="ponyDiffusionV6XL_v6StartWithThisOne.safetensors")

    # ── IP-Adapter ──────────────────────────────────────────────────
    usar_ip = (modo in ("txt2anim", "injection")) or (ipadapter_ref is not None)

    if usar_ip:
        nombre_ip = nombre_img
        ip_src_name = imagen_ref.name
        if ipadapter_ref is not None and ipadapter_ref != imagen_ref:
            try:
                nombre_ip = _upload_image(api_url, ipadapter_ref)
                ip_src_name = ipadapter_ref.name
            except Exception:
                print(f"  WARN: fallo subir ipadapter_ref, usando imagen_ref")

        print(f"  IP-Adapter activo (weight={ipadapter_weight}, ref={ip_src_name})")
        ref_loader = g.node("LoadImage", image=nombre_ip)
        unified = g.node("IPAdapterUnifiedLoader",
                         model=ckpt.out(0),
                         preset="STANDARD (medium strength)")
        ipa = g.node("IPAdapterAdvanced",
                     model=unified.out(0),
                     ipadapter=unified.out(1),
                     image=ref_loader.out(0),
                     weight=ipadapter_weight,
                     weight_type="linear",
                     combine_embeds="concat",
                     start_at=0.0,
                     end_at=1.0,
                     embeds_scaling="K+mean(V) w/ C penalty")
        model_para_anim = ipa.out(0)
    else:
        model_para_anim = ckpt.out(0)

    # ── Latente de entrada y config de sampling ─────────────────────
    if modo == "injection":
        # Injection: ruido diferente por frame + imagen inyectada en sampling
        loader = g.node("LoadImage", image=nombre_img)
        inj = g.node("ADE_NoisedImageInjection",
                      image=loader.out(0), vae=ckpt.out(2),
                      invert_mask=False, resize_image=True,
                      start_percent=0.0, guarantee_steps=1)
        sample_settings = g.node("ADE_AnimateDiffSamplingSettings",
                                  batch_offset=0, noise_type="default",
                                  seed_gen="comfy", seed_offset=0,
                                  image_inject=inj.out(0))
        latent = g.node("ADE_EmptyLatentImageLarge",
                         width=1024, height=1024,
                         batch_size=num_frames)
        denoise_real = 1.0
        print(f"  Modo injection: {num_frames} frames, imagen inyectada en sampling")

    elif modo == "txt2anim":
        # Ruido puro + IP-Adapter
        ancho = 1024
        alto = 1024
        latent = g.node("EmptyLatentImage",
                         width=ancho, height=alto,
                         batch_size=num_frames)
        sample_settings = None
        denoise_real = 1.0
        print(f"  Modo txt2anim: {num_frames} frames desde ruido puro ({ancho}x{alto})")

    else:
        # img2img: codificar imagen y repetir latente
        loader = g.node("LoadImage", image=nombre_img)
        encoded = g.node("VAEEncode", pixels=loader.out(0), vae=ckpt.out(2))
        latent = g.node("RepeatLatentBatch",
                         samples=encoded.out(0),
                         amount=num_frames)
        sample_settings = None
        denoise_real = denoise
        print(f"  Modo img2img: {num_frames} frames desde img ({denoise=})")

    # CLIP
    pos = g.node("CLIPTextEncode", text=prompt, clip=ckpt.out(1))
    neg = g.node("CLIPTextEncode", text=negative_prompt, clip=ckpt.out(1))

    # ── AnimateDiff ─────────────────────────────────────────────────
    motion = g.node("ADE_LoadAnimateDiffModel",
                     model_name=motion_model)
    m_models = g.node("ADE_ApplyAnimateDiffModelSimple",
                       motion_model=motion.out(0))

    ctx = g.node("ADE_StandardStaticContextOptions",
                  context_length=min(num_frames, 16),
                  context_overlap=min(num_frames // 4, 4))

    anim_kwargs = {
        "model": model_para_anim,
        "beta_schedule": "autoselect",
        "m_models": m_models.out(0),
        "context_options": ctx.out(0),
    }
    if sample_settings is not None:
        anim_kwargs["sample_settings"] = sample_settings.out(0)

    anim_model = g.node("ADE_UseEvolvedSampling", **anim_kwargs)

    # KSampler
    sampler = g.node("KSampler",
                     seed=seed, steps=pasos, cfg=cfg,
                     sampler_name="euler", scheduler="simple",
                     denoise=denoise_real,
                     model=anim_model.out(0),
                     positive=pos.out(0),
                     negative=neg.out(0),
                     latent_image=latent.out(0))

    decoded = g.node("VAEDecode", samples=sampler.out(0), vae=ckpt.out(2))
    g.node("SaveImage",
           filename_prefix=f"anim{client_id[:4]}",
           images=decoded.out(0))

    data = {"prompt": g.finalize(), "client_id": client_id}

    print(f"  Enviando animacion a ComfyUI (modo={modo}, seed={seed}, frames={num_frames})...")
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

    print(f"  Animacion generada: {len(frames)} frames ({modo=})")
    return frames
