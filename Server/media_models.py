"""
media_models.py — image / video generation models (diffusers), kept separate
from the chat (LLM) model code so the two never get mixed up.

What lives here:
  * scan_media_models()  finds image/video models under models/ (and any
                         extra paths the admin added) and classifies them
  * sniff_safetensors()  reads ONLY the header of a single-file checkpoint
                         to tell SD1.5 / SDXL / FLUX / LoRA / VAE apart
  * filter_repo_files()  which files of a Hugging Face diffusers repo are
                         worth downloading (skips duplicate fp32/ckpt/onnx)
  * load_pipeline()      builds a ready-to-use pipeline on the right device
  * run_with_progress()  runs a generation in a worker thread and yields
                         step progress, so the UI can show a live bar

Nothing heavy (torch / diffusers) is imported at module import time, so the
scanning/classifying parts work — and are testable — without a GPU stack.
"""
import importlib.util
import json
import queue
import threading
from pathlib import Path

# ── classification of diffusers pipelines (from model_index.json) ──────────
_VIDEO_TEXT_TOKENS = ("TextToVideo", "AnimateDiff", "CogVideoX", "Mochi", "LTXPipeline",
                      "HunyuanVideo", "Wan", "Allegro", "OpenSora")
_VIDEO_NEEDS_IMAGE_TOKENS = ("ImageToVideo", "Image2Video", "I2VGen", "StableVideoDiffusion")
_IMAGE_NEEDS_INPUT_TOKENS = ("Inpaint", "Upscale", "InstructPix2Pix", "ControlNet", "Img2Img",
                             "Depth2Img", "Pix2Pix", "Variation")

MIN_CHECKPOINT_BYTES = 1_000_000_000   # smaller .safetensors are LoRAs/VAEs/embeddings


def classify_pipeline_class(class_name: str):
    """-> (kind, usable, note). kind is 'image' or 'video'."""
    c = class_name or ""
    if any(t in c for t in _VIDEO_NEEDS_IMAGE_TOKENS):
        return "video", False, "Image-to-video model: it needs a starting picture, so it can't make a video from a text prompt alone."
    if any(t in c for t in _VIDEO_TEXT_TOKENS):
        return "video", True, ""
    if any(t in c for t in _IMAGE_NEEDS_INPUT_TOKENS):
        return "image", False, "Refiner / inpaint / img2img / ControlNet model: it needs an input image, so it can't make a picture from a prompt alone."
    return "image", True, ""


def sniff_safetensors(path) -> str:
    """Reads only the JSON header of a .safetensors file and returns one of:
    'sd15', 'sd2', 'sdxl', 'flux', 'lora', 'vae', 'unknown'."""
    try:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if n <= 0 or n > 100_000_000:
                return "unknown"
            header = json.loads(f.read(n))
    except Exception:
        return "unknown"
    keys = [k for k in header.keys() if k != "__metadata__"]
    if not keys:
        return "unknown"
    if any(k.startswith("lora_") or ".lora_" in k or "lora_down" in k or "lora_A" in k for k in keys[:400]):
        return "lora"
    if any("double_blocks." in k for k in keys):
        return "flux"
    if any("conditioner.embedders.1" in k or "text_encoder_2" in k for k in keys):
        return "sdxl"
    if any(k.startswith("cond_stage_model.model.") for k in keys):
        return "sd2"
    if any(k.startswith("cond_stage_model.") or "model.diffusion_model.input_blocks" in k for k in keys):
        return "sd15"
    # diffusers-layout single files and bare VAEs
    if all(k.startswith(("encoder.", "decoder.", "quant_conv", "post_quant_conv")) for k in keys):
        return "vae"
    return "unknown"


_CHECKPOINT_LABEL = {
    "sd15": ("Stable Diffusion 1.x checkpoint", True, "Loads from this one file. The first load fetches a few small config files from Hugging Face (internet needed once)."),
    "sd2": ("Stable Diffusion 2.x checkpoint", True, "The first load fetches a few small config files from Hugging Face (internet needed once)."),
    "sdxl": ("Stable Diffusion XL checkpoint", True, "The first load fetches a few small config files from Hugging Face (internet needed once)."),
    "flux": ("FLUX checkpoint", True, "FLUX is very large — expect it to be slow or not fit on a 12 GB card."),
}


def _has_weights(d: Path) -> bool:
    return any(d.glob("*.safetensors")) or any(d.glob("*.bin"))


def _folder_weight_bytes(d: Path) -> int:
    """Approximate bytes that will actually be loaded from a diffusers folder
    (prefers the fp16 files when present, like load_pipeline does)."""
    files = [f for f in d.rglob("*") if f.is_file() and f.suffix in (".safetensors", ".bin")]
    fp16 = [f for f in files if ".fp16." in f.name]
    chosen = fp16 or [f for f in files if ".fp16." not in f.name] or files
    return sum(f.stat().st_size for f in chosen)


def _inspect_folder(d: Path):
    try:
        idx = json.loads((d / "model_index.json").read_text(encoding="utf-8", errors="ignore"))
    except Exception as e:
        return {"kind": "image", "usable": False, "pipeline": "?", "note": f"model_index.json couldn't be read ({e})."}
    cls = str(idx.get("_class_name", ""))
    kind, usable, note = classify_pipeline_class(cls)
    if usable and not _has_weights(d) and not any(d.rglob("*.safetensors")) and not any(d.rglob("*.bin")):
        usable, note = False, "The folder has no weight files — the download may be incomplete."
    return {"kind": kind, "usable": usable, "pipeline": cls or "?", "note": note}


def scan_media_models(models_root, extra_paths=(), selected=None):
    """Returns a list of dicts: {kind, format, path, name, pipeline, size_gb, usable, note}.
    models_root is searched recursively; extra_paths are admin-added files or
    folders anywhere on disk; selected is {'image': path|None, 'video': path|None}
    — a selected model outside models_root is still listed so it never vanishes."""
    root = Path(models_root) if models_root else None
    found, seen = [], set()

    def add(entry):
        if entry["path"] not in seen:
            seen.add(entry["path"])
            found.append(entry)

    diffusers_roots = set()
    candidates = []
    if root and root.exists():
        for mi in root.rglob("model_index.json"):
            candidates.append(mi.parent)
    for ep in list(extra_paths or []) + [p for p in (selected or {}).values() if p]:
        p = Path(ep)
        if p.is_dir() and (p / "model_index.json").exists():
            candidates.append(p)

    for d in candidates:
        d = d.resolve() if d.exists() else d
        if str(d) in seen:
            continue
        diffusers_roots.add(d)
        info = _inspect_folder(d)
        add({"format": "folder", "path": str(d), "name": d.name, "size_gb": round(_folder_weight_bytes(d) / 1e9, 1), **info})

    def inside_known_model(f: Path) -> bool:
        for parent in f.parents:
            if parent in diffusers_roots or (parent / "config.json").exists() or (parent / "model_index.json").exists():
                return True
            if root and parent == root:
                break
        return False

    files = []
    if root and root.exists():
        for ext in ("*.safetensors", "*.ckpt"):
            files.extend(root.rglob(ext))
    for ep in list(extra_paths or []) + [p for p in (selected or {}).values() if p]:
        p = Path(ep)
        if p.is_file():
            files.append(p)
    explicit = {str(Path(p)) for p in list(extra_paths or []) + [q for q in (selected or {}).values() if q] if Path(p).is_file()}

    for f in files:
        try:
            size = f.stat().st_size
        except OSError:
            continue
        is_explicit = str(f) in explicit
        if not is_explicit and (size < MIN_CHECKPOINT_BYTES or inside_known_model(f)):
            continue
        if f.suffix.lower() == ".ckpt":
            add({"kind": "image", "format": "checkpoint", "path": str(f), "name": f.name, "pipeline": ".ckpt",
                 "size_gb": round(size / 1e9, 1), "usable": False,
                 "note": ".ckpt files are Python pickles, which can run code when opened — not loaded. Convert it to .safetensors first."})
            continue
        family = sniff_safetensors(f)
        if family in ("lora", "vae") and not is_explicit:
            continue
        label, usable, note = _CHECKPOINT_LABEL.get(family, (None, False, None))
        if label is None:
            label = {"lora": "LoRA (add-on, not a full model)", "vae": "VAE (add-on, not a full model)"}.get(family, "Unrecognised checkpoint")
            note = {"lora": "A LoRA modifies a full model; it can't generate images on its own.",
                    "vae": "A VAE is a component of a full model; it can't generate images on its own."}.get(
                        family, "Couldn't tell what this is — it doesn't look like a Stable Diffusion 1.x/2.x/XL or FLUX checkpoint.")
        add({"kind": "image", "format": "checkpoint", "path": str(f), "name": f.name, "pipeline": label,
             "size_gb": round(size / 1e9, 1), "usable": usable, "note": note})

    found.sort(key=lambda e: (e["kind"], e["name"].lower()))
    return found


# ── Hugging Face repo download filtering ───────────────────────────────────
_SKIP_SUFFIXES = (".ckpt", ".onnx", ".onnx_data", ".msgpack", ".h5", ".ot", ".pb", ".tflite", ".pt", ".pth", ".gguf")
_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def filter_repo_files(files):
    """Which files of a diffusers repo to download. Skips: root-level
    single-file duplicates of the whole model, .ckpt/.onnx/etc, and the
    fp32 copies when an fp16 variant exists — which roughly halves the
    download without changing what loads."""
    has_fp16 = any(f.endswith(".fp16.safetensors") for f in files)
    has_safetensors = any(f.endswith(".safetensors") for f in files)
    keep = []
    for f in files:
        low = f.lower()
        if low.endswith(_SKIP_SUFFIXES) or "openvino" in low:
            continue
        if "/" not in f and low.endswith(_WEIGHT_SUFFIXES):
            continue            # e.g. v1-5-pruned.safetensors at the repo root
        if low.endswith(_WEIGHT_SUFFIXES):
            if has_fp16 and ".fp16." not in low:
                continue
            if has_safetensors and low.endswith(".bin"):
                continue
        keep.append(f)
    return keep


# ── status of the optional libraries ───────────────────────────────────────
def diffusers_status():
    installed = importlib.util.find_spec("diffusers") is not None
    version = None
    if installed:
        try:
            from importlib.metadata import version as _v
            version = _v("diffusers")
        except Exception:
            pass
    video_export = (importlib.util.find_spec("imageio") is not None or importlib.util.find_spec("cv2") is not None)
    return {"installed": installed, "version": version, "video_export": video_export}


INSTALL_HINT = "pip install diffusers imageio imageio-ffmpeg"


# ── loading ────────────────────────────────────────────────────────────────
def load_pipeline(path, kind):
    """Builds a diffusers pipeline from a folder or a single .safetensors
    checkpoint and puts it on the GPU (or CPU). Raises RuntimeError with a
    plain-English message on anything it can explain."""
    import torch
    p = Path(path)
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32

    if p.is_file():
        if p.suffix.lower() != ".safetensors":
            raise RuntimeError("Only .safetensors checkpoints can be loaded (.ckpt files can run code when opened).")
        family = sniff_safetensors(p)
        import diffusers
        if family == "sdxl":
            cls = diffusers.StableDiffusionXLPipeline
        elif family in ("sd15", "sd2"):
            cls = diffusers.StableDiffusionPipeline
        elif family == "flux":
            cls = diffusers.FluxPipeline
            dtype = torch.bfloat16 if cuda else torch.float32
        else:
            raise RuntimeError("This file isn't a Stable Diffusion 1.x/2.x/XL or FLUX checkpoint, so I don't know how to run it.")
        try:
            pipe = cls.from_single_file(str(p), torch_dtype=dtype, use_safetensors=True)
        except Exception as e:
            msg = str(e)
            if any(w in msg.lower() for w in ("connection", "offline", "resolve", "couldn't connect", "max retries")):
                raise RuntimeError("Loading a single-file checkpoint needs a few small config files from Hugging Face the first "
                                   "time, and the download failed. Connect to the internet once and try again.") from e
            raise
        size = p.stat().st_size
    else:
        if not (p / "model_index.json").exists():
            raise RuntimeError("That folder has no model_index.json, so it isn't a diffusers model.")
        import diffusers
        variant = "fp16" if any(p.rglob("*.fp16.safetensors")) else None
        loader = diffusers.AutoPipelineForText2Image if kind == "image" else diffusers.DiffusionPipeline
        try:
            pipe = loader.from_pretrained(str(p), torch_dtype=dtype, variant=variant, local_files_only=True)
        except (ValueError, OSError):
            if variant:
                pipe = loader.from_pretrained(str(p), torch_dtype=dtype, local_files_only=True)
            else:
                raise
        size = _folder_weight_bytes(p)

    if cuda:
        vram = torch.cuda.get_device_properties(0).total_memory
        if size > vram * 0.7:
            pipe.enable_model_cpu_offload()      # streams parts to the GPU as needed (needs `accelerate`)
        else:
            pipe.to("cuda")
        for fn in ("enable_vae_slicing", "enable_vae_tiling"):
            try:
                getattr(pipe, fn)()
            except Exception:
                pass
    else:
        pipe.to("cpu")
    return pipe


# ── running, with live progress ────────────────────────────────────────────
def run_with_progress(pipe, prompt, steps, stop_event=None):
    """Runs pipe(prompt) in a worker thread. Yields ('progress', step, total)
    as denoising steps complete (if the pipeline supports step callbacks),
    then ('result', pipeline_output). If stop_event is set mid-run the
    pipeline is interrupted and the caller should discard the result."""
    import inspect
    q = queue.Queue()
    box = {}

    def cb(pipeline, step, timestep, kwargs):
        q.put(("progress", step + 1))
        if stop_event is not None and stop_event.is_set():
            try:
                pipeline._interrupt = True
            except Exception:
                pass
        return kwargs

    def worker():
        try:
            try:
                pipe._interrupt = False
            except Exception:
                pass
            kw = {"prompt": prompt, "num_inference_steps": steps}
            try:
                if "callback_on_step_end" in inspect.signature(pipe.__call__).parameters:
                    kw["callback_on_step_end"] = cb
            except (TypeError, ValueError):
                pass
            box["out"] = pipe(**kw)
        except BaseException as e:          # surfaced to the caller below
            box["err"] = e
        finally:
            q.put(("end", None))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        kind, val = q.get()
        if kind == "progress":
            yield ("progress", val, steps)
        else:
            break
    t.join()
    if "err" in box:
        raise box["err"]
    yield ("result", box["out"])


def output_to_png_bytes(out) -> bytes:
    import io
    image = out.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def output_to_mp4_bytes(out, tmp_dir) -> bytes:
    import uuid
    from diffusers.utils import export_to_video
    frames = out.frames[0]
    tmp = Path(tmp_dir) / f"_gen_video_{uuid.uuid4().hex[:8]}.mp4"
    try:
        export_to_video(frames, str(tmp))
        return tmp.read_bytes()
    except ImportError as e:
        raise RuntimeError(f"Saving a video needs extra packages. Run: {INSTALL_HINT}") from e
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
