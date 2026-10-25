import importlib.util
import json
import queue
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

_VIDEO_TEXT_TOKENS = ("TextToVideo", "AnimateDiff", "CogVideoX", "Mochi", "LTXPipeline",
                      "HunyuanVideo", "Wan", "Allegro", "OpenSora")
_VIDEO_NEEDS_IMAGE_TOKENS = ("ImageToVideo", "Image2Video", "I2VGen", "StableVideoDiffusion")
_IMAGE_NEEDS_INPUT_TOKENS = ("Inpaint", "Upscale", "InstructPix2Pix", "ControlNet", "Img2Img",
                             "Depth2Img", "Pix2Pix", "Variation")

MIN_CHECKPOINT_BYTES = 1_000_000_000
PIXEL_ART_ARCHITECTURE = "text-conditioned flow-matching UNet (rectified flow, velocity prediction)"

_NSFW_CHECKER_ID = "CompVis/stable-diffusion-safety-checker"
_NSFW_FEATURE_EXTRACTOR_ID = "openai/clip-vit-base-patch32"
_nsfw_checker_lock = threading.Lock()
_nsfw_checker = None
_nsfw_feature_extractor = None


def image_is_nsfw(image_data: bytes, model_path=None) -> bool:
    """Classify a generated image with the Stable Diffusion safety checker."""
    global _nsfw_checker, _nsfw_feature_extractor

    import io
    import numpy as np
    import torch
    from PIL import Image

    local_model = Path(model_path) if model_path else None
    local_checker = local_model / "safety_checker" if local_model and local_model.is_dir() else None
    local_processor = local_model / "feature_extractor" if local_model and local_model.is_dir() else None
    has_local_checker = (local_checker is not None
                         and (local_checker / "config.json").is_file()
                         and any((local_checker / name).is_file()
                                 for name in ("model.safetensors", "pytorch_model.bin")))
    has_local_processor = (local_processor is not None
                           and (local_processor / "preprocessor_config.json").is_file())

    with _nsfw_checker_lock:
        if _nsfw_checker is None or _nsfw_feature_extractor is None:
            from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
            from transformers import CLIPImageProcessor

            if has_local_checker and has_local_processor:
                checker = StableDiffusionSafetyChecker.from_pretrained(
                    str(local_checker), local_files_only=True
                ).to("cpu").eval()
                feature_extractor = CLIPImageProcessor.from_pretrained(
                    str(local_processor), local_files_only=True
                )
            else:
                checker = StableDiffusionSafetyChecker.from_pretrained(_NSFW_CHECKER_ID).to("cpu").eval()
                feature_extractor = CLIPImageProcessor.from_pretrained(_NSFW_FEATURE_EXTRACTOR_ID)
            _nsfw_checker = checker
            _nsfw_feature_extractor = feature_extractor

        with Image.open(io.BytesIO(image_data)) as source:
            image = source.convert("RGB")
        clip_input = _nsfw_feature_extractor(images=image, return_tensors="pt").pixel_values
        image_array = np.asarray(image, dtype=np.float32)[None, ...] / 255.0
        with torch.inference_mode():
            _, detected = _nsfw_checker(images=image_array, clip_input=clip_input)
        return bool(detected[0])


def is_pixelart_model(path) -> bool:
    d = Path(path)
    try:
        config = json.loads((d / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(config, dict)
            and config.get("architecture") == PIXEL_ART_ARCHITECTURE
            and isinstance(config.get("model_cfg"), dict)
            and (d / "model.safetensors").is_file()
            and (d / "pixelart" / "model.py").is_file()
            and (d / "pixelart" / "text.py").is_file())


def classify_pipeline_class(class_name: str):
    """-> (kind, usable, note). kind is 'image' or 'video'."""
    c = (class_name or "").casefold()
    if any(t.casefold() in c for t in _VIDEO_NEEDS_IMAGE_TOKENS):
        return "video", False, "Image-to-video model: it needs a starting picture, so it can't make a video from a text prompt alone."
    if any(t.casefold() in c for t in _VIDEO_TEXT_TOKENS):
        return "video", True, ""
    if any(t.casefold() in c for t in _IMAGE_NEEDS_INPUT_TOKENS):
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
    root = Path(models_root) if models_root else None
    found, seen = [], set()

    def add(entry):
        if entry["path"] not in seen:
            seen.add(entry["path"])
            found.append(entry)

    diffusers_roots = set()
    candidates = []
    pixelart_candidates = []
    if root and root.exists():
        for mi in root.rglob("model_index.json"):
            candidates.append(mi.parent)
        for config in root.rglob("config.json"):
            if is_pixelart_model(config.parent):
                pixelart_candidates.append(config.parent)
    for ep in list(extra_paths or []) + [p for p in (selected or {}).values() if p]:
        p = Path(ep)
        if p.is_dir() and (p / "model_index.json").exists():
            candidates.append(p)
        elif p.is_dir() and is_pixelart_model(p):
            pixelart_candidates.append(p)

    for d in pixelart_candidates:
        d = d.resolve()
        diffusers_roots.add(d)
        add({"kind": "image", "format": "custom", "path": str(d), "name": d.name,
             "size_gb": round(_folder_weight_bytes(d) / 1e9, 1), "usable": True,
             "pipeline": "PixelArtPipeline", "note": "Generates 32×32 RGBA pixel-art images."})
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

_SKIP_SUFFIXES = (".ckpt", ".onnx", ".onnx_data", ".msgpack", ".h5", ".ot", ".pb", ".tflite", ".pt", ".pth", ".gguf")
_WEIGHT_SUFFIXES = (".safetensors", ".bin")


def filter_repo_files(files):
    has_fp16 = any(f.endswith(".fp16.safetensors") for f in files)
    has_safetensors = any(f.endswith(".safetensors") for f in files)
    keep = []
    for f in files:
        low = f.lower()
        if low.endswith(_SKIP_SUFFIXES) or "openvino" in low:
            continue
        if "/" not in f and low.endswith(_WEIGHT_SUFFIXES):
            continue
        if low.endswith(_WEIGHT_SUFFIXES):
            if has_fp16 and ".fp16." not in low:
                continue
            if has_safetensors and low.endswith(".bin"):
                continue
        keep.append(f)
    return keep



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


def _load_pixelart_modules(model_dir):
    package_name = f"_terminal_ai_pixelart_{abs(hash(str(model_dir.resolve()))):x}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir / "pixelart")]
    sys.modules[package_name] = package
    loaded = {}
    module_names = []
    try:
        for name in ("model", "postprocess", "text"):
            module_name = f"{package_name}.{name}"
            module_names.append(module_name)
            spec = importlib.util.spec_from_file_location(module_name, model_dir / "pixelart" / f"{name}.py")
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load Pixel Art model component: {name}.")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
    finally:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        sys.modules.pop(package_name, None)
    return loaded


class _PixelArtPipeline:
    def __init__(self, model_dir, device):
        import torch
        from safetensors.torch import load_file

        components = _load_pixelart_modules(model_dir)
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        self.device = device
        self.torch = torch
        self.model = components["model"].UNet(**config["model_cfg"]).to(device).eval()
        self.model.load_state_dict(load_file(str(model_dir / "model.safetensors"), device=str(device)))
        self.encoder = components["text"].TextEncoder(str(device), config.get("text_encoder", "clip-b32"))
        self.sample = components["model"].sample
        self.caption_for_prompt = components["text"].caption_for_prompt
        self.to_rgba_u8 = components["postprocess"].to_rgba_u8
        self.defaults = config["sampler_defaults"]

    def __call__(self, prompt, num_inference_steps=None):
        torch = self.torch
        steps = num_inference_steps or self.defaults["steps"]
        caption = self.caption_for_prompt(prompt)
        ctx, mask, pooled = self.encoder.encode([caption])
        null_ctx, null_mask, null_pooled = self.encoder.encode([""])
        ctx, pooled = ctx.to(self.device).float(), pooled.to(self.device).float()
        null_ctx, null_pooled = null_ctx.to(self.device).float(), null_pooled.to(self.device).float()
        mask, null_mask = mask.to(self.device), null_mask.to(self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(torch.seed()))
        image = self.sample(
            self.model, ctx, mask, pooled,
            null_ctx, null_mask, null_pooled,
            steps=steps, cfg=self.defaults["cfg"], generator=generator,
            cfg_interval=tuple(self.defaults["cfg_interval"]),
        ).cpu()
        return SimpleNamespace(images=self.to_rgba_u8(image))


def _load_pixelart_pipeline(model_dir, device):
    return _PixelArtPipeline(model_dir, device)


def _upcast_stable_diffusion_vae(pipe, torch):
    vae = getattr(pipe, "vae", None)
    config = getattr(vae, "config", None)
    if (vae is None or not getattr(config, "force_upcast", False)
            or vae.dtype != torch.float16):
        return

    post_quant_conv = getattr(vae, "post_quant_conv", None)
    if post_quant_conv is None:
        raise RuntimeError("This Stable Diffusion VAE requires FP32 decoding but has no post-quantization layer.")

    vae.to(dtype=torch.float32)

    def cast_latents_to_vae_dtype(module, inputs):
        if inputs and torch.is_tensor(inputs[0]):
            return (inputs[0].to(dtype=module.weight.dtype), *inputs[1:])
        return inputs

    post_quant_conv.register_forward_pre_hook(cast_latents_to_vae_dtype)


def load_pipeline(path, kind, device=None):
    import torch
    p = Path(path)
    if device not in (None, "cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    cuda = torch.cuda.is_available() and device != "cpu"
    if device == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU placement was requested, but CUDA is not available.")
    dtype = torch.float16 if cuda else torch.float32

    if kind == "image" and is_pixelart_model(p):
        return _load_pixelart_pipeline(p, torch.device("cuda" if cuda else "cpu"))

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
        model_index = json.loads((p / "model_index.json").read_text(encoding="utf-8"))
        # Completely disable safety checker
        requires_safety_checker = False
        variant = "fp16" if any(p.rglob("*.fp16.safetensors")) else None
        loader = diffusers.AutoPipelineForText2Image if kind == "image" else diffusers.DiffusionPipeline
        load_kwargs = {
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        try:
            pipe = loader.from_pretrained(
                str(p), torch_dtype=dtype, variant=variant, local_files_only=True, **load_kwargs
            )
        except (ValueError, OSError):
            if variant:
                pipe = loader.from_pretrained(
                    str(p), torch_dtype=dtype, local_files_only=True, **load_kwargs
                )
            else:
                raise

        # Force disable after load
        if hasattr(pipe, "safety_checker"):
            pipe.safety_checker = None
        if hasattr(pipe, "requires_safety_checker"):
            pipe.requires_safety_checker = False

        # Monkey-patch so any remaining call can never black the image
        def _disabled_safety_checker(images, clip_input=None, **kwargs):
            # Always claim the image is safe
            if hasattr(images, "shape") and len(images.shape) == 4:
                return images, [False] * images.shape[0]
            return images, False

        # Patch both the attribute and the class method if present
        pipe.safety_checker = _disabled_safety_checker
        if hasattr(pipe, "feature_extractor"):
            # some pipelines still expect this; leave it alone
            pass

        size = _folder_weight_bytes(p)

    stable_diffusion_cls = getattr(diffusers, "StableDiffusionPipeline", None)
    if (cuda and dtype == torch.float16 and stable_diffusion_cls is not None
            and isinstance(pipe, stable_diffusion_cls)):
        _upcast_stable_diffusion_vae(pipe, torch)

    if cuda:
        free_vram, _ = torch.cuda.mem_get_info()
        estimated_bytes = size * 1.2 + 768 * 1024 * 1024
        if device == "gpu" and estimated_bytes > free_vram:
            raise RuntimeError(
                f"Not enough free GPU memory for this image model: approximately "
                f"{estimated_bytes / 1024**3:.1f} GB is needed and "
                f"{free_vram / 1024**3:.1f} GB is available."
            )
        if estimated_bytes > free_vram:
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


def run_with_progress(pipe, prompt, steps, stop_event=None):
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
