"""
web_server.py — Terminal AI V14.0: multi-user web GUI.

Run via the desktop Control Panel (control_panel.py), or directly:
    python web_server.py

ACCOUNTS
  Two roles: "standard" and "administrator". A default administrator
  account (username "Admin", password "Admin") is created automatically
  the first time no administrator account exists — CHANGE THIS
  PASSWORD from the chat page settings panel immediately, it's public
  knowledge the moment this code is public. Self-signup (if enabled)
  always creates "standard" accounts; only an existing admin can
  promote someone to administrator, from /admin.
  Administrators see a link to switch between the Chat view and the
  Admin/Control Panel view; standard accounts only see chat.
  Guests ("Continue as Guest") stay fully anonymous; chat history is
  temporary unless the guest opts into browser-local saving.

FILES
  Signed-in accounts (not guests) can attach an image or text file to a
  message. It's saved to that account's own folder AND copied into a
  separate training-data folder for later use. Note: the chat model is
  text-only — it can't actually "see" an attached image, only store and
  reference it. Text-file contents ARE fed into the model's context for
  that message.
"""

import argparse, filecmp, io, ipaddress, json, os, re, shutil, socket, subprocess, sys, time, uuid, heapq, itertools, threading, logging, secrets, zipfile
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

# Force stdout/stderr to real UTF-8 regardless of how this process was
# launched. On Windows, Python's console output otherwise defaults to
# the system's ANSI codepage (often cp1252), which can't represent
# characters used throughout this file's log messages (em dashes,
# arrows, etc). Whoever's reading this process's output — the Control
# Panel, a piped log file, a plain terminal — decodes what it gets as
# UTF-8; if this process wrote cp1252 bytes instead, those characters
# come out as the "�" replacement character rather than the real thing.
# reconfigure() exists on Python 3.7+; guarded in case stdout has been
# replaced with something that doesn't support it (e.g. some IDEs).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    requests = None
    REQUESTS_OK = False
from pathlib import Path
from functools import wraps

# Model downloads print progress with carriage returns ('\r'), which
# looks completely frozen through a piped subprocess log that only
# flushes on real lines — set before transformers/huggingface_hub import
# below so a first-time download doesn't look like a hang.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

SERVER_DIR = Path(__file__).resolve().parent
ROOT = SERVER_DIR.parent if SERVER_DIR.name == "server" else SERVER_DIR
CONFIG_FILE = ROOT / "config.json"
LOG_DIR = ROOT / "logs"
POLICY_FILE = ROOT / "user_policies.json"
USERS_FILE = ROOT / "users.json"
CHAT_HISTORY_DIR = ROOT / "chat_history"
ACCOUNTS_DIR = ROOT / "accounts"
USER_DATA_DIR = ROOT / "user data"
TRAINING_UPLOAD_DIR = ROOT / "web_training_data" / "user_uploads"
CUSTOM_MODELS_FILE = ROOT / "custom_models.json"
for d in (LOG_DIR, CHAT_HISTORY_DIR, ACCOUNTS_DIR, USER_DATA_DIR, TRAINING_UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(SERVER_DIR))

def _load_json(path, default):
    try:
        return json.loads(path.read_text()) if path.exists() else default
    except Exception:
        return default

_cfg = _load_json(CONFIG_FILE, {})

_NEEDS_PRIORITY_MIGRATION = False
if "allow_self_registration" not in _cfg and "guest_priority" in _cfg:
    old_guest = _cfg.get("guest_priority")
    old_default = _cfg.get("default_priority", 5)
    if old_guest is not None and old_guest <= old_default:
        _cfg["guest_priority"] = 11 - old_guest if 1 <= old_guest <= 10 else 10
        _cfg["allow_self_registration"] = True
        _NEEDS_PRIORITY_MIGRATION = True
        try:
            CONFIG_FILE.write_text(json.dumps(_cfg, indent=2))
            print(f"[migration] Priority scale changed (1=highest now, was 1=lowest). "
                  f"Adjusted guest_priority {old_guest} -> {_cfg['guest_priority']} in config.json.")
        except Exception:
            pass

parser = argparse.ArgumentParser()
parser.add_argument("--host",                     type=str, default=_cfg.get("host", "0.0.0.0"))
parser.add_argument("--port",                     type=int, default=_cfg.get("port", 8080))
parser.add_argument("--model",                    type=str, default=_cfg.get("model") or None)
parser.add_argument("--default-max-tokens",       type=int, default=_cfg.get("default_max_tokens", 1024))
parser.add_argument("--hard-max-tokens",          type=int, default=_cfg.get("hard_max_tokens", 8192))
parser.add_argument("--default-priority",         type=int, default=_cfg.get("default_priority", 5))
parser.add_argument("--default-tokens-per-min",   type=int, default=_cfg.get("default_tokens_per_min", 6000))
parser.add_argument("--guest-priority",           type=int, default=_cfg.get("guest_priority", 10))
parser.add_argument("--guest-tokens-per-min",     type=int, default=_cfg.get("guest_tokens_per_min", 1500))
parser.add_argument("--allow-self-registration",  type=int, default=1 if _cfg.get("allow_self_registration", True) else 0)
parser.add_argument("--auto-discover-peers",      type=int, default=1 if _cfg.get("auto_discover_peers", True) else 0)
parser.add_argument("--share-models-with-peers",  type=int, default=1 if _cfg.get("share_models_with_peers", True) else 0)
parser.add_argument("--machine-role",             type=str, default=_cfg.get("machine_role", "master"))
args = parser.parse_args()
ALLOW_SELF_REGISTRATION = bool(args.allow_self_registration)
AUTO_DISCOVER_PEERS = bool(args.auto_discover_peers)
SHARE_MODELS_WITH_PEERS = bool(args.share_models_with_peers)
MACHINE_ROLE = args.machine_role if args.machine_role in ("master", "slave") else "master"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_DIR / "web_server.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("web")
# huggingface_hub/httpx log every HEAD/GET at INFO level by default — that's
# the wall of "HTTP Request: ..." lines cluttering the log during model
# downloads, drowning out our own status messages. Quiet them to warnings.
for _noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

try:
    from flask import Flask, request, jsonify, session, render_template, Response, redirect, url_for, send_file
    from werkzeug.security import generate_password_hash, check_password_hash
except ImportError:
    logger.error("Flask not installed. Run BUILD.bat (or: pip install flask)")
    sys.exit(1)

try:
    from model_registry import DEFAULT_MODEL, MODELS, slug as model_slug
except Exception:
    DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
    MODELS = [DEFAULT_MODEL]
    def model_slug(model_id): return model_id.replace("/", "__")

try:
    import hardware_scan
except Exception:
    hardware_scan = None

try:
    from download_models import already_downloaded, download_one, MODELS_ROOT as REGISTRY_MODELS_ROOT
except Exception:
    already_downloaded = None
    download_one = None
    REGISTRY_MODELS_ROOT = ROOT / "models"

MODEL_CACHE = ROOT / "model_cache"
MODEL_CACHE.mkdir(parents=True, exist_ok=True)
BASE_MODEL = args.model or DEFAULT_MODEL

app = Flask(__name__, template_folder=str(SERVER_DIR / "templates"), static_folder=str(SERVER_DIR / "static"))

def _load_session_secret() -> str:
    configured_secret = os.environ.get("TERMINAL_AI_SECRET")
    if configured_secret:
        return configured_secret

    secret_file = ROOT / "session_secret.key"
    try:
        with secret_file.open("x", encoding="utf-8") as secret_handle:
            secret = secrets.token_hex(32)
            secret_handle.write(secret)
            return secret
    except FileExistsError:
        try:
            secret = secret_file.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise RuntimeError(f"Could not read the session signing key at {secret_file}: {error}") from error
        if not secret:
            raise RuntimeError(f"The session signing key at {secret_file} is empty.")
        return secret
    except OSError as error:
        raise RuntimeError(f"Could not create the session signing key at {secret_file}: {error}") from error

app.secret_key = _load_session_secret()

# ══════════════════════════════════════════════════════════════════════
#  PEERS — other Terminal AI instances on the LAN this one can either
#  hand generation requests off to (spreads load across machines) or
#  fetch already-downloaded models from (saves re-downloading over a
#  slow internet connection when a LAN neighbor already has it).
#
#  Every instance runs the exact same code and can act as BOTH the
#  "primary" (the one a browser talks to) and a worker for someone
#  else's primary, at the same time — there's no separate role to
#  configure. Peers only accept authenticated requests via a per-machine
#  worker key (like train_worker.py's --api-key, same trust model:
#  LAN-only, share the key out of band, never expose this port to the
#  open internet).
# ══════════════════════════════════════════════════════════════════════

PEERS_FILE = ROOT / "peers.json"
WORKER_KEY_FILE = ROOT / "worker_key.txt"
peers_lock = threading.RLock()

def load_peers() -> list:
    return _load_json(PEERS_FILE, [])

def save_peers(peers: list):
    with peers_lock:
        try:
            PEERS_FILE.write_text(json.dumps(peers, indent=2))
        except Exception as e:
            logger.warning(f"Could not save peers: {e}")

PEERS = load_peers()  # [{"name", "url", "api_key", "discovered"}]

if WORKER_KEY_FILE.exists():
    WORKER_API_KEY = WORKER_KEY_FILE.read_text().strip() or secrets.token_hex(16)
else:
    WORKER_API_KEY = secrets.token_hex(16)
    try:
        WORKER_KEY_FILE.write_text(WORKER_API_KEY)
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════════

def detect_hardware():
    result = {"device": "cpu", "device_name": "CPU", "npu": False, "cuda": False, "directml": False,
              "cuda_warning": None}
    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core
        core = Core()
        if "NPU" in core.available_devices:
            result.update(device="npu", device_name="Intel NPU (OpenVINO)", npu=True)
            return result
    except Exception:
        pass
    try:
        import torch, warnings, subprocess as _sp
        # See hardware_scan.py's longer explanation of why this warning
        # capture matters: an unsupported-for-this-GPU CUDA build (e.g.
        # RTX 50-series needing cu128+) shows up as a UserWarning here,
        # not an exception, and is easy to miss if you only check
        # torch.cuda.is_available().
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cuda_ok = torch.cuda.is_available()
            for w in caught:
                if "not compatible" in str(w.message) or "CUDA capability" in str(w.message):
                    result["cuda_warning"] = str(w.message)
        if cuda_ok:
            result.update(device="cuda", device_name=f"CUDA: {torch.cuda.get_device_name(0)}", cuda=True)
            return result
        elif not result["cuda_warning"]:
            try:
                _sp.run(["nvidia-smi"], capture_output=True, timeout=3, check=True)
                result["cuda_warning"] = ("nvidia-smi found an NVIDIA GPU, but this is a CPU-only PyTorch "
                                           "build. Reinstall with: pip install torch torchvision torchaudio "
                                           "--index-url https://download.pytorch.org/whl/cu128")
            except Exception:
                pass
    except Exception:
        pass
    try:
        import torch_directml
        result.update(device="directml", device_name="GPU (DirectML)", directml=True)
        return result
    except Exception:
        pass
    return result

HARDWARE = detect_hardware()
logger.info(f"Hardware: {HARDWARE['device_name']}")
if HARDWARE.get("cuda_warning"):
    logger.warning(f"GPU acceleration unavailable: {HARDWARE['cuda_warning']}")

try:
    from transformers import (AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer,
                               StoppingCriteria, StoppingCriteriaList)
    import torch
except ImportError as e:
    logger.error(f"Missing package: {e}. Run BUILD.bat (or: pip install transformers torch accelerate peft)")
    sys.exit(1)

model_lock = threading.RLock()
tokenizer = None
model = None
TORCH_DEVICE = "cpu"
ACTIVE_MODEL_PATH = None
LAST_MODEL_PATH = None
MODEL_INSTANCES = {}
model_instances_lock = threading.RLock()
model_instance_load_lock = threading.Lock()

def _is_local_model_dir(path) -> bool:
    p = Path(path)
    return p.exists() and (p / "config.json").exists()

def resolve_model_path():
    if (MODEL_CACHE / "config.json").exists():
        return str(MODEL_CACHE)
    # If this model id has already been physically downloaded (via the
    # Downloads tab / download_models.py), always prefer that local copy
    # over the bare hub id — otherwise transformers pings the HF Hub for
    # an ETag/metadata check on every single load, even when nothing
    # would change, which is slow and needs a network connection for no
    # reason once the weights are already sitting on disk.
    local_copy = REGISTRY_MODELS_ROOT / model_slug(BASE_MODEL)
    if _is_local_model_dir(local_copy):
        return str(local_copy)
    return BASE_MODEL

def _model_weight_bytes(path: str) -> int:
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    try:
        files = [f for f in p.rglob("*") if f.is_file()
                 and f.suffix.lower() in (".safetensors", ".bin", ".pt", ".pth")]
        fp16 = [f for f in files if ".fp16." in f.name.lower()]
        safe = [f for f in files if f.suffix.lower() == ".safetensors"]
        chosen = fp16 or safe or [f for f in files if f.suffix.lower() in (".bin", ".pt", ".pth")] or files
        return sum(f.stat().st_size for f in chosen)
    except OSError:
        return 0

def _check_gpu_memory(path: str):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU placement was requested, but CUDA is not available.")
    model_bytes = _model_weight_bytes(path)
    if not model_bytes:
        return
    free_bytes, _ = torch.cuda.mem_get_info()
    estimated_bytes = int(model_bytes * 1.2) + 768 * 1024 * 1024
    if estimated_bytes > free_bytes:
        raise RuntimeError(
            f"Not enough free GPU memory for this model: approximately "
            f"{estimated_bytes / 1024**3:.1f} GB is needed and "
            f"{free_bytes / 1024**3:.1f} GB is available."
        )

def _check_cpu_memory(path: str, multiplier: float = 1.2, reserve_gb: float = 0.75):
    model_bytes = _model_weight_bytes(path)
    if not model_bytes:
        return
    import psutil
    available_bytes = psutil.virtual_memory().available
    estimated_bytes = int(model_bytes * multiplier) + int(reserve_gb * 1024**3)
    if estimated_bytes > available_bytes:
        raise RuntimeError(
            f"Not enough free system memory for this model: approximately "
            f"{estimated_bytes / 1024**3:.1f} GB is needed and "
            f"{available_bytes / 1024**3:.1f} GB is available."
        )

def _load_text_model(path: str, requested_device: str = None):
    logger.info(f"Loading model: {path}")
    device = requested_device or ("gpu" if HARDWARE["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    if device == "gpu":
        _check_gpu_memory(path)
        if requested_device == "gpu":
            _check_cpu_memory(path)
    elif requested_device == "cpu":
        _check_cpu_memory(path)
    torch_device = "cuda" if device == "gpu" else "cpu"
    # Only reach out to the Hub if `path` isn't already a real local model
    # folder (bare hub id that hasn't been downloaded yet, e.g. very first
    # run). Once it's local, local_files_only=True skips the network
    # entirely instead of doing an unauthenticated metadata check first.
    offline = _is_local_model_dir(path)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=offline)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # "auto" reads the checkpoint's own native dtype from its config
    # (bf16 for most modern models, incl. Gemma) instead of hardcoding
    # float32 on CPU — hardcoding float32 was silently DOUBLING memory
    # use for every bf16/fp16-native model (a 21GB-on-disk bf16 checkpoint
    # was ballooning to ~48GB in RAM for no benefit).
    mod = AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto", low_cpu_mem_usage=True,
                                                local_files_only=offline)

    adapter = Path(path) / "_lora_adapter"
    if adapter.exists() and (adapter / "adapter_config.json").exists():
        try:
            from peft import PeftModel
            mod = PeftModel.from_pretrained(mod, str(adapter))
            mod = mod.merge_and_unload()
            logger.info("LoRA adapter merged.")
        except Exception as e:
            logger.warning(f"Adapter merge failed: {e}")

    if torch_device != "cpu":
        mod = mod.to(torch_device)
    mod.eval()
    return tok, mod, torch_device

def load_model(path: str):
    global tokenizer, model, TORCH_DEVICE, ACTIVE_MODEL_PATH, LAST_MODEL_PATH
    model_instance_load_lock.acquire()
    try:
        tok, mod, device = _load_text_model(path)
    finally:
        model_instance_load_lock.release()

    with model_lock:
        old_tokenizer, old_model = tokenizer, model
        tokenizer = tok
        model = mod
        TORCH_DEVICE = device
        ACTIVE_MODEL_PATH = path
        LAST_MODEL_PATH = path
    with model_instances_lock:
        default = MODEL_INSTANCES.get("default")
        if default is not None:
            default.update(path=path, name=Path(path).name or path, device=device,
                           tokenizer=tok, model=mod, status="running")
        else:
            MODEL_INSTANCES["default"] = {
                "id": "default", "kind": "text", "name": Path(path).name or path,
                "path": path, "device": device, "tokenizer": tok, "model": mod,
                "pipeline": None, "generation_lock": threading.Lock(),
                "status": "running", "managed": False, "active_requests": 0,
                "idle_since": time.time(), "pool_managed": False,
            }
    # Simply reassigning the globals above doesn't reliably free the OLD
    # model from memory: nn.Module graphs commonly contain reference
    # cycles (submodules/parents, forward hooks), so plain refcounting
    # often won't collect them right away — they'd just sit in RAM until
    # some later, unpredictable GC pass, which in a long-running server
    # could be "never, until restart". Explicitly drop the references and
    # force a collection pass, and release any CUDA-cached memory too.
    if old_model is not None:
        del old_model
    if old_tokenizer is not None:
        del old_tokenizer
    import gc
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info(f"Model ready on {device.upper()}: {path}")

def _peer_headers(peer: dict) -> dict:
    return {"X-Worker-Key": peer.get("api_key", "")}

def _try_sync_model_from_peers(model_id: str, timeout: float = 3.0) -> str:
    """If model_id isn't downloaded locally yet, check any PAIRED peers
    (ones an admin has actually entered an api_key for — discovered-but-
    unpaired peers are ignored here, same trust boundary as everywhere
    else) for whether they already have it, and pull it over the LAN
    instead of the internet if so. Returns the local path if synced (or
    already present), otherwise ''."""
    if already_downloaded and already_downloaded(model_id):
        return str(REGISTRY_MODELS_ROOT / model_slug(model_id))
    if not REQUESTS_OK:
        return ""
    paired = [p for p in PEERS if p.get("api_key")]
    for peer in paired:
        try:
            r = requests.get(f"{peer['url']}/internal/model_status/{model_id}",
                              headers=_peer_headers(peer), timeout=timeout)
            if not r.ok or not r.json().get("available"):
                continue
            logger.info(f"Model '{model_id}' is already on peer '{peer.get('name', peer['url'])}' — "
                        f"fetching over the LAN instead of the internet...")
            rz = requests.get(f"{peer['url']}/internal/model_zip/{model_id}",
                               headers=_peer_headers(peer), timeout=max(timeout, 600))
            if not rz.ok:
                logger.warning(f"Peer had the model but the transfer failed (HTTP {rz.status_code}).")
                continue
            target = REGISTRY_MODELS_ROOT / model_slug(model_id)
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(rz.content)) as zf:
                zf.extractall(target)
            logger.info(f"Synced '{model_id}' from peer '{peer.get('name', peer['url'])}' -> {target}")
            return str(target)
        except Exception as e:
            logger.warning(f"Peer sync attempt with '{peer.get('name', peer.get('url'))}' failed: {e}")
            continue
    return ""

logger.info("=" * 60)
logger.info("Loading model — the web page will NOT be reachable until this finishes.")
logger.info("If this model isn't cached locally yet, it's downloading now (can take")
logger.info("several minutes on first run depending on model size and connection).")
logger.info("=" * 60)

def _download_heartbeat(stop_event):
    waited = 0
    while not stop_event.wait(20):
        waited += 20
        logger.info(f"...still loading/downloading the model ({waited}s elapsed, this is normal for a first-time download)")

_hb_stop = threading.Event()
_hb_thread = threading.Thread(target=_download_heartbeat, args=(_hb_stop,), daemon=True)
_hb_thread.start()

_target_path = resolve_model_path()
if _target_path == BASE_MODEL and not Path(_target_path).exists():
    # This looks like a bare HF model id (not a real local path already,
    # e.g. an admin-activated local folder) — worth a quick check of
    # paired peers before hitting the internet for it.
    _synced = _try_sync_model_from_peers(BASE_MODEL)
    if _synced:
        _target_path = _synced
load_model(_target_path)
_hb_stop.set()
logger.info("WEB_SERVER_READY")

with model_instances_lock:
    MODEL_INSTANCES["default"] = {
        "id": "default", "kind": "text", "name": Path(ACTIVE_MODEL_PATH).name or ACTIVE_MODEL_PATH,
        "path": ACTIVE_MODEL_PATH, "device": TORCH_DEVICE, "tokenizer": tokenizer,
        "model": model, "pipeline": None, "generation_lock": threading.Lock(),
        "status": "running", "managed": False, "active_requests": 0,
        "idle_since": time.time(), "pool_managed": False,
    }

def _instance_summary(instance):
    return {key: instance[key] for key in ("id", "kind", "name", "path", "device", "status", "managed")}

def _start_model_instance(kind: str, path: str, device: str) -> dict:
    if kind not in ("text", "image", "video"):
        raise ValueError("Model type must be text, image, or video.")
    if device not in ("cpu", "gpu", "auto"):
        raise ValueError("Choose CPU, GPU, or automatic placement for this instance.")

    if device == "gpu" and (not HARDWARE.get("cuda") or not torch.cuda.is_available()):
        raise ValueError("GPU placement is unavailable because CUDA was not detected.")

    if kind == "text":
        available = {entry["path"]: entry for entry in _scan_chat_models()}
        if path not in (ACTIVE_MODEL_PATH, LAST_MODEL_PATH) and path not in available:
            raise ValueError("That text model is not installed or available.")
        name = available.get(path, {}).get("name") or Path(path).name or path
    else:
        available = {entry["path"]: entry for entry in _scan_media()
                     if entry["kind"] == kind and entry["usable"]}
        if path not in available:
            raise ValueError(f"That {kind} model is not installed or usable.")
        name = available[path]["name"]
        if not media_models.is_pixelart_model(path) and not media_models.diffusers_status()["installed"]:
            raise RuntimeError(f"{kind.capitalize()} instances need diffusers. Install it with: {media_models.INSTALL_HINT}")

    model_instance_load_lock.acquire()
    try:
        if device == "auto":
            device = "cpu"
            if HARDWARE.get("cuda") and torch.cuda.is_available():
                try:
                    _check_gpu_memory(path)
                    device = "gpu"
                except RuntimeError as error:
                    if "Not enough free GPU memory" not in str(error):
                        raise
        if kind != "text":
            if device == "gpu":
                _check_gpu_memory(path)
                _check_cpu_memory(path)
            else:
                _check_cpu_memory(path, multiplier=2.4, reserve_gb=1.5)

        instance_id = uuid.uuid4().hex[:12]
        instance = {"id": instance_id, "kind": kind, "name": name, "path": path,
                    "device": device, "status": "loading", "managed": True,
                    "tokenizer": None, "model": None, "pipeline": None,
                    "generation_lock": threading.Lock(), "active_requests": 0,
                    "idle_since": time.time(), "pool_managed": False}
        with model_instances_lock:
            MODEL_INSTANCES[instance_id] = instance
        try:
            if kind == "text":
                tok, mod, _ = _load_text_model(path, device)
                instance.update(tokenizer=tok, model=mod)
            else:
                pipe = media_models.load_pipeline(path, kind, device=device)
                instance["pipeline"] = pipe
            instance["status"] = "running"
            logger.info(f"Started {kind} model instance {instance_id}: {path} on {device.upper()}")
            return _instance_summary(instance)
        except Exception:
            with model_instances_lock:
                MODEL_INSTANCES.pop(instance_id, None)
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    finally:
        model_instance_load_lock.release()

def _lease_model_instance(kind: str, path: str) -> dict:
    with model_instances_lock:
        instance = next((entry for entry in MODEL_INSTANCES.values()
                         if entry["kind"] == kind and entry["path"] == path
                         and entry["status"] == "running" and entry.get("active_requests", 0) == 0
                         and not entry["generation_lock"].locked()), None)
        if instance:
            instance["active_requests"] = 1
            return instance

    summary = _start_model_instance(kind, path, "auto")
    with model_instances_lock:
        instance = MODEL_INSTANCES.get(summary["id"])
        if not instance or instance["status"] != "running":
            raise RuntimeError(f"The {kind} model instance stopped before it could be used.")
        instance["active_requests"] = 1
        instance["pool_managed"] = True
        return instance

def _release_model_instance(instance_id: str):
    with model_instances_lock:
        instance = MODEL_INSTANCES.get(instance_id)
        if instance:
            instance["active_requests"] = max(0, instance.get("active_requests", 0) - 1)
            if instance["active_requests"] == 0:
                instance["idle_since"] = time.time()

def _reap_idle_model_instances():
    while True:
        time.sleep(5)
        now = time.time()
        with model_instances_lock:
            candidates = [
                instance for instance in MODEL_INSTANCES.values()
                if instance.get("pool_managed") and instance["status"] == "running"
                and instance.get("active_requests", 0) == 0
                and now - instance.get("idle_since", now) >= 45
                and sum(1 for other in MODEL_INSTANCES.values()
                        if other["kind"] == instance["kind"] and other["path"] == instance["path"]
                        and other["status"] == "running") > 1
            ]
        for instance in candidates:
            try:
                with model_instances_lock:
                    current = MODEL_INSTANCES.get(instance["id"])
                    same_model_count = sum(
                        1 for other in MODEL_INSTANCES.values()
                        if other["kind"] == instance["kind"] and other["path"] == instance["path"]
                        and other["status"] == "running"
                    )
                    if (current is not instance or instance.get("active_requests", 0) != 0
                            or same_model_count <= 1):
                        continue
                _stop_model_instance(instance["id"])
            except (KeyError, RuntimeError):
                continue
            except Exception:
                logger.exception(f"Could not reap idle model instance {instance['id']}")

threading.Thread(target=_reap_idle_model_instances, daemon=True, name="model-instance-reaper").start()

def _stop_model_instance(instance_id: str):
    with model_instances_lock:
        instance = MODEL_INSTANCES.get(instance_id)
    if instance is None:
        raise KeyError("Model instance not found.")
    if instance["status"] != "running":
        raise RuntimeError("Wait for this model instance to finish loading before stopping it.")
    if instance_id == "default":
        gen_lock.acquire(1)
    instance["generation_lock"].acquire()
    try:
        with model_instances_lock:
            if MODEL_INSTANCES.get(instance_id) is not instance:
                return
            del MODEL_INSTANCES[instance_id]
        if instance_id == "default":
            global tokenizer, model, ACTIVE_MODEL_PATH
            with model_lock:
                old_tokenizer, old_model = tokenizer, model
                tokenizer = model = ACTIVE_MODEL_PATH = None
            del old_tokenizer, old_model
        instance.update(tokenizer=None, model=None, pipeline=None, status="stopped")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Stopped model instance {instance_id}")
    finally:
        instance["generation_lock"].release()
        if instance_id == "default":
            gen_lock.release()

# ══════════════════════════════════════════════════════════════════════
#  PRIORITY QUEUE — 1 = highest priority, 10 = lowest
# ══════════════════════════════════════════════════════════════════════

class PriorityLock:
    def __init__(self):
        self._cond = threading.Condition()
        self._heap = []
        self._counter = itertools.count()
        self._busy = False

    def acquire(self, priority: int):
        with self._cond:
            seq = next(self._counter)
            entry = (priority, seq)
            heapq.heappush(self._heap, entry)
            while self._busy or self._heap[0] != entry:
                self._cond.wait()
            heapq.heappop(self._heap)
            self._busy = True

    def release(self):
        with self._cond:
            self._busy = False
            self._cond.notify_all()

    def queue_depth(self):
        with self._cond:
            return len(self._heap)

    def is_busy(self):
        with self._cond:
            return self._busy

gen_lock = PriorityLock()

# ══════════════════════════════════════════════════════════════════════
#  ACCOUNTS — role-based (standard / administrator)
# ══════════════════════════════════════════════════════════════════════

users_lock = threading.RLock()

def load_users() -> dict:
    return _load_json(USERS_FILE, {})

def save_users(users: dict):
    with users_lock:
        try:
            USERS_FILE.write_text(json.dumps(users, indent=2))
        except Exception as e:
            logger.warning(f"Could not save users: {e}")

USERS = load_users()
USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]{3,24}$")

def is_admin_account(username: str) -> bool:
    with users_lock:
        u = USERS.get(username)
    return bool(u and u.get("role") == "administrator")

# Seed a default administrator account if none exists yet.
if not any(v.get("role") == "administrator" for v in USERS.values()):
    with users_lock:
        USERS["Admin"] = {
            "password_hash": generate_password_hash("Admin"),
            "created_at": time.time(),
            "role": "administrator",
        }
        save_users(USERS)
    logger.warning("=" * 60)
    logger.warning("  No administrator account existed — created one:")
    logger.warning("    Username: Admin   Password: Admin")
    logger.warning("  CHANGE THIS PASSWORD NOW (sign in, then use the")
    logger.warning("  password field in the chat page's settings panel).")
    logger.warning("=" * 60)

# ══════════════════════════════════════════════════════════════════════
#  PER-ACCOUNT POLICY (priority / token budget) — RLock, see note below.
# ══════════════════════════════════════════════════════════════════════

policy_lock = threading.RLock()

def load_policies() -> dict:
    return _load_json(POLICY_FILE, {})

def save_policies(policies: dict):
    with policy_lock:
        try:
            POLICY_FILE.write_text(json.dumps(policies, indent=2))
        except Exception as e:
            logger.warning(f"Could not save policies: {e}")

POLICIES = load_policies()

if _NEEDS_PRIORITY_MIGRATION and POLICIES:
    for _identity, _p in POLICIES.items():
        old_p = _p.get("priority")
        if isinstance(old_p, int) and 1 <= old_p <= 10:
            _p["priority"] = 11 - old_p
    save_policies(POLICIES)
    logger.warning(f"[migration] Flipped priority values for {len(POLICIES)} existing account(s). Spot-check in /admin.")

def is_guest_identity(identity: str) -> bool:
    return identity.startswith("guest-")

# ══════════════════════════════════════════════════════════════════════
#  APP SETTINGS — upload limits per role, generation model paths.
#  Admin-editable from /admin (Settings panel). Persisted separately from
#  config.json (which is CLI/startup args) so it can be changed live
#  without touching server launch config.
# ══════════════════════════════════════════════════════════════════════
SETTINGS_FILE = ROOT / "app_settings.json"
settings_lock = threading.RLock()

DEFAULT_SETTINGS = {
    # Max number of files a role may have stored at once in their account
    # uploads folder. -1 = unlimited, 0 = uploads disabled for that role.
    "upload_limits": {"guest": 0, "user": 50, "admin": -1},
    "generation": {"text_model_path": "", "image_model_path": "", "video_model_path": "", "extra_paths": []},
    "model_switching": {"enabled": False, "allowed_paths": []},
    "model_access": {},
    "search": {"engine": "duckduckgo", "max_sites": 4},
}

def load_settings() -> dict:
    s = _load_json(SETTINGS_FILE, {})
    # merge onto defaults so new keys added later don't break old files
    merged = json.loads(json.dumps(DEFAULT_SETTINGS))
    for k, v in (s or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    return merged

def save_settings(s: dict):
    with settings_lock:
        try:
            SETTINGS_FILE.write_text(json.dumps(s, indent=2))
        except Exception as e:
            logger.warning(f"Could not save app settings: {e}")

SETTINGS = load_settings()

def _model_access_roles(path: str, kind: str = None) -> set:
    access = SETTINGS.get("model_access", {})
    if isinstance(access, dict) and path in access:
        roles = access[path]
        roles = {role for role in roles if role in ("administrator", "standard", "guest")} if isinstance(roles, list) else set()
        return roles | {"administrator"}
    switching = SETTINGS.get("model_switching", {})
    if kind == "text":
        legacy = switching.get("allowed_paths", []) if switching.get("enabled") else []
        if path in legacy:
            return {"administrator", "standard"}
        generation = SETTINGS.get("generation", {})
        if path in (ACTIVE_MODEL_PATH, LAST_MODEL_PATH, generation.get("text_model_path")):
            return {"administrator", "standard"}
    else:
        generation = SETTINGS.get("generation", {})
        if path in (generation.get("image_model_path"), generation.get("video_model_path")):
            return {"administrator", "standard"}
    return {"administrator"}

def _identity_model_role(identity: str) -> str:
    if is_guest_identity(identity):
        return "guest"
    return "administrator" if is_admin_account(identity) else "standard"

def _identity_can_access_model(identity: str, path: str, kind: str = None) -> bool:
    return is_admin_account(identity) or _identity_model_role(identity) in _model_access_roles(path, kind)

def role_for(identity: str) -> str:
    if is_guest_identity(identity):
        return "guest"
    with users_lock:
        user = USERS.get(identity)
    if user and user.get("role") == "administrator":
        return "admin"
    return "user"

def upload_limit_for(identity: str) -> int:
    with settings_lock:
        limits = SETTINGS.get("upload_limits", {})
    return int(limits.get(role_for(identity), 0))

def uploads_count_for(identity: str) -> int:
    try:
        return sum(1 for _ in account_upload_dir(identity).iterdir() if _.is_file())
    except Exception:
        return 0

def get_policy(identity: str) -> dict:
    with policy_lock:
        p = POLICIES.get(identity)
        if not p:
            # give the seeded admin account generous defaults so it's
            # never stuck behind everyone else while you're testing
            is_seed_admin = identity == "Admin"
            p = {
                "name": identity,
                "priority": 1 if is_seed_admin else args.default_priority,
                "tokens_per_min": 100000 if is_seed_admin else args.default_tokens_per_min,
                "max_tokens": args.default_max_tokens,
            }
            POLICIES[identity] = p
            if not is_guest_identity(identity):
                save_policies(POLICIES)
        return p

# ══════════════════════════════════════════════════════════════════════
#  CONNECTION / RATE-LIMIT TRACKING
# ══════════════════════════════════════════════════════════════════════

sessions_lock = threading.Lock()
SESSIONS = {}

def get_session_state(identity: str) -> dict:
    with sessions_lock:
        s = SESSIONS.get(identity)
        if not s:
            s = {"history": [], "conversations": {}, "token_log": [], "connected_at": time.time(), "last_seen": time.time()}
            SESSIONS[identity] = s
        s["last_seen"] = time.time()
        return s

def tokens_used_last_minute(identity: str) -> int:
    now = time.time()
    s = get_session_state(identity)
    s["token_log"] = [(t, n) for (t, n) in s["token_log"] if now - t < 60]
    return sum(n for _, n in s["token_log"])

def record_tokens(identity: str, n: int):
    get_session_state(identity)["token_log"].append((time.time(), n))

# ══════════════════════════════════════════════════════════════════════
#  PERSISTENT MULTI-CONVERSATION CHAT HISTORY (registered accounts only)
# ══════════════════════════════════════════════════════════════════════

conv_lock = threading.RLock()

def safe_username(username: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", username)

def _user_file(username: str) -> Path:
    return CHAT_HISTORY_DIR / f"{safe_username(username)}.json"

def load_user_data(username: str) -> dict:
    return _load_json(_user_file(username), {"conversations": {}})

def save_user_data(username: str, data: dict):
    with conv_lock:
        try:
            _user_file(username).write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Could not save chat history for {username}: {e}")

def new_conversation(username: str) -> str:
    with conv_lock:
        data = load_user_data(username)
        conv_id = uuid.uuid4().hex[:12]
        data["conversations"][conv_id] = {
            "title": "New chat", "created_at": time.time(), "updated_at": time.time(), "messages": [],
        }
        save_user_data(username, data)
        return conv_id

def new_guest_conversation(identity: str) -> str:
    with conv_lock:
        conversations = get_session_state(identity).setdefault("conversations", {})
        conv_id = uuid.uuid4().hex[:12]
        now = time.time()
        conversations[conv_id] = {
            "title": "New chat", "created_at": now, "updated_at": now, "messages": [],
        }
        return conv_id

# ══════════════════════════════════════════════════════════════════════
#  FILE UPLOADS — saved to the account's own folder AND a training folder
# ══════════════════════════════════════════════════════════════════════

ALLOWED_UPLOAD_EXTS = {".txt", ".md", ".csv", ".json", ".log", ".py",
                        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
VIDEO_EXTS = {".mp4", ".webm"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024

def account_upload_dir(username: str) -> Path:
    account_dir = USER_DATA_DIR / safe_username(username)
    d = account_dir / "current"
    d.mkdir(parents=True, exist_ok=True)
    (account_dir / "deleted").mkdir(parents=True, exist_ok=True)
    legacy_dir = ACCOUNTS_DIR / safe_username(username) / "uploads"
    if legacy_dir.is_dir():
        for legacy_file in legacy_dir.iterdir():
            target = d / legacy_file.name
            if legacy_file.is_file() and not target.exists():
                try:
                    shutil.move(str(legacy_file), target)
                except OSError as error:
                    logger.warning(f"Could not migrate legacy upload {legacy_file} for {username}: {error}")
    return d

def account_deleted_dir(username: str) -> Path:
    d = USER_DATA_DIR / safe_username(username) / "deleted"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _move_upload_to_deleted(username: str, target: Path) -> bool:
    if not target.exists():
        return False
    deleted_dir = account_deleted_dir(username)
    destination = deleted_dir / target.name
    if destination.exists():
        destination = deleted_dir / f"{target.stem}_{uuid.uuid4().hex[:8]}{target.suffix}"
    shutil.move(str(target), str(destination))
    return True

def training_upload_pool_dir() -> Path:
    # Deliberately NOT keyed by username. Files land here as anonymous
    # training material only — there's no folder-per-user, no username in
    # the filename, and nothing that maps a pool file back to an account.
    # That's intentional: it's what lets a user delete "their" uploaded
    # file from their own account folder without that action being able
    # to (or needing to) reach into training data tied to no one.
    TRAINING_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return TRAINING_UPLOAD_DIR

uploads_meta_lock = threading.RLock()

def _uploads_meta_file(username: str) -> Path:
    return ACCOUNTS_DIR / safe_username(username) / "uploads_meta.json"

def load_uploads_meta(username: str) -> dict:
    return _load_json(_uploads_meta_file(username), {})

def save_uploads_meta(username: str, meta: dict):
    with uploads_meta_lock:
        try:
            path = _uploads_meta_file(username)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(meta, indent=2))
        except Exception as e:
            logger.warning(f"Could not save upload metadata for {username}: {e}")


def _upload_storage_dir(identity: str, info: dict) -> Path:
    if is_guest_identity(identity) and info.get("guest_nsfw"):
        return GUEST_NSFW_DIR
    return account_upload_dir(identity)


guest_archive_lock = threading.RLock()
GUEST_ARCHIVE_DIR = USER_DATA_DIR / "Guests"
GUEST_NSFW_DIR = USER_DATA_DIR / "guests" / "nsfw"
GUEST_ARCHIVE_META = GUEST_ARCHIVE_DIR / "uploads_meta.json"

def _guest_archive_filename(identity: str, source_name: str, relative_path: Path) -> str:
    parts = [re.sub(r"[^a-zA-Z0-9_.-]", "_", part)[:100] for part in relative_path.parts]
    return f"{safe_username(identity)}_{source_name}_{'_'.join(parts)}"

def _remove_empty_guest_dirs(root: Path):
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass
    try:
        root.rmdir()
    except OSError:
        pass

def _read_guest_archive_metadata(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read guest upload metadata from {path}: {error}") from error
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Guest upload metadata must be a JSON object: {path}")
    return metadata

def archive_guest_data(identity: str):
    """Move a guest's files and upload metadata into the shared flat archive."""
    if not is_guest_identity(identity):
        raise ValueError("Only guest data can be archived.")

    with guest_archive_lock:
        guest_roots = (
            ("user_data", USER_DATA_DIR / safe_username(identity)),
            ("account_data", ACCOUNTS_DIR / safe_username(identity)),
        )
        source_meta_path = _uploads_meta_file(identity)
        source_meta = _read_guest_archive_metadata(source_meta_path)
        GUEST_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        moved_names = {}
        nsfw_filenames = {
            Path(str(filename)).name for filename, info in source_meta.items()
            if isinstance(info, dict) and info.get("is_nsfw")
        }

        for source_name, root in guest_roots:
            if not root.is_dir():
                continue
            for source in root.rglob("*"):
                if not source.is_file() or source.is_symlink() or source == source_meta_path:
                    continue
                relative_path = source.relative_to(root)
                if (source.name in nsfw_filenames
                        and relative_path.parts[:1] in (("current",), ("uploads",))):
                    GUEST_NSFW_DIR.mkdir(parents=True, exist_ok=True)
                    archive_name = source.name
                    destination = GUEST_NSFW_DIR / archive_name
                    if destination.exists():
                        if filecmp.cmp(source, destination, shallow=False):
                            source.unlink()
                        else:
                            archive_name = _guest_archive_filename(identity, "nsfw", relative_path)
                            destination = GUEST_NSFW_DIR / archive_name
                            suffix = 2
                            while destination.exists():
                                archive_name = _guest_archive_filename(
                                    identity, "nsfw",
                                    relative_path.with_name(
                                        f"{relative_path.stem}_{suffix}{relative_path.suffix}"
                                    ),
                                )
                                destination = GUEST_NSFW_DIR / archive_name
                                suffix += 1
                            shutil.move(str(source), str(destination))
                    else:
                        shutil.move(str(source), str(destination))
                    moved_names[(source_name, relative_path)] = archive_name
                    continue
                archive_name = _guest_archive_filename(identity, source_name, relative_path)
                destination = GUEST_ARCHIVE_DIR / archive_name
                if destination.exists():
                    if filecmp.cmp(source, destination, shallow=False):
                        source.unlink()
                    else:
                        suffix = 2
                        while destination.exists():
                            archive_name = _guest_archive_filename(
                                identity, source_name,
                                relative_path.with_name(f"{relative_path.stem}_{suffix}{relative_path.suffix}"),
                            )
                            destination = GUEST_ARCHIVE_DIR / archive_name
                            suffix += 1
                        shutil.move(str(source), str(destination))
                else:
                    shutil.move(str(source), str(destination))
                moved_names[(source_name, relative_path)] = archive_name

        archived_meta = _read_guest_archive_metadata(GUEST_ARCHIVE_META)
        for filename, info in source_meta.items():
            source_path = Path(str(filename)).name
            archive_name = next(
                (name for (root_name, relative_path), name in moved_names.items()
                 if relative_path.name == source_path and
                 (root_name == "account_data" and relative_path.parts[:1] == ("uploads",)
                  or root_name == "user_data" and relative_path.parts[:1] == ("current",))),
                None,
            )
            if (archive_name is None and isinstance(info, dict) and info.get("guest_nsfw")
                    and (GUEST_NSFW_DIR / source_path).is_file()):
                archive_name = source_path
            if archive_name is None:
                for source_name, relative_path in (
                    ("user_data", Path("current") / source_path),
                    ("account_data", Path("uploads") / source_path),
                ):
                    candidate = _guest_archive_filename(identity, source_name, relative_path)
                    if (GUEST_ARCHIVE_DIR / candidate).is_file():
                        archive_name = candidate
                        break
            if archive_name is None:
                archive_name = _guest_archive_filename(identity, "metadata", Path(source_path))
            archived_meta[archive_name] = info

        temporary_meta = GUEST_ARCHIVE_META.with_name(f"{GUEST_ARCHIVE_META.name}.tmp")
        temporary_meta.write_text(json.dumps(archived_meta, indent=2), encoding="utf-8")
        temporary_meta.replace(GUEST_ARCHIVE_META)
        if source_meta_path.is_file():
            source_meta_path.unlink()

        for _, root in guest_roots:
            _remove_empty_guest_dirs(root)

def _cleanup_guest_session(identity: str):
    archive_guest_data(identity)
    with sessions_lock:
        SESSIONS.pop(identity, None)
    with policy_lock:
        POLICIES.pop(identity, None)

def _archive_existing_guest_data():
    guest_ids = set()
    for root in (USER_DATA_DIR, ACCOUNTS_DIR):
        if root.is_dir():
            guest_ids.update(
                path.name for path in root.iterdir()
                if path.is_dir() and is_guest_identity(path.name)
            )
    for identity in guest_ids:
        try:
            archive_guest_data(identity)
        except Exception as error:
            logger.warning(f"Could not consolidate data for {identity}: {error}")

_archive_existing_guest_data()

# ══════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════

def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not session.get("identity"):
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return wrapper

def admin_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not session.get("is_admin"):
            if request.is_json or request.accept_mimetypes.best == "application/json":
                return jsonify({"error": "Administrator sign-in is required."}), 401
            return redirect(url_for("login"))
        return fn(*a, **kw)
    return wrapper

def worker_key_required(fn):
    """For /internal/* routes only — authenticates OTHER Terminal AI
    machines calling in, via a per-machine shared key (X-Worker-Key
    header), not a human login session."""
    @wraps(fn)
    def wrapper(*a, **kw):
        if not secrets.compare_digest(request.headers.get("X-Worker-Key", ""), WORKER_API_KEY):
            return jsonify({"error": "unauthorized"}), 401
        return fn(*a, **kw)
    return wrapper

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    show_register = False
    if request.method == "POST":
        action = request.form.get("action")

        if action == "guest":
            previous_identity = session.get("identity")
            if session.get("is_guest") and previous_identity:
                try:
                    _cleanup_guest_session(previous_identity)
                except Exception as e:
                    logger.error(f"Could not archive guest data for {previous_identity}: {e}")
                    error = "Could not archive the previous guest session. Please try again."
                    return render_template("login.html", error=error,
                                           allow_registration=ALLOW_SELF_REGISTRATION, show_register=False)
            gid = "guest-" + uuid.uuid4().hex[:8]
            session.clear()
            session["identity"] = gid
            session["is_guest"] = True
            session["is_admin"] = False
            with policy_lock:
                POLICIES[gid] = {
                    "name": f"Guest-{gid[-4:]}",
                    "priority": args.guest_priority,
                    "tokens_per_min": args.guest_tokens_per_min,
                    "max_tokens": args.default_max_tokens,
                    "is_guest": True,
                }
            return redirect(url_for("index"))

        if action == "register":
            show_register = True
            if not ALLOW_SELF_REGISTRATION:
                error = "Self-signup is turned off. Ask an admin to create your account."
            else:
                username = (request.form.get("new_username") or "").strip()
                password = request.form.get("new_password") or ""
                if not USERNAME_RE.match(username):
                    error = "Username must be 3-24 characters: letters, numbers, _ . -"
                elif len(password) < 4:
                    error = "Password must be at least 4 characters."
                else:
                    with users_lock:
                        if username in USERS:
                            error = "That username is already taken."
                        else:
                            previous_identity = session.get("identity")
                            if session.get("is_guest") and previous_identity:
                                try:
                                    _cleanup_guest_session(previous_identity)
                                except Exception as e:
                                    logger.error(f"Could not archive guest data for {previous_identity}: {e}")
                                    error = "Could not archive the previous guest session. Please try again."
                        if not error:
                            # Self-signup can NEVER create an administrator account.
                            USERS[username] = {"password_hash": generate_password_hash(password),
                                                "created_at": time.time(), "role": "standard"}
                            save_users(USERS)
                    if not error:
                        with policy_lock:
                            POLICIES[username] = {
                                "name": username,
                                "priority": args.default_priority,
                                "tokens_per_min": args.default_tokens_per_min,
                                "max_tokens": args.default_max_tokens,
                            }
                            save_policies(POLICIES)
                        session.clear()
                        session["identity"] = username
                        session["is_guest"] = False
                        session["is_admin"] = False
                        return redirect(url_for("index"))

        if action == "login" or (action is None and not error):
            username = (request.form.get("username") or "").strip()
            password = request.form.get("password") or ""
            with users_lock:
                user = USERS.get(username)
            if user and check_password_hash(user["password_hash"], password):
                previous_identity = session.get("identity")
                if session.get("is_guest") and previous_identity:
                    try:
                        _cleanup_guest_session(previous_identity)
                    except Exception as e:
                        logger.error(f"Could not archive guest data for {previous_identity}: {e}")
                        error = "Could not archive the previous guest session. Please try again."
                        return render_template("login.html", error=error,
                                               allow_registration=ALLOW_SELF_REGISTRATION, show_register=False)
                session.clear()
                session["identity"] = username
                session["is_guest"] = False
                session["is_admin"] = (user.get("role") == "administrator")
                return redirect(url_for("index"))
            elif action == "login":
                error = "Wrong username or password."

    return render_template("login.html", error=error, allow_registration=ALLOW_SELF_REGISTRATION, show_register=show_register)

@app.route("/logout")
def logout():
    identity = session.get("identity")
    if session.get("is_guest") and identity:
        try:
            _cleanup_guest_session(identity)
        except Exception as e:
            logger.error(f"Could not archive guest data for {identity}: {e}")
            return "Could not archive guest data. Please try again.", 500
    session.clear()
    return redirect(url_for("login"))

@app.route("/admin/login")
def admin_login():
    # kept only so any old bookmarks don't 404 — auth is unified into /login now
    return redirect(url_for("login"))

# ══════════════════════════════════════════════════════════════════════
#  GENERATION
# ══════════════════════════════════════════════════════════════════════

MEMORY_TURNS_DEFAULT = 8
MEMORY_TURNS_MAX = 60  # sane ceiling — this is raw message count, not exchange count; protects against runaway prompt size

def build_prompt(message: str, history: list, system_prompt: str = None, memory_turns: int = MEMORY_TURNS_DEFAULT) -> str:
    system = system_prompt or (
        "You are a helpful, knowledgeable AI assistant. Answer clearly and concisely. "
        "Format code in triple backticks with the language name."
    )
    prompt = system + "\n\n"
    window = history[-memory_turns:] if memory_turns > 0 else []
    for turn in window:
        role = "User" if turn.get("role") == "user" else "Assistant"
        prompt += f"{role}: {turn.get('content', '')}\n"
    prompt += f"User: {message}\nAssistant:"
    return prompt

def _strip_model_control_tokens(piece_iter, tok):
    reasoning_tags = set(THINK_OPENERS + THINK_CLOSERS)
    special_tokens = [token for token in getattr(tok, "all_special_tokens", [])
                      if token and token not in reasoning_tags]
    if not special_tokens:
        yield from piece_iter
        return
    max_token_len = max(map(len, special_tokens))
    buf = ""
    for piece in piece_iter:
        buf += piece
        while True:
            matches = [(buf.find(token), token) for token in special_tokens if buf.find(token) >= 0]
            if not matches:
                safe_len = max(0, len(buf) - (max_token_len - 1))
                if safe_len:
                    yield buf[:safe_len]
                    buf = buf[safe_len:]
                break
            idx, token = min(matches, key=lambda match: (match[0], -len(match[1])))
            if idx:
                yield buf[:idx]
            buf = buf[idx + len(token):]
    if buf:
        yield buf

ACTIVE_GENERATIONS = {}  # request_id -> threading.Event (set() to cancel that generation early)
active_generations_lock = threading.RLock()

class _StopFlagCriteria(StoppingCriteria):
    """Lets a generation be cancelled from another thread — model.generate()
    checks stopping criteria once per generated token, so setting the
    event stops real compute within one token instead of only stopping
    the browser from receiving any more of it."""
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event
    def __call__(self, input_ids, scores, **kwargs):
        return self.stop_event.is_set()

def _generate_pieces(message, history, max_tokens, temperature, top_p, system_prompt=None,
                      memory_turns=MEMORY_TURNS_DEFAULT, stop_event: threading.Event = None,
                      instance_id: str = None):
    prompt = build_prompt(message, history, system_prompt, memory_turns)
    instance = None
    if instance_id:
        with model_instances_lock:
            instance = MODEL_INSTANCES.get(instance_id)
        if not instance or instance["kind"] != "text" or instance["status"] != "running":
            raise RuntimeError("The selected text model instance is no longer running.")
        instance["generation_lock"].acquire()
        with model_instances_lock:
            if MODEL_INSTANCES.get(instance_id) is not instance or instance["status"] != "running":
                instance["generation_lock"].release()
                raise RuntimeError("The selected text model instance has stopped.")
        tok, mod = instance["tokenizer"], instance["model"]
    else:
        with model_instances_lock:
            instance = MODEL_INSTANCES.get("default")
            if instance is None:
                instance = next((item for item in MODEL_INSTANCES.values()
                                 if item["kind"] == "text" and item["status"] == "running"), None)
        if instance:
            instance["generation_lock"].acquire()
            with model_instances_lock:
                if MODEL_INSTANCES.get(instance["id"]) is not instance or instance["status"] != "running":
                    instance["generation_lock"].release()
                    raise RuntimeError("The selected text model instance has stopped.")
            tok, mod = instance["tokenizer"], instance["model"]
        else:
            with model_lock:
                tok, mod = tokenizer, model
    if tok is None or mod is None:
        if instance:
            instance["generation_lock"].release()
        raise RuntimeError("No text model instance is running. Start a model from the instance manager.")
    try:
        inputs = tok(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(next(mod.parameters()).device)
        # Preserve reasoning delimiters even if the tokenizer marks them special.
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=False)
        gen_kwargs = dict(input_ids=input_ids, max_new_tokens=max_tokens, do_sample=True,
                           temperature=temperature, top_p=top_p, repetition_penalty=1.15,
                           pad_token_id=tok.eos_token_id, streamer=streamer)
        if stop_event is not None:
            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopFlagCriteria(stop_event)])
        thread = threading.Thread(target=mod.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()
        for piece in _strip_model_control_tokens(streamer, tok):
            if piece:
                yield piece
            if stop_event is not None and stop_event.is_set():
                break
        thread.join()
    finally:
        if instance:
            instance["generation_lock"].release()

def _clean_reply(reply: str) -> str:
    for stop in ["User:", "Human:", "\nUser", "\nHuman"]:
        if stop in reply:
            reply = reply[:reply.index(stop)].strip()
    return reply.strip()

# Natural-language triggers for the same thing. The slash-command was the
# only way in before, which is exactly why "look up X" typed normally
# never actually searched — the message just went to the model like any
# other question, and it answered from its own (possibly wrong/outdated)
# training instead of touching the web at all. Anchored to the start of
# the message and requiring a lead-in verb phrase keeps this from
# firing on unrelated sentences that merely contain the word "search".
_LOOKUP_NL_RE = re.compile(
    r"^(?:hey[, ]+|please\s+|can you\s+|could you\s+|pls\s+)*"
    r"(?:look\s*up|search(?:\s+the\s+web)?\s+for|google|web\s*search)\s+(.+)",
    re.IGNORECASE,
)

def _detect_lookup(message: str):
    command = re.match(r"^/(?:lookup|search)(?:\s+|$)", message, re.IGNORECASE)
    if command:
        return True, message[command.end():].strip()
    m = _LOOKUP_NL_RE.match(message.strip())
    if m:
        topic = m.group(1).strip().rstrip("?.! ")
        if topic:
            return True, topic
    return False, None

# ── "Create a file" without a separate button ──────────────────────────
# Detected from plain chat wording so it just happens as part of the
# conversation, the same way a person would expect — no dedicated modal
# to remember. Deliberately conservative (anchored verb + noun) to avoid
# firing on something like "I filed a complaint" or "let's create a plan".
_FILE_INTENT_RE = re.compile(
    r"\b(?:create|make|write|generate|save|produce)\b[^!?\n]{0,60}?"
    r"\b(?:a|an|me a|me an|the)?\s*(?:file|script|document)\b",
    re.IGNORECASE,
)
_FILE_EXT_RE = re.compile(r"\.(txt|md|markdown|py|js|html|css|json|csv|yaml|yml|xml)\b", re.IGNORECASE)
_FILE_EXT_HINTS = [
    (re.compile(r"\bpython\b", re.IGNORECASE), ".py"),
    (re.compile(r"\bjavascript\b|\bnode(?:\.js)?\b", re.IGNORECASE), ".js"),
    (re.compile(r"\bhtml\b|\bweb ?page\b", re.IGNORECASE), ".html"),
    (re.compile(r"\bcss\b|\bstylesheet\b", re.IGNORECASE), ".css"),
    (re.compile(r"\bjson\b", re.IGNORECASE), ".json"),
    (re.compile(r"\bcsv\b|\bspreadsheet\b", re.IGNORECASE), ".csv"),
    (re.compile(r"\byaml\b|\byml\b", re.IGNORECASE), ".yaml"),
    (re.compile(r"\bxml\b", re.IGNORECASE), ".xml"),
    (re.compile(r"\bmarkdown\b|\breadme\b", re.IGNORECASE), ".md"),
]

def _detect_file_intent(message: str):
    if not _FILE_INTENT_RE.search(message):
        return False, None
    ext_m = _FILE_EXT_RE.search(message)
    if ext_m:
        ext = "." + ext_m.group(1).lower()
        if ext == ".markdown":
            ext = ".md"
        if ext == ".yml":
            ext = ".yaml"
        return True, ext
    for pattern, ext in _FILE_EXT_HINTS:
        if pattern.search(message):
            return True, ext
    return True, ".txt"

# ── Flash cards ─────────────────────────────────────────────────────────
# "make me some flash cards on morphemes" -> an interactive deck in the
# chat. The model is asked for plain "term :: definition" lines rather
# than JSON, because small local models break JSON constantly but follow
# a one-line-per-card format fine; parsing below is deliberately forgiving.
_FLASHCARD_RE = re.compile(
    r"\b(?:make|create|generate|give|write|build|produce|do)\b[^!?\n]{0,40}?\bflash\s?-?cards?\b",
    re.IGNORECASE,
)
_FLASHCARD_TOPIC_RE = re.compile(r"\bflash\s?-?cards?\s+(?:on|about|for|covering|over|of|regarding)\s+(.+)", re.IGNORECASE)
_FLASHCARD_COUNT_RE = re.compile(r"\b(\d{1,2})\s+(?:flash\s?-?)?cards?\b|\b(\d{1,2})\s+flash", re.IGNORECASE)

def _detect_flashcards(message: str):
    if not _FLASHCARD_RE.search(message):
        return False, None, 10
    topic = None
    m = _FLASHCARD_TOPIC_RE.search(message)
    if m:
        topic = re.sub(r"[\s,]*(?:please|pls|thanks|thank you)\s*[?.!]*$", "", m.group(1).strip(), flags=re.IGNORECASE).rstrip("?.! ")
    if not topic:
        # "make flash cards" with the subject elsewhere / earlier in the sentence
        topic = re.sub(r"\b(?:please|can you|could you|make|create|generate|give|write|build|produce|me|some|a|an|set of|few|flash\s?-?cards?)\b",
                       " ", message, flags=re.IGNORECASE)
        topic = re.sub(r"\s+", " ", topic).strip(" ?.!,") or "the topic from this conversation"
    count = 10
    cm = _FLASHCARD_COUNT_RE.search(message)
    if cm:
        count = int(cm.group(1) or cm.group(2))
    return True, topic, max(3, min(count, 30))

def _parse_flashcards(text: str):
    cards = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^(?:\d+[.)]|[-*\u2022])\s*", "", line)   # numbering / bullets
        line = line.replace("**", "").replace("__", "").strip()
        for sep in ("::", " — ", " – ", " - ", ": "):
            if sep in line:
                front, back = line.split(sep, 1)
                front, back = front.strip(" \t\"'`"), back.strip()
                if front and back and len(front) <= 120:
                    cards.append({"front": front, "back": back})
                break
    seen, unique = set(), []
    for c in cards:
        key = c["front"].lower()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique

# ── Images / video straight from chat ──────────────────────────────────
# "generate an image of a red fox" / "draw me a dragon" / "make a video of
# waves". Anchored to the start of the message and requiring a connector
# word after the noun ("of", "showing", "with"...) so that questions like
# "how do I make an image smaller?" or "make the image bigger" are NOT
# hijacked into generating a picture.
_MEDIA_LEAD = (r"^(?:hey[,!]?\s+|please\s+|pls\s+|can you\s+|could you\s+|would you\s+|will you\s+|"
               r"i (?:want|need|would like|'d like)\s+(?:you to\s+)?|go ahead and\s+)*")
_IMAGE_NOUNS = r"(?:images?|pictures?|photos?|photographs?|illustrations?|drawings?|paintings?|artworks?|renders?|portraits?|logos?|wallpapers?|sketch(?:es)?|icons?|posters?)"
_VIDEO_NOUNS = r"(?:videos?|clips?|animations?|gifs?|short films?)"
_MEDIA_CONNECT = r"\s*(?:of|showing|depicting|featuring|with|about|:|-)\s*.+"
_IMG_GEN_RE = re.compile(_MEDIA_LEAD + r"(?:generate|create|make|produce|render|paint|draw|design|give me)\s+(?:me\s+)?"
                         r"(?:an?\s+|the\s+|some\s+)?(?P<body>(?:[\w-]+\s+){0,2}?" + _IMAGE_NOUNS + _MEDIA_CONNECT + ")", re.IGNORECASE)
_VID_GEN_RE = re.compile(_MEDIA_LEAD + r"(?:generate|create|make|produce|render|animate)\s+(?:me\s+)?"
                         r"(?:an?\s+|the\s+|some\s+)?(?P<body>(?:[\w-]+\s+){0,2}?" + _VIDEO_NOUNS + _MEDIA_CONNECT + ")", re.IGNORECASE)
_DRAW_RE = re.compile(_MEDIA_LEAD + r"(?:draw|paint|sketch)\s+(?:me\s+)?(?:(?:an?|the|some)\s+)?"
                      r"(?!conclusion|line\b|distinction|comparison|parallel|card\b|blank|breath|attention|bath)(?P<body>.+)", re.IGNORECASE)
# Generic lead-ins add nothing for the model ("image of a fox" -> "a fox"),
# but style words are kept: "watercolor painting of a lighthouse" stays whole.
_GENERIC_LEAD_RE = re.compile(r"^(?:images?|pictures?|videos?|clips?)\s*(?:of|:|-)\s*", re.IGNORECASE)
_IMAGE_COUNT_RE = re.compile(
    r"^(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
    r"\s+" + _IMAGE_NOUNS + r"\s*(?:of|showing|depicting|featuring|with|about|:|-)\s*(?P<prompt>.+)$",
    re.IGNORECASE,
)
_COUNTED_IMAGE_REQUEST_RE = re.compile(
    _MEDIA_LEAD + r"(?P<body>(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)"
    r"\s+" + _IMAGE_NOUNS + _MEDIA_CONNECT + ")",
    re.IGNORECASE,
)
# A standalone image noun phrase is also a direct request in chat, e.g.
# "image of flowers in a field", without an explicit "make" or "generate".
_IMPLICIT_IMAGE_REQUEST_RE = re.compile(
    r"^(?:(?:an?|some)\s+)?(?P<body>" + _IMAGE_NOUNS + _MEDIA_CONNECT + ")",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    word: number for number, word in enumerate(
        ("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
         "eighteen", "nineteen", "twenty"), 1
    )
}

def _image_count_and_prompt(body: str):
    match = _IMAGE_COUNT_RE.match(body.strip())
    if not match:
        return 1, body
    raw_count = match.group("count").casefold()
    count = int(raw_count) if raw_count.isdigit() else _NUMBER_WORDS[raw_count]
    return count, match.group("prompt")

def _detect_media_request(message: str):
    text = message.strip()
    for kind, rx in (("video", _VID_GEN_RE), ("image", _IMG_GEN_RE),
                     ("image", _COUNTED_IMAGE_REQUEST_RE),
                     ("image", _IMPLICIT_IMAGE_REQUEST_RE), ("image", _DRAW_RE)):
        m = rx.match(text)
        if m:
            body = m.group("body").strip()
            image_count = 1
            if kind == "image":
                image_count, body = _image_count_and_prompt(body)
            prompt = _GENERIC_LEAD_RE.sub("", body).strip().rstrip("?.! ")
            if prompt:
                return kind, prompt[:400], image_count
    return None, None, 1


def _detect_illustrated_story_request(message: str) -> bool:
    text = message.casefold()
    asks_for_story = re.search(r"\b(?:story|stories|tale|tales|narrative|fable)\b", text)
    asks_for_images = re.search(r"\b(?:images?|pictures?|illustrations?|artworks?|drawings?)\b", text)
    asks_to_generate = re.search(
        r"\b(?:generate|create|make|include|add|accompany|alongside|illustrate|illustrated|illustrating)\b"
        r"|\bto go with\b",
        text,
    )
    return bool(asks_for_story and asks_for_images and asks_to_generate)


def _story_reply_events(piece_iter, max_images: int):
    """Hide image prompts while yielding story text and inline thinking events."""
    open_markers = ("[[IMAGE_PROMPT]]", "[IMAGE_PROMPT]", "[IMAGE PROMPT]")
    open_marker_pattern = re.compile(
        r"\[\[IMAGE_PROMPT\]\]|\[IMAGE_PROMPT\]|\[IMAGE PROMPT\]", re.IGNORECASE
    )
    close_marker_pattern = re.compile(
        r"\[\[/IMAGE_PROMPT\]\]|\[/IMAGE_PROMPT\]|\[/IMAGE PROMPT\]", re.IGNORECASE
    )
    prompt_label = re.compile(
        r"(?im)^[ \t]*(?:(?:[-*+]\s+)|>\s*)*(?:\*\*)?image[ _-]?prompt(?:\*\*)?\s*:\s*"
    )
    buffer = ""
    image_count = 0
    limit_reported = False

    def clean_prompt(value):
        return re.sub(r"^(?:(?:[-*+]\s+)|>\s*)+", "", value.strip()).strip(" \t*_`")

    def emit_prompt(prompt):
        nonlocal image_count, limit_reported
        prompt = clean_prompt(prompt)
        if not prompt:
            return
        if image_count >= max_images:
            if not limit_reported:
                yield "text", f"\n\n(Reached the limit of {max_images} story illustrations.)\n\n"
                limit_reported = True
            return
        image_count += 1
        yield "image_prompt", prompt[:1000]

    def parse_buffer(final=False):
        nonlocal buffer
        while True:
            marker_match = open_marker_pattern.search(buffer)
            marker_start = marker_match.start() if marker_match else -1
            label_match = prompt_label.search(buffer)
            label_start = label_match.start() if label_match else -1
            if marker_start < 0 and label_start < 0:
                if final:
                    partial_marker = max(
                        (length for marker in open_markers
                         for length in range(1, min(len(buffer), len(marker) - 1) + 1)
                         if marker.casefold().startswith(buffer[-length:].casefold())),
                        default=0,
                    )
                    safe_buffer = buffer[:-partial_marker] if partial_marker else buffer
                    if safe_buffer:
                        yield "text", safe_buffer
                    buffer = ""
                else:
                    line_start = buffer.rfind("\n") + 1
                    current_line = buffer[line_start:]
                    candidate = re.sub(r"^[ \t]*(?:(?:[-*+]\s+)|>\s*)*", "", current_line)
                    if candidate.startswith("**"):
                        candidate = candidate[2:]
                    candidate = candidate.casefold()
                    if candidate and ("image prompt".startswith(candidate)
                                      or candidate.startswith("image prompt")):
                        safe_length = line_start
                    else:
                        partial_marker = max(
                            (length for marker in open_markers
                             for length in range(1, min(len(buffer), len(marker) - 1) + 1)
                             if marker.casefold().startswith(buffer[-length:].casefold())),
                            default=0,
                        )
                        safe_length = len(buffer) - partial_marker
                    if safe_length:
                        yield "text", buffer[:safe_length]
                        buffer = buffer[safe_length:]
                return
            if label_start >= 0 and (marker_start < 0 or label_start < marker_start):
                if label_start:
                    yield "text", buffer[:label_start]
                tail_start = label_match.end()
                line_end = buffer.find("\n", tail_start)
                if line_end < 0 and not final:
                    return
                if line_end < 0:
                    line_end = len(buffer)
                prompt = buffer[tail_start:line_end].strip()
                consume_end = line_end + (1 if line_end < len(buffer) else 0)
                if not clean_prompt(prompt):
                    next_line_end = buffer.find("\n", consume_end)
                    if next_line_end < 0 and not final:
                        buffer = buffer[label_start:]
                        return
                    if next_line_end < 0:
                        next_line_end = len(buffer)
                    next_line = buffer[consume_end:next_line_end]
                    if re.match(r"^[ \t]*(?:(?:[-*+]\s+)|>\s*)", next_line):
                        prompt = next_line
                        consume_end = next_line_end + (1 if next_line_end < len(buffer) else 0)
                buffer = buffer[consume_end:]
                yield from emit_prompt(prompt)
                continue
            if marker_start:
                yield "text", buffer[:marker_start]
            marker_end = marker_match.end()
            close_match = close_marker_pattern.search(buffer, marker_end)
            if close_match is None:
                if final:
                    buffer = ""
                return
            prompt = buffer[marker_end:close_match.start()]
            buffer = buffer[close_match.end():]
            yield from emit_prompt(prompt)

    for kind, piece in piece_iter:
        if kind == "thinking":
            yield from parse_buffer(final=True)
            buffer = ""
            yield kind, piece
            continue
        buffer += piece
        yield from parse_buffer()
    yield from parse_buffer(final=True)


# ── "Thinking" trace ─────────────────────────────────────────────────
# Asks the model to briefly plan its answer inside <thinking>...</thinking>
# before writing the actual reply, then splits the two apart as they
# stream so the UI can show the plan in a collapsible dropdown and the
# real answer as the normal reply. This is prompted behavior, not a
# built-in reasoning mode — a small model won't always use the tags, and
# when it doesn't, everything is just treated as the answer with no
# dropdown shown. No hidden compute happens either way; it's the same
# generation, just asked to narrate a little before it answers.
THINK_OPEN, THINK_CLOSE = "<thinking>", "</thinking>"
THINK_OPENERS = (THINK_OPEN, "<think>", "<Thinking>", "<Think>")
THINK_CLOSERS = (THINK_CLOSE, "</think>", "</Thinking>", "</Think>")

def _system_prompt_for(show_thinking: bool) -> str:
    base = ("You are a helpful, knowledgeable AI assistant. Answer clearly and concisely. "
            "Format code in triple backticks with the language name.")
    if not show_thinking:
        return base
    return base + (
        f" Before your final answer, briefly plan it inside {THINK_OPEN}...{THINK_CLOSE} tags — "
        f"1 to 3 short sentences on what's actually being asked and how you'll approach it. "
        f"Then write the real answer OUTSIDE those tags, without repeating the thinking there."
    )

def _split_thinking_stream(piece_iter):
    """Separates common <think> and <thinking> tags into thinking/answer
    tuples. Handles
    the tags landing across separate streamed chunks by holding back a
    small tail of the buffer whenever a full tag hasn't been seen yet, in
    case it's a partial tag rather than plain text."""
    buf, state = "", "answer"
    max_tag_len = max(map(len, THINK_OPENERS + THINK_CLOSERS))
    for piece in piece_iter:
        buf += piece
        while True:
            tags = THINK_OPENERS + THINK_CLOSERS if state == "answer" else THINK_CLOSERS
            folded_buf = buf.casefold()
            matches = [(folded_buf.find(tag.casefold()), tag)
                       for tag in tags if folded_buf.find(tag.casefold()) >= 0]
            if not matches:
                safe_len = max(0, len(buf) - (max_tag_len - 1))
                if safe_len > 0:
                    yield (state, buf[:safe_len])
                    buf = buf[safe_len:]
                break
            idx, tag = min(matches, key=lambda match: match[0])
            if idx > 0:
                yield (state, buf[:idx])
            buf = buf[idx + len(tag):]
            if tag.casefold() in {opener.casefold() for opener in THINK_OPENERS}:
                state = "thinking"
            else:
                state = "answer"
    if buf:
        yield (state, buf)

_CODE_HINTS = ("```", "def ", "class ", "function ", "import ", "SELECT ", "select ",
               "<html", "</", "public static", "#include", "console.log", "== null",
               "npm install", "pip install", "traceback", "stack trace", "exception")

def _classify_prompt_type(message: str) -> str:
    """Rough heuristic used only for routing preference, not correctness
    — worst case a code question gets routed like a chat question, which
    is harmless. "chat" and "code" are the two types peers can be given
    different priority weights for on the Peers admin page."""
    m = message or ""
    if any(h in m for h in _CODE_HINTS):
        return "code"
    return "chat"

def _peer_priority(peer: dict, prompt_type: str) -> int:
    return int((peer.get("priorities") or {}).get(prompt_type, 5))

def _pick_generation_peer(prompt_type: str = "chat", timeout: float = 1.0):
    """Picks where a request should actually run: this machine, or one of
    its PAIRED peers. Two things factor in, not just raw queue depth:

    - This machine's own role. A "master" keeps work local unless it's
      genuinely busier than a peer. A "slave" is meant to hand work off
      to a peer whenever one is available and not clearly more loaded —
      without this, a lone user sending prompts one at a time to a
      machine that's essentially always idle between messages would
      NEVER see the peer get used at all (0 < 0 is never true), which is
      exactly why pairing looked like it did nothing.
    - Per-peer, per-prompt-type priority (1-10, set on the Peers admin
      page once a peer is paired) — a peer configured with high priority
      for "code" gets preferred for code-shaped prompts even if it's
      carrying a little more load than another peer or than local.

    Local and each peer get an effective score = queue_depth - bias,
    where bias comes from priority (peers) or role (local). Lowest score
    wins; local keeps ties unless this machine is a "slave" and a peer
    is available, in which case the peer keeps ties instead."""
    with peers_lock:
        paired = [p for p in PEERS if p.get("api_key")]
    local_depth = gen_lock.queue_depth() + (1 if gen_lock.is_busy() else 0)
    local_bias = -1.5 if MACHINE_ROLE == "slave" else 0.0
    best_peer, best_score = None, local_depth - local_bias
    if not paired or not REQUESTS_OK:
        return None
    for peer in paired:
        try:
            r = requests.get(f"{peer['url']}/internal/status", headers=_peer_headers(peer), timeout=timeout)
            if not r.ok:
                continue
            d = r.json()
            depth = d.get("queue_depth", 0) + (1 if d.get("busy") else 0)
            priority = _peer_priority(peer, prompt_type)
            bias = (priority - 5) * 0.4  # 1-10 scale -> roughly -1.6..+2.0
            score = depth - bias
            strictly_better = score < best_score
            tie_goes_to_peer = (score == best_score and best_peer is None and MACHINE_ROLE == "slave")
            if strictly_better or tie_goes_to_peer:
                best_score, best_peer = score, peer
        except Exception:
            continue
    return best_peer

def _relay_from_peer(peer: dict, message: str, history: list, max_tokens: int,
                      temperature: float, top_p: float, memory_turns: int, request_id: str = None,
                      show_thinking: bool = True):
    """Streams a generation from a peer's /internal/generate and yields
    the SAME ndjson lines our own local generation produces, so the
    browser-side code needs zero changes to work with either. Raises on
    any failure so the caller can fall back to local generation. The
    caller is responsible for parsing the final {"done": true, ...} line
    out of what this yields — a generator's `return` value isn't
    reachable through a plain for-loop, so this doesn't try to smuggle
    the reply out that way. request_id (if given) lets a later Stop
    request find its way to the peer that's actually doing the work —
    see /api/chat/stop and /internal/generate/stop."""
    resp = requests.post(f"{peer['url']}/internal/generate", headers=_peer_headers(peer), timeout=(3, 120),
                          json={"message": message, "history": history, "max_tokens": max_tokens,
                                "temperature": temperature, "top_p": top_p, "memory_turns": memory_turns,
                                "request_id": request_id, "show_thinking": show_thinking},
                          stream=True)
    resp.raise_for_status()
    got_done = False
    for line in resp.iter_lines():
        if not line:
            continue
        # decode_unicode=True on iter_lines() doesn't actually guarantee a
        # str back — when requests can't confidently determine an encoding
        # it silently falls back to bytes. Decode explicitly rather than
        # trusting it, since a bytes/str concat here throws and looked
        # exactly like "the peer failed" even though the peer worked fine.
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")
        yield line
        try:
            if json.loads(line).get("done"):
                got_done = True
        except Exception:
            pass
    if not got_done:
        raise RuntimeError("peer stream ended without a final reply")

# ══════════════════════════════════════════════════════════════════════
#  ROUTES — chat
# ══════════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def index():
    identity = session["identity"]
    policy = get_policy(identity)
    with settings_lock:
        configured_text_model = (SETTINGS.get("generation", {}).get("text_model_path") or "").strip()
    active_model_path = configured_text_model or ACTIVE_MODEL_PATH or resolve_model_path()
    swappable_models = [entry for entry in _scan_chat_models()
                        if _identity_can_access_model(identity, entry["path"], "text")]
    if active_model_path and _identity_can_access_model(identity, active_model_path, "text") and not any(
            entry["path"] == active_model_path for entry in swappable_models):
        swappable_models.insert(0, {"path": active_model_path, "name": Path(active_model_path).name or active_model_path})
    can_switch_models = len(swappable_models) > 1 or bool(
        swappable_models and swappable_models[0]["path"] != active_model_path)
    media_models = _scan_media()
    return render_template("chat.html", name=policy["name"], max_tokens_cap=policy["max_tokens"],
                            hard_cap=args.hard_max_tokens, is_guest=session.get("is_guest", False),
                            is_admin=session.get("is_admin", False), upload_limit=upload_limit_for(identity),
                            active_model_path=active_model_path, swappable_models=swappable_models,
                            can_switch_models=can_switch_models,
                            has_image_model=any(e["kind"] == "image" and e["usable"] and _identity_can_access_model(identity, e["path"], "image") for e in media_models),
                            has_video_model=any(e["kind"] == "video" and e["usable"] and _identity_can_access_model(identity, e["path"], "video") for e in media_models))

@app.route("/brand-icon")
def brand_icon():
    return send_file(str(ROOT / "TerminalAI.ico"), mimetype="image/vnd.microsoft.icon")

@app.route("/api/set_name", methods=["POST"])
@login_required
def api_set_name():
    identity = session["identity"]
    name = (request.json or {}).get("name", "").strip()[:40]
    if name:
        with policy_lock:
            get_policy(identity)["name"] = name
            if not is_guest_identity(identity):
                save_policies(POLICIES)
    return jsonify({"ok": True})

@app.route("/api/change_password", methods=["POST"])
@login_required
def api_change_password():
    identity = session["identity"]
    if session.get("is_guest"):
        return jsonify({"error": "Guests don't have a password."}), 400
    new_password = (request.json or {}).get("password") or ""
    if len(new_password) < 4:
        return jsonify({"error": "Password must be at least 4 characters."}), 400
    with users_lock:
        if identity not in USERS:
            return jsonify({"error": "Account not found."}), 400
        USERS[identity]["password_hash"] = generate_password_hash(new_password)
        save_users(USERS)
    return jsonify({"ok": True})

@app.route("/api/conversations")
@login_required
def api_conversations():
    identity = session["identity"]
    if session.get("is_guest"):
        with conv_lock:
            conversations = get_session_state(identity).setdefault("conversations", {})
            convs = sorted(
                [{"id": cid, "title": c["title"], "updated_at": c["updated_at"], "messages": c["messages"]}
                 for cid, c in conversations.items()],
                key=lambda c: -c["updated_at"],
            )
        return jsonify({"conversations": convs, "anonymous": True})
    data = load_user_data(identity)
    convs = sorted(
        [{"id": cid, "title": c["title"], "updated_at": c["updated_at"]} for cid, c in data["conversations"].items()],
        key=lambda c: -c["updated_at"],
    )
    return jsonify({"conversations": convs, "anonymous": False})

@app.route("/api/conversations/new", methods=["POST"])
@login_required
def api_new_conversation():
    if session.get("is_guest"):
        return jsonify({"id": new_guest_conversation(session["identity"])})
    return jsonify({"id": new_conversation(session["identity"])})

@app.route("/api/conversations/<conv_id>/delete", methods=["POST"])
@login_required
def api_delete_conversation(conv_id):
    identity = session["identity"]
    if session.get("is_guest"):
        with conv_lock:
            conversations = get_session_state(identity).setdefault("conversations", {})
            conversations.pop(conv_id, None)
        return jsonify({"ok": True})
    with conv_lock:
        data = load_user_data(identity)
        data["conversations"].pop(conv_id, None)
        save_user_data(identity, data)
    return jsonify({"ok": True})

@app.route("/api/conversations/<conv_id>/delete_if_empty", methods=["POST"])
@login_required
def api_delete_empty_conversation(conv_id):
    identity = session["identity"]
    if session.get("is_guest"):
        with conv_lock:
            conversations = get_session_state(identity).setdefault("conversations", {})
            conversation = conversations.get(conv_id)
            if conversation is None:
                return jsonify({"error": "Conversation not found."}), 404
            deleted = not conversation.get("messages")
            if deleted:
                conversations.pop(conv_id)
        return jsonify({"ok": True, "deleted": deleted})
    with conv_lock:
        data = load_user_data(identity)
        conversation = data["conversations"].get(conv_id)
        if conversation is None:
            return jsonify({"error": "Conversation not found."}), 404
        deleted = not conversation.get("messages")
        if deleted:
            data["conversations"].pop(conv_id)
            save_user_data(identity, data)
    return jsonify({"ok": True, "deleted": deleted})

@app.route("/api/conversations/<conv_id>/rename", methods=["POST"])
@login_required
def api_rename_conversation(conv_id):
    identity = session["identity"]
    if session.get("is_guest"):
        return jsonify({"error": "Guests don't have saved conversations."}), 400
    title = ((request.json or {}).get("title") or "").strip()[:80]
    if not title:
        return jsonify({"error": "Enter a title."}), 400
    with conv_lock:
        data = load_user_data(identity)
        if conv_id not in data["conversations"]:
            return jsonify({"error": "Conversation not found."}), 404
        data["conversations"][conv_id]["title"] = title
        save_user_data(identity, data)
    return jsonify({"ok": True, "title": title})

@app.route("/api/conversations/<conv_id>/download")
@login_required
def api_download_conversation(conv_id):
    identity = session["identity"]
    if session.get("is_guest"):
        return jsonify({"error": "Guests don't have saved conversations."}), 400
    with conv_lock:
        data = load_user_data(identity)
        conv = data["conversations"].get(conv_id)
    if not conv:
        return jsonify({"error": "Conversation not found."}), 404
    lines = [f"{conv.get('title', 'Chat')}", ""]
    for turn in conv.get("messages", []):
        who = "You" if turn.get("role") == "user" else "Assistant"
        lines.append(f"{who}: {turn.get('content', '')}")
        attachments = turn.get("attachments") or []
        att = turn.get("attachment")
        if att:
            lines.append(f"[attachment: {att.get('original_name', att.get('filename', ''))}]")
        for att in attachments:
            lines.append(f"[attachment: {att.get('original_name', att.get('filename', ''))}]")
        lines.append("")
    body = "\n".join(lines)
    safe_title = re.sub(r"[^a-zA-Z0-9_.-]", "_", conv.get("title", "chat"))[:60] or "chat"
    resp = Response(body, mimetype="text/plain")
    resp.headers["Content-Disposition"] = f'attachment; filename="{safe_title}.txt"'
    return resp

@app.route("/api/history")
@login_required
def api_history():
    identity = session["identity"]
    if session.get("is_guest"):
        with conv_lock:
            conversations = get_session_state(identity).setdefault("conversations", {})
            conv_id = request.args.get("conv_id")
            if not conv_id or conv_id not in conversations:
                conv_id = max(conversations, key=lambda cid: conversations[cid]["updated_at"], default=None)
            if not conv_id:
                conv_id = new_guest_conversation(identity)
            conversation = conversations[conv_id]
            return jsonify({"history": conversation["messages"], "conv_id": conv_id, "anonymous": True})
    conv_id = request.args.get("conv_id")
    with conv_lock:
        data = load_user_data(identity)
        if not conv_id or conv_id not in data["conversations"]:
            if data["conversations"]:
                conv_id = max(data["conversations"].items(), key=lambda kv: kv[1]["updated_at"])[0]
            else:
                return jsonify({"history": [], "conv_id": None, "anonymous": False})
        return jsonify({"history": data["conversations"][conv_id]["messages"], "conv_id": conv_id, "anonymous": False})

@app.route("/api/clear", methods=["POST"])
@login_required
def api_clear():
    identity = session["identity"]
    if not session.get("is_guest"):
        return jsonify({"error": "Accounts use New Chat instead of Clear."}), 400
    conv_id = (request.json or {}).get("conv_id")
    with conv_lock:
        conversations = get_session_state(identity).setdefault("conversations", {})
        if conv_id and conv_id in conversations:
            conversations[conv_id]["messages"] = []
            conversations[conv_id]["title"] = "New chat"
            conversations[conv_id]["updated_at"] = time.time()
    return jsonify({"ok": True})

@app.route("/api/guest/conversations/restore", methods=["POST"])
@login_required
def api_restore_guest_conversations():
    if not session.get("is_guest"):
        return jsonify({"error": "Guest chat restore is only available to guests."}), 403
    body = request.json or {}
    raw_conversations = body.get("conversations")
    if not isinstance(raw_conversations, list) or len(raw_conversations) > 30:
        return jsonify({"error": "Guest chat data must contain at most 30 conversations."}), 400
    conversations = {}
    total_chars = 0

    def clean_attachment(attachment):
        if not isinstance(attachment, dict):
            return None
        url = attachment.get("url")
        if not isinstance(url, str) or not url.startswith(f"/files/{session['identity']}/"):
            return None
        clean = {
            key: attachment.get(key)
            for key in ("filename", "original_name", "is_image", "is_video", "is_nsfw",
                        "story_image_index", "url")
            if key in attachment
        }
        generated_prompt = attachment.get("generated_prompt")
        if isinstance(generated_prompt, str):
            clean["generated_prompt"] = generated_prompt[:400]
        return clean

    for raw in raw_conversations:
        if not isinstance(raw, dict):
            return jsonify({"error": "Invalid guest conversation data."}), 400
        conv_id = raw.get("id")
        messages = raw.get("messages")
        if not isinstance(conv_id, str) or not re.fullmatch(r"[a-f0-9]{12}", conv_id):
            return jsonify({"error": "Invalid guest conversation ID."}), 400
        if not isinstance(messages, list) or len(messages) > 100:
            return jsonify({"error": "Guest conversations may contain at most 100 messages each."}), 400
        clean_messages = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
                return jsonify({"error": "Invalid guest message data."}), 400
            content = message.get("content")
            if not isinstance(content, str) or len(content) > 20000:
                return jsonify({"error": "Guest messages must be text under 20,000 characters."}), 400
            total_chars += len(content)
            if total_chars > 500000:
                return jsonify({"error": "Guest chat data exceeds the 500,000-character limit."}), 400
            clean_message = {"role": message["role"], "content": content}
            attachment = clean_attachment(message.get("attachment"))
            if attachment:
                clean_message["attachment"] = attachment
            attachments = message.get("attachments")
            if isinstance(attachments, list):
                clean_attachments = [item for item in (clean_attachment(value) for value in attachments) if item]
                if clean_attachments:
                    clean_message["attachments"] = clean_attachments
            story_thinking = message.get("story_thinking")
            if isinstance(story_thinking, list) and len(story_thinking) <= 25:
                clean_thinking = []
                for item in story_thinking:
                    if (isinstance(item, dict) and isinstance(item.get("index"), int)
                            and 1 <= item["index"] <= 25 and isinstance(item.get("text"), str)):
                        clean_thinking.append({"index": item["index"], "text": item["text"][:4000]})
                if clean_thinking:
                    clean_message["story_thinking"] = clean_thinking
                    total_chars += sum(len(item["text"]) for item in clean_thinking)
                    if total_chars > 500000:
                        return jsonify({"error": "Guest chat data exceeds the 500,000-character limit."}), 400
            flashcards = message.get("flashcards")
            if isinstance(flashcards, list) and len(flashcards) <= 100:
                clean_message["flashcards"] = flashcards
            clean_messages.append(clean_message)
        now = time.time()
        conversations[conv_id] = {
            "title": str(raw.get("title") or "New chat")[:80],
            "created_at": now,
            "updated_at": now,
            "messages": clean_messages,
        }
    state = get_session_state(session["identity"])
    with conv_lock:
        state["conversations"] = conversations
        state["history"] = []
    current_id = body.get("current_conv_id")
    if current_id not in conversations:
        current_id = max(conversations, key=lambda cid: conversations[cid]["updated_at"], default=None)
    return jsonify({"ok": True, "conv_id": current_id})

@app.route("/api/guest/leave", methods=["POST"])
@login_required
def api_guest_leave():
    if not session.get("is_guest"):
        return jsonify({"error": "Only guest sessions can be cleared this way."}), 403
    identity = session["identity"]
    try:
        _cleanup_guest_session(identity)
    except Exception as e:
        logger.error(f"Could not archive guest data for {identity}: {e}")
        return jsonify({"error": "Could not archive guest data. Please try again."}), 500
    session.clear()
    return jsonify({"ok": True})

@app.route("/api/upload", methods=["POST"])
@login_required
def api_upload():
    identity = session["identity"]
    limit = upload_limit_for(identity)
    if limit == 0:
        return jsonify({"error": "File uploads aren't available for your account type. "
                                  "Create a full account to attach files." if session.get("is_guest")
                                  else "File uploads are disabled for your account."}), 403

    files = request.files.getlist("files") or ([request.files["file"]] if request.files.get("file") else [])
    if not files:
        return jsonify({"error": "No file provided."}), 400

    used = uploads_count_for(identity)
    saved, errors = [], []
    for f in files:
        if not f.filename:
            continue
        if limit >= 0 and used >= limit:
            errors.append(f"{f.filename}: upload limit reached ({limit} files max)")
            continue
        ext = Path(f.filename).suffix.lower()
        if ext not in ALLOWED_UPLOAD_EXTS:
            errors.append(f"{f.filename}: file type '{ext or '(none)'}' isn't allowed")
            continue
        data = f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            errors.append(f"{f.filename}: too large (8MB max)")
            continue

        clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(f.filename).name)
        stamped = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{clean_name}"
        try:
            (account_upload_dir(identity) / stamped).write_bytes(data)
            (training_upload_pool_dir() / stamped).write_bytes(data)
        except Exception as e:
            logger.error(f"Upload save failed for {identity}: {e}")
            errors.append(f"{f.filename}: couldn't save on the server")
            continue

        is_image = ext in IMAGE_EXTS
        is_video = ext in VIDEO_EXTS
        with uploads_meta_lock:
            meta = load_uploads_meta(identity)
            meta[stamped] = {"original_name": f.filename, "is_image": is_image, "is_video": is_video,
                              "uploaded_at": time.time()}
            save_uploads_meta(identity, meta)
        used += 1
        saved.append({"filename": stamped, "original_name": f.filename, "is_image": is_image,
                       "is_video": is_video, "url": f"/files/{identity}/{stamped}"})

    if not saved and errors:
        return jsonify({"error": "; ".join(errors)}), 400
    # Kept for backwards compatibility with older single-file callers, which
    # expect these fields at the top level for the one file they sent.
    resp = {"ok": True, "saved": saved, "errors": errors}
    if saved:
        resp.update(saved[0])
    return jsonify(resp)

@app.route("/api/upload_limits")
@login_required
def api_upload_limits():
    identity = session["identity"]
    limit = upload_limit_for(identity)
    return jsonify({"limit": limit, "used": uploads_count_for(identity), "role": role_for(identity)})

@app.route("/api/uploads")
@login_required
def api_uploads():
    identity = session["identity"]
    if upload_limit_for(identity) == 0:
        return jsonify({"uploads": []})
    meta = load_uploads_meta(identity)
    items = []
    for fname, info in meta.items():
        upload_path = _upload_storage_dir(identity, info) / fname
        if upload_path.is_file() and not info.get("nsfw_check_pending"):
            items.append({
                "filename": fname,
                "original_name": info.get("original_name", fname),
                "is_image": info.get("is_image", False),
                "is_video": info.get("is_video", False),
                "is_nsfw": info.get("is_nsfw", False),
                "uploaded_at": info.get("uploaded_at", 0),
                "url": f"/files/{identity}/{fname}",
            })
    items.sort(key=lambda x: -x["uploaded_at"])
    return jsonify({"uploads": items})

@app.route("/api/uploads/delete", methods=["POST"])
@login_required
def api_delete_upload():
    identity = session["identity"]
    if upload_limit_for(identity) == 0:
        return jsonify({"error": "File uploads are disabled for this account."}), 403
    filename = Path((request.json or {}).get("filename", "")).name  # strip any path components
    if not filename:
        return jsonify({"error": "filename required"}), 400
    d = account_upload_dir(identity)
    target = d / filename
    # This only ever touches the user's OWN account folder. The copy that
    # was made into the anonymous training-data pool at upload time is
    # deliberately not associated with any user and is not reachable or
    # removable from here — see training_upload_pool_dir() above.
    if target.exists():
        try:
            _move_upload_to_deleted(identity, target)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    with uploads_meta_lock:
        meta = load_uploads_meta(identity)
        meta.pop(filename, None)
        save_uploads_meta(identity, meta)
    return jsonify({"ok": True})

@app.route("/api/uploads/delete_all", methods=["POST"])
@login_required
def api_delete_all_uploads():
    identity = session["identity"]
    if upload_limit_for(identity) == 0:
        return jsonify({"error": "File uploads are disabled for this account."}), 403
    upload_dir = account_upload_dir(identity)
    deleted = 0
    try:
        for target in upload_dir.iterdir():
            if target.is_file():
                deleted += int(_move_upload_to_deleted(identity, target))
    except OSError as error:
        logger.warning(f"Could not delete all uploads for {identity}: {error}")
        return jsonify({"error": f"Could not delete all files: {error}"}), 500
    with uploads_meta_lock:
        save_uploads_meta(identity, {})
    return jsonify({"ok": True, "deleted": deleted})

@app.route("/api/account/delete", methods=["POST"])
@login_required
def api_delete_account():
    identity = session["identity"]
    if session.get("is_guest"):
        return jsonify({"error": "Guests don't have accounts to delete."}), 403
    password = (request.json or {}).get("password", "")
    with users_lock:
        user = USERS.get(identity)
        if not user or not check_password_hash(user["password_hash"], password):
            return jsonify({"error": "Incorrect password."}), 400
        if user.get("role") == "administrator":
            remaining_admins = sum(1 for u, v in USERS.items()
                                    if v.get("role") == "administrator" and u != identity)
            if remaining_admins == 0:
                return jsonify({"error": "Can't delete the last administrator account. "
                                          "Promote another admin first."}), 400
        USERS.pop(identity, None)
        save_users(USERS)
    with policy_lock:
        POLICIES.pop(identity, None)
        save_policies(POLICIES)
    with sessions_lock:
        SESSIONS.pop(identity, None)
    # Removes everything in this account's own folder: uploaded files,
    # upload metadata, all of it. Does NOT touch the anonymous training
    # data pool — those copies were never associated with this (or any)
    # account, so there's nothing to trace back and remove there.
    try:
        shutil.rmtree(ACCOUNTS_DIR / safe_username(identity), ignore_errors=True)
    except Exception as e:
        logger.warning(f"Could not fully remove account folder for {identity}: {e}")
    try:
        shutil.rmtree(USER_DATA_DIR / safe_username(identity), ignore_errors=True)
    except Exception as e:
        logger.warning(f"Could not fully remove user data folder for {identity}: {e}")
    try:
        _user_file(identity).unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Could not remove chat history for {identity}: {e}")
    session.clear()
    return jsonify({"ok": True})

# ══════════════════════════════════════════════════════════════════════
#  IMAGE / VIDEO GENERATION
#  Optional: only works if an admin has configured a local diffusers-
#  compatible model path in Admin → Settings and `diffusers` is
#  installed. Otherwise these endpoints return a clear explanation
#  instead of pretending to generate something.
# ══════════════════════════════════════════════════════════════════════
import media_models

IMAGE_PIPE_CACHE = {"path": None, "pipe": None}
VIDEO_PIPE_CACHE = {"path": None, "pipe": None}
generation_lock = threading.RLock()
MEDIA_STEPS = {"image": 25, "video": 25}
MEDIA_NOUN = {"image": "image", "video": "video"}

def _media_cache(kind):
    return IMAGE_PIPE_CACHE if kind == "image" else VIDEO_PIPE_CACHE

def _free_media_pipe(kind):
    """Drops a loaded image/video pipeline and gives its memory back (same
    idea as the chat-model switch fix: del + gc + empty the CUDA cache)."""
    with generation_lock:
        _media_cache(kind).update(path=None, pipe=None)
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _get_media_pipe(kind, instance_id=None):
    """-> (pipeline, None) or (None, plain-English reason it can't run)."""
    noun = MEDIA_NOUN[kind]
    if instance_id:
        with model_instances_lock:
            instance = MODEL_INSTANCES.get(instance_id)
        if not instance or instance["kind"] != kind or instance["status"] != "running":
            return None, f"The selected {noun} model instance is no longer running."
        if instance["pipeline"] is None:
            return None, f"The selected {noun} model instance is not ready."
        return instance["pipeline"], None
    with settings_lock:
        path = (SETTINGS.get("generation", {}).get(f"{kind}_model_path") or "").strip()
    if not path:
        return None, (f"No {noun} model is selected yet. An admin can pick one under Admin \u2192 Models \u2192 "
                      f"\u201cImage & video models\u201d (click \u201cUse for {noun}s\u201d).")
    if not Path(path).exists():
        return None, f"The selected {noun} model is missing from disk ({path}). Pick another one under Admin \u2192 Models."
    st = media_models.diffusers_status()
    if not media_models.is_pixelart_model(path) and not st["installed"]:
        return None, (f"{noun.capitalize()} generation needs the 'diffusers' package, which isn't installed. "
                      f"An admin can install it with: {media_models.INSTALL_HINT}")
    cache = _media_cache(kind)
    with generation_lock:
        if cache["path"] != path or cache["pipe"] is None:
            if cache["pipe"] is not None:
                _free_media_pipe(kind)
            try:
                logger.info(f"Loading {noun} generation model: {path}")
                pipe = media_models.load_pipeline(path, kind)
                cache.update(path=path, pipe=pipe)
                logger.info(f"{noun.capitalize()} model ready: {Path(path).name}")
            except Exception as e:
                logger.error(f"Couldn't load {noun} model {path}: {e}")
                return None, f"Couldn't load the {noun} model \u201c{Path(path).name}\u201d: {e}"
        return cache["pipe"], None

def _run_media_blocking(pipe, kind, prompt, instance_id=None):
    out = None
    with model_instances_lock:
        instance = MODEL_INSTANCES.get(instance_id) if instance_id else None
    lock = instance["generation_lock"] if instance else generation_lock
    with lock:
        if instance_id:
            with model_instances_lock:
                still_running = (MODEL_INSTANCES.get(instance_id) is instance
                                 and instance is not None and instance["status"] == "running")
            if not still_running:
                raise RuntimeError(f"The selected {kind} model instance has stopped.")
        for ev in media_models.run_with_progress(pipe, prompt, MEDIA_STEPS[kind]):
            if ev[0] == "result":
                out = ev[1]
    return out

def _save_generated_upload(identity: str, data: bytes, ext: str, is_image: bool, is_video: bool,
                           is_nsfw: bool | None = False) -> str:
    clean_name = f"generated{ext}"
    stamped = f"{int(time.time())}_{uuid.uuid4().hex[:6]}_{clean_name}"
    (account_upload_dir(identity) / stamped).write_bytes(data)
    with uploads_meta_lock:
        meta = load_uploads_meta(identity)
        meta[stamped] = {"original_name": clean_name, "is_image": is_image, "is_video": is_video,
                          "is_nsfw": is_nsfw, "uploaded_at": time.time()}
        if is_nsfw is None:
            meta[stamped]["nsfw_check_pending"] = True
        save_uploads_meta(identity, meta)
    return stamped

def _finish_generated_upload_safety_check(identity: str, filename: str, is_nsfw: bool):
    with uploads_meta_lock:
        meta = load_uploads_meta(identity)
        info = meta.get(filename)
        if info is None:
            raise RuntimeError("Generated image metadata was removed before its safety check completed.")
        if is_nsfw and is_guest_identity(identity):
            source = account_upload_dir(identity) / filename
            if not source.is_file():
                raise RuntimeError("Generated image file was removed before its safety check completed.")
            GUEST_NSFW_DIR.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(GUEST_NSFW_DIR / filename))
            info["guest_nsfw"] = True
        info["is_nsfw"] = is_nsfw
        info.pop("nsfw_check_pending", None)
        save_uploads_meta(identity, meta)

def _persist_generation_turn(identity: str, conv_id: str, prompt: str, attachment, kind_label: str = None):
    """Records generated media in an account chat or temporary guest chat."""
    attachments = attachment if isinstance(attachment, list) else [attachment]
    with conv_lock:
        if is_guest_identity(identity):
            conversations = get_session_state(identity).setdefault("conversations", {})
            if not conv_id or conv_id not in conversations:
                conv_id = new_guest_conversation(identity)
                conversations = get_session_state(identity)["conversations"]
            conv = conversations.get(conv_id)
        else:
            if not conv_id:
                conv_id = new_conversation(identity)
            data = load_user_data(identity)
            conv = data["conversations"].get(conv_id)
        if not conv:
            return None
        conv["messages"].append({"role": "user", "content": prompt})
        reply = {"role": "assistant", "content": ""}
        if len(attachments) == 1:
            reply["attachment"] = attachments[0]
        elif attachments:
            reply["attachments"] = attachments
        conv["messages"].append(reply)
        if conv["title"] == "New chat":
            kind = kind_label or ("Video" if attachments[0].get("is_video") else "Image")
            conv["title"] = f"{kind}: {prompt[:32]}"
        conv["updated_at"] = time.time()
        if not is_guest_identity(identity):
            save_user_data(identity, data)
    return conv_id

@app.route("/api/generate/image", methods=["POST"])
@login_required
def api_generate_image():
    identity = session["identity"]
    body = request.json or {}
    request_prompt = (body.get("prompt") or "").strip()
    prompt = request_prompt
    media_kind, parsed_prompt, parsed_count = _detect_media_request(prompt)
    if media_kind == "image":
        prompt = parsed_prompt
        image_count = parsed_count
    else:
        image_count, prompt = _image_count_and_prompt(prompt)
        prompt = _GENERIC_LEAD_RE.sub("", prompt).strip()
    conv_id = body.get("conv_id")
    image_instance_id = body.get("image_instance_id") or None
    if not prompt:
        return jsonify({"error": "Enter a description first."}), 400
    if image_count < 1:
        return jsonify({"error": "Request at least one image."}), 400
    with settings_lock:
        configured_path = (SETTINGS.get("generation", {}).get("image_model_path") or "").strip()
    requested_path = (body.get("image_model_path") or "").strip()
    if requested_path:
        selected_path = requested_path
    elif image_instance_id:
        with model_instances_lock:
            selected_instance = MODEL_INSTANCES.get(image_instance_id)
        if not selected_instance or selected_instance["kind"] != "image" or selected_instance["status"] != "running":
            return jsonify({"error": "The selected image model instance is no longer running."}), 400
        selected_path = selected_instance["path"]
    else:
        selected_path = configured_path
    if not selected_path or not _identity_can_access_model(identity, selected_path, "image"):
        return jsonify({"error": "You don't have access to the selected image model."}), 403
    leased_instance = None
    try:
        leased_instance = _lease_model_instance("image", selected_path)
        pipe, err = _get_media_pipe("image", leased_instance["id"])
        if err:
            return jsonify({"error": err}), 400
        image_data = []
        for _ in range(image_count):
            out = _run_media_blocking(pipe, "image", prompt, leased_instance["id"])
            data = media_models.output_to_png_bytes(out)
            stamped = _save_generated_upload(identity, data, ".png", is_image=True, is_video=False,
                                               is_nsfw=None)
            is_nsfw = media_models.image_is_nsfw(data, selected_path)
            _finish_generated_upload_safety_check(identity, stamped, is_nsfw)
            image_data.append((data, is_nsfw, stamped))
        attachments = []
        for index, (data, is_nsfw, stamped) in enumerate(image_data, 1):
            attachments.append({"filename": stamped, "original_name": f"generated-{index}.png",
                                "is_image": True, "is_video": False, "is_nsfw": is_nsfw,
                                "url": f"/files/{identity}/{stamped}", "generated_prompt": prompt})
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if leased_instance:
            _release_model_instance(leased_instance["id"])
    conv_id = _persist_generation_turn(identity, conv_id, request_prompt, attachments)
    return jsonify({"ok": True, "url": attachments[0]["url"], "filename": attachments[0]["filename"],
                    "attachments": attachments, "conv_id": conv_id})

@app.route("/api/generate/video", methods=["POST"])
@login_required
def api_generate_video():
    identity = session["identity"]
    body = request.json or {}
    prompt = (body.get("prompt") or "").strip()
    conv_id = body.get("conv_id")
    video_instance_id = body.get("video_instance_id") or None
    if not prompt:
        return jsonify({"error": "Enter a description first."}), 400
    with settings_lock:
        configured_path = (SETTINGS.get("generation", {}).get("video_model_path") or "").strip()
    requested_path = (body.get("video_model_path") or "").strip()
    if requested_path:
        selected_path = requested_path
    elif video_instance_id:
        with model_instances_lock:
            selected_instance = MODEL_INSTANCES.get(video_instance_id)
        if not selected_instance or selected_instance["kind"] != "video" or selected_instance["status"] != "running":
            return jsonify({"error": "The selected video model instance is no longer running."}), 400
        selected_path = selected_instance["path"]
    else:
        selected_path = configured_path
    if not selected_path or not _identity_can_access_model(identity, selected_path, "video"):
        return jsonify({"error": "You don't have access to the selected video model."}), 403
    leased_instance = None
    try:
        leased_instance = _lease_model_instance("video", selected_path)
        pipe, err = _get_media_pipe("video", leased_instance["id"])
        if err:
            return jsonify({"error": err}), 400
        out = _run_media_blocking(pipe, "video", prompt, leased_instance["id"])
        stamped = _save_generated_upload(identity, media_models.output_to_mp4_bytes(out, ROOT / "logs"), ".mp4",
                                           is_image=False, is_video=True)
    except Exception as e:
        logger.error(f"Video generation failed: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if leased_instance:
            _release_model_instance(leased_instance["id"])
    attachment = {"filename": stamped, "original_name": "generated.mp4", "is_image": False, "is_video": True,
                  "url": f"/files/{identity}/{stamped}"}
    conv_id = _persist_generation_turn(identity, conv_id, prompt, attachment, "Video")
    return jsonify({"ok": True, "url": attachment["url"], "filename": stamped, "conv_id": conv_id})

@app.route("/files/<username>/<path:filename>")
@login_required
def serve_uploaded_file(username, filename):
    identity = session["identity"]
    if identity != username and not session.get("is_admin"):
        return jsonify({"error": "forbidden"}), 403
    info = load_uploads_meta(username).get(filename, {})
    d = _upload_storage_dir(username, info)
    target = (d / filename).resolve()
    if not str(target).startswith(str(d.resolve())) or not target.is_file():
        return jsonify({"error": "not found"}), 404
    if info.get("nsfw_check_pending"):
        return jsonify({"error": "Image is unavailable until its safety check completes."}), 404
    # ?dl=1 forces a real download using the human-readable original name
    # instead of the timestamp/uuid-stamped on-disk name. HTML is ALWAYS
    # sent as a download, never rendered: files here include model-written
    # .html, and rendering one inline would run its scripts on this app's
    # own origin (with the viewer's login session) — not something a
    # generated file should ever get to do.
    ext = target.suffix.lower()
    force_download = request.args.get("dl") == "1" or ext in (".html", ".htm", ".svg", ".xml")
    kwargs = {}
    if force_download:
        original = info.get("original_name") or target.name
        kwargs = {"as_attachment": True, "download_name": original}
    resp = send_file(str(target), **kwargs)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp

@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    identity = session["identity"]
    is_guest = session.get("is_guest", False)
    policy = get_policy(identity)
    body = request.json or {}
    requested_instance_id = body.get("instance_id") or None
    requested_image_instance_id = body.get("image_instance_id") or None
    requested_text_model_path = (body.get("text_model_path") or "").strip()
    requested_image_model_path = (body.get("image_model_path") or "").strip()
    requested_video_model_path = (body.get("video_model_path") or "").strip()
    message = (body.get("message") or "").strip()
    attachment = body.get("attachment")
    if not message and not attachment:
        return jsonify({"error": "empty message"}), 400

    used = tokens_used_last_minute(identity)
    if used >= policy["tokens_per_min"]:
        return jsonify({"error": f"You've hit your {policy['tokens_per_min']} tokens/min limit. "
                                  f"Wait a moment — an admin can raise it in /admin."}), 429

    requested_max = int(body.get("max_tokens") or policy["max_tokens"])
    max_tokens = max(16, min(requested_max, policy["max_tokens"], args.hard_max_tokens))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.9))
    memory_turns = max(0, min(int(body.get("memory_turns", MEMORY_TURNS_DEFAULT)), MEMORY_TURNS_MAX))

    is_lookup, lookup_topic = _detect_lookup(message)
    if is_lookup and is_guest:
        return jsonify({"error": "Web lookups are only available to signed-in accounts. "
                                  "Create a free account to use /lookup."}), 403
    if is_lookup and not lookup_topic:
        return jsonify({"error": "Usage: /lookup <topic>"}), 400

    is_flashcards, fc_topic, fc_count = (False, None, 10) if is_lookup else _detect_flashcards(message)
    is_illustrated_story = (not is_lookup and not is_flashcards
                            and _detect_illustrated_story_request(message))
    media_kind, media_prompt, media_count = (
        (None, None, 1) if (is_lookup or is_flashcards or is_illustrated_story)
        else _detect_media_request(message)
    )
    if media_kind == "image" and media_count < 1:
        return jsonify({"error": "Request at least one image."}), 400
    is_file_request, file_ext = ((False, None) if
                                 (is_lookup or is_flashcards or media_kind or is_illustrated_story)
                                 else _detect_file_intent(message))
    if is_file_request and is_guest:
        return jsonify({"error": "Creating files is only available to signed-in accounts. Create a free account to do that."}), 403

    selected_model_path = None

    show_thinking = bool(body.get("show_thinking", True))

    # Text attachments get their content fed into the model's context for
    # this message. Image attachments are stored/shown but NOT actually
    # visible to the model — it's text-only. Say so rather than pretending.
    # Gated on the account's actual upload permission (not just "not a
    # guest") since a guest role can now be given upload access too.
    effective_message = message
    if attachment and not attachment.get("is_image") and upload_limit_for(identity) != 0:
        try:
            fpath = account_upload_dir(identity) / attachment["filename"]
            if fpath.exists():
                content = fpath.read_text(encoding="utf-8", errors="ignore")[:4000]
                effective_message = (f"{message}\n\n[Attached file: "
                                      f"{attachment.get('original_name', attachment['filename'])}]\n{content}")
        except Exception as e:
            logger.warning(f"Could not read attachment for {identity}: {e}")

    conv_id = body.get("conv_id")
    if is_guest:
        with conv_lock:
            conversations = get_session_state(identity).setdefault("conversations", {})
            if not conv_id or conv_id not in conversations:
                conv_id = new_guest_conversation(identity)
                conversations = get_session_state(identity)["conversations"]
            guest_conversation = conversations[conv_id]
            history = guest_conversation["messages"]
    else:
        with conv_lock:
            data = load_user_data(identity)
            if not conv_id or conv_id not in data["conversations"]:
                conv_id = new_conversation(identity)
                data = load_user_data(identity)
            history = data["conversations"][conv_id]["messages"]

    if not media_kind and not is_illustrated_story and re.fullmatch(r"(?:please\s+)?(?:try again|again|one more time|regenerate)[.! ]*",
                                       message, re.IGNORECASE):
        if len(history) >= 2:
            previous_user, previous_assistant = history[-2:]
            previous_attachments = previous_assistant.get("attachments")
            if not isinstance(previous_attachments, list):
                previous_attachments = []
            previous_attachment = previous_assistant.get("attachment") or (
                previous_attachments[0] if previous_attachments else {}
            )
            if (previous_user.get("role") == "user" and previous_assistant.get("role") == "assistant"
                    and previous_attachment.get("is_image")):
                retry_prompt = (previous_attachment.get("generated_prompt")
                                or previous_user.get("content", "").strip())
                if retry_prompt:
                    media_kind, media_prompt = "image", retry_prompt[:400]
                    media_count = max(1, len(previous_attachments)) if previous_attachments else 1

    if is_illustrated_story:
        with settings_lock:
            story_image_path = (SETTINGS.get("generation", {}).get("image_model_path") or "").strip()
        if requested_image_model_path:
            story_image_path = requested_image_model_path
        elif requested_image_instance_id:
            with model_instances_lock:
                selected_instance = MODEL_INSTANCES.get(requested_image_instance_id)
            if not selected_instance or selected_instance["kind"] != "image" or selected_instance["status"] != "running":
                return jsonify({"error": "The selected image model instance is no longer running."}), 400
            story_image_path = selected_instance["path"]
        if not story_image_path or not _identity_can_access_model(identity, story_image_path, "image"):
            return jsonify({"error": "Choose an image model you can access before requesting story illustrations."}), 403
    if media_kind:
        with settings_lock:
            media_path = (SETTINGS.get("generation", {}).get(f"{media_kind}_model_path") or "").strip()
        requested_media_path = (requested_image_model_path if media_kind == "image"
                                else requested_video_model_path if media_kind == "video" else "")
        if media_kind == "image" and requested_image_instance_id and not requested_media_path:
            with model_instances_lock:
                selected_instance = MODEL_INSTANCES.get(requested_image_instance_id)
            if not selected_instance or selected_instance["kind"] != "image" or selected_instance["status"] != "running":
                return jsonify({"error": "The selected image model instance is no longer running."}), 400
            media_path = selected_instance["path"]
        elif requested_media_path:
            media_path = requested_media_path
        if not media_path or not _identity_can_access_model(identity, media_path, media_kind):
            return jsonify({"error": f"You don't have access to the configured {media_kind} model."}), 403
        selected_model_path = media_path
    elif requested_text_model_path:
        if not _identity_can_access_model(identity, requested_text_model_path, "text"):
            return jsonify({"error": "You don't have access to that chat model."}), 403
        selected_model_path = requested_text_model_path
    elif requested_instance_id:
        with model_instances_lock:
            selected_instance = MODEL_INSTANCES.get(requested_instance_id)
        if not selected_instance or selected_instance["kind"] != "text":
            return jsonify({"error": "The selected text model instance is no longer running."}), 400
        if not _identity_can_access_model(identity, selected_instance["path"], "text"):
            return jsonify({"error": "You don't have access to that chat model."}), 403
        selected_model_path = selected_instance["path"]
    else:
        with settings_lock:
            configured_text_path = (SETTINGS.get("generation", {}).get("text_model_path") or "").strip()
        default_text_path = configured_text_path or ACTIVE_MODEL_PATH
        if not default_text_path or not _identity_can_access_model(identity, default_text_path, "text"):
            return jsonify({"error": "You don't have access to the server's default chat model."}), 403
        selected_model_path = default_text_path

    persisted_media_reply = None
    streamed_image_attachments = []

    def _persist_streamed_images(user_msg, image_attachments):
        nonlocal persisted_media_reply
        entry = {"role": "user", "content": user_msg}
        if attachment:
            entry["attachment"] = attachment
        reply_entry = {"role": "assistant", "content": "Generating images...",
                       "attachments": list(image_attachments)}
        with conv_lock:
            if persisted_media_reply is None:
                history.extend((entry, reply_entry))
                persisted_media_reply = reply_entry
            else:
                persisted_media_reply["attachments"] = list(image_attachments)
            if is_guest:
                guest_conversation = get_session_state(identity)["conversations"].get(conv_id)
                if guest_conversation:
                    if guest_conversation["title"] == "New chat":
                        guest_conversation["title"] = user_msg[:40]
                    guest_conversation["updated_at"] = time.time()
            else:
                conv = data["conversations"][conv_id]
                if conv["title"] == "New chat":
                    conv["title"] = user_msg[:40]
                conv["updated_at"] = time.time()
                save_user_data(identity, data)

    def _persist(user_msg, reply, n_tokens, reply_attachment=None, reply_extra=None):
        nonlocal persisted_media_reply
        record_tokens(identity, n_tokens)
        entry = {"role": "user", "content": user_msg}
        if attachment:
            entry["attachment"] = attachment
        reply_entry = {"role": "assistant", "content": reply}
        if reply_attachment:
            reply_entry["attachment"] = reply_attachment
        if reply_extra:
            reply_entry.update(reply_extra)
        if persisted_media_reply is not None:
            with conv_lock:
                persisted_media_reply.update(reply_entry)
                if reply_extra and "attachments" in reply_extra:
                    persisted_media_reply.pop("attachment", None)
                if is_guest:
                    guest_conversation = get_session_state(identity)["conversations"].get(conv_id)
                    if guest_conversation:
                        guest_conversation["updated_at"] = time.time()
                else:
                    data["conversations"][conv_id]["updated_at"] = time.time()
                    save_user_data(identity, data)
            return
        if is_guest:
            with conv_lock:
                history.extend((entry, reply_entry))
                guest_conversation = get_session_state(identity)["conversations"].get(conv_id)
                if guest_conversation:
                    if guest_conversation["title"] == "New chat":
                        guest_conversation["title"] = (user_msg or attachment.get("original_name", "Attachment"))[:40]
                    guest_conversation["updated_at"] = time.time()
        else:
            with conv_lock:
                history.extend((entry, reply_entry))
                conv = data["conversations"][conv_id]
                if conv["title"] == "New chat":
                    conv["title"] = (user_msg or attachment.get("original_name", "Attachment"))[:40]
                conv["updated_at"] = time.time()
                save_user_data(identity, data)

    def stream():
        request_id = uuid.uuid4().hex
        stop_event = threading.Event()
        leased_instance_id = None
        with active_generations_lock:
            ACTIVE_GENERATIONS[request_id] = {"stop_event": stop_event, "peer": None}
        yield json.dumps({"request_id": request_id}) + "\n"
        leased_instance = None
        story_image_instance = None
        story_reply_parts = []
        story_attachments = []
        story_thinking = []
        story_image_count = 0
        try:
            # Peer dispatch is decided BEFORE touching our own local lock —
            # holding gen_lock while waiting on a remote machine would
            # serialize every request through our own queue regardless of
            # where the actual compute happens, which defeats the entire
            # point of spreading load across machines. Local generation
            # (including /lookup, which stays local-only — it's cheap and
            # tied to this machine's own web-search step) is the only path
            # that acquires gen_lock, and only right when it's about to use it.
            peer = None if ((requested_instance_id and requested_instance_id != "default")
                            or requested_text_model_path or requested_image_model_path or requested_video_model_path
                            or is_guest or is_lookup or is_file_request or is_flashcards or media_kind) else _pick_generation_peer(_classify_prompt_type(effective_message or message))

            if peer:
                with active_generations_lock:
                    ACTIVE_GENERATIONS[request_id]["peer"] = peer
                try:
                    logger.info(f"Routing this reply to peer '{peer.get('name', peer['url'])}' (less busy than local right now)")
                    n_tokens, parts, reply = 0, [], None
                    for line in _relay_from_peer(peer, effective_message, history, max_tokens, temperature, top_p,
                                                  memory_turns, request_id=request_id, show_thinking=show_thinking):
                        yield line + "\n"
                        try:
                            obj = json.loads(line)
                            if "delta" in obj:
                                parts.append(obj["delta"]); n_tokens += 1
                            if obj.get("done"):
                                reply = obj.get("reply") or _clean_reply("".join(parts)) or "I'm not sure how to answer that. Could you rephrase?"
                        except (ValueError, KeyError):
                            pass
                    if reply is not None:
                        _persist(message, reply, n_tokens)
                    return
                except Exception as e:
                    logger.warning(f"Peer '{peer.get('name', peer['url'])}' failed mid-dispatch, falling back to local: {e}")
                    # Falls through to local generation below. Peer failures
                    # surface fast (connection refused/timeout) almost always
                    # before any real token deltas reached the browser, so
                    # restarting locally here doesn't produce visible duplication.

            instance_kind = media_kind or "text"
            try:
                leased_instance = _lease_model_instance(instance_kind, selected_model_path)
                leased_instance_id = leased_instance["id"]
            except Exception as error:
                logger.exception(f"Could not get a {instance_kind} model instance for {identity}")
                reply = f"Could not start the selected {instance_kind} model: {error}"
                _persist(message, reply, 0)
                yield json.dumps({"delta": reply}) + "\n"
                yield json.dumps({"done": True, "reply": reply, "conv_id": conv_id}) + "\n"
                return
            try:
                if media_kind:
                    noun = MEDIA_NOUN[media_kind]
                    yield json.dumps({"tool_call": {"type": f"generate_{noun}", "query": media_prompt}}) + "\n"
                    yield json.dumps({"media_status": f"Loading the {noun} model\u2026 (the first one after a restart takes a while)"}) + "\n"
                    pipe, err = _get_media_pipe(media_kind, leased_instance_id)
                    if err:
                        _persist(message, err, 0)
                        yield json.dumps({"delta": err}) + "\n"
                        yield json.dumps({"done": True, "reply": err, "conv_id": conv_id}) + "\n"
                        return
                    reply_attachments = streamed_image_attachments
                    out = None
                    with model_instances_lock:
                        image_instance = MODEL_INSTANCES.get(leased_instance_id) if leased_instance_id else None
                    media_lock = image_instance["generation_lock"] if image_instance else generation_lock
                    with media_lock:
                        with model_instances_lock:
                            still_running = (MODEL_INSTANCES.get(leased_instance_id) is image_instance
                                             and image_instance["status"] == "running") if leased_instance_id else True
                        if not still_running:
                            raise RuntimeError("The selected image model instance has stopped.")
                        generation_count = media_count if media_kind == "image" else 1
                        for image_index in range(generation_count):
                            if media_kind == "image" and generation_count > 1:
                                status = f"Generating image {image_index + 1} of {generation_count}\u2026"
                            else:
                                status = f"Generating the {noun}\u2026"
                            yield json.dumps({"media_status": status}) + "\n"
                            for ev in media_models.run_with_progress(pipe, media_prompt, MEDIA_STEPS[media_kind], stop_event):
                                if ev[0] == "progress":
                                    yield json.dumps({"media_progress": {"step": ev[1], "total": ev[2]}}) + "\n"
                                else:
                                    out = ev[1]
                            if stop_event.is_set():
                                break
                            if media_kind == "image":
                                image_data = media_models.output_to_png_bytes(out)
                                stamped = _save_generated_upload(identity, image_data, ".png",
                                                                   is_image=True, is_video=False, is_nsfw=None)
                                yield json.dumps({"media_status": "Checking image for NSFW content\u2026"}) + "\n"
                                is_nsfw = media_models.image_is_nsfw(image_data, selected_model_path)
                                _finish_generated_upload_safety_check(identity, stamped, is_nsfw)
                                image_attachment = {
                                    "filename": stamped, "original_name": f"generated-{image_index + 1}.png",
                                    "is_image": True, "is_video": False, "is_nsfw": is_nsfw,
                                    "url": f"/files/{identity}/{stamped}", "generated_prompt": media_prompt,
                                }
                                reply_attachments.append(image_attachment)
                                _persist_streamed_images(message, reply_attachments)
                                yield json.dumps({"image_attachment": image_attachment}) + "\n"
                    if stop_event.is_set():
                        media_label = "images" if media_kind == "image" and media_count > 1 else noun
                        stopped_verb = "weren't" if media_label == "images" else "wasn't"
                        reply_text = f"Stopped \u2014 the {media_label} {stopped_verb} finished."
                        if reply_attachments:
                            _persist(message, reply_text, 0, reply_extra={"attachments": reply_attachments})
                        else:
                            _persist(message, reply_text, 0)
                        yield json.dumps({"delta": reply_text}) + "\n"
                        yield json.dumps({"done": True, "reply": reply_text, "conv_id": conv_id,
                                          "attachments": reply_attachments}) + "\n"
                        return
                    if media_kind == "image":
                        if not reply_attachments:
                            raise RuntimeError("Image generation finished without producing an image.")
                    else:
                        stamped = _save_generated_upload(identity, media_models.output_to_mp4_bytes(out, ROOT / "logs"), ".mp4",
                                                           is_image=False, is_video=True)
                        reply_attachments = [{"filename": stamped, "original_name": "generated.mp4",
                                              "is_image": False, "is_video": True,
                                              "url": f"/files/{identity}/{stamped}"}]
                    reply_text = (f"Here are your {media_count} images." if media_kind == "image" and media_count > 1
                                  else f"Here's your {noun}.")
                    if persisted_media_reply is not None:
                        _persist(message, reply_text, 0, reply_extra={"attachments": reply_attachments})
                    elif len(reply_attachments) == 1:
                        _persist(message, reply_text, 0, reply_attachment=reply_attachments[0])
                    else:
                        _persist(message, reply_text, 0, reply_extra={"attachments": reply_attachments})
                    yield json.dumps({"delta": reply_text}) + "\n"
                    result_payload = {"done": True, "reply": reply_text, "conv_id": conv_id}
                    result_payload["attachment" if len(reply_attachments) == 1 else "attachments"] = (
                        reply_attachments[0] if len(reply_attachments) == 1 else reply_attachments
                    )
                    yield json.dumps(result_payload) + "\n"
                    return

                if is_flashcards:
                    yield json.dumps({"tool_call": {"type": "flashcards", "query": fc_topic}}) + "\n"
                    fc_prompt = (
                        f"Create exactly {fc_count} study flash cards about: {fc_topic}\n\n"
                        f"Output ONE card per line in exactly this format:\nterm :: short definition or answer\n\n"
                        f"Rules: no numbering, no bullets, no headings, no intro or closing text — only the card lines. "
                        f"Keep every definition to one or two sentences."
                    )
                    parts = []
                    for piece in _generate_pieces(fc_prompt, [], min(max_tokens, 2048), 0.5, 0.9,
                                                  stop_event=stop_event, instance_id=leased_instance_id):
                        parts.append(piece)
                    cards = _parse_flashcards("".join(parts))[:fc_count]
                    if len(cards) < 2:
                        yield json.dumps({"done": True, "error": "I couldn't turn that into flash cards — try naming the topic more specifically."}) + "\n"
                        return
                    deck = {"topic": fc_topic, "cards": cards}
                    reply_text = f"Here are {len(cards)} flash cards on {fc_topic}."
                    _persist(message, reply_text, 0, reply_extra={"flashcards": deck})
                    yield json.dumps({"flashcards": deck}) + "\n"
                    yield json.dumps({"delta": reply_text}) + "\n"
                    yield json.dumps({"done": True, "reply": reply_text, "conv_id": conv_id}) + "\n"
                    return

                if is_file_request:
                    yield json.dumps({"tool_call": {"type": "create_file", "query": file_ext}}) + "\n"
                    file_prompt = (
                        f"Write ONLY the raw contents of a {file_ext} file for this request — no explanation, "
                        f"no markdown code fences, no commentary before or after, just the file's actual content "
                        f"from the first character:\n\n{message}"
                    )
                    # Stream the file as it's written so the browser can show the
                    # code appearing live, instead of nothing until it's all done.
                    yield json.dumps({"file_start": {"ext": file_ext, "name": f"generated{file_ext}"}}) + "\n"
                    parts = []
                    for piece in _generate_pieces(file_prompt, [], min(max_tokens, 2048), 0.6, 0.9,
                                                   stop_event=stop_event, instance_id=leased_instance_id):
                        parts.append(piece)
                        yield json.dumps({"file_delta": piece}) + "\n"
                    content = _clean_reply("".join(parts))
                    fence_match = re.match(r"^```[a-zA-Z0-9]*\n(.*)\n```\s*$", content, re.DOTALL)
                    if fence_match:
                        content = fence_match.group(1)
                    if not content.strip():
                        yield json.dumps({"done": True, "error": "Couldn't generate that file — try rephrasing the request."}) + "\n"
                        return
                    stamped = _save_generated_upload(identity, content.encode("utf-8"), file_ext,
                                                       is_image=False, is_video=False)
                    reply_attachment = {"filename": stamped, "original_name": f"generated{file_ext}",
                                         "is_image": False, "is_video": False,
                                         "url": f"/files/{identity}/{stamped}"}
                    if stop_event.is_set():
                        reply_text = f"Stopped early \u2014 here's the partial {file_ext} file."
                    else:
                        reply_text = f"Here's the {file_ext} file you asked for."
                    _persist(message, reply_text, 0, reply_attachment=reply_attachment)
                    yield json.dumps({"delta": reply_text}) + "\n"
                    # file_content = exactly what was saved (fences stripped), so the
                    # live view can swap to the final text.
                    yield json.dumps({"done": True, "reply": reply_text, "conv_id": conv_id,
                                       "attachment": reply_attachment, "file_content": content}) + "\n"
                    return

                if is_lookup:
                    with settings_lock:
                        search_cfg = dict(SETTINGS.get("search", DEFAULT_SETTINGS["search"]))
                    engine = search_cfg.get("engine", "duckduckgo")
                    max_sites = max(1, min(int(search_cfg.get("max_sites", 4)), 10))
                    yield json.dumps({"tool_call": {"type": "web_search", "query": lookup_topic}}) + "\n"
                    try:
                        from search_engine import search
                        from webscraper import fetch_page, make_session
                        engine_used = {"name": None}
                        engine_attempts = []
                        def _note_engine(name, err):
                            engine_attempts.append((name, err))
                            if err is None and engine_used["name"] is None:
                                engine_used["name"] = name
                        results = search(lookup_topic, engine=engine, n=max(max_sites, 6), on_engine_used=_note_engine)
                        if not results:
                            details = "; ".join(f"{name}: {error}" for name, error in engine_attempts)
                            lookup_error = "No search results found for that topic."
                            if details:
                                lookup_error += f" Search engines tried: {details}"
                            yield json.dumps({"done": True, "error": lookup_error}) + "\n"
                            return
                        http = make_session()
                        texts, used = [], []
                        for r in results[:max_sites]:
                            page = fetch_page(http, r["url"])
                            if page:
                                texts.append(f"Source: {page['title']} ({page['url']})\n{page['text'][:1500]}")
                                used.append(r)
                        if not texts:
                            yield json.dumps({"done": True, "error": "Found results but couldn't fetch any pages."}) + "\n"
                            return
                        result_payload = [{"title": r.get("title") or r["url"], "url": r["url"],
                                            "domain": urllib.parse.urlparse(r["url"]).netloc} for r in used]
                        yield json.dumps({"tool_result": {"type": "web_search", "query": lookup_topic,
                                                            "engine": engine_used["name"] or engine,
                                                            "results": result_payload}}) + "\n"
                        combined = "\n\n---\n\n".join(texts)[:6000]
                        summarize_prompt = (
                            f"Summarize the following web search results about '{lookup_topic}' into a clear, "
                            f"well-organized answer. Mention which source each key fact came from by title. "
                            f"Be concise.\n\n{combined}"
                        )
                        n_tokens, parts = 0, []
                        gen = _generate_pieces(summarize_prompt, [], max_tokens, temperature, top_p,
                                                system_prompt=_system_prompt_for(show_thinking), stop_event=stop_event,
                                                instance_id=leased_instance_id)
                        for kind, text in _split_thinking_stream(gen):
                            if kind == "thinking":
                                yield json.dumps({"thinking_delta": text}) + "\n"
                            else:
                                parts.append(text); n_tokens += 1
                                yield json.dumps({"delta": text}) + "\n"
                        reply = _clean_reply("".join(parts)) or "Couldn't produce a summary from those sources."
                        _persist(message, reply, n_tokens)
                        yield json.dumps({"done": True, "reply": reply, "conv_id": conv_id}) + "\n"
                    except Exception as e:
                        logger.error(f"Lookup failed for {identity}: {e}")
                        yield json.dumps({"done": True, "error": f"Lookup failed: {e}"}) + "\n"
                    return

                if is_illustrated_story:
                    story_system_prompt = _system_prompt_for(show_thinking) + (
                        "\nWhen the user asks for a story with images, write the complete story as your answer. "
                        "At up to 25 important scene breaks, insert one image request using exactly this format: "
                        "[[IMAGE_PROMPT]]a concise, self-contained visual description of that scene[[/IMAGE_PROMPT]]. "
                        "Place each marker on its own line where its illustration belongs. Make the prompt describe "
                        "only visual content and avoid text, captions, or lettering in the image. Never explain, quote, "
                        "or otherwise reveal these markers or their prompt text in the story. Include at least one "
                        "illustration, then choose as many more as the story benefits from, up to 25 total; do not "
                        "force extra images."
                    )
                    n_tokens = 0
                    gen = _generate_pieces(
                        effective_message, history, max_tokens, temperature, top_p,
                        system_prompt=story_system_prompt, memory_turns=memory_turns,
                        stop_event=stop_event, instance_id=leased_instance_id,
                    )
                    for event_kind, event_text in _story_reply_events(
                            _split_thinking_stream(gen), 25):
                        if event_kind == "thinking":
                            if not story_thinking or not story_thinking[-1].get("_active"):
                                thinking_index = len(story_thinking) + 1
                                story_thinking.append({"index": thinking_index, "text": "", "_active": True})
                                story_reply_parts.append(f"[[STORY_THINKING_{thinking_index}]]")
                            thought = story_thinking[-1]
                            thought["text"] += event_text
                            yield json.dumps({"story_thinking_delta": {
                                "index": thought["index"], "text": event_text
                            }}) + "\n"
                            continue
                        if story_thinking and story_thinking[-1].get("_active"):
                            story_thinking[-1].pop("_active", None)
                        if event_kind == "text":
                            story_reply_parts.append(event_text)
                            n_tokens += 1
                            yield json.dumps({"delta": event_text}) + "\n"
                            continue
                        if stop_event.is_set():
                            break

                        story_image_count += 1
                        marker = f"[[STORY_IMAGE_{story_image_count}]]"
                        story_reply_parts.append(marker)
                        yield json.dumps({"delta": marker}) + "\n"
                        yield json.dumps({"tool_call": {
                            "type": "generate_image", "query": "Story illustration"
                        }}) + "\n"
                        yield json.dumps({"media_status":
                                          f"Preparing story illustration {story_image_count}\u2026"}) + "\n"
                        if stop_event.is_set():
                            story_reply_parts[-1] = "(Illustration skipped because generation was stopped.)"
                            yield json.dumps({"story_image_failure": {
                                "index": story_image_count,
                                "message": story_reply_parts[-1],
                            }}) + "\n"
                            break
                        try:
                            if story_image_instance is None:
                                story_image_instance = _lease_model_instance("image", story_image_path)
                            image_pipe, image_error = _get_media_pipe("image", story_image_instance["id"])
                            if image_error:
                                raise RuntimeError(image_error)
                            yield json.dumps({"media_status":
                                              f"Generating story illustration {story_image_count}\u2026"}) + "\n"
                            image_out = _run_media_blocking(
                                image_pipe, "image", event_text, story_image_instance["id"]
                            )
                            if stop_event.is_set():
                                story_reply_parts[-1] = "(Illustration skipped because generation was stopped.)"
                                yield json.dumps({"story_image_failure": {
                                    "index": story_image_count,
                                    "message": story_reply_parts[-1],
                                }}) + "\n"
                                break
                            image_data = media_models.output_to_png_bytes(image_out)
                            is_nsfw = media_models.image_is_nsfw(image_data, story_image_path)
                            stamped = _save_generated_upload(
                                identity, image_data, ".png", is_image=True, is_video=False,
                                is_nsfw=is_nsfw,
                            )
                            _finish_generated_upload_safety_check(identity, stamped, is_nsfw)
                            image_attachment = {
                                "filename": stamped,
                                "original_name": f"story-illustration-{story_image_count}.png",
                                "is_image": True, "is_video": False, "is_nsfw": is_nsfw,
                                "story_image_index": story_image_count,
                                "url": f"/files/{identity}/{stamped}",
                            }
                            story_attachments.append(image_attachment)
                            yield json.dumps({"image_attachment": image_attachment}) + "\n"
                        except Exception as image_error:
                            logger.warning(
                                f"Story illustration {story_image_count} failed for {identity}: {image_error}"
                            )
                            failure_message = "(Illustration could not be generated.)"
                            story_reply_parts[-1] = failure_message
                            yield json.dumps({"story_image_failure": {
                                "index": story_image_count,
                                "message": failure_message,
                            }}) + "\n"

                    reply = _clean_reply("".join(story_reply_parts))
                    if stop_event.is_set() and reply:
                        reply += "\n\n[stopped]"
                    if not reply:
                        reply = "I couldn't write the story. Please try again."
                    persisted_thinking = [
                        {"index": item["index"], "text": item["text"]}
                        for item in story_thinking
                    ]
                    reply_extra = {}
                    if story_attachments:
                        reply_extra["attachments"] = story_attachments
                    if persisted_thinking:
                        reply_extra["story_thinking"] = persisted_thinking
                    _persist(message, reply, n_tokens, reply_extra=reply_extra)
                    result_payload = {"done": True, "reply": reply, "conv_id": conv_id}
                    if story_attachments:
                        result_payload["attachments"] = story_attachments
                    if persisted_thinking:
                        result_payload["story_thinking"] = persisted_thinking
                    yield json.dumps(result_payload) + "\n"
                    return

                n_tokens, parts = 0, []
                gen = _generate_pieces(effective_message, history, max_tokens, temperature, top_p,
                                        system_prompt=_system_prompt_for(show_thinking),
                                        memory_turns=memory_turns, stop_event=stop_event,
                                        instance_id=leased_instance_id)
                for kind, text in _split_thinking_stream(gen):
                    if kind == "thinking":
                        yield json.dumps({"thinking_delta": text}) + "\n"
                    else:
                        parts.append(text); n_tokens += 1
                        yield json.dumps({"delta": text}) + "\n"
                reply = _clean_reply("".join(parts)) or "I'm not sure how to answer that. Could you rephrase?"
                if stop_event.is_set() and reply:
                    reply += "\n\n[stopped]"
                _persist(message, reply, n_tokens)
                yield json.dumps({"done": True, "reply": reply, "conv_id": conv_id}) + "\n"
            except Exception as e:
                logger.error(f"Generation error for {identity}: {e}")
                if media_kind == "image" and persisted_media_reply is not None:
                    partial_reply = f"Image generation failed after {len(streamed_image_attachments)} image(s): {e}"
                    _persist(message, partial_reply, 0, reply_extra={"attachments": streamed_image_attachments})
                elif is_illustrated_story and story_reply_parts:
                    partial_reply = _clean_reply("".join(story_reply_parts))
                    partial_reply += f"\n\n(Illustration generation stopped: {e})"
                    reply_extra = {}
                    if story_attachments:
                        reply_extra["attachments"] = story_attachments
                    if story_thinking:
                        reply_extra["story_thinking"] = [
                            {"index": item["index"], "text": item["text"]}
                            for item in story_thinking
                        ]
                    _persist(message, partial_reply, 0, reply_extra=reply_extra)
                yield json.dumps({"done": True, "error": str(e)}) + "\n"
            finally:
                if story_image_instance:
                    _release_model_instance(story_image_instance["id"])
                if leased_instance:
                    _release_model_instance(leased_instance["id"])
        finally:
            with active_generations_lock:
                ACTIVE_GENERATIONS.pop(request_id, None)

    return Response(stream(), mimetype="application/x-ndjson")

@app.route("/api/chat/stop", methods=["POST"])
@login_required
def api_chat_stop():
    """Cancels an in-progress generation by request_id (handed to the
    browser as the first line of /api/chat's stream). Works whether the
    generation ended up running locally or got routed to a paired peer —
    in the peer case this just forwards the same stop to wherever the
    work actually landed."""
    request_id = (request.json or {}).get("request_id")
    if not request_id:
        return jsonify({"error": "request_id required"}), 400
    with active_generations_lock:
        entry = ACTIVE_GENERATIONS.get(request_id)
    if not entry:
        return jsonify({"ok": True, "note": "already finished"})
    peer = entry.get("peer")
    if peer:
        try:
            requests.post(f"{peer['url']}/internal/generate/stop", headers=_peer_headers(peer),
                           json={"request_id": request_id}, timeout=3)
        except Exception as e:
            logger.warning(f"Could not forward stop to peer '{peer.get('name', peer['url'])}': {e}")
    entry["stop_event"].set()
    return jsonify({"ok": True})

@app.route("/api/queue_depth")
@login_required
def api_queue_depth():
    return jsonify({"waiting": gen_lock.queue_depth()})

@app.route("/api/hw_stats")
@admin_required
def api_hw_stats():
    if hardware_scan is None:
        return jsonify({"error": "hardware_scan.py not available"}), 500
    return jsonify(hardware_scan.get_runtime_stats())

# ══════════════════════════════════════════════════════════════════════
#  ROUTES — admin dashboard
# ══════════════════════════════════════════════════════════════════════

@app.route("/admin")
@admin_required
def admin_panel():
    with sessions_lock:
        snapshot = {identity: dict(s) for identity, s in SESSIONS.items()}
    active = {
        identity: {
            "name": get_policy(identity)["name"],
            "connected_at": s["connected_at"],
            "last_seen": s["last_seen"],
            "tokens_last_min": tokens_used_last_minute(identity),
        }
        for identity, s in snapshot.items()
    }
    with policy_lock:
        policies = dict(POLICIES)
    with users_lock:
        accounts = {u: {"created_at": v.get("created_at"), "role": v.get("role", "standard")} for u, v in USERS.items()}
    return render_template("admin.html", active=active, policies=policies, accounts=accounts,
                            hardware=HARDWARE, model_path=ACTIVE_MODEL_PATH or resolve_model_path(),
                            hard_cap=args.hard_max_tokens, allow_registration=ALLOW_SELF_REGISTRATION,
                            settings=SETTINGS)

@app.route("/admin/settings/upload_limits", methods=["POST"])
@admin_required
def admin_set_upload_limits():
    body = request.json or {}
    with settings_lock:
        limits = SETTINGS.setdefault("upload_limits", {})
        for role in ("guest", "user", "admin"):
            if role in body:
                try:
                    v = int(body[role])
                except (TypeError, ValueError):
                    return jsonify({"error": f"'{role}' limit must be a whole number."}), 400
                if v < -1:
                    return jsonify({"error": f"'{role}' limit must be -1 (unlimited), 0 (disabled), or a positive count."}), 400
                limits[role] = v
        save_settings(SETTINGS)
    return jsonify({"ok": True, "upload_limits": SETTINGS["upload_limits"]})

@app.route("/admin/settings/generation", methods=["POST"])
@admin_required
def admin_set_generation_settings():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Expected generation settings as a JSON object."}), 400
    if "text_model_path" in body:
        text_path = body["text_model_path"]
        if not isinstance(text_path, str):
            return jsonify({"error": "The default text model path must be a string."}), 400
        text_path = text_path.strip()
        available_text_paths = {entry["path"] for entry in _scan_chat_models()}
        if ACTIVE_MODEL_PATH:
            available_text_paths.add(ACTIVE_MODEL_PATH)
        if text_path and text_path not in available_text_paths:
            return jsonify({"error": "The default text model must be installed on this server."}), 400
    if "image_model_path" in body and not isinstance(body["image_model_path"], str):
        return jsonify({"error": "The default image model path must be a string."}), 400
    if "video_model_path" in body and not isinstance(body["video_model_path"], str):
        return jsonify({"error": "The default video model path must be a string."}), 400
    with settings_lock:
        gen = SETTINGS.setdefault("generation", {})
        if "text_model_path" in body:
            gen["text_model_path"] = text_path
        if "image_model_path" in body:
            gen["image_model_path"] = str(body["image_model_path"]).strip()
        if "video_model_path" in body:
            gen["video_model_path"] = str(body["video_model_path"]).strip()
        save_settings(SETTINGS)
    # Model paths changed — drop any cached pipeline so the next generation
    # request picks up the new path instead of quietly reusing the old one.
    _free_media_pipe("image")
    _free_media_pipe("video")
    return jsonify({"ok": True, "generation": SETTINGS["generation"]})

# ── Image & video model selector (Admin -> Models) ───────────────────────────
def _norm_path(p):
    return os.path.normcase(os.path.realpath(str(p)))

def _scan_media():
    with settings_lock:
        gen = dict(SETTINGS.get("generation", {}))
    selected = {"image": (gen.get("image_model_path") or "").strip() or None,
                "video": (gen.get("video_model_path") or "").strip() or None}
    extras = [e for e in gen.get("extra_paths", []) if e]
    entries = media_models.scan_media_models(REGISTRY_MODELS_ROOT, extras, selected)
    extra_norm = {_norm_path(e) for e in extras}
    for e in entries:
        np_ = _norm_path(e["path"])
        e["active_for"] = [k for k, v in selected.items() if v and _norm_path(v) == np_]
        e["extra"] = np_ in extra_norm
    return entries

def _scan_chat_models():
    """Return locally installed transformer models, excluding diffusers components."""
    roots = [REGISTRY_MODELS_ROOT, MODEL_CACHE]
    media_roots = set()
    for root in roots:
        if root.exists():
            media_roots.update(mi.parent.resolve() for mi in root.rglob("model_index.json"))
            media_roots.update(cfg.parent.resolve() for cfg in root.rglob("config.json")
                               if media_models.is_pixelart_model(cfg.parent))

    models = {}
    for root in roots:
        if not root.exists():
            continue
        for config in root.rglob("config.json"):
            folder = config.parent.resolve()
            if any(part.casefold() == "_openvino" for part in folder.parts):
                continue
            if any(media_root == folder or media_root in folder.parents for media_root in media_roots):
                continue
            has_weights = any(folder.glob("*.safetensors")) or any(folder.glob("*.bin"))
            has_sharded_weights = ((folder / "model.safetensors.index.json").exists()
                                   or (folder / "pytorch_model.bin.index.json").exists())
            if not has_weights and not has_sharded_weights:
                continue
            models[str(folder)] = {"path": str(folder), "name": folder.name}
    return sorted(models.values(), key=lambda entry: entry["name"].casefold())

@app.route("/api/model-instances")
@login_required
def api_model_instances():
    identity = session["identity"]
    can_manage = is_admin_account(identity)
    with model_instances_lock:
        instances = [_instance_summary(entry) for entry in MODEL_INSTANCES.values()
                     if can_manage or _identity_can_access_model(identity, entry["path"], entry["kind"])]
    text_models = [entry for entry in _scan_chat_models()
                   if can_manage or _identity_can_access_model(identity, entry["path"], "text")]
    media_entries = [entry for entry in _scan_media() if entry["usable"]]
    if not can_manage:
        media_entries = [entry for entry in media_entries
                         if _identity_can_access_model(identity, entry["path"], entry["kind"])]
    image_models = [entry for entry in media_entries if entry["kind"] == "image"]
    video_models = [entry for entry in media_entries if entry["kind"] == "video"]
    with settings_lock:
        generation = SETTINGS.get("generation", {})
        configured_text_model = (generation.get("text_model_path") or "").strip()
        configured_media = {kind: (generation.get(f"{kind}_model_path") or "").strip()
                            for kind in ("image", "video")}
    if configured_text_model and not (can_manage or _identity_can_access_model(identity, configured_text_model, "text")):
        configured_text_model = ""
    configured_media = {kind: path for kind, path in configured_media.items()
                        if path and (can_manage or _identity_can_access_model(identity, path, kind))}
    return jsonify({"instances": instances, "text_models": text_models, "image_models": image_models,
                    "video_models": video_models, "configured_text_model": configured_text_model,
                    "configured_media": configured_media,
                    "gpu_available": bool(HARDWARE.get("cuda") and torch.cuda.is_available()),
                    "gpu_name": HARDWARE.get("device_name") if HARDWARE.get("cuda") else None})

@app.route("/api/model-instances/start", methods=["POST"])
@admin_required
def api_start_model_instance():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Expected model instance settings as a JSON object."}), 400
    kind, path, device = body.get("kind"), body.get("path"), body.get("device")
    if not all(isinstance(value, str) and value.strip() for value in (kind, path, device)):
        return jsonify({"error": "Choose a model, model type, and CPU, GPU, or automatic placement."}), 400
    if kind not in ("text", "image", "video"):
        return jsonify({"error": "Unsupported model type."}), 400
    if not _identity_can_access_model(session["identity"], path, kind):
        return jsonify({"error": f"You don't have access to that {kind} model."}), 403
    try:
        instance = _start_model_instance(kind, path, device)
        return jsonify({"ok": True, "instance": instance})
    except (ValueError, RuntimeError) as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Could not start model instance")
        return jsonify({"error": f"Could not start model instance: {e}"}), 500

@app.route("/api/model-instances/<instance_id>/stop", methods=["POST"])
@admin_required
def api_stop_model_instance(instance_id):
    try:
        _stop_model_instance(instance_id)
        return jsonify({"ok": True})
    except KeyError as e:
        return jsonify({"error": str(e).strip("'")}), 404
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 409
    except Exception as e:
        logger.exception(f"Could not stop model instance {instance_id}")
        return jsonify({"error": f"Could not stop model instance: {e}"}), 500

@app.route("/admin/model_switching/list")
@admin_required
def admin_model_switching_list():
    models = [{"kind": "text", "usable": True, **entry} for entry in _scan_chat_models()]
    models.extend({"kind": entry["kind"], **entry} for entry in _scan_media())
    if ACTIVE_MODEL_PATH and not any(entry["path"] == ACTIVE_MODEL_PATH for entry in models):
        models.insert(0, {"kind": "text", "usable": True, "path": ACTIVE_MODEL_PATH,
                          "name": Path(ACTIVE_MODEL_PATH).name or ACTIVE_MODEL_PATH})
    for entry in models:
        entry["roles"] = list(_model_access_roles(entry["path"], entry["kind"]))
    with model_instances_lock:
        instances = [_instance_summary(entry) for entry in MODEL_INSTANCES.values()]
    return jsonify({"models": models, "instances": instances,
                    "gpu_available": bool(HARDWARE.get("cuda") and torch.cuda.is_available())})

@app.route("/admin/settings/model_switching", methods=["POST"])
@admin_required
def admin_set_model_switching():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Expected model switching settings as a JSON object."}), 400
    enabled = body.get("enabled")
    paths = body.get("allowed_paths")
    if not isinstance(enabled, bool):
        return jsonify({"error": "enabled must be true or false"}), 400
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        return jsonify({"error": "allowed_paths must be a list of model paths"}), 400
    available = {entry["path"] for entry in _scan_chat_models()}
    if any(path not in available for path in paths):
        return jsonify({"error": "Only locally installed text models can be enabled for user switching."}), 400
    with settings_lock:
        SETTINGS["model_switching"] = {"enabled": enabled, "allowed_paths": list(dict.fromkeys(paths))}
        model_access = SETTINGS.setdefault("model_access", {})
        for model in _scan_chat_models():
            model_access[model["path"]] = ["administrator", "standard"] if enabled and model["path"] in paths else []
        save_settings(SETTINGS)
    return jsonify({"ok": True})

@app.route("/admin/settings/model_access", methods=["POST"])
@admin_required
def admin_set_model_access():
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not isinstance(body.get("access"), dict):
        return jsonify({"error": "Expected a model access mapping."}), 400
    models = {entry["path"]: "text" for entry in _scan_chat_models()}
    models.update({entry["path"]: entry["kind"] for entry in _scan_media()})
    if ACTIVE_MODEL_PATH:
        models.setdefault(ACTIVE_MODEL_PATH, "text")
    roles_by_path = body["access"]
    if any(path not in models for path in roles_by_path):
        return jsonify({"error": "Model access contains a model that is not installed."}), 400
    valid_roles = {"administrator", "standard", "guest"}
    access = {}
    for path, roles in roles_by_path.items():
        if not isinstance(roles, list) or any(not isinstance(role, str) or role not in valid_roles for role in roles):
            return jsonify({"error": "Each model must have a list of valid roles."}), 400
        access[path] = list(dict.fromkeys(roles))
    with settings_lock:
        SETTINGS["model_access"] = access
        text_paths = [path for path, kind in models.items()
                      if kind == "text" and "standard" in access.get(path, [])]
        SETTINGS["model_switching"] = {"enabled": bool(text_paths), "allowed_paths": text_paths}
        save_settings(SETTINGS)
    return jsonify({"ok": True})

@app.route("/api/model/switch", methods=["POST"])
@admin_required
def api_switch_model():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Expected the model path as a JSON object."}), 400
    path = body.get("path")
    if not isinstance(path, str) or not path:
        return jsonify({"error": "Select a model to switch to."}), 400
    if not any(entry["path"] == path for entry in _scan_chat_models()):
        return jsonify({"error": "That model is no longer available on this server."}), 400
    if not _identity_can_access_model(session["identity"], path, "text"):
        return jsonify({"error": "You don't have access to that chat model."}), 403
    try:
        with model_instances_lock:
            existing = next((entry for entry in MODEL_INSTANCES.values()
                             if entry["kind"] == "text" and entry["path"] == path
                             and entry["status"] == "running"), None)
        instance = _instance_summary(existing) if existing else _start_model_instance("text", path, "auto")
    except Exception as e:
        logger.error(f"User model switch failed: {e}")
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "active_model": path, "instance": instance})

@app.route("/admin/media_models/list")
@admin_required
def admin_media_models_list():
    return jsonify({"models": _scan_media(), "diffusers": media_models.diffusers_status(),
                    "install_hint": media_models.INSTALL_HINT})

@app.route("/admin/media_models/select", methods=["POST"])
@admin_required
def admin_media_models_select():
    body = request.json or {}
    kind = body.get("kind")
    if kind not in ("image", "video"):
        return jsonify({"error": "kind must be 'image' or 'video'"}), 400
    path = (body.get("path") or "").strip()
    if path:
        match = next((e for e in _scan_media() if _norm_path(e["path"]) == _norm_path(path)), None)
        if not match:
            return jsonify({"error": "That model isn't in the list. Add it first."}), 400
        if not match["usable"]:
            return jsonify({"error": match["note"] or "That model can't be used for prompt-only generation."}), 400
        if match["kind"] != kind:
            return jsonify({"error": f"That's {'a video' if match['kind'] == 'video' else 'an image'} model, not {'a video' if kind == 'video' else 'an image'} model."}), 400
        path = match["path"]
    with settings_lock:
        SETTINGS.setdefault("generation", {})[f"{kind}_model_path"] = path
        save_settings(SETTINGS)
    _free_media_pipe(kind)
    return jsonify({"ok": True, "models": _scan_media()})

@app.route("/admin/media_models/add", methods=["POST"])
@admin_required
def admin_media_models_add():
    raw = ((request.json or {}).get("path") or "").strip().strip('"').strip("'")
    if not raw:
        return jsonify({"error": "Paste the full path to a model folder or a .safetensors file."}), 400
    p = Path(raw)
    if not p.exists():
        return jsonify({"error": f"Nothing found at: {raw}"}), 400
    if p.is_dir() and not (p / "model_index.json").exists():
        return jsonify({"error": "That folder has no model_index.json, so it isn't a diffusers image/video model. "
                                  "(Chat models are added from the chat-model section above.)"}), 400
    if p.is_file() and p.suffix.lower() not in (".safetensors", ".ckpt"):
        return jsonify({"error": "Only .safetensors checkpoints (or a diffusers folder) can be added."}), 400
    with settings_lock:
        extras = SETTINGS.setdefault("generation", {}).setdefault("extra_paths", [])
        if _norm_path(p) not in {_norm_path(e) for e in extras}:
            extras.append(str(p))
        save_settings(SETTINGS)
    return jsonify({"ok": True, "models": _scan_media()})

@app.route("/admin/media_models/forget", methods=["POST"])
@admin_required
def admin_media_models_forget():
    target = _norm_path((request.json or {}).get("path") or "")
    with settings_lock:
        gen = SETTINGS.setdefault("generation", {})
        gen["extra_paths"] = [e for e in gen.get("extra_paths", []) if _norm_path(e) != target]
        for k in ("image", "video"):
            cur = (gen.get(f"{k}_model_path") or "").strip()
            if cur and _norm_path(cur) == target:
                gen[f"{k}_model_path"] = ""
                _free_media_pipe(k)
        save_settings(SETTINGS)
    return jsonify({"ok": True, "models": _scan_media()})

@app.route("/admin/settings/search", methods=["POST"])
@admin_required
def admin_set_search_settings():
    body = request.json or {}
    with settings_lock:
        s = SETTINGS.setdefault("search", dict(DEFAULT_SETTINGS["search"]))
        if "engine" in body:
            engine = str(body["engine"]).strip().lower()
            valid_engines = {"duckduckgo", "google", "bing", "yahoo"}
            if engine not in valid_engines:
                return jsonify({"error": f"engine must be one of: {', '.join(sorted(valid_engines))}"}), 400
            s["engine"] = engine
        if "max_sites" in body:
            try:
                v = int(body["max_sites"])
            except (TypeError, ValueError):
                return jsonify({"error": "max_sites must be a whole number"}), 400
            if not (1 <= v <= 10):
                return jsonify({"error": "max_sites must be between 1 and 10"}), 400
            s["max_sites"] = v
        save_settings(SETTINGS)
    return jsonify({"ok": True, "search": SETTINGS["search"]})

@app.route("/admin/policy", methods=["POST"])
@admin_required
def admin_update_policy():
    body = request.json or {}
    identity = body.get("sid")
    if not identity:
        return jsonify({"error": "sid required"}), 400
    with policy_lock:
        p = POLICIES.setdefault(identity, get_policy(identity))
        if "priority" in body:
            p["priority"] = max(1, min(10, int(body["priority"])))
        if "tokens_per_min" in body:
            p["tokens_per_min"] = max(0, int(body["tokens_per_min"]))
        if "max_tokens" in body:
            p["max_tokens"] = max(16, min(int(body["max_tokens"]), args.hard_max_tokens))
        if not is_guest_identity(identity):
            save_policies(POLICIES)
    if "role" in body and not is_guest_identity(identity):
        role = body["role"] if body["role"] in ("standard", "administrator") else "standard"
        with users_lock:
            if identity in USERS:
                USERS[identity]["role"] = role
                save_users(USERS)
    return jsonify({"ok": True, "policy": POLICIES[identity]})

@app.route("/admin/users/create", methods=["POST"])
@admin_required
def admin_create_user():
    body = request.json or {}
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    role = body.get("role") if body.get("role") in ("standard", "administrator") else "standard"
    if not USERNAME_RE.match(username):
        return jsonify({"error": "Username must be 3-24 characters: letters, numbers, _ . -"}), 400
    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters."}), 400
    with users_lock:
        if username in USERS:
            return jsonify({"error": "that username already exists"}), 400
        USERS[username] = {"password_hash": generate_password_hash(password), "created_at": time.time(), "role": role}
        save_users(USERS)
    with policy_lock:
        POLICIES[username] = {
            "name": username,
            "priority": max(1, min(10, int(body.get("priority", args.default_priority)))),
            "tokens_per_min": max(0, int(body.get("tokens_per_min", args.default_tokens_per_min))),
            "max_tokens": max(16, min(int(body.get("max_tokens", args.default_max_tokens)), args.hard_max_tokens)),
        }
        save_policies(POLICIES)
    return jsonify({"ok": True})

@app.route("/admin/users/delete", methods=["POST"])
@admin_required
def admin_delete_user():
    body = request.json or {}
    username = (body.get("username") or "").strip()
    with users_lock:
        remaining_admins = sum(1 for u, v in USERS.items() if v.get("role") == "administrator" and u != username)
        if USERS.get(username, {}).get("role") == "administrator" and remaining_admins == 0:
            return jsonify({"error": "Can't delete the last administrator account."}), 400
        USERS.pop(username, None)
        save_users(USERS)
    with policy_lock:
        POLICIES.pop(username, None)
        save_policies(POLICIES)
    with sessions_lock:
        SESSIONS.pop(username, None)
    return jsonify({"ok": True})

@app.route("/admin/logout")
def admin_logout():
    session.clear()
    return redirect(url_for("login"))

# ══════════════════════════════════════════════════════════════════════
#  ROUTES — admin: Training tab (download / switch models)
# ══════════════════════════════════════════════════════════════════════

DOWNLOAD_STATUS = {}   # model_id -> {state, message, progress_pct}
DOWNLOAD_PROCS = {}    # model_id -> subprocess.Popen, only while actively running
download_status_lock = threading.Lock()
custom_models_lock = threading.Lock()
PCT_RE = re.compile(r"(\d{1,3})%\|")  # matches tqdm's own percentage marker

def load_custom_models() -> list:
    return _load_json(CUSTOM_MODELS_FILE, [])

def save_custom_models(items: list):
    with custom_models_lock:
        try:
            CUSTOM_MODELS_FILE.write_text(json.dumps(items, indent=2))
        except Exception as e:
            logger.warning(f"Could not save custom models list: {e}")

def parse_hf_model_id(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    m = re.match(r"https?://huggingface\.co/([^/?#\s]+/[^/?#\s]+)", raw)
    if m:
        return m.group(1).rstrip("/")
    return raw.strip("/")

def _download_worker(model_id):
    # Runs download_models.py as a SEPARATE PROCESS (not a plain thread)
    # specifically so pause/cancel are real: psutil can suspend/resume an
    # OS process (same technique resource_throttle.py already uses for
    # training), and terminate() can actually stop it. A thread calling
    # into transformers directly can't be paused or killed cleanly.
    with download_status_lock:
        DOWNLOAD_STATUS[model_id] = {"state": "downloading", "message": "Starting…", "progress_pct": 0}

    child_env = dict(os.environ)
    child_env["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"  # we WANT the progress bar here, to parse it
    child_env["PYTHONUNBUFFERED"] = "1"
    try:
        proc = subprocess.Popen(
            [sys.executable, str(SERVER_DIR / "download_models.py"), model_id],
            cwd=str(SERVER_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0, env=child_env,
        )
    except Exception as e:
        with download_status_lock:
            DOWNLOAD_STATUS[model_id] = {"state": "error", "message": str(e), "progress_pct": 0}
        return

    DOWNLOAD_PROCS[model_id] = proc
    try:
        fd = proc.stdout.fileno()
        buf = ""
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            # tqdm progress updates use '\r', not '\n' — split on either so
            # percentage updates are seen as they happen, not only at EOF.
            segments = re.split(r"[\r\n]", buf)
            buf = segments.pop()
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                m = PCT_RE.search(seg)
                with download_status_lock:
                    if model_id not in DOWNLOAD_STATUS:
                        continue
                    if DOWNLOAD_STATUS[model_id]["state"] == "cancelled":
                        continue
                    if m:
                        DOWNLOAD_STATUS[model_id]["progress_pct"] = int(m.group(1))
                        DOWNLOAD_STATUS[model_id]["state"] = "downloading"
                    DOWNLOAD_STATUS[model_id]["message"] = seg[:200]
        proc.wait()
    finally:
        DOWNLOAD_PROCS.pop(model_id, None)

    with download_status_lock:
        if DOWNLOAD_STATUS.get(model_id, {}).get("state") == "cancelled":
            pass
        elif proc.returncode == 0:
            DOWNLOAD_STATUS[model_id] = {"state": "done", "message": "Downloaded.", "progress_pct": 100}
        else:
            prior_pct = DOWNLOAD_STATUS.get(model_id, {}).get("progress_pct", 0)
            DOWNLOAD_STATUS[model_id] = {"state": "error",
                                          "message": f"Download failed (exit code {proc.returncode}) — see server log.",
                                          "progress_pct": prior_pct}

@app.route("/admin/training")
@admin_required
def admin_training():
    all_ids = list(MODELS) + [m for m in load_custom_models() if m not in MODELS]
    registry_rows = []
    for mid in all_ids:
        with download_status_lock:
            live = dict(DOWNLOAD_STATUS.get(mid, {}))
        state = live.get("state")
        if state in ("downloading", "paused", "error"):
            status = state
        else:
            status = "downloaded" if (already_downloaded and already_downloaded(mid)) else "not_downloaded"
        registry_rows.append({"id": mid, "status": status, "progress_pct": live.get("progress_pct", 0)})

    local_models, gguf_files, seen = [], [], set()
    if REGISTRY_MODELS_ROOT.exists():
        # Recursive, not a single-level check: models downloaded or copied
        # in by hand don't always land at a consistent folder depth (e.g.
        # a zip extract can add its own wrapping folder), and a shallow
        # scan silently misses those even though the model is right there.
        # A diffusers (image/video) model keeps config.json files inside its
        # unet/ vae/ text_encoder/ sub-folders — those are NOT chat models and
        # must not be offered under "Switch the active chat model".
        media_roots = [mi.parent for mi in REGISTRY_MODELS_ROOT.rglob("model_index.json")]
        media_roots.extend(cfg.parent for cfg in REGISTRY_MODELS_ROOT.rglob("config.json")
                           if media_models.is_pixelart_model(cfg.parent))
        for cfg in REGISTRY_MODELS_ROOT.rglob("config.json"):
            if any(r == cfg.parent or r in cfg.parent.parents for r in media_roots):
                continue
            d = str(cfg.parent)
            if d not in seen:
                seen.add(d)
                local_models.append(d)
        gguf_files = [str(f) for f in REGISTRY_MODELS_ROOT.rglob("*.gguf")]
    if MODEL_CACHE.exists() and (MODEL_CACHE / "config.json").exists() and str(MODEL_CACHE) not in seen:
        local_models.append(str(MODEL_CACHE))
    local_models.sort()

    return render_template("admin_training.html", registry_rows=registry_rows, local_models=local_models,
                            gguf_files=gguf_files, current_model=ACTIVE_MODEL_PATH or resolve_model_path(),
                            models_root=str(REGISTRY_MODELS_ROOT), media_list=_scan_media(),
                            diffusers_info=media_models.diffusers_status(), install_hint=media_models.INSTALL_HINT)

@app.route("/admin/models/download", methods=["POST"])
@admin_required
def admin_download_model():
    model_id = (request.json or {}).get("model_id")
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    with download_status_lock:
        if DOWNLOAD_STATUS.get(model_id, {}).get("state") in ("downloading", "paused"):
            return jsonify({"error": "already downloading"}), 400
    threading.Thread(target=_download_worker, args=(model_id,), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/admin/models/download_custom", methods=["POST"])
@admin_required
def admin_download_custom():
    raw = (request.json or {}).get("model") or ""
    model_id = parse_hf_model_id(raw)
    if not model_id:
        return jsonify({"error": "Paste a Hugging Face model id (org/model) or a huggingface.co URL."}), 400
    custom = load_custom_models()
    if model_id not in custom and model_id not in MODELS:
        custom.append(model_id)
        save_custom_models(custom)
    threading.Thread(target=_download_worker, args=(model_id,), daemon=True).start()
    return jsonify({"ok": True, "model_id": model_id})

@app.route("/admin/models/download/pause", methods=["POST"])
@admin_required
def admin_pause_download():
    model_id = (request.json or {}).get("model_id")
    proc = DOWNLOAD_PROCS.get(model_id)
    if not proc:
        return jsonify({"error": "No active download for that model."}), 400
    try:
        import psutil
        psutil.Process(proc.pid).suspend()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    with download_status_lock:
        if model_id in DOWNLOAD_STATUS:
            DOWNLOAD_STATUS[model_id]["state"] = "paused"
    return jsonify({"ok": True})

@app.route("/admin/models/download/resume", methods=["POST"])
@admin_required
def admin_resume_download():
    model_id = (request.json or {}).get("model_id")
    proc = DOWNLOAD_PROCS.get(model_id)
    if not proc:
        return jsonify({"error": "No active download for that model."}), 400
    try:
        import psutil
        psutil.Process(proc.pid).resume()
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    with download_status_lock:
        if model_id in DOWNLOAD_STATUS:
            DOWNLOAD_STATUS[model_id]["state"] = "downloading"
    return jsonify({"ok": True})

@app.route("/admin/models/download/cancel", methods=["POST"])
@admin_required
def admin_cancel_download():
    model_id = (request.json or {}).get("model_id")
    with download_status_lock:
        if model_id in DOWNLOAD_STATUS:
            DOWNLOAD_STATUS[model_id] = {"state": "cancelled", "message": "Cancelled.", "progress_pct": 0}
    proc = DOWNLOAD_PROCS.pop(model_id, None)
    if proc:
        try:
            proc.terminate()
        except Exception:
            pass
    return jsonify({"ok": True})

@app.route("/admin/models/status")
@admin_required
def admin_models_status():
    with download_status_lock:
        return jsonify(dict(DOWNLOAD_STATUS))

def _activate_model_path(path: str):
    """Waits for the generation queue to clear, then hot-swaps the live
    model. Raises on failure. Used both by the admin 'Activate' button
    and automatically after a training run completes."""
    logger.info(f"Switching active model to: {path} (waiting for the queue to clear)")
    gen_lock.acquire(1)
    try:
        load_model(path)
        cfg = _load_json(CONFIG_FILE, {})
        cfg["model"] = path
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    finally:
        gen_lock.release()

@app.route("/admin/models/activate", methods=["POST"])
@admin_required
def admin_activate_model():
    path = (request.json or {}).get("path", "").strip()
    if not path or not Path(path).exists():
        return jsonify({"error": "model path not found"}), 400
    try:
        _activate_model_path(path)
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True, "active_model": path})

# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
#  ROUTES — admin: web scraping (wraps search_engine.py / webscraper.py)
# ══════════════════════════════════════════════════════════════════════

SCRAPE_STATUS = {"state": "idle", "log": "", "saved_to": None, "page_count": 0}
scrape_lock = threading.Lock()

def _scrape_log(msg: str):
    with scrape_lock:
        SCRAPE_STATUS["log"] += msg + "\n"
        SCRAPE_STATUS["log"] = SCRAPE_STATUS["log"][-8000:]  # cap growth

def _scrape_worker(mode: str, params: dict):
    with scrape_lock:
        SCRAPE_STATUS["state"] = "running"
        SCRAPE_STATUS["log"] = ""
        SCRAPE_STATUS["saved_to"] = None
        SCRAPE_STATUS["page_count"] = 0
    try:
        from webscraper import scrape_url_list, scrape_website, make_session, fetch_page, save_scraped_data
        pages = []
        if mode == "urls":
            urls = [u.strip() for u in (params.get("urls") or "").splitlines() if u.strip()]
            if not urls:
                _scrape_log("No URLs provided.")
            else:
                pages = scrape_url_list(urls, on_progress=_scrape_log, skip_visited=bool(params.get("skip_visited", True)))
        elif mode == "crawl":
            start_url = (params.get("start_url") or "").strip()
            if not start_url:
                _scrape_log("No start URL provided.")
            else:
                max_pages = max(1, min(int(params.get("max_pages", 10)), 100))
                pages = scrape_website(start_url, max_pages, on_progress=_scrape_log,
                                        skip_visited=bool(params.get("skip_visited", True)))
        elif mode == "search":
            topic = (params.get("topic") or "").strip()
            if not topic:
                _scrape_log("No topic provided.")
            else:
                from search_engine import search
                from webscraper import load_visited, save_visited
                max_pages = max(1, min(int(params.get("max_pages", 10)), 50))
                engine = params.get("engine", "duckduckgo")
                results = search(topic, engine=engine, n=max_pages,
                                  on_engine_used=lambda name, err: _scrape_log(f"search[{name}]: {'FAILED - ' + err if err else 'OK'}"))
                visited_before = load_visited() if params.get("skip_visited", True) else set()
                if params.get("skip_visited", True):
                    results = [result for result in results if result["url"] not in visited_before]
                _scrape_log(f"Found {len(results)} result(s) for '{topic}'")
                http = make_session()
                for r in results[:max_pages]:
                    _scrape_log(f"Crawling: {r['url'][:80]}")
                    page = fetch_page(http, r["url"], on_error=_scrape_log)
                    if page:
                        pages.append(page)
                        _scrape_log(f"  OK: {page['title'][:50]} ({len(page['text']):,} chars)")
                    time.sleep(0.5)
                if params.get("skip_visited", True):
                    save_visited({page["url"] for page in pages})

        if pages:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_dir = save_scraped_data(pages, ROOT / "web_training_data", f"admin_{ts}")
            _scrape_log(f"\nSaved {len(pages)} page(s) to {out_dir}")
            with scrape_lock:
                SCRAPE_STATUS["saved_to"] = str(out_dir)
                SCRAPE_STATUS["page_count"] = len(pages)
        else:
            _scrape_log("\nNo pages were saved.")
        with scrape_lock:
            SCRAPE_STATUS["state"] = "done"
    except Exception as e:
        logger.error(f"Scrape job failed: {e}")
        _scrape_log(f"ERROR: {e}")
        with scrape_lock:
            SCRAPE_STATUS["state"] = "error"

@app.route("/admin/scrape/start", methods=["POST"])
@admin_required
def admin_scrape_start():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Expected scraping settings as a JSON object."}), 400
    mode = body.get("mode")
    if mode not in ("urls", "crawl", "search"):
        return jsonify({"error": "invalid mode"}), 400
    if mode == "search" and body.get("engine", "duckduckgo") not in ("duckduckgo", "google", "bing", "yahoo"):
        return jsonify({"error": "Choose a supported search engine."}), 400
    with scrape_lock:
        if SCRAPE_STATUS["state"] == "running":
            return jsonify({"error": "A scrape job is already running — wait for it to finish."}), 400
    threading.Thread(target=_scrape_worker, args=(mode, body), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/admin/scrape/status")
@admin_required
def admin_scrape_status():
    with scrape_lock:
        return jsonify(dict(SCRAPE_STATUS))

# ══════════════════════════════════════════════════════════════════════
#  ROUTES — admin: training (wraps train.py, same invocation auto_trainer.py uses)
# ══════════════════════════════════════════════════════════════════════

TRAIN_STATUS = {"state": "idle", "log": "", "progress_pct": 0, "eta": None}
TRAIN_PROC = None
train_lock = threading.Lock()
PROGRESS_RE = re.compile(r"^PROGRESS_PCT:([\d.]+)(?:\s+ETA:(\d+)s)?")

# ── Training-data uploads (single/multiple files, or an entire folder) ──
# Accepts .csv/.tsv/.json/.jsonl/.parquet prompt+answer files as well as
# raw text/code files — train.py sorts out which is which. A "folder"
# upload from the browser arrives as a flat list of File objects each
# carrying its own relative path (webkitRelativePath), which the client
# sends alongside as the parallel "paths" field so the structure can be
# preserved on disk.
TRAIN_DATA_UPLOAD_DIR = TRAINING_UPLOAD_DIR / "admin_uploads"
MAX_TRAIN_UPLOAD_BYTES = 50 * 1024 * 1024

@app.route("/admin/train/upload", methods=["POST"])
@admin_required
def admin_train_upload():
    files = request.files.getlist("files")
    rel_paths = request.form.getlist("paths")
    if not files:
        return jsonify({"error": "No files provided."}), 400
    TRAIN_DATA_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    saved, errors = 0, []
    for i, f in enumerate(files):
        if not f.filename:
            continue
        rel = rel_paths[i] if i < len(rel_paths) and rel_paths[i] else f.filename
        parts = [re.sub(r"[^a-zA-Z0-9_.-]", "_", p) for p in Path(rel).parts if p not in ("", ".", "..")]
        if not parts:
            errors.append(f"{rel}: invalid path")
            continue
        target = TRAIN_DATA_UPLOAD_DIR.joinpath(*parts)
        try:
            data = f.read()
            if len(data) > MAX_TRAIN_UPLOAD_BYTES:
                errors.append(f"{rel}: too large (50MB max)")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            saved += 1
        except Exception as e:
            errors.append(f"{rel}: {e}")
    return jsonify({"ok": True, "saved": saved, "errors": errors, "dest": str(TRAIN_DATA_UPLOAD_DIR)})

@app.route("/admin/train/data_summary")
@admin_required
def admin_train_data_summary():
    data_dir = Path((request.args.get("data_dir") or str(ROOT / "web_training_data")).strip())
    if not data_dir.exists():
        return jsonify({"exists": False})
    qa_exts = {".csv", ".tsv", ".json", ".jsonl", ".parquet"}
    text_exts = {".txt", ".md", ".py", ".js", ".ts", ".java", ".cs", ".cpp", ".c", ".go", ".rs", ".html", ".css"}
    qa_count, text_count, other_count = 0, 0, 0
    for f in data_dir.rglob("*"):
        if not f.is_file() or f.name.startswith("_"):
            continue
        ext = f.suffix.lower()
        if ext in qa_exts:
            qa_count += 1
        elif ext in text_exts:
            text_count += 1
        else:
            other_count += 1
    return jsonify({"exists": True, "qa_files": qa_count, "text_files": text_count, "other_files": other_count})

def _train_worker(data_dir: str, epochs: int, auto_activate: bool):
    global TRAIN_PROC
    with train_lock:
        TRAIN_STATUS["state"] = "running"
        TRAIN_STATUS["log"] = ""
        TRAIN_STATUS["progress_pct"] = 0
        TRAIN_STATUS["eta"] = None
    try:
        proc = subprocess.Popen(
            [sys.executable, str(SERVER_DIR / "train.py"),
             "--data_dir", str(data_dir), "--epochs", str(epochs),
             "--model", resolve_model_path(), "--output", str(MODEL_CACHE),
             "--batch", "1", "--maxlen", "256"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
        )
        with train_lock:
            TRAIN_PROC = proc

        fd = proc.stdout.fileno()
        buf = ""
        while True:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            parts = buf.split("\n")
            buf = parts.pop()
            for line in parts:
                line = line.strip()
                if not line:
                    continue
                m = PROGRESS_RE.match(line)
                with train_lock:
                    if m:
                        TRAIN_STATUS["progress_pct"] = float(m.group(1))
                        # Only overwrite ETA when this line actually carries a
                        # new one. Plenty of progress() calls (model loading,
                        # saving, etc) don't compute an ETA and used to reset
                        # this to None every time, making the number flicker
                        # in and out during training instead of counting down.
                        if m.group(2):
                            TRAIN_STATUS["eta"] = int(m.group(2))
                    else:
                        TRAIN_STATUS["log"] += line + "\n"
                        TRAIN_STATUS["log"] = TRAIN_STATUS["log"][-8000:]
        proc.wait()
        with train_lock:
            TRAIN_PROC = None
            TRAIN_STATUS["eta"] = None
            if TRAIN_STATUS["state"] == "stopping":
                TRAIN_STATUS["state"] = "stopped"
            elif proc.returncode == 0:
                TRAIN_STATUS["state"] = "done"
                TRAIN_STATUS["progress_pct"] = 100
            else:
                TRAIN_STATUS["state"] = "error"
        if proc.returncode == 0 and auto_activate:
            try:
                _activate_model_path(str(MODEL_CACHE))
                with train_lock:
                    TRAIN_STATUS["log"] += "\nNew adapter automatically activated as the live model.\n"
            except Exception as e:
                with train_lock:
                    TRAIN_STATUS["log"] += f"\nTraining finished but auto-activate failed: {e}\n"
    except Exception as e:
        logger.error(f"Training job failed: {e}")
        with train_lock:
            TRAIN_STATUS["state"] = "error"
            TRAIN_STATUS["log"] += f"ERROR: {e}\n"
            TRAIN_PROC = None

@app.route("/admin/train/start", methods=["POST"])
@admin_required
def admin_train_start():
    body = request.json or {}
    data_dir = (body.get("data_dir") or str(ROOT / "web_training_data")).strip()
    epochs = max(1, min(int(body.get("epochs", 2)), 20))
    auto_activate = bool(body.get("auto_activate", True))
    if not Path(data_dir).exists():
        return jsonify({"error": f"data_dir not found: {data_dir}"}), 400
    with train_lock:
        if TRAIN_STATUS["state"] == "running":
            return jsonify({"error": "Training is already running."}), 400
    threading.Thread(target=_train_worker, args=(data_dir, epochs, auto_activate), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/admin/train/status")
@admin_required
def admin_train_status():
    with train_lock:
        return jsonify(dict(TRAIN_STATUS))

@app.route("/admin/train/stop", methods=["POST"])
@admin_required
def admin_train_stop():
    with train_lock:
        if TRAIN_PROC:
            TRAIN_STATUS["state"] = "stopping"
            try:
                TRAIN_PROC.terminate()
            except Exception:
                pass
        else:
            return jsonify({"error": "no training job running"}), 400
    return jsonify({"ok": True})


# ══════════════════════════════════════════════════════════════════════
#  ROUTES — internal (machine-to-machine, X-Worker-Key auth)
# ══════════════════════════════════════════════════════════════════════

@app.route("/internal/ping")
def internal_ping():
    # Deliberately unauthenticated and minimal — this is purely for LAN
    # discovery to identify "something running Terminal AI is here",
    # not for accessing anything. Everything that matters requires the
    # worker key.
    return jsonify({"ok": True, "app": "terminal-ai", "hostname": socket.gethostname()})

@app.route("/internal/status")
@worker_key_required
def internal_status():
    return jsonify({
        "busy": gen_lock.queue_depth() > 0,
        "queue_depth": gen_lock.queue_depth(),
        "model": ACTIVE_MODEL_PATH,
        "hostname": socket.gethostname(),
    })

@app.route("/internal/generate", methods=["POST"])
@worker_key_required
def internal_generate():
    body = request.json or {}
    message = body.get("message", "")
    history = body.get("history", [])
    max_tokens = max(16, min(int(body.get("max_tokens", 512)), args.hard_max_tokens))
    temperature = float(body.get("temperature", 0.7))
    top_p = float(body.get("top_p", 0.9))
    memory_turns = max(0, min(int(body.get("memory_turns", MEMORY_TURNS_DEFAULT)), MEMORY_TURNS_MAX))
    priority = max(1, min(10, int(body.get("priority", 3))))
    show_thinking = bool(body.get("show_thinking", True))
    # Shared with the primary machine that relayed this request, so a Stop
    # click there can reach the generation actually running here.
    request_id = body.get("request_id") or uuid.uuid4().hex
    stop_event = threading.Event()
    with active_generations_lock:
        ACTIVE_GENERATIONS[request_id] = {"stop_event": stop_event, "peer": None}

    def stream():
        gen_lock.acquire(priority)
        try:
            parts = []
            gen = _generate_pieces(message, history, max_tokens, temperature, top_p,
                                    system_prompt=_system_prompt_for(show_thinking),
                                    memory_turns=memory_turns, stop_event=stop_event)
            for kind, text in _split_thinking_stream(gen):
                if kind == "thinking":
                    yield json.dumps({"thinking_delta": text}) + "\n"
                else:
                    parts.append(text)
                    yield json.dumps({"delta": text}) + "\n"
            reply = _clean_reply("".join(parts)) or "I'm not sure how to answer that. Could you rephrase?"
            if stop_event.is_set() and reply:
                reply += "\n\n[stopped]"
            yield json.dumps({"done": True, "reply": reply}) + "\n"
        except Exception as e:
            logger.error(f"Internal generate failed: {e}")
            yield json.dumps({"done": True, "error": str(e)}) + "\n"
        finally:
            gen_lock.release()
            with active_generations_lock:
                ACTIVE_GENERATIONS.pop(request_id, None)

    return Response(stream(), mimetype="application/x-ndjson")

@app.route("/internal/generate/stop", methods=["POST"])
@worker_key_required
def internal_generate_stop():
    request_id = (request.json or {}).get("request_id")
    with active_generations_lock:
        entry = ACTIVE_GENERATIONS.get(request_id)
    if entry:
        entry["stop_event"].set()
    return jsonify({"ok": True})

@app.route("/internal/model_status/<path:model_id>")
@worker_key_required
def internal_model_status(model_id):
    if not SHARE_MODELS_WITH_PEERS:
        return jsonify({"available": False, "sharing_disabled": True})
    target = REGISTRY_MODELS_ROOT / model_slug(model_id)
    available = (target / "config.json").exists()
    return jsonify({"available": available})

@app.route("/internal/model_zip/<path:model_id>")
@worker_key_required
def internal_model_zip(model_id):
    if not SHARE_MODELS_WITH_PEERS:
        return jsonify({"error": "model sharing is turned off on this machine"}), 403
    target = REGISTRY_MODELS_ROOT / model_slug(model_id)
    if not (target / "config.json").exists():
        return jsonify({"error": "not available"}), 404
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in target.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(target))
    buf.seek(0)
    return send_file(buf, download_name=f"{model_slug(model_id)}.zip", mimetype="application/zip")

# ══════════════════════════════════════════════════════════════════════
#  ROUTES — admin: Peers (network / load distribution)
# ══════════════════════════════════════════════════════════════════════

def _lan_scan(port: int = None):
    """Best-effort scan of this machine's local /24 for other Terminal AI
    instances (same technique as remote_train.py's parallel worker-status
    checks). Runs in a background thread, never blocks server startup.
    Finds candidates only — an admin still has to enter the api_key to
    actually pair with one (LAN discovery isn't the same as trust)."""
    if not REQUESTS_OK:
        return
    port = port or args.port
    try:
        my_ip = detect_lan_ip()
        if my_ip == "127.0.0.1":
            return
        network = ipaddress.ip_network(f"{my_ip}/24", strict=False)
        with peers_lock:
            known_urls = {p["url"] for p in PEERS}
        my_url = f"http://{my_ip}:{port}"
        candidates = [str(ip) for ip in network.hosts() if str(ip) != my_ip]

        found = []
        def probe(ip):
            url = f"http://{ip}:{port}"
            if url in known_urls or url == my_url:
                return
            try:
                r = requests.get(f"{url}/internal/ping", timeout=0.4)
                if r.ok and r.json().get("app") == "terminal-ai":
                    found.append({"name": r.json().get("hostname", ip), "url": url, "api_key": "", "discovered": True})
            except Exception:
                pass

        with ThreadPoolExecutor(max_workers=40) as ex:
            list(ex.map(probe, candidates))

        if found:
            with peers_lock:
                existing = {p["url"] for p in PEERS}
                for f in found:
                    if f["url"] not in existing:
                        PEERS.append(f)
                        existing.add(f["url"])
                save_peers(PEERS)
            logger.info(f"LAN scan found {len(found)} other Terminal AI instance(s) — "
                        f"pair them with an api_key in /admin/peers to actually use them.")
        else:
            logger.info("LAN scan found no other Terminal AI instances.")
    except Exception as e:
        logger.warning(f"LAN scan failed: {e}")

@app.route("/admin/peers")
@admin_required
def admin_peers():
    with peers_lock:
        peers = list(PEERS)
    return render_template("admin_peers.html", peers=peers, own_key=WORKER_API_KEY,
                            own_url=f"http://{detect_lan_ip()}:{args.port}",
                            auto_discover=AUTO_DISCOVER_PEERS, share_models=SHARE_MODELS_WITH_PEERS,
                            machine_role=MACHINE_ROLE)

@app.route("/admin/machine_role", methods=["POST"])
@admin_required
def admin_set_machine_role():
    global MACHINE_ROLE
    role = (request.json or {}).get("role", "").strip().lower()
    if role not in ("master", "slave"):
        return jsonify({"error": "role must be 'master' or 'slave'"}), 400
    MACHINE_ROLE = role
    cfg = _load_json(CONFIG_FILE, {})
    cfg["machine_role"] = role
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        logger.warning(f"Could not persist machine_role: {e}")
    return jsonify({"ok": True, "role": MACHINE_ROLE})

@app.route("/admin/peers/priority", methods=["POST"])
@admin_required
def admin_peers_priority():
    body = request.json or {}
    url = body.get("url")
    with peers_lock:
        for p in PEERS:
            if p["url"] == url:
                priorities = p.setdefault("priorities", {})
                for ptype in ("chat", "code"):
                    if ptype in body:
                        try:
                            v = int(body[ptype])
                        except (TypeError, ValueError):
                            return jsonify({"error": f"'{ptype}' priority must be a whole number 1-10"}), 400
                        if not (1 <= v <= 10):
                            return jsonify({"error": f"'{ptype}' priority must be between 1 and 10"}), 400
                        priorities[ptype] = v
                save_peers(PEERS)
                return jsonify({"ok": True, "priorities": priorities})
    return jsonify({"error": "unknown peer"}), 400

@app.route("/admin/peers/add", methods=["POST"])
@admin_required
def admin_peers_add():
    body = request.json or {}
    url = (body.get("url") or "").strip().rstrip("/")
    if not url:
        return jsonify({"error": "url required"}), 400
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "http://" + url
    name = (body.get("name") or "").strip() or url
    api_key = (body.get("api_key") or "").strip()
    with peers_lock:
        if any(p["url"] == url for p in PEERS):
            return jsonify({"error": "that peer is already in the list"}), 400
        PEERS.append({"name": name, "url": url, "api_key": api_key, "discovered": False})
        save_peers(PEERS)
    return jsonify({"ok": True})

@app.route("/admin/peers/update", methods=["POST"])
@admin_required
def admin_peers_update():
    body = request.json or {}
    url = body.get("url")
    with peers_lock:
        for p in PEERS:
            if p["url"] == url:
                if "api_key" in body:
                    p["api_key"] = body["api_key"].strip()
                if "name" in body:
                    p["name"] = body["name"].strip() or p["url"]
                save_peers(PEERS)
                return jsonify({"ok": True})
    return jsonify({"error": "unknown peer"}), 400

@app.route("/admin/peers/remove", methods=["POST"])
@admin_required
def admin_peers_remove():
    url = (request.json or {}).get("url")
    with peers_lock:
        PEERS[:] = [p for p in PEERS if p["url"] != url]
        save_peers(PEERS)
    return jsonify({"ok": True})

@app.route("/admin/peers/ping", methods=["POST"])
@admin_required
def admin_peers_ping():
    if not REQUESTS_OK:
        return jsonify({"error": "'requests' not installed on this server"}), 500
    url = (request.json or {}).get("url")
    with peers_lock:
        peer = next((p for p in PEERS if p["url"] == url), None)
    if not peer:
        return jsonify({"error": "unknown peer"}), 400
    result = {"reachable": False, "paired": False, "status": None}
    try:
        r = requests.get(f"{url}/internal/ping", timeout=2)
        result["reachable"] = r.ok
    except Exception as e:
        result["error"] = str(e)
        return jsonify(result)
    if peer.get("api_key"):
        try:
            r2 = requests.get(f"{url}/internal/status", headers=_peer_headers(peer), timeout=2)
            if r2.ok:
                result["paired"] = True
                result["status"] = r2.json()
            else:
                result["error"] = f"HTTP {r2.status_code} — check the api_key matches that machine's worker key"
        except Exception as e:
            result["error"] = str(e)
    return jsonify(result)

@app.route("/admin/peers/scan", methods=["POST"])
@admin_required
def admin_peers_scan():
    threading.Thread(target=_lan_scan, daemon=True).start()
    return jsonify({"ok": True})

def detect_lan_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def _warm_configured_model_pools():
    with settings_lock:
        generation = dict(SETTINGS.get("generation", {}))
    configured = [
        ("text", (generation.get("text_model_path") or "").strip() or ACTIVE_MODEL_PATH),
        ("image", (generation.get("image_model_path") or "").strip()),
    ]
    for kind, path in configured:
        if not path or not (_model_access_roles(path, kind) & {"standard", "guest"}):
            continue
        instance = None
        try:
            instance = _lease_model_instance(kind, path)
            logger.info(f"Shared {kind} model pool ready for {Path(path).name or path}")
        except Exception:
            logger.exception(f"Could not warm the shared {kind} model pool for {path}")
        finally:
            if instance:
                _release_model_instance(instance["id"])

def run():
    lan_ip = detect_lan_ip() if args.host == "0.0.0.0" else args.host
    logger.info("=" * 60)
    logger.info("  Terminal AI web GUI is up.")
    logger.info(f"  On this machine:  http://127.0.0.1:{args.port}/")
    logger.info(f"  On your LAN:      http://{lan_ip}:{args.port}/   <- give this to other people")
    logger.info(f"  Self-signup:      {'ON — anyone can create an account' if ALLOW_SELF_REGISTRATION else 'OFF — admin must create accounts'}")
    logger.info(f"  Worker key:       {WORKER_API_KEY}  (share this with another machine to let it use this one as a peer)")
    logger.info("=" * 60)
    threading.Thread(target=_warm_configured_model_pools, daemon=True, name="model-pool-warmer").start()
    if AUTO_DISCOVER_PEERS:
        # Runs in the background AFTER the server is already up — never
        # delays the page becoming reachable, unlike model loading above.
        threading.Thread(target=_lan_scan, daemon=True).start()
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == "__main__":
    run()
