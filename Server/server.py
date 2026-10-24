"""
Terminal AI V14.0 - AI Inference Server
Run: python server.py --port 5001

Hardware priority: NPU (OpenVINO/ONNX) > CUDA > DirectML > CPU
Supports: chat, cancel, model switch, model restore, status,
          LoRA adapter merge, lookup/search helper
"""

import argparse, json, os, sys, threading, time, uuid, logging, shutil
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "0")  # allow first download; can force offline later

parser = argparse.ArgumentParser()
parser.add_argument("--port",  type=int, default=5001)
parser.add_argument("--model", type=str, default=None)
args = parser.parse_args()

ROOT          = Path(__file__).resolve().parent.parent
SERVER_DIR    = Path(__file__).resolve().parent
MODEL_CACHE   = ROOT / "model_cache"
MODEL_ARCHIVE = ROOT / "model_archive"
LOG_DIR       = ROOT / "logs"
MODELS_ROOT   = ROOT / "models"   # multi-model support from V8

for d in [MODEL_CACHE, MODEL_ARCHIVE, LOG_DIR, MODELS_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SERVER_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "server.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("server")
logger.info(f"ROOT:          {ROOT}")
logger.info(f"MODEL_CACHE:   {MODEL_CACHE}")

# ── Hardware detection ────────────────────────────────────────
def detect_hardware():
    result = {"device": "cpu", "device_name": "CPU",
              "npu": False, "cuda": False, "directml": False, "openvino": False}

    try:
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core
        core = Core()
        devices = core.available_devices
        logger.info(f"OpenVINO devices: {devices}")
        result["openvino"] = True
        result["openvino_devices"] = devices
        if "NPU" in devices:
            result.update(device="npu", device_name="Intel NPU (OpenVINO)", npu=True)
            logger.info("Hardware: Intel NPU via OpenVINO")
            return result
    except Exception as e:
        logger.debug(f"OpenVINO check: {e}")

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"ONNX providers: {providers}")
        for ep in ("QNNExecutionProvider", "OpenVINOExecutionProvider"):
            if ep in providers:
                result.update(device="npu", device_name=f"NPU ({ep})", npu=True)
                logger.info(f"Hardware: NPU via {ep}")
                return result
    except Exception as e:
        logger.debug(f"ONNX check: {e}")

    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            result.update(device="cuda", device_name=f"CUDA: {name}", cuda=True)
            logger.info(f"Hardware: CUDA GPU: {name}")
            return result
    except Exception as e:
        logger.debug(f"CUDA check: {e}")

    try:
        import torch_directml
        result.update(device="directml", device_name="GPU (DirectML)", directml=True)
        logger.info("Hardware: DirectML GPU")
        return result
    except Exception as e:
        logger.debug(f"DirectML check: {e}")

    logger.info("Hardware: CPU only")
    return result

HARDWARE = detect_hardware()
logger.info(f"Active device: {HARDWARE['device_name']}")

# ── Imports ───────────────────────────────────────────────────
logger.info("Loading model libraries...")
try:
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TextIteratorStreamer, pipeline
    )
    import torch
    logger.info(f"PyTorch {torch.__version__}")
except ImportError as e:
    logger.error(f"Missing package: {e}")
    logger.error("Run: pip install transformers torch accelerate peft")
    sys.exit(1)

try:
    from model_registry import MODELS, DEFAULT_MODEL, slug as model_slug
except Exception:
    MODELS = ["distilgpt2", "gpt2", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
               "Qwen/Qwen2.5-0.5B-Instruct"]
    DEFAULT_MODEL = "distilgpt2"
    def model_slug(mid): return mid.replace("/", "__")

BASE_MODEL = args.model or DEFAULT_MODEL

def resolve_model_path():
    # Prefer model_cache if it has weights
    if MODEL_CACHE.exists() and (MODEL_CACHE / "config.json").exists():
        logger.info(f"Found model at {MODEL_CACHE}")
        return str(MODEL_CACHE)
    # Multi-model folder from V8 style
    mid = BASE_MODEL
    d = MODELS_ROOT / model_slug(mid)
    if (d / "config.json").exists():
        return str(d)
    logger.info(f"Using base model id: {BASE_MODEL}")
    return BASE_MODEL

model_lock = threading.RLock()
tokenizer  = None
model      = None
generator  = None
MODEL_PATH = resolve_model_path()

def get_torch_device():
    d = HARDWARE["device"]
    if d == "cuda":
        return "cuda"
    if d == "directml":
        try:
            import torch_directml
            return torch_directml.device()
        except Exception:
            pass
    return "cpu"

TORCH_DEVICE = get_torch_device()
logger.info(f"Torch device: {TORCH_DEVICE}")

def adapter_available(path: Path) -> bool:
    return (path / "_lora_adapter" / "adapter_config.json").exists()

def load_model(path: str):
    """Load base model and merge LoRA adapter if present (V8 style)."""
    global tokenizer, model, generator, MODEL_PATH
    logger.info(f"Loading: {path}")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if str(TORCH_DEVICE) == "cuda" else torch.float32
    mod = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=dtype, low_cpu_mem_usage=True
    )

    # Merge LoRA knowledge adapter if it exists next to the model
    p = Path(path)
    adapter = p / "_lora_adapter"
    if adapter.exists() and (adapter / "adapter_config.json").exists():
        try:
            from peft import PeftModel
            logger.info("Merging LoRA knowledge adapter...")
            mod = PeftModel.from_pretrained(mod, str(adapter))
            mod = mod.merge_and_unload()
            logger.info(f"Adapter merged from {adapter}")
        except Exception as e:
            logger.info(f"Adapter load failed (using base only): {e}")

    device_arg = TORCH_DEVICE if str(TORCH_DEVICE) != "cpu" else "cpu"
    if str(device_arg) != "cpu":
        mod = mod.to(device_arg)
    mod.eval()

    pipe = pipeline(
        "text-generation", model=mod, tokenizer=tok,
        device=0 if str(TORCH_DEVICE) == "cuda" else -1
    )

    with model_lock:
        tokenizer = tok
        model     = mod
        generator = pipe
        MODEL_PATH = path

    params = sum(p.numel() for p in mod.parameters())
    logger.info(f"Model loaded: {path} | {params:,} params | device={TORCH_DEVICE}")

logger.info("Loading model...")
try:
    load_model(MODEL_PATH)
    logger.info("SERVER_READY")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    sys.exit(1)

# ── Active generation registry ────────────────────────────────
_active: dict[str, threading.Event] = {}
_active_lock = threading.Lock()

def build_prompt(message: str, history: list, system_prompt: str = None) -> str:
    system = system_prompt or (
        "You are a helpful, knowledgeable AI assistant. "
        "Answer clearly and concisely. "
        "Format code in triple backticks with the language name."
    )
    prompt = system + "\n\n"
    for turn in history[-8:]:
        role = "User" if turn.get("role") == "user" else "Assistant"
        prompt += f"{role}: {turn.get('content','')}\n"
    prompt += f"User: {message}\nAssistant:"
    return prompt

def generate_reply(message: str, history: list, gen_id: str,
                   max_tokens: int = 300, temperature: float = 0.7,
                   top_p: float = 0.9, repeat_penalty: float = 1.2,
                   system_prompt: str = None) -> str:
    parts = []
    for item in generate_reply_stream(message, history, gen_id,
                                      max_tokens=max_tokens,
                                      temperature=temperature,
                                      top_p=top_p,
                                      repeat_penalty=repeat_penalty,
                                      system_prompt=system_prompt):
        if item.get("done"):
            return item.get("reply") or "".join(parts).strip() or "I'm not sure how to answer that. Could you rephrase?"
        if item.get("delta"):
            parts.append(item["delta"])
    return "".join(parts).strip() or "I'm not sure how to answer that. Could you rephrase?"


def generate_reply_stream(message: str, history: list, gen_id: str,
                          max_tokens: int = 300, temperature: float = 0.7,
                          top_p: float = 0.9, repeat_penalty: float = 1.2,
                          system_prompt: str = None):
    cancel_ev = threading.Event()
    with _active_lock:
        _active[gen_id] = cancel_ev

    try:
        prompt = build_prompt(message, history, system_prompt)
        with model_lock:
            tok = tokenizer
            mod = model

        inputs = tok(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(next(mod.parameters()).device)

        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repeat_penalty,
            pad_token_id=tok.eos_token_id,
            streamer=streamer,
        )

        thread = threading.Thread(target=mod.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        parts = []
        for token in streamer:
            if cancel_ev.is_set():
                token = " [stopped]"
            if token:
                parts.append(token)
                yield {"delta": token}
            if cancel_ev.is_set():
                break

        thread.join(timeout=3)
        reply = "".join(parts).strip()

        for stop in ["User:", "Human:", "\nUser", "\nHuman", "\n\nUser"]:
            if stop in reply:
                reply = reply[:reply.index(stop)].strip()

        reply = reply or "I'm not sure how to answer that. Could you rephrase?"
        yield {"done": True, "reply": reply}

    except Exception as e:
        logger.error(f"Generation error: {e}")
        yield {"done": True, "error": f"Generation error: {e}"}
    finally:
        with _active_lock:
            _active.pop(gen_id, None)

def switch_model(new_model_name: str) -> dict:
    try:
        logger.info(f"Switching model to: {new_model_name}")
        if MODEL_CACHE.exists() and (MODEL_CACHE / "config.json").exists():
            archive_name = f"model_{int(time.time())}"
            archive_path = MODEL_ARCHIVE / archive_name
            logger.info(f"Archiving current model → {archive_path}")
            shutil.copytree(str(MODEL_CACHE), str(archive_path), dirs_exist_ok=True)

        logger.info(f"Downloading {new_model_name} ...")
        tok = AutoTokenizer.from_pretrained(new_model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.float16 if str(TORCH_DEVICE) == "cuda" else torch.float32
        mod = AutoModelForCausalLM.from_pretrained(new_model_name, torch_dtype=dtype, low_cpu_mem_usage=True)

        if MODEL_CACHE.exists():
            shutil.rmtree(str(MODEL_CACHE))
        MODEL_CACHE.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(MODEL_CACHE))

        import platform
        safe = platform.system() != "Windows"
        mod.save_pretrained(str(MODEL_CACHE), safe_serialization=safe)

        load_model(str(MODEL_CACHE))
        logger.info(f"Model switched to: {new_model_name}")
        return {"success": True, "model": new_model_name}
    except Exception as e:
        logger.error(f"Model switch failed: {e}")
        return {"success": False, "error": str(e)}

def restore_model(archive_name: str) -> dict:
    archive_path = MODEL_ARCHIVE / archive_name
    if not archive_path.exists():
        return {"success": False, "error": f"Archive not found: {archive_name}"}
    try:
        if MODEL_CACHE.exists():
            shutil.rmtree(str(MODEL_CACHE))
        shutil.copytree(str(archive_path), str(MODEL_CACHE))
        load_model(str(MODEL_CACHE))
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ── HTTP handler ──────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a): pass

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            adapter = adapter_available(Path(MODEL_PATH))
            self.send_json({
                "status": "ready",
                "model": MODEL_PATH,
                "hardware": HARDWARE,
                "has_lora_adapter": adapter,
                "models_available": MODELS,
            })
        elif self.path == "/archived_models":
            archives = sorted(
                [d.name for d in MODEL_ARCHIVE.iterdir() if d.is_dir()],
                reverse=True
            )
            self.send_json({"models": archives})
        elif self.path == "/ping":
            self.send_json({"ok": True})
        elif self.path == "/auto_train_state":
            state_file = ROOT / "auto_train_state.json"
            try:
                state = json.loads(state_file.read_text()) if state_file.exists() else {}
            except Exception:
                state = {}
            self.send_json(state)
        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            self.send_json({"error": f"Bad request: {e}"}, 400)
            return

        if self.path == "/chat":
            message = body.get("message", "").strip()
            history = body.get("history", [])
            gen_id  = body.get("gen_id") or str(uuid.uuid4())
            max_tok = int(body.get("max_tokens", 300))
            temp    = float(body.get("temperature", 0.7))
            top_p   = float(body.get("top_p", 0.9))
            rep_pen = float(body.get("repeat_penalty", 1.2))
            sys_pmt = body.get("system_prompt", None)
            stream  = bool(body.get("stream", False))

            if not message:
                self.send_json({"reply": "Please send a message."})
                return

            logger.info(f"Chat [{gen_id[:8]}]: {message[:60]}")
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for item in generate_reply_stream(message, history, gen_id,
                                                  max_tokens=max_tok, temperature=temp,
                                                  top_p=top_p, repeat_penalty=rep_pen,
                                                  system_prompt=sys_pmt):
                    chunk = json.dumps(item).encode("utf-8") + b"\n"
                    self.wfile.write(chunk)
                    self.wfile.flush()
                return

            reply = generate_reply(message, history, gen_id,
                                   max_tokens=max_tok, temperature=temp,
                                   top_p=top_p, repeat_penalty=rep_pen,
                                   system_prompt=sys_pmt)
            self.send_json({"reply": reply, "gen_id": gen_id})

        elif self.path == "/cancel":
            gen_id = body.get("gen_id", "")
            with _active_lock:
                if gen_id in _active:
                    _active[gen_id].set()
                    logger.info(f"Cancelled: {gen_id[:8]}")
                    self.send_json({"cancelled": True})
                else:
                    self.send_json({"cancelled": False})

        elif self.path == "/switch_model":
            name = body.get("model", "").strip()
            if not name:
                self.send_json({"error": "No model specified"}, 400)
                return
            result = switch_model(name)
            self.send_json(result)

        elif self.path == "/restore_model":
            arch = body.get("archive", "").strip()
            self.send_json(restore_model(arch))

        elif self.path == "/lookup":
            # Quick search + summarize (from V8 lookup)
            query = body.get("query", "").strip()
            if not query:
                self.send_json({"error": "query required"}, 400)
                return
            try:
                from search_engine import search, summarize_results
                import webscraper
                results = search(query, engine=body.get("engine", "duckduckgo"), n=8)
                session = webscraper.make_session()
                pages = []
                for r in results[:3]:
                    page = webscraper.fetch_page(session, r["url"])
                    if page:
                        pages.append(page["text"])
                    time.sleep(0.3)
                summary = summarize_results(query, results, pages)
                self.send_json({"summary": summary, "results": results[:5]})
            except Exception as e:
                self.send_json({"error": str(e)}, 500)

        elif self.path == "/auto_train_control":
            # pause / resume / stop the auto-trainer via state file
            action = body.get("action", "").lower()
            state_file = ROOT / "auto_train_state.json"
            try:
                state = json.loads(state_file.read_text()) if state_file.exists() else {}
            except Exception:
                state = {}
            if action == "pause":
                state["paused"] = True
            elif action == "resume":
                state["paused"] = False
            elif action == "stop":
                state["stopped"] = True
                state["running"] = False
            state_file.write_text(json.dumps(state, indent=2))
            self.send_json({"ok": True, "state": state})

        else:
            self.send_json({"error": "Unknown endpoint"}, 404)


def run():
    server = HTTPServer(("127.0.0.1", args.port), Handler)
    logger.info(f"Listening on http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")

if __name__ == "__main__":
    run()
