"""
train_worker.py — Runs on each remote machine you want to contribute
training compute (e.g. your Ubuntu Server box, your Linux tower).

What it does:
  - Exposes a small HTTP API so a controller machine (the one running
    chatbot.py) can hand it a training job: some scraped text + which
    base model to train + the current LoRA adapter to continue.
  - Runs the exact same train.py used locally, so training behaves
    identically no matter which machine does the work.
  - Sends back the updated LoRA adapter when done, so the controller can
    merge it in and the *next* job (possibly on a different machine)
    picks up right where this one left off.

Only the training DATA and the small LoRA adapter travel over the
network — base model weights are downloaded/cached locally on each
worker the first time it trains a given model, same as the main app.

SETUP (on the remote Linux machine):
    python3 -m venv venv && source venv/bin/activate
    pip install flask transformers torch accelerate datasets peft psutil
    python3 train_worker.py --port 5000 --api-key CHANGE_ME

SECURITY: this accepts uploads and runs a training subprocess. Only run
it on a trusted private network (LAN / VPN like Tailscale), and always
set --api-key to something private. Do not expose it to the open
internet.
"""
import argparse, os, sys, json, uuid, shutil, subprocess, threading, zipfile, io, socket
from pathlib import Path
from datetime import datetime

BASE_DIR    = Path(__file__).parent
MODELS_ROOT = BASE_DIR / "models"
JOBS_ROOT   = BASE_DIR / "jobs"
MODELS_ROOT.mkdir(exist_ok=True)
JOBS_ROOT.mkdir(exist_ok=True)

def slug(model_id: str) -> str:
    return model_id.replace("/", "__")

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--api-key", type=str, default=os.environ.get("TRAIN_WORKER_KEY", ""))
args = parser.parse_args()

if not args.api_key:
    print("ERROR: no API key set. Pass --api-key YOURSECRET or set TRAIN_WORKER_KEY.")
    sys.exit(1)

try:
    from flask import Flask, request, jsonify, send_file
except ImportError:
    print("ERROR: flask not installed. Run: pip install flask")
    sys.exit(1)

app = Flask(__name__)
JOBS = {}  # job_id -> dict(state, log, returncode, proc, result_zip)
JOBS_LOCK = threading.Lock()


def check_auth():
    return request.headers.get("X-API-Key") == args.api_key


@app.before_request
def _auth():
    if not check_auth():
        return jsonify({"error": "unauthorized"}), 401


@app.route("/status")
def status():
    busy = any(j["state"] == "running" for j in JOBS.values())
    try:
        import psutil
        cpu_cores = os.cpu_count()
        ram_gb    = round(psutil.virtual_memory().total / (1024**3), 1)
        cpu_pct   = psutil.cpu_percent(interval=0.2)
    except Exception:
        cpu_cores = os.cpu_count(); ram_gb = None; cpu_pct = None
    return jsonify({
        "hostname":  socket.gethostname(),
        "busy":      busy,
        "cpu_cores": cpu_cores,
        "ram_gb":    ram_gb,
        "cpu_pct":   cpu_pct,
        "active_jobs": sum(1 for j in JOBS.values() if j["state"] == "running"),
    })


def _run_job(job_id, model_id, epochs, job_dir):
    j = JOBS[job_id]
    model_path = MODELS_ROOT / slug(model_id)
    model_path.mkdir(parents=True, exist_ok=True)
    log_path = job_dir / "worker.log"

    def log(msg):
        with JOBS_LOCK:
            j["log"] += msg + "\n"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    log(f"[worker] Starting job {job_id} — model={model_id} epochs={epochs}")
    try:
        proc = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "train.py"),
             "--data_dir", str(job_dir / "data"),
             "--epochs", str(epochs),
             "--model", str(model_path),
             "--output", str(model_path),
             "--batch", "1", "--maxlen", "256"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        j["proc"] = proc
        success = False
        for line in proc.stdout:
            log(line.rstrip())
            if "TRAINING_SUCCESS" in line: success = True
            if "TRAINING_ERROR" in line: break
        proc.wait()

        if success or proc.returncode == 0:
            # Zip up the resulting adapter so the controller can pull it.
            adapter_dir = model_path / "_lora_adapter"
            buf = io.BytesIO()
            if adapter_dir.exists():
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in adapter_dir.rglob("*"):
                        if f.is_file():
                            zf.write(f, f.relative_to(adapter_dir))
            buf.seek(0)
            with JOBS_LOCK:
                j["state"] = "done"
                j["returncode"] = proc.returncode
                j["result_zip"] = buf.getvalue()
            log("[worker] Job complete.")
        else:
            with JOBS_LOCK:
                j["state"] = "error"
                j["returncode"] = proc.returncode
            log(f"[worker] Job failed, returncode={proc.returncode}")
    except Exception as e:
        with JOBS_LOCK:
            j["state"] = "error"
        log(f"[worker] EXCEPTION: {e}")


@app.route("/train", methods=["POST"])
def train():
    model_id = request.form.get("model_id")
    epochs   = int(request.form.get("epochs", 2))
    if not model_id:
        return jsonify({"error": "model_id required"}), 400
    if "data" not in request.files:
        return jsonify({"error": "training data zip required"}), 400

    job_id  = uuid.uuid4().hex[:12]
    job_dir = JOBS_ROOT / job_id
    (job_dir / "data").mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(request.files["data"]) as zf:
        zf.extractall(job_dir / "data")

    # If the controller sent along the current adapter state, unpack it
    # into place BEFORE training starts so train.py resumes/continues it
    # instead of starting a brand new one (this is what keeps knowledge
    # continuous across machines/jobs).
    if "adapter" in request.files:
        model_path  = MODELS_ROOT / slug(model_id)
        adapter_dir = model_path / "_lora_adapter"
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        adapter_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(request.files["adapter"]) as zf:
            zf.extractall(adapter_dir)

    JOBS[job_id] = {"state": "running", "log": "", "returncode": None,
                     "proc": None, "result_zip": None}
    t = threading.Thread(target=_run_job, args=(job_id, model_id, epochs, job_dir), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/train/<job_id>/status")
def train_status(job_id):
    j = JOBS.get(job_id)
    if not j: return jsonify({"error": "unknown job"}), 404
    return jsonify({"state": j["state"], "returncode": j["returncode"]})


@app.route("/train/<job_id>/log")
def train_log(job_id):
    j = JOBS.get(job_id)
    if not j: return jsonify({"error": "unknown job"}), 404
    offset = int(request.args.get("offset", 0))
    with JOBS_LOCK:
        text = j["log"]
    return jsonify({"log": text[offset:], "offset": len(text), "state": j["state"]})


@app.route("/train/<job_id>/result")
def train_result(job_id):
    j = JOBS.get(job_id)
    if not j: return jsonify({"error": "unknown job"}), 404
    if j["state"] != "done":
        return jsonify({"error": f"job not finished (state={j['state']})"}), 409
    return send_file(io.BytesIO(j["result_zip"]), download_name=f"{job_id}_adapter.zip",
                      mimetype="application/zip")


if __name__ == "__main__":
    print(f"train_worker.py listening on 0.0.0.0:{args.port}  (hostname={socket.gethostname()})")
    print("Make sure train.py and model_registry.py are copied into this same folder.")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
