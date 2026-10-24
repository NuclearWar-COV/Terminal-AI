"""
remote_train.py — Client-side dispatcher for distributed training.

Runs on the controller machine (wherever chatbot.py lives). Given a list
of remote train_worker.py instances, it:
  1. Asks each one if it's free (GET /status).
  2. Picks the best available machine (idle, then most RAM/cores).
  3. Uploads the training data + current LoRA adapter (if any).
  4. Polls for progress, mirroring log lines back through the same
     on_progress/on_bar callbacks run_training() already uses locally.
  5. Downloads the finished adapter and unpacks it into place, so the
     next job -- local or remote -- continues from here.

If no worker is reachable or all are busy, callers should fall back to
local training; this module never raises for "no worker available", it
just returns None so that's easy to detect.
"""
import json, time, zipfile, io, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    import requests
except ImportError:
    requests = None

BASE_DIR      = Path(__file__).resolve().parent
WORKERS_FILE  = BASE_DIR.parent / "remote_workers.json"
PCT_RE        = re.compile(r"^PROGRESS_PCT:([\d.]+)(?:\s+ETA:(\d+)s)?")


def load_workers():
    """remote_workers.json format:
    [
      {"name": "ubuntu-server", "url": "http://192.168.1.50:5000", "api_key": "..."},
      {"name": "linux-tower",   "url": "http://192.168.1.60:5000", "api_key": "..."}
    ]
    """
    if not WORKERS_FILE.exists():
        return []
    try:
        return json.loads(WORKERS_FILE.read_text())
    except Exception:
        return []


def _check_worker(worker, timeout=3):
    if requests is None:
        return None
    try:
        r = requests.get(f"{worker['url']}/status",
                          headers={"X-API-Key": worker.get("api_key", "")},
                          timeout=timeout)
        if r.status_code == 200:
            info = r.json()
            info["_worker"] = worker
            return info
    except Exception:
        pass
    return None


def list_worker_status(workers=None, timeout=3):
    """Check every configured worker in parallel. Returns list of status
    dicts (only for workers that responded)."""
    workers = workers if workers is not None else load_workers()
    if not workers:
        return []
    with ThreadPoolExecutor(max_workers=max(1, len(workers))) as ex:
        results = list(ex.map(lambda w: _check_worker(w, timeout), workers))
    return [r for r in results if r is not None]


def pick_best_worker(workers=None, timeout=3):
    """Pick the strongest idle worker: most RAM, then most CPU cores.
    Returns a worker config dict, or None if nothing usable is free."""
    statuses = list_worker_status(workers, timeout)
    idle = [s for s in statuses if not s.get("busy")]
    if not idle:
        return None
    idle.sort(key=lambda s: (s.get("ram_gb") or 0, s.get("cpu_cores") or 0), reverse=True)
    return idle[0]["_worker"]


def _zip_dir(path: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in Path(path).rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(path))
    return buf.getvalue()


def dispatch_remote_training(worker, data_dir, model_id, epochs, adapter_dir,
                              on_progress, on_bar=None, poll_interval=2.0,
                              timeout=15):
    """
    Run one training job on `worker`. `adapter_dir` is the LOCAL path to
    the current LoRA adapter (Path), if any -- it gets uploaded so the
    remote job continues it rather than starting fresh. Returns the raw
    adapter zip bytes on success, or raises Exception with a readable
    message on failure (callers should catch and fall back to local
    training).
    """
    if requests is None:
        raise RuntimeError("'requests' package not installed on controller machine")

    headers = {"X-API-Key": worker.get("api_key", "")}
    on_progress(f"Uploading training data to {worker.get('name', worker['url'])}...")

    files = {"data": ("data.zip", _zip_dir(data_dir), "application/zip")}
    if adapter_dir and Path(adapter_dir).exists() and any(Path(adapter_dir).iterdir()):
        files["adapter"] = ("adapter.zip", _zip_dir(adapter_dir), "application/zip")
        on_progress("  (including current knowledge adapter to resume it remotely)")

    r = requests.post(f"{worker['url']}/train", headers=headers,
                       data={"model_id": model_id, "epochs": epochs},
                       files=files, timeout=timeout)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    on_progress(f"Remote job started: {job_id}")

    offset = 0
    while True:
        time.sleep(poll_interval)
        lr = requests.get(f"{worker['url']}/train/{job_id}/log",
                           headers=headers, params={"offset": offset}, timeout=timeout)
        lr.raise_for_status()
        payload = lr.json()
        new_text = payload.get("log", "")
        offset   = payload.get("offset", offset)
        for line in new_text.splitlines():
            m = PCT_RE.match(line)
            if m and on_bar:
                pct  = float(m.group(1))
                secs = int(m.group(2)) if m.group(2) else -1
                eta  = f"{secs//60}m {secs%60:02d}s" if secs >= 60 else (f"{secs}s" if secs >= 0 else "Calc...")
                on_bar(pct, eta)
            elif line.strip():
                on_progress(f"[remote] {line}")

        state = payload.get("state")
        if state == "done":
            break
        if state == "error":
            raise RuntimeError(f"Remote training job failed on {worker.get('name', worker['url'])}")

    rr = requests.get(f"{worker['url']}/train/{job_id}/result", headers=headers, timeout=timeout)
    rr.raise_for_status()
    on_progress("Downloaded updated adapter from remote worker.")
    return rr.content


def apply_adapter_zip(zip_bytes: bytes, adapter_dir: Path):
    """Unpack a received adapter zip into the local adapter directory,
    replacing whatever was there."""
    import shutil
    adapter_dir = Path(adapter_dir)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(adapter_dir)
