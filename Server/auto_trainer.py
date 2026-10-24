"""
auto_trainer.py — Background auto-training for Terminal AI V14.0.
Runs as a separate process. Communicates via a JSON state file.
"""

import sys, os, json, time, argparse, subprocess, traceback
from pathlib import Path
from datetime import datetime

BASE_DIR   = Path(__file__).resolve().parent          # server/
ROOT       = BASE_DIR.parent                          # project root
LOG_DIR    = ROOT / "logs"
STATE_FILE = ROOT / "auto_train_state.json"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE   = LOG_DIR / f"auto_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

parser = argparse.ArgumentParser()
parser.add_argument("--topics",    type=str, default="")
parser.add_argument("--engine",    type=str, default="duckduckgo")
parser.add_argument("--epochs",    type=int, default=2)
parser.add_argument("--max_pages", type=int, default=5)
parser.add_argument("--interval",  type=int, default=3600)
parser.add_argument("--cpu_limit", type=int, default=50)
parser.add_argument("--ram_limit", type=int, default=2048)
parser.add_argument("--model_src", type=str, default="")
parser.add_argument("--model_out", type=str, default="")
parser.add_argument("--model_id",  type=str, default="")
parser.add_argument("--remote", type=int, default=0)
parser.add_argument("--skip_visited", type=int, default=1)
args = parser.parse_args()

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def write_state(state: dict):
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception:
        pass

def read_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}
    except Exception:
        return {}

def check_paused() -> bool:
    return read_state().get("paused", False)

def check_stopped() -> bool:
    return read_state().get("stopped", False)

def apply_resource_limits():
    try:
        import psutil
        p = psutil.Process(os.getpid())
        if hasattr(p, "nice"):
            p.nice(10)
    except Exception:
        pass

apply_resource_limits()
sys.path.insert(0, str(BASE_DIR))

topics = [t.strip() for t in args.topics.split(",") if t.strip()]
if not topics:
    log("No topics provided — using default: 'artificial intelligence'")
    topics = ["artificial intelligence"]

log(f"Auto-trainer started. Topics: {topics}")
log(f"Engine: {args.engine}  Epochs: {args.epochs}  MaxPages: {args.max_pages}")
log(f"Interval: {args.interval}s  CPU limit: {args.cpu_limit}%")

write_state({
    "running": True, "paused": False, "stopped": False,
    "current_topic": "", "last_run": "", "sessions": [],
    "pid": os.getpid(),
})

def run_one_cycle():
    import random
    topic = random.choice(topics)
    log(f"\n=== Starting cycle: '{topic}' ===")
    write_state({**read_state(), "current_topic": topic, "last_run": datetime.now().isoformat()})

    try:
        from search_engine import search
        from webscraper import fetch_page, save_scraped_data, make_session
        results = search(topic, engine=args.engine, n=10,
                          on_engine_used=lambda name, err: log(
                              f"  search[{name}]: {'FAILED - ' + err if err else 'OK'}"))
        log(f"Found {len(results)} results for '{topic}'")
    except Exception as e:
        log(f"Search failed: {e}")
        return

    if not results:
        log("No results — skipping cycle")
        return

    from webscraper import load_visited, save_visited
    visited_before = load_visited() if args.skip_visited else set()
    session = make_session()
    pages   = []
    skipped = 0
    for res in results:
        if len(pages) >= args.max_pages:
            break
        if check_paused() or check_stopped():
            log("Paused/stopped during crawl")
            return
        url = res["url"]
        if args.skip_visited and url in visited_before:
            skipped += 1
            continue
        log(f"  Crawling: {url[:80]}")
        page = fetch_page(session, url)
        if page:
            pages.append(page)
            log(f"    OK: {page['title'][:50]} ({len(page['text']):,} chars)")
        time.sleep(1.0)
    if skipped:
        log(f"  Skipped {skipped} previously-scraped URL(s)")

    if not pages:
        log("No pages fetched — skipping training")
        return

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = save_scraped_data(pages, ROOT / "web_training_data", f"auto_{ts}")
    log(f"Saved {len(pages)} pages to {data_dir}")
    if args.skip_visited:
        save_visited({p["url"] for p in pages})

    log("Starting training...")
    si = None
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0

    model_src = args.model_src or str(ROOT / "model_cache")
    model_out = args.model_out or model_src

    if args.remote and args.model_id:
        try:
            import remote_train
            worker = remote_train.pick_best_worker()
            if worker:
                log(f"Dispatching to remote worker: {worker.get('name', worker['url'])}")
                adapter_dir = Path(model_out) / "_lora_adapter"
                zip_bytes = remote_train.dispatch_remote_training(
                    worker, str(data_dir), args.model_id, args.epochs,
                    adapter_dir=adapter_dir, on_progress=log,
                )
                remote_train.apply_adapter_zip(zip_bytes, adapter_dir)
                log("Remote training cycle complete.")
                sessions = read_state().get("sessions", [])
                sessions.append({"timestamp": ts, "topic": topic, "pages": len(pages),
                                  "epochs": args.epochs, "success": True,
                                  "worker": worker.get("name", worker["url"])})
                write_state({**read_state(), "sessions": sessions})
                return
            else:
                log("No idle remote worker — training locally.")
        except Exception as e:
            log(f"Remote dispatch failed ({e}) — falling back to local.")

    proc = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "train.py"),
         "--data_dir", str(data_dir),
         "--epochs",   str(args.epochs),
         "--model",    model_src,
         "--output",   model_out,
         "--batch",    "1", "--maxlen", "256"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, startupinfo=si,
    )
    from resource_throttle import start_throttle
    log(f"Enforcing limits: CPU <= {args.cpu_limit}%, RAM <= {args.ram_limit}MB")
    throttle_stop = start_throttle(proc.pid, args.cpu_limit, args.ram_limit, log)
    for line in proc.stdout:
        line = line.rstrip()
        if line and not line.startswith("PROGRESS_PCT"):
            log(f"  [train] {line}")
        if check_paused():
            log("Paused — killing training process")
            proc.kill()
            throttle_stop.set()
            return
        if check_stopped():
            proc.kill()
            throttle_stop.set()
            return
    proc.wait()
    throttle_stop.set()

    sessions = read_state().get("sessions", [])
    sessions.append({
        "timestamp": ts, "topic": topic,
        "pages": len(pages), "epochs": args.epochs,
        "success": proc.returncode == 0,
    })
    write_state({**read_state(), "sessions": sessions})
    log(f"Cycle complete. returncode={proc.returncode}")

while True:
    if check_stopped():
        log("Stopped by user.")
        write_state({**read_state(), "running": False})
        break

    while check_paused():
        log("Paused — waiting...")
        time.sleep(5)
        if check_stopped():
            break

    if check_stopped():
        break

    try:
        run_one_cycle()
    except Exception as e:
        log(f"Cycle error: {e}\n{traceback.format_exc()}")

    log(f"Sleeping {args.interval}s until next cycle...")
    for _ in range(args.interval):
        time.sleep(1)
        if check_paused() or check_stopped():
            break
