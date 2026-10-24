"""
Terminal AI V14.0 - Online Conversation Trainer
Connects to Claude / OpenAI / custom AI and runs a structured conversation.
Transcript is saved as training data in web_training_data/conv_<timestamp>/
"""

import argparse, json, os, sys, time, logging, urllib.request, urllib.error
from pathlib import Path
from datetime import datetime

ROOT         = Path(__file__).resolve().parent.parent
LOG_DIR      = ROOT / "logs"
TRAINING_DIR = ROOT / "web_training_data"
LOG_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_DIR / "conv_train.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("conv")

parser = argparse.ArgumentParser()
parser.add_argument("--provider",   choices=["claude","openai","custom"], default="claude")
parser.add_argument("--api_key",    type=str, default="")
parser.add_argument("--endpoint",   type=str, default="http://localhost:11434/api/chat")
parser.add_argument("--model",      type=str, default="")
parser.add_argument("--topic",      type=str, default="general programming and coding best practices")
parser.add_argument("--turns",      type=int, default=10)
parser.add_argument("--local_port", type=int, default=5001)
args = parser.parse_args()

DEFAULT_MODELS = {
    "claude": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "custom": ""
}
REMOTE_MODEL = args.model or DEFAULT_MODELS.get(args.provider, "")

# ── Local AI ──────────────────────────────────────────────────
def query_local(message: str, history: list) -> str:
    payload = json.dumps({"message": message, "history": history}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{args.local_port}/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read()).get("reply", "[no reply]")
    except Exception as e:
        return f"[local error: {e}]"

# ── Remote AI ─────────────────────────────────────────────────
def query_remote(message: str, history: list) -> str:
    if args.provider == "claude":  return _claude(message, history)
    if args.provider == "openai":  return _openai(message, history)
    return _custom(message, history)

def _claude(message: str, history: list) -> str:
    msgs = [{"role": h["role"], "content": h["content"]} for h in history]
    msgs.append({"role": "user", "content": message})
    payload = json.dumps({
        "model": REMOTE_MODEL,
        "max_tokens": 600,
        "messages": msgs
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": args.api_key,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["content"][0]["text"]
    except urllib.error.HTTPError as e:
        return f"[Claude HTTP {e.code}: {e.read().decode()[:200]}]"
    except Exception as e:
        return f"[Claude error: {e}]"

def _openai(message: str, history: list) -> str:
    msgs = [{"role": "system", "content": f"You are a helpful assistant discussing: {args.topic}."}]
    msgs += [{"role": h["role"], "content": h["content"]} for h in history]
    msgs.append({"role": "user", "content": message})
    payload = json.dumps({"model": REMOTE_MODEL, "messages": msgs, "max_tokens": 600}).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        return f"[OpenAI HTTP {e.code}: {e.read().decode()[:200]}]"
    except Exception as e:
        return f"[OpenAI error: {e}]"

def _custom(message: str, history: list) -> str:
    msgs = [{"role": h["role"], "content": h["content"]} for h in history]
    msgs.append({"role": "user", "content": message})
    payload = json.dumps({
        "model": REMOTE_MODEL or "llama3",
        "messages": msgs,
        "stream": False
    }).encode()
    req = urllib.request.Request(
        args.endpoint, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
            if "message" in data:    return data["message"]["content"]
            if "choices" in data:    return data["choices"][0]["message"]["content"]
            return str(data)
    except Exception as e:
        return f"[custom error: {e}]"

# ── Run ───────────────────────────────────────────────────────
def run():
    log.info("=" * 60)
    log.info("  AI-to-AI Conversation Training")
    log.info(f"  Provider: {args.provider}  Remote model: {REMOTE_MODEL}")
    log.info(f"  Topic: {args.topic}")
    log.info(f"  Turns: {args.turns}")
    log.info("=" * 60)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir   = TRAINING_DIR / f"conv_{session_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    local_hist  = []
    remote_hist = []
    lines = [
        f"# AI-to-AI Conversation Training",
        f"# Topic:    {args.topic}",
        f"# Provider: {args.provider} / {REMOTE_MODEL}",
        f"# Date:     {session_id}",
        ""
    ]

    seed = f"Let's have a detailed technical discussion about: {args.topic}. Please start with an overview."

    for turn in range(args.turns):
        log.info(f"\n[Turn {turn+1}/{args.turns}]")

        # Remote AI speaks
        if turn == 0:
            remote_msg = seed
        else:
            log.info(f"  {args.provider} thinking...")
            last_local = local_hist[-1]["content"] if local_hist else seed
            remote_msg = query_remote(last_local, remote_hist)
            remote_hist.append({"role": "assistant", "content": remote_msg})
        log.info(f"  [{args.provider.upper()}]: {remote_msg[:100]}...")

        # Local AI responds
        log.info("  Local AI thinking...")
        local_reply = query_local(remote_msg, local_hist)
        local_hist.append({"role": "user",      "content": remote_msg})
        local_hist.append({"role": "assistant", "content": local_reply})
        remote_hist.append({"role": "user",     "content": local_reply})
        log.info(f"  [LOCAL]: {local_reply[:100]}...")

        lines.append(f"[{args.provider.upper()}]: {remote_msg}")
        lines.append(f"[LOCAL AI]: {local_reply}")
        lines.append("")

        time.sleep(0.5)

    out_file = save_dir / f"conversation_{session_id}.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")

    log.info(f"\nTranscript saved: {out_file}")
    log.info("CONV_TRAIN_SUCCESS")
    print(f"SAVE_PATH:{save_dir}")

if __name__ == "__main__":
    run()
