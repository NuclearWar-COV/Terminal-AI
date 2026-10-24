"""
Terminal AI V14.0 - LoRA / knowledge-only training
Base model weights stay frozen. Only a small LoRA adapter is trained and
saved, so personality / instruction-following is preserved while new
material is absorbed.

Two kinds of training data are supported, and can be mixed in the same
data_dir:
  - Structured prompt/answer examples: .csv, .tsv, .json, .jsonl, .parquet
    files with a prompt-like column (prompt/question/input/instruction)
    and an answer-like column (answer/response/output/completion). Each
    example is trained as supervised Q&A: the loss is only computed on
    the answer tokens (the prompt tokens are masked out with label -100),
    so the model actually learns to answer like the examples instead of
    just memorizing prompt text as filler.
  - Raw text files (.txt/.md/code files/etc): treated as a plain text
    corpus and chunked for continued-pretraining-style learning, same as
    previous versions.

Fixes from V8a/V8b/V8c:
  - safe_serialization=False on Windows (Errno 22)
  - use_cpu instead of deprecated no_cuda
  - real loss from state.log_history
  - continual learning: resumes existing adapter if present
  - FIXED: DataCollatorForLanguageModeling silently *recomputed* labels
    from input_ids on every batch, discarding any custom labels a caller
    set on the dataset. That's harmless for the old plain-text-only path
    (labels == input_ids either way) but would have silently destroyed
    prompt-masking for Q&A examples. Replaced with a small fixed-length
    collator that respects whatever labels were actually computed.
  - FIXED: padded positions in the raw-text path were previously
    scored in the loss (labels included the pad tokens); now masked to
    -100 like the Q&A path.
"""

import argparse, os, sys, glob, json, traceback, random, shutil, logging, time, math, platform
from pathlib import Path
from datetime import datetime

os.environ["TQDM_DISABLE"]           = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",      required=True)
parser.add_argument("--epochs",        type=int,   default=3)
parser.add_argument("--output",        type=str,   default=None)
parser.add_argument("--model",         type=str,   default=None)
parser.add_argument("--batch",         type=int,   default=1)
parser.add_argument("--batch_size",    type=int,   default=None)  # alias for C# UI
parser.add_argument("--maxlen",        type=int,   default=256)
parser.add_argument("--max_length",    type=int,   default=None)  # alias
parser.add_argument("--lr",            type=float, default=1e-4)
parser.add_argument("--learning_rate", type=float, default=None)  # alias
parser.add_argument("--lora_r",        type=int,   default=8)
parser.add_argument("--lora_alpha",    type=int,   default=16)
parser.add_argument("--lora_dropout",  type=float, default=0.05)
parser.add_argument("--freeze_base",   type=int,   default=1)
args = parser.parse_args()

# Normalize aliases from C# TrainingPanel
if args.batch_size is not None: args.batch = args.batch_size
if args.max_length is not None: args.maxlen = args.max_length
if args.learning_rate is not None: args.lr = args.learning_rate

ROOT       = Path(__file__).resolve().parent.parent
LOG_DIR    = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE   = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
OUTPUT_DIR = Path(args.output) if args.output else ROOT / "model_cache"
_requested_model = args.model or (str(OUTPUT_DIR) if (OUTPUT_DIR / "config.json").exists() else "distilgpt2")

def _is_local_model_dir(p) -> bool:
    p = Path(p)
    return p.exists() and (p / "config.json").exists()

if _is_local_model_dir(_requested_model):
    MODEL_SRC = _requested_model
else:
    # Not a path that exists on disk — probably a bare hub id like
    # "Qwen/Qwen2.5-0.5B-Instruct". Check the project's own local models/
    # folder (populated by the Downloads tab / download_models.py) before
    # falling back to the hub id itself, so an already-downloaded model
    # doesn't get re-fetched or hit the Hub for a metadata check.
    _local_copy = ROOT / "models" / _requested_model.replace("/", "__")
    MODEL_SRC = str(_local_copy) if _is_local_model_dir(_local_copy) else _requested_model
MODEL_SRC_IS_LOCAL = _is_local_model_dir(MODEL_SRC)
ADAPTER_DIR = OUTPUT_DIR / "_lora_adapter"

def log(msg: str):
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def progress(pct: float, eta_secs: float = -1):
    eta_str = f" ETA:{int(eta_secs)}s" if eta_secs >= 0 else ""
    print(f"PROGRESS_PCT:{pct:.2f}{eta_str}", flush=True)

log("=" * 60)
log("  Terminal AI V14.0 - Training (LoRA / knowledge-only)")
log(f"  Data:   {args.data_dir}")
log(f"  Output: {OUTPUT_DIR}")
log(f"  Model:  {MODEL_SRC}")
log(f"  Epochs: {args.epochs}  Batch: {args.batch}  MaxLen: {args.maxlen}  LR: {args.lr}")
log(f"  Freeze base: {'YES' if args.freeze_base else 'NO'}  LoRA r={args.lora_r} alpha={args.lora_alpha}")
log("=" * 60)

try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
        TrainerCallback,
        logging as hf_log,
    )
    from datasets import Dataset
    hf_log.set_verbosity_error()
    log(f"Libraries OK. PyTorch {torch.__version__}")
except ImportError as e:
    log(f"IMPORT ERROR: {e}")
    log("Run: pip install transformers torch accelerate datasets peft")
    sys.exit(1)

# ── Structured Q&A file support (csv/tsv/json/jsonl/parquet) ───
# pandas is only needed for csv/tsv/parquet; imported lazily below so a
# pure raw-text training run doesn't require it at all.
PROMPT_KEYS   = ("prompt", "question", "input", "instruction", "user", "text", "query", "context")
ANSWER_KEYS   = ("answer", "response", "output", "completion", "assistant", "target", "label", "chosen")
MESSAGES_KEYS = ("messages", "conversations", "conversation", "dialogue", "turns")

def _pick_col(columns, wanted):
    lower = {str(c).lower().strip(): c for c in columns}
    for w in wanted:
        if w in lower:
            return lower[w]
    return None

def _flatten_content(content):
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(str(c.get("text", c.get("content", ""))))
            else:
                parts.append(str(c))
        return " ".join(p for p in parts if p).strip()
    return str(content).strip()

def _pairs_from_messages(msgs):
    """A chat-format 'messages' cell/value -> list of (prompt, answer)
    pairs, pairing each user turn with the assistant turn that follows
    it. Accepts a list of {'role','content'} dicts, or the same thing
    JSON-encoded as a string (common when it comes out of a parquet
    cell)."""
    if isinstance(msgs, str):
        try:
            msgs = json.loads(msgs)
        except Exception:
            return []
    try:
        msgs = list(msgs)
    except Exception:
        return []
    pairs, pending_user = [], None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", m.get("from", ""))).lower()
        content = _flatten_content(m.get("content", m.get("value", "")))
        if role in ("user", "human"):
            pending_user = content
        elif role in ("assistant", "gpt", "bot", "ai") and pending_user is not None and content:
            pairs.append((pending_user, content))
            pending_user = None
    return pairs

def _extract_pair(obj):
    if not isinstance(obj, dict):
        return None, None
    lower = {str(k).lower().strip(): k for k in obj.keys()}
    pk = next((lower[w] for w in PROMPT_KEYS if w in lower), None)
    ak = next((lower[w] for w in ANSWER_KEYS if w in lower), None)
    if not pk or not ak:
        return None, None
    return str(obj.get(pk, "")).strip(), str(obj.get(ak, "")).strip()

def _pairs_from_dataframe(df, source_name):
    """Shared column-matching logic for csv/tsv/parquet dataframes."""
    mcol = _pick_col(df.columns, MESSAGES_KEYS)
    if mcol:
        pairs = []
        for _, row in df.iterrows():
            pairs.extend(_pairs_from_messages(row[mcol]))
        if pairs:
            return pairs, None
        return [], f"'{mcol}' column found but no usable user/assistant turns in it"

    pcol = _pick_col(df.columns, PROMPT_KEYS)
    acol = _pick_col(df.columns, ANSWER_KEYS)
    if pcol and acol and pcol != acol:
        pairs = [(str(r[pcol]).strip(), str(r[acol]).strip()) for _, r in df.iterrows()]
        return [p for p in pairs if p[0] and p[1]], None

    # Nothing matched a known name. If it's exactly two columns, assume
    # they're (prompt, answer) in file order rather than skip the file —
    # log the guess so it's easy to check the columns weren't swapped.
    if len(df.columns) == 2:
        c0, c1 = df.columns[0], df.columns[1]
        log(f"  (guessing '{c0}' = prompt, '{c1}' = answer in {source_name} — "
            f"rename columns if that's backwards)")
        pairs = [(str(r[c0]).strip(), str(r[c1]).strip()) for _, r in df.iterrows()]
        return [p for p in pairs if p[0] and p[1]], None

    return [], f"no prompt/answer-style columns found (has: {', '.join(map(str, df.columns))})"

def load_qa_pairs(path: Path):
    """Returns (pairs, error). pairs is a list of (prompt, answer) tuples
    found in a single structured file; error is a short string if the
    file couldn't be used at all (missing columns, bad format, etc)."""
    ext = path.suffix.lower()
    try:
        if ext in (".csv", ".tsv"):
            import pandas as pd
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
            # A file with no real header row (data starts on line 1) makes
            # pandas treat the first data row as column names — none of
            # which will match. Re-read headerless and try again.
            if not _pick_col(df.columns, PROMPT_KEYS + ANSWER_KEYS + MESSAGES_KEYS) and len(df.columns) <= 3:
                df_noheader = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, header=None)
                if len(df_noheader.columns) == 2:
                    df = df_noheader
            return _pairs_from_dataframe(df, path.name)
        elif ext == ".parquet":
            import pandas as pd
            df = pd.read_parquet(path)
            return _pairs_from_dataframe(df, path.name)
        elif ext == ".jsonl":
            pairs = []
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    mkey = next((k for k in obj if str(k).lower() in MESSAGES_KEYS), None) if isinstance(obj, dict) else None
                    if mkey:
                        pairs.extend(_pairs_from_messages(obj[mkey]))
                        continue
                    p, a = _extract_pair(obj)
                    if p and a:
                        pairs.append((p, a))
            return pairs, (None if pairs else "no prompt/answer-style fields or usable 'messages' found")
        elif ext == ".json":
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            items = obj if isinstance(obj, list) else [obj]
            pairs = []
            for it in items:
                mkey = next((k for k in it if str(k).lower() in MESSAGES_KEYS), None) if isinstance(it, dict) else None
                if mkey:
                    pairs.extend(_pairs_from_messages(it[mkey]))
                    continue
                p, a = _extract_pair(it)
                if p and a:
                    pairs.append((p, a))
            return pairs, (None if pairs else "no prompt/answer-style fields or usable 'messages' found")
    except ImportError:
        return [], "pandas (and pyarrow, for .parquet) isn't installed — run: pip install pandas pyarrow"
    except Exception as e:
        return [], str(e)
    return [], "unrecognized format"

PEFT_OK = True
if args.freeze_base:
    try:
        from peft import LoraConfig, get_peft_model, PeftModel
        log("PEFT (LoRA) library OK")
    except ImportError:
        PEFT_OK = False
        log("WARNING: 'peft' not installed — falling back to full fine-tune.")
        log("         Run: pip install peft")

# ── Collect training files ────────────────────────────────────
RAW_TEXT_EXTS = [".txt",".md",".py",".js",".ts",".java",".cs",".cpp",".c",".go",".rs",".html",".css"]
QA_EXTS       = [".csv",".tsv",".json",".jsonl",".parquet"]
data_dir = Path(args.data_dir)
if not data_dir.exists():
    log(f"ERROR: data_dir not found: {data_dir}")
    sys.exit(1)

log(f"\nScanning {data_dir} for structured Q&A files ({', '.join(QA_EXTS)})...")
qa_files = []
for ext in QA_EXTS:
    qa_files.extend(data_dir.rglob(f"*{ext}"))

qa_pairs = []
for f in qa_files:
    if f.name.startswith("_"):
        continue
    pairs, err = load_qa_pairs(f)
    if err:
        log(f"  ! Skip {f.name}: {err}")
        continue
    if pairs:
        qa_pairs.extend(pairs)
        log(f"  + {f.name} ({len(pairs)} prompt/answer example(s))")
log(f"Q&A examples loaded: {len(qa_pairs)}")

log(f"\nScanning {data_dir} for raw text files ({', '.join(RAW_TEXT_EXTS)})...")
files = []
for ext in RAW_TEXT_EXTS:
    files.extend(data_dir.rglob(f"*{ext}"))

texts, skipped = [], 0
for f in files:
    try:
        if f.name.startswith("_"):
            skipped += 1
            continue
        content = f.read_text(encoding="utf-8", errors="ignore").strip()
        if len(content) < 30:
            skipped += 1
            continue
        texts.append(content)
        log(f"  + {f.name} ({len(content):,} chars)")
    except Exception as e:
        log(f"  ! Skip {f.name}: {e}")
        skipped += 1

log(f"\nRaw text files loaded: {len(texts)}  Skipped: {skipped}")
log(f"Total chars: {sum(len(t) for t in texts):,}")
if not qa_pairs and not texts:
    log(f"ERROR: No usable training data found. Supported: prompt/answer files "
        f"({', '.join(QA_EXTS)}) or raw text files ({', '.join(RAW_TEXT_EXTS)}).")
    sys.exit(1)

# ── Load model ────────────────────────────────────────────────
progress(2.0)
log(f"\nLoading model: {MODEL_SRC}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_SRC, local_files_only=MODEL_SRC_IS_LOCAL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log("Tokenizer OK")
except Exception as e:
    log(f"ERROR: tokenizer: {e}")
    sys.exit(1)

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # "auto" uses the checkpoint's own native dtype (bf16 for most modern
    # models) instead of upcasting the frozen base model to float32 on
    # CPU, which was needlessly doubling RAM use during training too —
    # same root cause as the live-chat model load.
    model  = AutoModelForCausalLM.from_pretrained(MODEL_SRC, torch_dtype="auto", local_files_only=MODEL_SRC_IS_LOCAL)

    use_lora = bool(args.freeze_base) and PEFT_OK
    if use_lora:
        if ADAPTER_DIR.exists() and (ADAPTER_DIR / "adapter_config.json").exists():
            log(f"Resuming existing LoRA adapter from {ADAPTER_DIR}")
            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR), is_trainable=True)
        else:
            log("Creating new LoRA adapter (base model stays frozen)")
            lora_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj","k_proj","v_proj","o_proj",
                                 "gate_proj","up_proj","down_proj",
                                 "c_attn","c_proj","c_fc"],
            )
            model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        log("FULL fine-tune mode (all weights trainable)")

    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log(f"Model OK — {total:,} total, {trainable:,} trainable "
        f"({trainable/total*100:.2f}%) on {device.upper()}")
except Exception as e:
    log(f"ERROR: model: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# ── Chunk ─────────────────────────────────────────────────────
progress(8.0)
log("\nTokenizing and chunking...")
CHUNK   = args.maxlen
OVERLAP = CHUNK // 4

def chunk_texts(text_list):
    """Raw-text corpus -> fixed-length LM chunks. Padded positions are
    masked out of the loss (-100), not scored against the pad token."""
    chunks = []
    for text in text_list:
        ids  = tokenizer.encode(text, add_special_tokens=False)
        step = CHUNK - OVERLAP
        for start in range(0, max(1, len(ids) - OVERLAP), step):
            cids = ids[start : start + CHUNK]
            if len(cids) < 16:
                continue
            n_real = len(cids)
            pad = CHUNK - n_real
            input_ids = cids + [tokenizer.pad_token_id] * pad
            attention_mask = [1] * n_real + [0] * pad
            labels = cids[:] + [-100] * pad
            chunks.append({
                "input_ids":      input_ids,
                "attention_mask": attention_mask,
                "labels":         labels,
            })
    return chunks

def build_qa_examples(pairs):
    """Prompt/answer pairs -> fixed-length supervised examples. Loss is
    only computed on the answer tokens (and the eos that ends them) —
    prompt tokens and padding are masked out with label -100, so the
    model learns to produce that answer rather than to just predict the
    prompt text back."""
    examples = []
    eos = tokenizer.eos_token or ""
    for prompt, answer in pairs:
        prefix = f"User: {prompt}\nAssistant: "
        full = prefix + answer + eos
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(full, add_special_tokens=False)
        if len(full_ids) <= len(prefix_ids):
            continue  # answer got tokenized away to nothing usable
        if len(full_ids) > CHUNK:
            full_ids = full_ids[:CHUNK]
        n_real = len(full_ids)
        pad = CHUNK - n_real
        input_ids = full_ids + [tokenizer.pad_token_id] * pad
        attention_mask = [1] * n_real + [0] * pad
        plen = min(len(prefix_ids), n_real)
        labels = ([-100] * plen) + full_ids[plen:] + ([-100] * pad)
        if all(l == -100 for l in labels):
            continue
        examples.append({
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        })
    return examples

class FixedLengthCollator:
    """All examples (Q&A and raw-text alike) are already padded/truncated
    to exactly CHUNK tokens with their own precomputed labels above, so
    batching is just a stack — no dynamic padding needed. Deliberately
    NOT using DataCollatorForLanguageModeling here: it recomputes labels
    from input_ids on every call (cloning input_ids and masking the pad
    token), which would silently throw away the prompt-masking done in
    build_qa_examples above."""
    def __call__(self, features):
        return {
            "input_ids":      torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
            "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
            "labels":         torch.tensor([f["labels"] for f in features], dtype=torch.long),
        }

try:
    qa_chunks = build_qa_examples(qa_pairs) if qa_pairs else []
    if qa_pairs and not qa_chunks:
        log("WARNING: Q&A examples were found but none produced usable training chunks.")
    text_chunks = chunk_texts(texts) if texts else []
    chunks = qa_chunks + text_chunks
    if not chunks:
        log("ERROR: No chunks produced.")
        sys.exit(1)
    random.shuffle(chunks)
    n_eval   = max(1, min(20, len(chunks) // 10))
    n_train  = len(chunks) - n_eval
    train_ds = Dataset.from_list(chunks[:n_train])
    eval_ds  = Dataset.from_list(chunks[n_train:])
    log(f"Chunks: {len(chunks)} total (Q&A: {len(qa_chunks)}, text: {len(text_chunks)})  |  Train: {n_train}  Eval: {n_eval}")
except Exception as e:
    log(f"ERROR tokenizing: {e}\n{traceback.format_exc()}")
    sys.exit(1)

# ── Train ─────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
total_steps = math.ceil(n_train / args.batch) * args.epochs
progress(10.0)
log(f"\nStarting {args.epochs} epoch(s) on {device.upper()}")
log(f"Total steps: {total_steps}\n")

class ProgressCallback(TrainerCallback):
    def __init__(self):
        self._start = time.time()
        self._step  = 0
        self._total = total_steps
        self._epoch_start = time.time()

    def on_epoch_begin(self, args, state, control, **kwargs):
        ep = int(state.epoch or 0) + 1
        self._epoch_start = time.time()
        log(f"Epoch {ep}/{int(args.num_train_epochs)} started")

    def on_step_end(self, args, state, control, **kwargs):
        self._step += 1
        elapsed = time.time() - self._start
        train_pct = self._step / max(self._total, 1)
        pct = 10.0 + train_pct * 78.0
        eta = (elapsed / self._step) * (self._total - self._step) if self._step > 0 else -1
        progress(pct, eta)

    def on_epoch_end(self, args, state, control, **kwargs):
        ep = int(state.epoch or 0)
        hist = [h for h in state.log_history if "loss" in h]
        loss = f"{hist[-1]['loss']:.4f}" if hist else "?"
        secs = time.time() - self._epoch_start
        log(f"Epoch {ep} done  |  loss: {loss}  |  time: {secs:.0f}s")

try:
    import inspect
    valid_params = set(inspect.signature(TrainingArguments.__init__).parameters)
    desired = {
        "output_dir":                   str(OUTPUT_DIR / "_trainer_tmp"),
        "num_train_epochs":              args.epochs,
        "per_device_train_batch_size":   args.batch,
        "per_device_eval_batch_size":    args.batch,
        "warmup_steps":                  min(50, n_train // 2),
        "weight_decay":                  0.01,
        "learning_rate":                 args.lr,
        "logging_steps":                 999999,
        "eval_strategy":                 "epoch",
        "evaluation_strategy":           "epoch",
        "save_strategy":                 "no",
        "load_best_model_at_end":        False,
        "report_to":                     "none",
        "no_cuda":                       (device == "cpu"),
        "use_cpu":                       (device == "cpu"),
        "fp16":                          (device == "cuda"),
        "dataloader_pin_memory":         False,
        "remove_unused_columns":         False,
        "disable_tqdm":                  True,
        "log_level":                     "error",
    }
    filtered = {k: v for k, v in desired.items() if k in valid_params}
    t_args = TrainingArguments(**filtered)

    trainer = Trainer(
        model=model,
        args=t_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=FixedLengthCollator(),
        callbacks=[ProgressCallback()],
    )
    trainer.train()
    progress(88.0)

    ON_WINDOWS = platform.system() == "Windows"
    if use_lora:
        log("\nTraining complete. Saving LoRA adapter (base model untouched)...")
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        progress(92.0)
        model.save_pretrained(str(ADAPTER_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        progress(97.0)
        log(f"Adapter saved to: {ADAPTER_DIR}")
        log(f"(Base model weights in {MODEL_SRC} were never modified.)")
    else:
        log("\nTraining complete. Saving full model...")
        TMP = OUTPUT_DIR.parent / "_model_tmp"
        if TMP.exists(): shutil.rmtree(TMP)
        TMP.mkdir(parents=True)
        progress(90.0)
        model.save_pretrained(str(TMP), safe_serialization=not ON_WINDOWS)
        tokenizer.save_pretrained(str(TMP))
        progress(95.0)
        for src in TMP.iterdir():
            try:
                shutil.copy2(str(src), str(OUTPUT_DIR / src.name))
                log(f"  copied: {src.name}")
            except Exception as ce:
                log(f"  WARNING: {src.name}: {ce}")
        shutil.rmtree(TMP, ignore_errors=True)

    shutil.rmtree(OUTPUT_DIR / "_trainer_tmp", ignore_errors=True)
    progress(100.0)
    log(f"\nModel + adapter saved under: {OUTPUT_DIR}")
    log("TRAINING_SUCCESS")

except Exception as e:
    log(f"\nTRAINING_ERROR: {e}\n{traceback.format_exc()}")
    sys.exit(1)
