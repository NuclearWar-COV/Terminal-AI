import sys, os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

BASE_DIR    = Path(__file__).parent
MODELS_ROOT = BASE_DIR.parent / "models"
MODELS_ROOT.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))
from model_registry import MODELS, slug


def already_downloaded(model_id: str) -> bool:
    d = MODELS_ROOT / slug(model_id)
    return (d / "config.json").exists() or (d / "model_index.json").exists()


def _download_diffusers(model_id: str, d: Path):
    """Returns True/False if model_id is an image/video (diffusers) repo and
    the download succeeded/failed, or None if it isn't a diffusers repo at all
    (so the caller falls back to the chat-model path)."""
    try:
        from huggingface_hub import list_repo_files, snapshot_download
        files = list_repo_files(model_id)
    except Exception:
        return None
    if "model_index.json" not in files:
        return None
    from media_models import filter_repo_files
    keep = filter_repo_files(files)
    print(f"  {model_id} is an image/video (diffusers) model — fetching {len(keep)} of {len(files)} files "
          f"(skipping duplicate fp32/.ckpt/.onnx copies)...")
    try:
        d.mkdir(parents=True, exist_ok=True)
        snapshot_download(model_id, local_dir=str(d), allow_patterns=keep)
        print(f"  OK — saved to {d}")
        return True
    except Exception as e:
        hint = ""
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            hint = " (this repo is gated: accept its license on huggingface.co and log in with `huggingface-cli login`)"
        print(f"  FAILED: {model_id} — {e}{hint}")
        return False


def download_one(model_id: str) -> bool:
    d = MODELS_ROOT / slug(model_id)
    if already_downloaded(model_id):
        print(f"  [skip] {model_id} — already have a physical copy at {d}")
        return True
    print(f"  Downloading {model_id} ...")
    media = _download_diffusers(model_id, d)
    if media is not None:
        return media
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        d.mkdir(parents=True, exist_ok=True)
        AutoTokenizer.from_pretrained(model_id).save_pretrained(str(d))
        AutoModelForCausalLM.from_pretrained(model_id).save_pretrained(str(d))
        print(f"  OK — saved to {d}")
        return True
    except Exception as e:
        print(f"  FAILED: {model_id} — {e}")
        return False


def main():
    targets = sys.argv[1:] or MODELS
    print(f"Fetching {len(targets)} model(s) into: {MODELS_ROOT}\n")
    ok = 0
    for model_id in targets:
        if download_one(model_id):
            ok += 1
        print()
    print(f"Done: {ok}/{len(targets)} models available offline in {MODELS_ROOT}")
    if ok < len(targets):
        print("Some models failed to download — check your internet connection "
              "and try re-running this script (already-downloaded ones will be skipped).")


if __name__ == "__main__":
    main()
