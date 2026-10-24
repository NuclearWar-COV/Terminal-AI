"""
model_registry.py — Single source of truth for which base chat models
Terminal AI V14.0 supports.

Both chatbot.py (for the model picker in Settings) and download_models.py
(for pre-fetching physical local copies) import from here, so the list
never goes out of sync between the two.

IMPORTANT: distilgpt2/gpt2 are NOT instruction/chat-tuned models — they
only continue text and were never trained to hold a conversation or
follow instructions. If a model "won't respond properly" to chat, that's
often why. They're kept here only as tiny, fast legacy options. Every
other entry below IS an instruct/chat-tuned model.
"""

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# (model_id, approx download size, note) — size/note are just for display.
MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",           # ~1GB   - good default, fast, actually chats
    "Qwen/Qwen2.5-1.5B-Instruct",           # ~3GB   - noticeably smarter, still CPU-friendly
    "HuggingFaceTB/SmolLM2-360M-Instruct",  # ~750MB - fastest option, weakest hardware
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # ~3.5GB
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",   # ~2.2GB
    "microsoft/Phi-3.5-mini-instruct",      # ~7.5GB - best quality, needs more RAM/CPU
    "distilgpt2",                           # ~330MB - legacy, NOT chat-tuned
    "gpt2",                                 # ~500MB - legacy, NOT chat-tuned
]

def slug(model_id: str) -> str:
    """Filesystem-safe folder name for a model id, e.g. 'Qwen/Qwen2.5-0.5B-Instruct'
    -> 'Qwen__Qwen2.5-0.5B-Instruct'."""
    return model_id.replace("/", "__")
