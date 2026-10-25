

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",           # ~1GB
    "Qwen/Qwen2.5-1.5B-Instruct",           # ~3GB
    "HuggingFaceTB/SmolLM2-360M-Instruct",  # ~750MB
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",  # ~3.5GB
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",   # ~2.2GB
    "microsoft/Phi-3.5-mini-instruct",      # ~7.5GB
    "distilgpt2",                           # ~330MB
    "gpt2",                                 # ~500MB
]

def slug(model_id: str) -> str:
    """Filesystem-safe folder name for a model id, e.g. 'Qwen/Qwen2.5-0.5B-Instruct'
    -> 'Qwen__Qwen2.5-0.5B-Instruct'."""
    return model_id.replace("/", "__")
