"""
hardware_scan.py — Scans all available hardware and AI acceleration options.
Called once at startup, results cached in hw_info dict.
"""
import sys, os, subprocess, platform, json
from pathlib import Path

def scan() -> dict:
    hw = {
        "cpu": {}, "ram": {}, "gpu": [], "npu": None,
        "torch_cuda": False, "torch_directml": False,
        "openvino": False, "openvino_devices": [],
        "best_device": "cpu", "best_label": "CPU",
        "python": sys.version,
        "platform": platform.platform(),
    }

    # ── CPU ───────────────────────────────────────────────────────────────────
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        hw["cpu"]["name"]   = info.get("brand_raw", "Unknown CPU")
        hw["cpu"]["cores"]  = os.cpu_count() or 1
        hw["cpu"]["arch"]   = info.get("arch", "?")
    except Exception:
        hw["cpu"]["name"]  = platform.processor() or "Unknown CPU"
        hw["cpu"]["cores"] = os.cpu_count() or 1
        hw["cpu"]["arch"]  = platform.machine()

    # ── RAM ───────────────────────────────────────────────────────────────────
    try:
        import psutil
        vm = psutil.virtual_memory()
        hw["ram"]["total_gb"] = round(vm.total / (1024**3), 1)
        hw["ram"]["avail_gb"] = round(vm.available / (1024**3), 1)
        hw["psutil"] = True
    except Exception:
        hw["ram"]["total_gb"] = "?"
        hw["ram"]["avail_gb"] = "?"
        hw["psutil"] = False

    # ── NVIDIA CUDA ───────────────────────────────────────────────────────────
    hw["cuda_warning"] = None
    try:
        import torch, warnings, subprocess as _sp
        hw["torch_version"] = torch.__version__
        # An NVIDIA GPU physically present but not supported by THIS torch
        # build (wrong CUDA version for the card's compute capability —
        # RTX 50-series/Blackwell needs a cu128+ build, for example) shows
        # up as a UserWarning right here rather than an exception, and
        # torch.cuda.is_available() can still come back False after it —
        # which otherwise looks identical to "no GPU at all". Capture the
        # warning so the real reason shows up instead of just "CPU".
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cuda_ok = torch.cuda.is_available()
            for w in caught:
                msg = str(w.message)
                if "not compatible" in msg or "CUDA capability" in msg:
                    hw["cuda_warning"] = msg
        if cuda_ok:
            hw["torch_cuda"] = True
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                hw["gpu"].append({
                    "name":    props.name,
                    "vram_gb": round(props.total_memory / (1024**3), 1),
                    "type":    "CUDA",
                })
        elif not hw["cuda_warning"]:
            # is_available() was False and printed no warning — most
            # likely this is a genuinely CPU-only torch build (the
            # default from `pip install torch` with no index URL) rather
            # than a real absence of hardware. Check nvidia-smi directly
            # to tell those two cases apart for whoever reads this later.
            try:
                _sp.run(["nvidia-smi"], capture_output=True, timeout=3, check=True)
                hw["cuda_warning"] = ("nvidia-smi found an NVIDIA GPU, but this is a CPU-only PyTorch "
                                       "build (installed without a CUDA index URL). Reinstall with: "
                                       "pip install torch torchvision torchaudio "
                                       "--index-url https://download.pytorch.org/whl/cu128")
            except Exception:
                pass  # no nvidia-smi -> genuinely no NVIDIA GPU, nothing to warn about
    except Exception:
        hw["torch_version"] = "not installed"

    # ── DirectML (AMD / Intel Arc on Windows) ─────────────────────────────────
    try:
        import torch_directml
        hw["torch_directml"] = True
        hw["gpu"].append({"name": "DirectML GPU", "vram_gb": "?", "type": "DirectML"})
    except Exception:
        pass

    # ── OpenVINO (Intel NPU + iGPU) ───────────────────────────────────────────
    try:
        # OpenVINO moved Core out of the 'runtime' submodule in newer
        # releases; try the modern path first, then fall back to the
        # older one so this works across versions.
        try:
            from openvino import Core
        except ImportError:
            from openvino.runtime import Core
        core = Core()
        devs = core.available_devices
        hw["openvino"]         = True
        hw["openvino_devices"] = devs
        hw["openvino_version"] = core.get_versions("CPU").get("CPU", {}).get("build_number", "?")

        for d in devs:
            try:
                name = core.get_property(d, "FULL_DEVICE_NAME")
            except Exception:
                name = d
            if "NPU" in d:
                hw["npu"] = {"name": name, "device_id": d}
                hw["gpu"].append({"name": name, "vram_gb": "N/A", "type": "NPU"})
            elif "GPU" in d:
                hw["gpu"].append({"name": name, "vram_gb": "?", "type": "iGPU/OpenVINO"})
    except Exception:
        pass

    # ── Optimum Intel ─────────────────────────────────────────────────────────
    try:
        import optimum
        hw["optimum_version"] = optimum.__version__
        hw["optimum_intel"]   = True
    except Exception:
        hw["optimum_intel"] = False

    # ── Pick best device ──────────────────────────────────────────────────────
    if hw["torch_cuda"] and hw["gpu"]:
        cuda_gpu = next((g for g in hw["gpu"] if g["type"] == "CUDA"), None)
        if cuda_gpu:
            hw["best_device"] = "cuda"
            hw["best_label"]  = f"CUDA — {cuda_gpu['name']}"
    elif hw["npu"]:
        hw["best_device"] = "NPU"
        hw["best_label"]  = f"Intel NPU — {hw['npu']['name']}"
    elif hw["openvino"] and any("GPU" in d for d in hw["openvino_devices"]):
        hw["best_device"] = "GPU"
        hw["best_label"]  = "Intel iGPU (OpenVINO)"
    elif hw["torch_directml"]:
        hw["best_device"] = "directml"
        hw["best_label"]  = "DirectML GPU"
    else:
        hw["best_device"] = "cpu"
        hw["best_label"]  = f"CPU — {hw['cpu']['name']}"

    return hw


# ── Integrated GPU usage (Windows only) ────────────────────────────────────────
# torch.cuda/NVML only exists for discrete NVIDIA cards. Intel/AMD
# integrated GPUs don't expose usage that way at all -- Windows tracks it
# through a completely different mechanism: the "GPU Engine" performance
# counters (this is also how Task Manager itself gets its iGPU numbers).
# Requires 'pywin32' (optional install); degrades to None if unavailable.
_igpu_query      = None
_igpu_counters   = []
_igpu_last_paths = None

def _igpu_percent():
    global _igpu_query, _igpu_counters, _igpu_last_paths
    try:
        import win32pdh
    except ImportError:
        return None  # pywin32 not installed -- caller shows "N/A"
    try:
        paths = win32pdh.ExpandCounterPath(r"\GPU Engine(*)\Utilization Percentage")
    except Exception:
        return None
    if not paths:
        return None

    # (Re)build the query if this is the first call or the set of GPU
    # engine instances changed (engines come and go as apps use the GPU).
    if _igpu_query is None or paths != _igpu_last_paths:
        if _igpu_query is not None:
            try: win32pdh.CloseQuery(_igpu_query)
            except Exception: pass
        _igpu_query    = win32pdh.OpenQuery()
        _igpu_counters = []
        for p in paths:
            try:
                _igpu_counters.append(win32pdh.AddCounter(_igpu_query, p))
            except Exception:
                pass
        _igpu_last_paths = paths
        try:
            win32pdh.CollectQueryData(_igpu_query)
        except Exception:
            pass
        return None  # PDH rate counters need a 2nd sample before they're valid

    try:
        win32pdh.CollectQueryData(_igpu_query)
        total = 0.0
        for c in _igpu_counters:
            try:
                _, val = win32pdh.GetFormattedCounterValue(c, win32pdh.PDH_FMT_DOUBLE)
                total += val
            except Exception:
                pass
        # Multiple GPU engines (3D, Copy, Video Decode...) can be active
        # at once, so this is a rough "how busy is the GPU overall"
        # figure rather than a single precise number -- same idea as the
        # combined bar Task Manager shows.
        return round(min(total, 100.0), 1)
    except Exception:
        return None

def get_runtime_stats() -> dict:
    """Live CPU / RAM / GPU usage — called every second for the graph."""
    stats = {"cpu_pct": 0, "ram_pct": 0, "ram_used_gb": 0,
             "gpu_pct": None, "vram_used_gb": None}
    try:
        import psutil
        # interval=None reports the delta since whenever cpu_percent() was
        # last called anywhere in the process — with irregular polling
        # (browser tab throttling, JS timer drift, Flask request overhead)
        # that produces stale, laggy, over-smoothed numbers. A short
        # blocking sample is slightly slower per call but actually
        # reflects "CPU usage right now."
        stats["cpu_pct"]     = psutil.cpu_percent(interval=0.2)
        vm = psutil.virtual_memory()
        stats["ram_pct"]     = vm.percent
        stats["ram_used_gb"] = round(vm.used / (1024**3), 1)
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            mem = torch.cuda.memory_stats()
            used = mem.get("allocated_bytes.all.current", 0)
            tot  = torch.cuda.get_device_properties(0).total_memory
            stats["gpu_pct"]      = round(used / tot * 100, 1) if tot else 0
            stats["vram_used_gb"] = round(used / (1024**3), 2)
    except Exception:
        pass

    if stats["gpu_pct"] is None and sys.platform == "win32":
        # No discrete NVIDIA GPU found -- try the integrated GPU path.
        # (VRAM isn't reported here: integrated GPUs share system RAM
        # rather than having dedicated VRAM, so there's no separate
        # number to show -- RAM usage above already reflects that.)
        igpu = _igpu_percent()
        if igpu is not None:
            stats["gpu_pct"] = igpu

    return stats
