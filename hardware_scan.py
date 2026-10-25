"""
hardware_scan.py — Scans all available hardware and AI acceleration options.
Called once at startup, results cached in hw_info dict.
"""
import sys, os, subprocess, platform, json, re
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


    hw["cuda_warning"] = None
    try:
        import torch, warnings, subprocess as _sp
        hw["torch_version"] = torch.__version__
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
            try:
                _sp.run(["nvidia-smi"], capture_output=True, timeout=3, check=True)
                hw["cuda_warning"] = ("nvidia-smi found an NVIDIA GPU, but this is a CPU-only PyTorch "
                                       "build (installed without a CUDA index URL). Reinstall with: "
                                       "pip install torch torchvision torchaudio "
                                       "--index-url https://download.pytorch.org/whl/cu128")
            except Exception:
                pass
    except Exception:
        hw["torch_version"] = "not installed"
    try:
        import torch_directml
        hw["torch_directml"] = True
        hw["gpu"].append({"name": "DirectML GPU", "vram_gb": "?", "type": "DirectML"})
    except Exception:
        pass

    try:
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

    try:
        import optimum
        hw["optimum_version"] = optimum.__version__
        hw["optimum_intel"]   = True
    except Exception:
        hw["optimum_intel"] = False

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


_igpu_query      = None
_igpu_counters   = []
_igpu_last_paths = None

try:
    import psutil as _psutil
    _psutil.cpu_percent(interval=None)
except ImportError:
    _psutil = None

def _igpu_percent():
    global _igpu_query, _igpu_counters, _igpu_last_paths
    try:
        import win32pdh
    except ImportError:
        return None
    try:
        paths = win32pdh.ExpandCounterPath(r"\GPU Engine(*)\Utilization Percentage")
    except Exception:
        return None
    if not paths:
        return None

    if _igpu_query is None or paths != _igpu_last_paths:
        if _igpu_query is not None:
            try: win32pdh.CloseQuery(_igpu_query)
            except Exception: pass
        _igpu_query    = win32pdh.OpenQuery()
        _igpu_counters = []
        for p in paths:
            try:
                _igpu_counters.append((p, win32pdh.AddCounter(_igpu_query, p)))
            except Exception:
                pass
        _igpu_last_paths = paths
        try:
            win32pdh.CollectQueryData(_igpu_query)
        except Exception:
            pass
        return None

    try:
        win32pdh.CollectQueryData(_igpu_query)
        engine_usage = {}
        for path, counter in _igpu_counters:
            try:
                status, val = win32pdh.GetFormattedCounterValue(counter, win32pdh.PDH_FMT_DOUBLE)
                if status not in (win32pdh.PDH_CSTATUS_VALID_DATA, win32pdh.PDH_CSTATUS_NEW_DATA):
                    continue
                match = re.search(r"luid_(.+?)_phys_(\d+)_.*?engtype_(.+?)\)", path, re.IGNORECASE)
                engine = (match.group(1), match.group(2), match.group(3)) if match else path
                engine_usage[engine] = engine_usage.get(engine, 0.0) + max(0.0, val)
            except Exception:
                pass
        if not engine_usage:
            return None
        # Task Manager reports the busiest GPU engine, rather than summing
        # overlapping 3D, copy, video decode, and per-process counters.
        return round(min(max(engine_usage.values()), 100.0), 1)
    except Exception:
        return None


def _nvidia_stats():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=1, check=True,
        )
        line = next((row.strip() for row in result.stdout.splitlines() if row.strip()), "")
        name, gpu, used, total = (part.strip() for part in line.split(",", 3))
        return {
            "gpu_pct": float(gpu),
            "vram_used_gb": round(float(used) / 1024, 2),
            "vram_total_gb": round(float(total) / 1024, 2),
            "gpu_name": name,
            "gpu_source": "NVIDIA driver",
        }
    except (OSError, subprocess.SubprocessError, ValueError, StopIteration):
        return None


def get_runtime_stats() -> dict:
    """Return system CPU, physical RAM, and GPU engine utilization."""
    stats = {"cpu_pct": None, "ram_pct": None, "ram_used_gb": None,
             "gpu_pct": None, "vram_used_gb": None, "vram_total_gb": None,
             "gpu_name": None, "gpu_source": None}
    if _psutil is not None:
        try:
            stats["cpu_pct"] = _psutil.cpu_percent(interval=None)
            vm = _psutil.virtual_memory()
            stats["ram_pct"] = vm.percent
            stats["ram_used_gb"] = round(vm.used / (1024**3), 1)
        except Exception:
            pass

    nvidia = _nvidia_stats()
    if nvidia:
        stats.update(nvidia)
    elif sys.platform == "win32":
        igpu = _igpu_percent()
        if igpu is not None:
            stats["gpu_pct"] = igpu
            stats["gpu_source"] = "Windows GPU Engine counters"

    return stats
