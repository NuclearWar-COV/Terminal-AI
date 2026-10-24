"""
Terminal AI V14.0 - Hardware Detection
Output is parsed by InstallerCheck.cs / InstallForm.cs
Lines printed: NPU:true|false, CUDA:true|false, DML:true|false, NAME:<device name>
"""
import sys

npu  = False
cuda = False
dml  = False
name = "CPU"

# 1. Intel NPU via OpenVINO
try:
    from openvino.runtime import Core
    core = Core()
    devs = core.available_devices
    if "NPU" in devs:
        npu  = True
        name = f"Intel NPU (OpenVINO devices: {devs})"
except Exception:
    pass

# 2. NPU via ONNX Runtime
if not npu:
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        for ep in ("QNNExecutionProvider", "OpenVINOExecutionProvider"):
            if ep in providers:
                npu  = True
                name = f"NPU ({ep})"
                break
    except Exception:
        pass

# 3. CUDA
if not npu:
    try:
        import torch
        if torch.cuda.is_available():
            cuda = True
            name = torch.cuda.get_device_name(0)
    except Exception:
        pass

# 4. DirectML
if not npu and not cuda:
    try:
        import torch_directml
        dml  = True
        name = "GPU (DirectML)"
    except Exception:
        pass

print(f"NPU:{'true' if npu else 'false'}")
print(f"CUDA:{'true' if cuda else 'false'}")
print(f"DML:{'true' if dml else 'false'}")
print(f"NAME:{name}")
