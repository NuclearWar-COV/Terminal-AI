set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_MARKER="$ROOT/.built"
VENV_DIR="$ROOT/venv"
SERVER_PY="$ROOT/server/web_server.py"

echo
echo " ====================================================="
echo "                      Terminal AI"
echo " ====================================================="
echo


if [ ! -f "$SERVER_PY" ]; then
    echo " [ERROR] server/web_server.py not found."
    echo " Expected: $SERVER_PY"
    exit 1
fi


PYTHON_BIN=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" --version >/dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ] && command -v py >/dev/null 2>&1 && py -3 --version >/dev/null 2>&1; then
    PYTHON_BIN="py"
fi
if [ -z "$PYTHON_BIN" ]; then
    echo " [ERROR] Python 3 is not installed or not on your PATH."
    echo " Install it with your package manager, e.g.:"
    echo "   Debian/Ubuntu: sudo apt install python3 python3-venv python3-tk"
    echo "   Fedora:        sudo dnf install python3 python3-tkinter"
    echo "   macOS:         brew install python-tk"
    echo "   Windows:       install Python 3.10+ and enable the Python Launcher (py)"
    exit 1
fi
python_cmd() {
    if [ "$PYTHON_BIN" = "py" ]; then
        py -3 "$@"
    else
        "$PYTHON_BIN" "$@"
    fi
}
echo " [OK] $(python_cmd --version)"

if [ ! -f "$VENV_DIR/bin/python" ]; then
    echo " Creating virtual environment..."
    python_cmd -m venv "$VENV_DIR"
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo " [ERROR] Failed to create virtual environment."
        echo " On Debian/Ubuntu you may need: sudo apt install python3-venv"
        exit 1
    fi
else
    echo " [OK] Virtual environment already exists."
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip >/dev/null


echo " Checking for an NVIDIA GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    echo " [OK] NVIDIA GPU detected — installing CUDA-enabled PyTorch (this can take a while)..."
    if ! "$VENV_DIR/bin/python" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128; then
        echo " [WARN] CUDA PyTorch install failed — falling back to CPU-only PyTorch."
        echo " (DGPU acceleration setup failed; re-run build.sh.)"
        "$VENV_DIR/bin/python" -m pip install torch torchvision torchaudio
    fi
else
    echo " No NVIDIA GPU detected — installing CPU-only PyTorch."
    "$VENV_DIR/bin/python" -m pip install torch torchvision torchaudio
fi
if ! "$VENV_DIR/bin/python" -c "import torch" >/dev/null 2>&1; then
    echo " [ERROR] PyTorch install failed. See the output above."
    exit 1
fi
echo " [OK] PyTorch installed."


echo " Installing remaining dependencies (this can take a while)..."
if ! "$VENV_DIR/bin/python" -m pip install -r "$ROOT/requirements.txt"; then
    echo " [ERROR] Dependency install failed. See the output above."
    exit 1
fi
echo " [OK] Dependencies installed."


if ! "$VENV_DIR/bin/python" -c "import tkinter" >/dev/null 2>&1; then
    echo " [WARN] tkinter isn't available — the Desktop Control Panel window won't"
    echo "        open. start.sh will fall back to running the server directly"
    echo "        in this terminal instead. To get the Control Panel GUI, install:"
    echo "          Debian/Ubuntu: sudo apt install python3-tk"
    echo "          Fedora:        sudo dnf install python3-tkinter"
    echo "          macOS:         brew install python-tk"
fi


cd "$ROOT" || exit 1
mkdir -p model_cache model_archive models training_data web_training_data logs
echo "Directories verified."


if [ ! -f "$ROOT/config.json" ]; then
    "$VENV_DIR/bin/python" -c "
import json, pathlib
pathlib.Path('config.json').write_text(json.dumps({
    'host': '0.0.0.0', 'port': 8080, 'model': 'Qwen/Qwen2.5-0.5B-Instruct',
    'default_max_tokens': 1024, 'hard_max_tokens': 8192,
    'default_priority': 5, 'default_tokens_per_min': 6000,
    'guest_priority': 10, 'guest_tokens_per_min': 1500,
    'allow_self_registration': True,
    'auto_discover_peers': True, 'share_models_with_peers': True,
    'machine_role': 'master',
}, indent=2))
"
    echo "Created default config.json."
    echo "Defualt Admin Credentials, Usr: Admin, Pwd: Admin"
    echo "Do your self a favor and change it in the Admin Pannel"
fi

echo "V-built" > "$BUILD_MARKER"

echo
echo " ====================================================="
echo "  Setup complete!"
echo "  Run ./start.sh to launch."
echo " ====================================================="
