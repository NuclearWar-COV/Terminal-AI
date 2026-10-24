set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_MARKER="$ROOT/.built"
VENV_DIR="$ROOT/venv"
CONTROL_PANEL="$ROOT/control_panel.py"
SERVER_PY="$ROOT/server/web_server.py"


if [ ! -f "$CONTROL_PANEL" ]; then
    echo
    echo " [ERROR] control_panel.py not found."
    echo " Expected: $CONTROL_PANEL"
    exit 1
fi
if [ ! -f "$BUILD_MARKER" ] || [ ! -f "$VENV_DIR/bin/python" ]; then
    echo
    echo " [INFO] First run — setting up. This runs build.sh once."
    echo
    bash "$ROOT/build.sh" || exit 1
    if [ ! -f "$VENV_DIR/bin/python" ]; then
        echo " [ERROR] Setup did not complete. Cannot launch."
        exit 1
    fi
fi
echo " [OK] Ready: $VENV_DIR/bin/python"


if "$VENV_DIR/bin/python" -c "import tkinter" >/dev/null 2>&1; then
    echo " Launching Control Panel..."
    nohup "$VENV_DIR/bin/python" "$CONTROL_PANEL" >/dev/null 2>&1 &
    disown
    echo " Control Panel started in the background (PID $!)."
else
    echo " [INFO] tkinter isn't installed — no desktop Control Panel available."
    echo "        Install it for the full GUI (see build.sh output above for the"
    echo "        exact command for your distro), or just use the server directly:"
    echo
    echo " Launching the web server directly in this terminal..."
    echo " (settings come from config.json — edit it by hand, or install tkinter"
    echo " and re-run this script to get the Control Panel's Settings tab)"
    echo
    exec "$VENV_DIR/bin/python" "$SERVER_PY"
fi