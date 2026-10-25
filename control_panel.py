"""
control_panel.py — Terminal AI desktop Control Panel.

This is what START.bat launches. It's a small local GUI (not the web
chat itself) that lets you:
  - Edit server settings (port, model, admin password, priority/token
    defaults for new accounts and guests) — saved to config.json,
    which web_server.py reads on startup.
  - Start / stop the actual web server as a background process.
  - Watch its console output live, without a separate terminal window.

Uses only the Python standard library (tkinter) — nothing extra to
install beyond what BUILD.bat already sets up.
"""

import json, os, subprocess, socket, queue, threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

ROOT = Path(__file__).resolve().parent
CONFIG_FILE = ROOT / "config.json"
VENV_PY = ROOT / "venv" / "Scripts" / "python.exe"
SERVER_PY = ROOT / "server" / "web_server.py"

DEFAULT_CONFIG = {
    "host": "0.0.0.0",
    "port": 8080,
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "default_max_tokens": 1024,
    "hard_max_tokens": 8192,
    "default_priority": 5,
    "default_tokens_per_min": 6000,
    "guest_priority": 10,
    "guest_tokens_per_min": 1500,
    "allow_self_registration": True,
    "auto_discover_peers": True,
    "share_models_with_peers": True,
    # "master" keeps work local unless it's genuinely busier than a
    # paired peer; "slave" prefers handing work off to a paired peer
    # (typically a "master") whenever one is available and not clearly
    # more loaded, treating itself as an auxiliary compute node rather
    # than the machine people are meant to talk to directly.
    "machine_role": "master",
}

FIELDS = [
    ("host", "Host  (0.0.0.0 = whole LAN, 127.0.0.1 = this PC only)"),
    ("port", "Port"),
    ("model", "Model  (HuggingFace id, or leave — model_cache/ is used if present)"),
    ("allow_self_registration", "Let anyone create a Standard account (yes/no)"),
    ("auto_discover_peers", "Scan the LAN for other Terminal AI machines on startup (yes/no)"),
    ("share_models_with_peers", "Let paired peers fetch models you've already downloaded (yes/no)"),
    ("machine_role", "This machine's role when paired with others (master/slave)"),
    ("default_priority", "Default priority for new accounts (1=highest, 10=lowest)"),
    ("default_tokens_per_min", "Default tokens/min budget for new accounts"),
    ("guest_priority", "Guest priority (1=highest, 10=lowest — guests should usually be 10)"),
    ("guest_tokens_per_min", "Guest tokens/min budget"),
    ("default_max_tokens", "Default max tokens per reply"),
    ("hard_max_tokens", "Absolute hard ceiling — nobody can exceed this"),
]


def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return {**DEFAULT_CONFIG, **json.loads(CONFIG_FILE.read_text())}
        except Exception:
            pass
    return dict(DEFAULT_CONFIG)


def save_config(cfg: dict):
    CONFIG_FILE.write_text(json.dumps(cfg, indent=2))


def lan_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Terminal AI — Control Panel")
        self.geometry("780x580")
        self.minsize(620, 440)
        self.proc = None
        self.log_queue = queue.Queue()
        self.cfg = load_config()
        if not CONFIG_FILE.exists():
            save_config(self.cfg)

        self._build_ui()
        self.after(150, self._drain_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ────────────────────────────────────────
    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        server_tab = ttk.Frame(nb)
        nb.add(server_tab, text="Server")
        self._build_server_tab(server_tab)

        settings_tab = ttk.Frame(nb)
        nb.add(settings_tab, text="Settings")
        self._build_settings_tab(settings_tab)

    def _build_server_tab(self, parent):
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=(8, 4), padx=8)
        self.status_var = tk.StringVar(value="Stopped")
        ttk.Label(top, text="Status:").pack(side="left")
        ttk.Label(top, textvariable=self.status_var, font=("Segoe UI", 10, "bold")).pack(side="left", padx=(4, 20))
        self.start_btn = ttk.Button(top, text="Start Server", command=self.start_server)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(top, text="Stop Server", command=self.stop_server, state="disabled")
        self.stop_btn.pack(side="left", padx=4)

        url_frame = ttk.Frame(parent)
        url_frame.pack(fill="x", padx=8, pady=(4, 8))
        ttk.Label(url_frame, text="LAN address for others:").pack(side="left")
        self.url_var = tk.StringVar(value="Not running")
        entry = ttk.Entry(url_frame, textvariable=self.url_var, width=42)
        entry.pack(side="left", padx=6)
        entry.configure(state="readonly")
        ttk.Button(url_frame, text="Copy", command=self._copy_url).pack(side="left")

        ttk.Label(parent, text="Log output:").pack(anchor="w", padx=8)
        log_frame = ttk.Frame(parent)
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_text = tk.Text(log_frame, bg="#0d1210", fg="#d7e4dc", insertbackground="#d7e4dc",
                                 font=("Consolas", 9), wrap="word", state="disabled")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _build_settings_tab(self, parent):
        self.vars = {}
        self.bool_keys = {"allow_self_registration", "auto_discover_peers", "share_models_with_peers"}
        self.choice_keys = {"machine_role": ["master", "slave"]}
        container = ttk.Frame(parent)
        container.pack(fill="both", expand=True, padx=12, pady=12)
        for i, (key, label) in enumerate(FIELDS):
            ttk.Label(container, text=label).grid(row=i, column=0, sticky="w", pady=5)
            if key in self.bool_keys:
                var = tk.BooleanVar(value=bool(self.cfg.get(key, True)))
                ttk.Checkbutton(container, variable=var).grid(row=i, column=1, sticky="w", padx=10, pady=5)
            elif key in self.choice_keys:
                var = tk.StringVar(value=str(self.cfg.get(key, self.choice_keys[key][0])))
                ttk.Combobox(container, textvariable=var, values=self.choice_keys[key],
                             state="readonly", width=35).grid(row=i, column=1, sticky="w", padx=10, pady=5)
            else:
                var = tk.StringVar(value=str(self.cfg.get(key, "")))
                ttk.Entry(container, textvariable=var, width=38).grid(row=i, column=1, sticky="w", padx=10, pady=5)
            self.vars[key] = var
        container.grid_columnconfigure(1, weight=1)

        ttk.Label(parent, text="Admin access is now account-based, not a separate password. A default\n"
                                "Administrator account (Admin / Admin) is created on first run — sign in\n"
                                "with it and change the password from the chat page's settings panel.",
                  foreground="#4fd68c", justify="left").pack(anchor="w", padx=12, pady=(0, 8))
        ttk.Label(parent, text="Changes apply next time the server starts — restart it if it's already running.",
                  foreground="#888").pack(anchor="w", padx=12)
        ttk.Button(parent, text="Save Settings", command=self.save_settings).pack(anchor="w", padx=12, pady=10)

    # ── Settings ────────────────────────────────────────────────
    def save_settings(self):
        int_fields = {"port", "default_priority", "default_tokens_per_min",
                      "guest_priority", "guest_tokens_per_min",
                      "default_max_tokens", "hard_max_tokens"}
        cfg = dict(self.cfg)
        try:
            for key, _ in FIELDS:
                if key in self.bool_keys:
                    cfg[key] = bool(self.vars[key].get())
                    continue
                raw = self.vars[key].get().strip()
                cfg[key] = int(raw) if key in int_fields else raw
        except ValueError:
            messagebox.showerror("Invalid value", "Port, priority, and token fields must be whole numbers.")
            return
        if not (1 <= cfg["default_priority"] <= 10) or not (1 <= cfg["guest_priority"] <= 10):
            messagebox.showerror("Invalid value", "Priorities must be between 1 (highest) and 10 (lowest).")
            return
        save_config(cfg)
        self.cfg = cfg
        messagebox.showinfo("Saved", "Settings saved to config.json.")

    # ── Server process control ─────────────────────────────────
    def start_server(self):
        if self.proc is not None:
            return
        if not VENV_PY.exists():
            messagebox.showerror("Not set up", f"Couldn't find:\n{VENV_PY}\n\nRun BUILD.bat first.")
            return
        self._append_log(f"Starting server: {SERVER_PY}\n")
        # python.exe is a console-subsystem executable — even though this
        # Control Panel window itself is GUI-only, Windows still pops a
        # separate console for the subprocess unless explicitly told not
        # to. CREATE_NO_WINDOW suppresses that window while stdout/stderr
        # are still piped to us normally below.
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        child_env = dict(os.environ)
        child_env["PYTHONUNBUFFERED"] = "1"  # belt-and-suspenders against any buffered writes on the child side
        # Force the child's stdout to actually BE UTF-8 bytes. Without
        # this, Python on Windows encodes console output using the
        # system's ANSI codepage (often cp1252) by default. We decode
        # what comes back as UTF-8 below (_read_proc_output) — feeding it
        # cp1252 bytes for any non-ASCII character (em dashes, arrows,
        # etc. show up throughout the server's log messages) produces
        # invalid UTF-8 sequences that decode as the "�" replacement
        # character. This was never a display glitch, it was two ends of
        # the pipe assuming two different encodings.
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"
        self.proc = subprocess.Popen(
            [str(VENV_PY), str(SERVER_PY)],
            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0, creationflags=creationflags, env=child_env,
        )
        self.status_var.set("Running")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.url_var.set(f"http://{lan_ip()}:{self.cfg.get('port', 8080)}/")
        threading.Thread(target=self._read_proc_output, daemon=True).start()

    def stop_server(self):
        if self.proc is None:
            return
        self._append_log("\nStopping server...\n")
        try:
            self.proc.terminate()
        except Exception:
            pass

    def _read_proc_output(self):
        # Read raw bytes off the file descriptor directly with os.read().
        # This returns as soon as ANY data is available (even 1 byte) —
        # unlike TextIOWrapper.read(n), which blocks trying to fill the
        # full n characters before returning anything. That difference
        # is exactly what caused the log to go completely silent: short
        # log lines never accumulated enough to satisfy a fixed-size
        # text read, so nothing ever came through.
        proc = self.proc
        try:
            fd = proc.stdout.fileno()
            while True:
                try:
                    chunk = os.read(fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                self.log_queue.put(chunk.decode("utf-8", errors="replace"))
        except Exception:
            pass
        self.log_queue.put("\n[server process exited]\n")
        self.log_queue.put("__STOPPED__")

    def _drain_log_queue(self):
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line == "__STOPPED__":
                    self.status_var.set("Stopped")
                    self.start_btn.config(state="normal")
                    self.stop_btn.config(state="disabled")
                    self.url_var.set("Not running")
                    self.proc = None
                else:
                    self._append_log(line)
        except queue.Empty:
            pass
        self.after(150, self._drain_log_queue)

    def _append_log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _copy_url(self):
        self.clipboard_clear()
        self.clipboard_append(self.url_var.get())

    def _on_close(self):
        if self.proc is not None:
            if not messagebox.askyesno("Server running", "The server is still running. Stop it and quit?"):
                return
            self.stop_server()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
