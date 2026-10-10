"""Small helpers shared by the Jeff notebooks: locate builds, download, compile, start/stop system1-ane-serve, decide.

Configuration (environment variables, all optional):
    JEFF_MODELS_DIR    where download_model.py puts builds (default ~/Models/jeff-ane)
    JEFF_TETRIS_BUILD  a dir with coreai/ + model/ (default $JEFF_MODELS_DIR/tetris)
    JEFF_SNAKE_BUILD   (default $JEFF_MODELS_DIR/snake)
    JEFF_BASE_BUILD    (default $JEFF_MODELS_DIR/base)
    COREAI_PYTHON      Core AI SDK interpreter (required to compile/serve)
    SYSTEM1_ANE_URL   use an already running system1-ane-serve instead of starting one (JEFF_URL still works, deprecated)
    JEFF_NB_PORT       port for the notebook's own server (default 8797)
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for sub in ("coreai", "scripts", ""):
    p = str(REPO / sub) if sub else str(REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


def models_dir() -> Path:
    return Path(os.environ.get("JEFF_MODELS_DIR", Path.home() / "Models" / "jeff-ane")).expanduser()


def build_dir(name: str) -> Path:
    env = os.environ.get(f"JEFF_{name.upper()}_BUILD")
    return Path(env).expanduser() if env else models_dir() / name


def is_build(path: Path) -> bool:
    return (path / "coreai" / "manifest.json").is_file() and (path / "model" / "decision_config.json").is_file()


def ensure(name: str, download: bool = True) -> Path:
    """Return the build dir for ``name``; download it from Hugging Face if missing and allowed."""
    path = build_dir(name)
    if is_build(path):
        return path
    if not download:
        raise FileNotFoundError(f"{name}: no build at {path}; set JEFF_{name.upper()}_BUILD or download it")
    subprocess.run([sys.executable, str(REPO / "scripts" / "download_model.py"), name, "--dir", str(path.parent)],
                   check=True, cwd=REPO)
    if path.name != name:
        path = path.parent / name
    return path


def coreai_python() -> str:
    py = os.environ.get("COREAI_PYTHON")
    if not py or not Path(py).expanduser().exists():
        raise RuntimeError("Set COREAI_PYTHON to your Core AI SDK interpreter (see README).")
    return str(Path(py).expanduser())


def compile_build(path: Path) -> None:
    """Specialize the packages for this Mac's ANE (fast when already cached)."""
    env = {**os.environ, "COREAI_PYTHON": coreai_python()}
    subprocess.run([sys.executable, "forge.py", "compile", "--build", str(path / "coreai")], check=True, cwd=REPO, env=env)


def _free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def reuse_url() -> str | None:
    """URL of an already running server: SYSTEM1_ANE_URL, else the deprecated JEFF_URL (one-line notice)."""
    url = os.environ.get("SYSTEM1_ANE_URL")
    if not url and os.environ.get("JEFF_URL"):
        print("note: JEFF_URL is deprecated; use SYSTEM1_ANE_URL", file=sys.stderr, flush=True)
        url = os.environ["JEFF_URL"]
    return url or None


def start_server(base: Path, adapters: dict[str, Path] | None = None, port: int | None = None,
                 log: Path | None = None, timeout: float = 180):
    """Start system1-ane-serve in its own session; returns (process, url). Reuses SYSTEM1_ANE_URL if set."""
    if reuse_url():
        return None, reuse_url().rstrip("/")
    port = port or int(os.environ.get("JEFF_NB_PORT", "8797"))
    if not _free(port):
        raise RuntimeError(f"port {port} is busy; set JEFF_NB_PORT or SYSTEM1_ANE_URL")
    cmd = [sys.executable, "forge.py", "system1-ane-serve", "--model", str(base / "model"), "--build", str(base / "coreai"),
           "--host", "127.0.0.1", "--port", str(port)]
    for name, path in (adapters or {}).items():
        cmd += ["--adapter", f"{name}={path / 'coreai'}"]
    env = {**os.environ, "COREAI_PYTHON": coreai_python()}
    log = log or Path(os.environ.get("TMPDIR", "/tmp")) / f"jeff-nb-serve-{port}.log"
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT,
                            start_new_session=True)
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"system1-ane-serve exited ({proc.returncode}); see {log}")
        try:
            if health(url).get("status") == "ready":
                return proc, url
        except OSError:
            pass
        time.sleep(1)
    stop_server(proc)
    raise TimeoutError(f"system1-ane-serve not ready after {timeout}s; see {log}")


def stop_server(proc) -> None:
    if proc is None or proc.poll() is not None:
        return
    # forge.py launches system1_ane_serve.py as a child; the server runs in its own session, so stop the whole group.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)


def health(url: str) -> dict:
    with urllib.request.urlopen(url + "/health", timeout=5) as r:
        return json.load(r)


def decide(url: str, body: dict, timeout: float = 120) -> dict:
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)
