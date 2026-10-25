"""
HTTP server: ``python -m webui`` (stdlib only; one training job at a time).
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(os.environ.get("NAM_WEB_DIR", "/outputs/web"))
UPLOADS = ROOT / "uploads"
JOBS = ROOT / "jobs"
STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = int(os.environ.get("NAM_WEB_MAX_UPLOAD_MB", "1024")) * 1024 * 1024
KINDS = ("input", "output")

_lock = threading.Lock()
_proc = None  # type: subprocess.Popen | None


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # Zombie children of this process count as dead:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


def _job_dirs():
    return sorted((p for p in JOBS.glob("*") if p.is_dir()), reverse=True)


def _status(job_dir: Path) -> dict:
    """Status of a job, reconciling a worker that died without reporting."""
    global _proc
    status = _read_json(job_dir / "status.json", {"state": "queued"})
    if status.get("state") in ("queued", "running"):
        alive = (_proc is not None and _proc.poll() is None) or _pid_alive(
            status.get("pid")
        )
        if not alive and status.get("pid"):
            status.update(
                state="failed",
                message="Worker exited unexpectedly (out of memory?).",
                ended=time.time(),
            )
            _write_json(job_dir / "status.json", status)
    return status


def _running_job():
    for job_dir in _job_dirs()[:1]:
        if _status(job_dir).get("state") in ("queued", "running"):
            return job_dir
    return None


_ckpt_re = re.compile(r"epoch=(\d+)_step=\d+_ESR=([0-9.e+-]+)")
_tb_cache = {}


def _esr_history(job_dir: Path) -> dict:
    """Validation ESR by epoch: checkpoint names (immediate) + TensorBoard (complete)."""
    history = {}
    logs = job_dir / "train" / "lightning_logs"
    for ckpt in logs.glob("version_*/checkpoints/*.ckpt"):
        m = _ckpt_re.search(ckpt.name)
        if m:
            history[int(m.group(1))] = float(m.group(2))
    for events in logs.glob("version_*/events.out.tfevents.*"):
        key = (str(events), events.stat().st_mtime)
        if key not in _tb_cache:
            try:
                from tensorboard.backend.event_processing.event_accumulator import (
                    EventAccumulator,
                )

                acc = EventAccumulator(str(events), size_guidance={"scalars": 0})
                acc.Reload()
                _tb_cache[key] = (
                    [e.value for e in acc.Scalars("ESR")]
                    if "ESR" in acc.Tags()["scalars"]
                    else []
                )
            except Exception:
                _tb_cache[key] = []
        for epoch, value in enumerate(_tb_cache[key]):
            history[epoch] = value
    return history


def _tail_log(job_dir: Path, lines: int = 200) -> str:
    try:
        data = (job_dir / "log.txt").read_bytes()[-64_000:].decode(errors="replace")
    except OSError:
        return ""
    data = data.replace("\r", "\n")
    return "\n".join(data.split("\n")[-lines:])


def _job_summary(job_dir: Path, with_details: bool = False) -> dict:
    status = _status(job_dir)
    params = _read_json(job_dir / "params.json", {})
    out = {"id": job_dir.name, "params": params, **status}
    out["has_model"] = bool(
        status.get("model") and (job_dir / status["model"]).exists()
    )
    if with_details:
        out["esr_history"] = {str(k): v for k, v in sorted(_esr_history(job_dir).items())}
        tail = _tail_log(job_dir, 400)
        out["log"] = "\n".join(tail.split("\n")[-120:])
    return out


def _uploads_state() -> dict:
    meta = _read_json(UPLOADS / "meta.json", {})
    return {
        kind: {
            "name": meta.get(kind, {}).get("name"),
            "size": (UPLOADS / f"{kind}.wav").stat().st_size
            if (UPLOADS / f"{kind}.wav").exists()
            else None,
        }
        for kind in KINDS
    }


def _clean_str(value, limit=80) -> str:
    return re.sub(r"[\x00-\x1f]", "", str(value or "")).strip()[:limit]


def _start_job(body: dict) -> dict:
    global _proc
    uploads = _uploads_state()
    if any(uploads[k]["size"] is None for k in KINDS):
        raise ValueError("Upload both input.wav and output.wav first.")
    try:
        epochs = int(body.get("epochs", 100))
        batch_size = int(body.get("batch_size", 16))
        seed = int(body.get("seed", 0))
        latency = body.get("latency")
        latency = None if latency in (None, "", "auto") else int(latency)
    except (TypeError, ValueError):
        raise ValueError("epochs, batch_size, seed and latency must be integers.")
    if not 1 <= epochs <= 1000:
        raise ValueError("epochs must be between 1 and 1000.")
    if not 1 <= batch_size <= 256:
        raise ValueError("batch_size must be between 1 and 256.")
    name = _clean_str(body.get("name"))
    basename = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "model"
    params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "latency": latency,
        "ignore_checks": bool(body.get("ignore_checks")),
        "name": name,
        "modeled_by": _clean_str(body.get("modeled_by")),
        "basename": basename,
        "input_name": uploads["input"]["name"],
        "output_name": uploads["output"]["name"],
    }
    with _lock:
        if _running_job() is not None:
            raise RuntimeError("A training job is already running.")
        job_dir = JOBS / time.strftime("%Y%m%d-%H%M%S")
        job_dir.mkdir(parents=True)
        for kind in KINDS:
            shutil.copyfile(UPLOADS / f"{kind}.wav", job_dir / f"{kind}.wav")
        _write_json(job_dir / "params.json", params)
        _write_json(job_dir / "status.json", {"state": "queued"})
        log = open(job_dir / "log.txt", "wb")
        _proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "webui.worker", str(job_dir)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        threading.Thread(target=_proc.wait, daemon=True).start()
    return {"id": job_dir.name}


def _cancel() -> bool:
    job_dir = _running_job()
    if job_dir is None:
        return False
    pid = _read_json(job_dir / "status.json", {}).get("pid") or (
        _proc.pid if _proc else None
    )
    if pid:
        os.kill(pid, signal.SIGINT)
    return True


class Handler(BaseHTTPRequestHandler):
    server_version = "nam-webui"

    def log_message(self, fmt, *args):
        if "/api/status" not in (args[0] if args else ""):
            super().log_message(fmt, *args)

    def _send(self, code, body: bytes, ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def _error(self, code, message):
        self._json({"error": message}, code)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/status":
            jobs = _job_dirs()
            self._json(
                {
                    "now": time.time(),
                    "uploads": _uploads_state(),
                    "job": _job_summary(jobs[0], True) if jobs else None,
                    "history": [_job_summary(j) for j in jobs[:20]],
                }
            )
        elif url.path == "/api/download":
            job_id = (query.get("job") or [""])[0]
            job_dir = JOBS / job_id
            if not re.fullmatch(r"[0-9]{8}-[0-9]{6}", job_id) or not job_dir.is_dir():
                return self._error(404, "Unknown job.")
            model = _status(job_dir).get("model")
            path = job_dir / model if model else None
            if path is None or not path.exists():
                return self._error(404, "This job has no model.")
            self._send(
                200,
                path.read_bytes(),
                "application/octet-stream",
                {"Content-Disposition": f'attachment; filename="{path.name}"'},
            )
        else:
            self._error(404, "Not found.")

    def do_POST(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path == "/api/upload":
            return self._upload(query)
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._error(400, "Invalid JSON.")
        if url.path == "/api/train":
            try:
                self._json(_start_job(body))
            except ValueError as e:
                self._error(400, str(e))
            except RuntimeError as e:
                self._error(409, str(e))
        elif url.path == "/api/cancel":
            self._json({"cancelled": _cancel()})
        else:
            self._error(404, "Not found.")

    def _upload(self, query):
        kind = (query.get("kind") or [""])[0]
        if kind not in KINDS:
            return self._error(400, "kind must be input or output.")
        if _running_job() is not None:
            return self._error(409, "Wait for the running job to finish.")
        length = int(self.headers.get("Content-Length") or 0)
        if not 44 < length <= MAX_UPLOAD_BYTES:
            return self._error(413, f"File must be a WAV of at most {MAX_UPLOAD_BYTES >> 20} MB.")
        UPLOADS.mkdir(parents=True, exist_ok=True)
        tmp = UPLOADS / f"{kind}.wav.part"
        remaining = length
        with open(tmp, "wb") as f:
            while remaining:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                f.write(chunk)
                remaining -= len(chunk)
        with open(tmp, "rb") as f:
            header = f.read(12)
        if remaining or header[:4] not in (b"RIFF", b"RF64") or header[8:12] != b"WAVE":
            tmp.unlink(missing_ok=True)
            return self._error(400, "Not a complete WAV file.")
        tmp.replace(UPLOADS / f"{kind}.wav")
        meta = _read_json(UPLOADS / "meta.json", {})
        meta[kind] = {"name": _clean_str((query.get("name") or [""])[0], 120)}
        _write_json(UPLOADS / "meta.json", meta)
        self._json({"ok": True})


def main():
    JOBS.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get("NAM_WEB_PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"NAM web UI on http://0.0.0.0:{port} (data in {ROOT})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
