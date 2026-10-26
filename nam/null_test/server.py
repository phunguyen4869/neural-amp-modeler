"""Local null-test page.

    python -m nam.null_test

The server listens on 127.0.0.1. It renders uploaded .nam files on the uploaded
DI, aligns the two resulting signals, and returns ESR plus WAV of the residual.
"""

import argparse as _argparse
import io as _io
import json as _json
import tempfile as _tempfile
import threading as _threading
import traceback as _traceback
import uuid as _uuid
import wave as _wave
import webbrowser as _webbrowser
from email.parser import BytesParser as _BytesParser
from http.server import BaseHTTPRequestHandler as _BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as _ThreadingHTTPServer
from pathlib import Path as _Path
from typing import Dict as _Dict
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import numpy as _np

from nam.null_test.core import compare_model_to_wet as _compare_model_to_wet
from nam.null_test.core import compare_models as _compare_models
from nam.null_test.core import compare_wavs as _compare_wavs
from nam.null_test.core import load_nam as _load_nam
from nam.null_test.core import load_wav as _load_wav

_STATIC_DIR = _Path(__file__).resolve().parent / "static"
_MAX_BODY_BYTES = 512 * 1024 * 1024
_MAX_LAG_MS = 2000.0
_JOB_LIMIT = 2
_AUDIO_NAMES = {"a", "b", "residual"}


class _JobStore:
    """Keep the last few rendered WAVs so the page can play them."""

    def __init__(self, limit: int = _JOB_LIMIT):
        self._limit = limit
        self._jobs: _Dict[str, dict] = {}
        self._order: list = []
        self._lock = _threading.Lock()

    def put(self, job_id: str, rate: int, audio: _Dict[str, _np.ndarray]) -> None:
        with self._lock:
            self._jobs[job_id] = {"rate": rate, "audio": audio}
            self._order.append(job_id)
            while len(self._order) > self._limit:
                expired = self._order.pop(0)
                self._jobs.pop(expired, None)

    def get(self, job_id: str, which: str) -> _Optional[_Tuple[_np.ndarray, int]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or which not in job["audio"]:
                return None
            return job["audio"][which], job["rate"]


class _NullTestHandler(_BaseHTTPRequestHandler):
    store: _JobStore = _JobStore()
    infer_lock = _threading.Lock()

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send_bytes(
                200,
                (_STATIC_DIR / "index.html").read_bytes(),
                "text/html; charset=utf-8",
            )
            return
        if path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        if path.startswith("/api/audio/"):
            self._send_audio(path)
            return
        self._send_json(404, {"error": "Page not found."})

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/null-test":
            self._send_json(404, {"error": "API not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Content-Length is invalid."})
            return
        if length <= 0 or length > _MAX_BODY_BYTES:
            self._send_json(
                413, {"error": "The request is empty or larger than 512 MB."}
            )
            return
        body = self.rfile.read(length)
        try:
            fields, files = _parse_multipart(self.headers.get("Content-Type", ""), body)
            with self.infer_lock:
                result = _run_request(fields, files)
            job_id = _uuid.uuid4().hex
            self.store.put(
                job_id,
                result.sample_rate,
                {
                    "a": result.aligned_reference.astype(_np.float32),
                    "b": result.aligned_other.astype(_np.float32),
                    "residual": result.residual.astype(_np.float32),
                },
            )
            self._send_json(200, result.public_dict(job_id))
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            _traceback.print_exc()
            self._send_json(500, {"error": f"Null test failed: {exc}"})

    def log_message(self, fmt, *args):
        print("[null-test]", fmt % args, flush=True)

    def _send_audio(self, path: str):
        pieces = path.strip("/").split("/")
        # api / audio / job_id / which
        if len(pieces) != 4 or pieces[3] not in _AUDIO_NAMES:
            self._send_json(404, {"error": "That audio file does not exist."})
            return
        found = self.store.get(pieces[2], pieces[3])
        if found is None:
            self._send_json(
                404, {"error": "This result has expired. Run the null test again."}
            )
            return
        audio, rate = found
        payload = _wav_bytes(audio, rate)
        filename = {"a": "a.wav", "b": "b.wav", "residual": "residual.wav"}[pieces[3]]
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Disposition", f'inline; filename="{filename}"')
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, status: int, payload: dict):
        raw = _json.dumps(payload).encode("utf-8")
        self._send_bytes(status, raw, "application/json; charset=utf-8")

    def _send_bytes(self, status: int, payload: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)


def make_server(host: str, port: int) -> _ThreadingHTTPServer:
    _NullTestHandler.store = _JobStore()
    server = _ThreadingHTTPServer((host, port), _NullTestHandler)
    server.daemon_threads = True
    return server


def main(argv=None) -> None:
    parser = _argparse.ArgumentParser(description="Open the NAM null-test page.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--no-open", action="store_true", help="Do not open a browser window."
    )
    args = parser.parse_args(argv)
    server = make_server(args.host, args.port)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Null test: {url}", flush=True)
    if not args.no_open:
        _webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        server.server_close()


def _run_request(fields: dict, files: dict):
    mode = fields.get("mode", "")
    match_gain = _flag(fields, "match_gain", default=True)
    auto_polarity = _flag(fields, "auto_polarity", default=True)
    invert = _flag(fields, "invert", default=False)
    max_lag_ms = _float_field(fields.get("max_lag_ms", "0"), "Alignment window")
    if max_lag_ms < 0 or max_lag_ms > _MAX_LAG_MS:
        raise ValueError(
            f"The alignment window must be from 0 to {_MAX_LAG_MS:.0f} ms."
        )

    if mode == "two_wavs":
        return _compare_uploaded_wavs(
            files,
            match_gain=match_gain,
            auto_polarity=auto_polarity,
            invert=invert and not auto_polarity and not match_gain,
            max_lag_ms=max_lag_ms,
        )

    di_part = _require_file(files, "di", "DI file")
    di_path = _as_temp(di_part, ".wav")
    try:
        di, di_rate = _load_wav(di_path, name=di_part.filename)
        max_lag_samples = int(round(max_lag_ms / 1000.0 * di_rate))
        options = dict(
            match_gain=match_gain,
            auto_polarity=auto_polarity,
            invert=invert and not auto_polarity and not match_gain,
            max_lag_samples=max_lag_samples,
        )
        if mode == "model_vs_wet":
            model_part = _require_file(files, "model", "model file")
            wet_part = _require_file(files, "wet", "reamp")
            # Reject a bad reamp before building the model. Loading a WaveNet
            # is the slow part, and a stereo or mismatched file cannot be nulled.
            wet_path = _as_temp(wet_part, ".wav")
            model_path = None
            try:
                wet, wet_rate = _load_wav(wet_path, name=wet_part.filename)
                if wet_rate != di_rate:
                    raise ValueError(
                        f"The DI is {di_rate} Hz but the reamp is {wet_rate} Hz. "
                        "The two WAV files must share a sample rate."
                    )
                model_path = _as_temp(model_part, ".nam")
                model = _load_nam(model_path, name=model_part.filename)
                return _compare_model_to_wet(
                    di,
                    di_rate,
                    model,
                    wet,
                    wet_rate,
                    wet_name=wet_part.filename or "reamp",
                    **options,
                )
            finally:
                if model_path is not None:
                    model_path.unlink(missing_ok=True)
                wet_path.unlink(missing_ok=True)
        if mode == "two_models":
            part_a = _require_file(files, "model_a", "model A")
            part_b = _require_file(files, "model_b", "model B")
            path_a = _as_temp(part_a, ".nam")
            path_b = None
            try:
                path_b = _as_temp(part_b, ".nam")
                model_a = _load_nam(path_a, name=part_a.filename)
                model_b = _load_nam(path_b, name=part_b.filename)
                return _compare_models(di, di_rate, model_a, model_b, **options)
            finally:
                path_a.unlink(missing_ok=True)
                if path_b is not None:
                    path_b.unlink(missing_ok=True)
        raise ValueError(
            "Choose a mode: a model against a reamp, two models, or two WAVs."
        )
    finally:
        di_path.unlink(missing_ok=True)


def _compare_uploaded_wavs(
    files: dict,
    *,
    match_gain: bool,
    auto_polarity: bool,
    invert: bool,
    max_lag_ms: float,
):
    part_a = _require_file(files, "wav_a", "WAV A")
    part_b = _require_file(files, "wav_b", "WAV B")
    path_a = _as_temp(part_a, ".wav")
    path_b = None
    try:
        path_b = _as_temp(part_b, ".wav")
        wav_a, rate_a = _load_wav(path_a, name=part_a.filename)
        wav_b, rate_b = _load_wav(path_b, name=part_b.filename)
        return _compare_wavs(
            wav_a,
            rate_a,
            wav_b,
            rate_b,
            name_a=part_a.filename or "WAV A",
            name_b=part_b.filename or "WAV B",
            match_gain=match_gain,
            auto_polarity=auto_polarity,
            invert=invert,
            max_lag_samples=int(round(max_lag_ms / 1000.0 * rate_a)),
        )
    finally:
        path_a.unlink(missing_ok=True)
        if path_b is not None:
            path_b.unlink(missing_ok=True)


def _flag(fields: dict, name: str, default: bool) -> bool:
    if name not in fields:
        return default
    return fields[name].strip().lower() in {"1", "true", "on", "yes"}


def _float_field(value: str, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not a number.") from exc


def _require_file(files: dict, key: str, label: str):
    part = files.get(key)
    if part is None or not part.data:
        raise ValueError(f"Missing {label}.")
    return part


def _as_temp(part, suffix: str) -> _Path:
    handle = _tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(part.data)
    finally:
        handle.close()
    return _Path(handle.name)


class _FormPart:
    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self.data = data


def _parse_multipart(content_type: str, body: bytes):
    """Split a multipart body into text fields and file parts.

    Standard-library ``cgi`` is gone on newer Python, so this parses the
    small set of form fields the page sends.
    """
    boundary = _boundary(content_type)
    token = b"--" + boundary
    fields = {}
    files = {}
    for part in body.split(token):
        if part.startswith(b"\r\n"):
            part = part[2:]
        if part.endswith(b"\r\n"):
            part = part[:-2]
        if not part or part == b"--":
            continue
        if part.startswith(b"--"):
            continue
        header_blob, separator, data = part.partition(b"\r\n\r\n")
        if not separator:
            continue
        header = _BytesParser().parsebytes(header_blob + b"\r\n\r\n")
        disposition = header.get("Content-Disposition", "")
        name = _disposition_param(disposition, "name")
        if not name:
            continue
        filename = _disposition_param(disposition, "filename")
        if filename:
            files[name] = _FormPart(_Path(filename).name, data)
        else:
            charset = header.get_content_charset() or "utf-8"
            fields[name] = data.decode(charset, errors="replace")
    return fields, files


def _boundary(content_type: str) -> bytes:
    for piece in content_type.split(";"):
        piece = piece.strip()
        if piece.lower().startswith("boundary="):
            value = piece.split("=", 1)[1].strip().strip('"')
            if value:
                return value.encode("utf-8")
    raise ValueError("The request is not multipart.")


def _disposition_param(disposition: str, key: str) -> str:
    marker = key + "="
    for piece in disposition.split(";"):
        piece = piece.strip()
        if piece.startswith(marker):
            return piece[len(marker) :].strip().strip('"')
    return ""


def _wav_bytes(audio: _np.ndarray, rate: int) -> bytes:
    clipped = _np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(_np.int16)
    buffer = _io.BytesIO()
    with _wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(rate))
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()
