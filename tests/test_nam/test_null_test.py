"""Null-test alignment, rendering, and the local page's request parser."""

import json as _json
import threading as _threading
import urllib.request as _request
from pathlib import Path as _Path

import numpy as _np
import pytest as _pytest
import torch as _torch

from nam.data import np_to_wav as _np_to_wav
from nam.models import losses as _losses
from nam.models.linear import Linear as _Linear
from nam.null_test import core as _core
from nam.null_test import server as _server
from tests.resources import resource_path as _resource_path


def _noise(length, seed):
    return _np.random.default_rng(seed).normal(size=length)


def test_esr_matches_training_loss():
    prediction = _np.array([0.0, 0.5, -0.25, 0.125])
    target = _np.array([0.1, 0.4, -0.2, 0.0])
    expected = _losses.esr(
        _torch.tensor(prediction, dtype=_torch.float64),
        _torch.tensor(target, dtype=_torch.float64),
    )
    assert _core._esr(prediction, target) == _pytest.approx(float(expected))


def test_other_late_by_n_samples_is_a_positive_lag():
    reference = _noise(8000, seed=0)
    delay = 17
    other = _np.concatenate([_np.zeros(delay), reference[:-delay]])
    aligned_ref, scaled, residual, lag, gain, correlation = _core.null_signals(
        reference,
        other,
        max_lag=100,
        match_gain=False,
        auto_polarity=True,
        invert=False,
    )
    assert lag == delay
    assert gain == _pytest.approx(1.0)
    assert correlation == _pytest.approx(1.0, abs=1e-6)
    assert _core._esr(scaled, aligned_ref) < 1e-10
    assert _np.max(_np.abs(residual)) < 1e-8


def test_other_early_is_a_negative_lag():
    reference = _noise(8000, seed=1)
    advance = 11
    other = _np.concatenate([reference[advance:], _np.zeros(advance)])
    _, _, _, lag, _, _ = _core.null_signals(
        reference,
        other,
        max_lag=40,
        match_gain=False,
        auto_polarity=True,
        invert=False,
    )
    assert lag == -advance


def test_gain_match_removes_a_level_difference():
    reference = _noise(4000, seed=2)
    other = 0.5 * reference
    matched_ref, matched_other, _, lag, gain, _ = _core.null_signals(
        reference,
        other,
        max_lag=0,
        match_gain=True,
        auto_polarity=True,
        invert=False,
    )
    raw_ref, raw_other, _, _, raw_gain, _ = _core.null_signals(
        reference,
        other,
        max_lag=0,
        match_gain=False,
        auto_polarity=False,
        invert=False,
    )
    assert lag == 0
    assert gain == _pytest.approx(2.0)
    assert _core._esr(matched_other, matched_ref) < 1e-12
    assert raw_gain == _pytest.approx(1.0)
    assert _core._esr(raw_other, raw_ref) == _pytest.approx(0.25)


def test_auto_polarity_flips_an_inverted_copy():
    reference = _noise(3000, seed=3)
    _, scaled, _, _, gain, correlation = _core.null_signals(
        reference,
        -reference,
        max_lag=0,
        match_gain=False,
        auto_polarity=True,
        invert=False,
    )
    assert gain == _pytest.approx(-1.0)
    assert correlation == _pytest.approx(-1.0)
    assert _np.allclose(scaled, reference)


def test_lag_window_must_fit_the_audio():
    with _pytest.raises(ValueError):
        _core.null_signals(
            _noise(100, seed=4),
            _noise(100, seed=5),
            max_lag=100,
            match_gain=False,
            auto_polarity=True,
            invert=False,
        )


def _identity_model():
    model = _Linear(receptive_field=1, sample_rate=48_000)
    model.import_weights(_torch.tensor([1.0]))
    return _core.LoadedNam(model, "identity.nam", "Linear", 48_000.0)


def test_chunked_render_matches_one_shot():
    model = _Linear(receptive_field=8, sample_rate=48_000)
    model.import_weights(_torch.linspace(0.05, 0.4, 8))
    audio = _noise(2537, seed=6) * 0.2
    full = _core.render_audio(model, audio, chunk_samples=None)
    chunked = _core.render_audio(model, audio, chunk_samples=100)
    assert full.shape == audio.shape
    assert _np.allclose(chunked, full, atol=1e-6)


def test_identity_model_nulls_a_delayed_reamp():
    loaded = _identity_model()
    di = _noise(6000, seed=7) * 0.2
    rendered = _core.render_audio(loaded.model, di)
    delay = 23
    wet = _np.concatenate([_np.zeros(delay), rendered[:-delay]])
    result = _core.compare_model_to_wet(
        di,
        48_000,
        loaded,
        wet,
        48_000,
        wet_name="output.wav",
        match_gain=True,
        auto_polarity=True,
        invert=False,
        max_lag_samples=100,
    )
    assert result.lag_samples == delay
    assert result.esr < 1e-8
    assert result.other_name == "output.wav"
    assert result.residual_db is None or result.residual_db < -80


def test_two_identity_models_null():
    di = _noise(2000, seed=8) * 0.15
    result = _core.compare_models(
        di,
        48_000,
        _identity_model(),
        _identity_model(),
        match_gain=False,
        auto_polarity=True,
        invert=False,
        max_lag_samples=0,
    )
    assert result.lag_samples == 0
    assert result.esr < 1e-10
    payload = result.public_dict("job")
    assert payload["job_id"] == "job"
    assert len(payload["preview"]["residual"]["max"]) > 10


def _calibration_tone(rate, seconds, dbfs=-23.0):
    """Mono sine whose peak is ``dbfs``. A −23 dBFS 1 kHz tone is about −26.0 LUFS."""
    times = _np.arange(int(rate * seconds)) / rate
    amplitude = 10 ** (dbfs / 20.0)
    return amplitude * _np.sin(2 * _np.pi * 1000.0 * times)


def test_integrated_lufs_of_a_mono_calibration_tone():
    loudness = _core.integrated_lufs(_calibration_tone(48_000, seconds=2.0), 48_000)
    assert loudness == _pytest.approx(-26.01, abs=0.2)


def test_integrated_lufs_is_undefined_when_silent_or_shorter_than_one_block():
    rate = 48_000
    assert _core.integrated_lufs(_np.zeros(rate * 2), rate) is None
    assert _core.integrated_lufs(_np.ones(int(rate * 0.2)), rate) is None


def test_null_result_reports_lufs_before_gain_on_b():
    rate = 48_000
    tone = _calibration_tone(rate, seconds=2.0)
    result = _core.compare_model_to_wet(
        tone,
        rate,
        _identity_model(),
        0.5 * tone,
        rate,
        match_gain=True,
        auto_polarity=True,
        invert=False,
        max_lag_samples=0,
    )
    payload = result.public_dict("job")
    assert payload["lufs"]["a"] == _pytest.approx(-26.01, abs=0.2)
    assert payload["lufs"]["b"] == _pytest.approx(-32.03, abs=0.3)
    assert payload["lufs"]["residual"] is None
    assert any("BS.1770" in note for note in payload["notes"])


def test_short_overlap_says_lufs_cannot_be_measured():
    di = _noise(1000, seed=14) * 0.2
    result = _core.compare_models(
        di,
        48_000,
        _identity_model(),
        _identity_model(),
        match_gain=False,
        auto_polarity=True,
        invert=False,
        max_lag_samples=0,
    )
    assert result.lufs_a is None
    assert result.lufs_residual is None
    assert any("400 ms" in note for note in result.notes)


def test_stereo_wav_is_rejected(tmp_path: _Path):
    stereo = _np.column_stack([_noise(1000, seed=15), _noise(1000, seed=16)]) * 0.1
    path = tmp_path / "stereo.wav"
    _np_to_wav(stereo, path, rate=48_000, sampwidth=2)
    with _pytest.raises(ValueError, match="stereo.wav must be a mono WAV"):
        _core.load_wav(path)
    with _pytest.raises(ValueError, match="ban-thu.wav must be a mono WAV"):
        _core.load_wav(path, name="ban-thu.wav")


def test_di_and_wet_must_share_a_sample_rate(tmp_path: _Path):
    di_path = tmp_path / "input.wav"
    wet_path = tmp_path / "output.wav"
    _np_to_wav(_noise(1000, seed=17) * 0.1, di_path, rate=48_000, sampwidth=2)
    _np_to_wav(_noise(1000, seed=18) * 0.1, wet_path, rate=44_100, sampwidth=2)
    di, di_rate = _core.load_wav(di_path)
    wet, wet_rate = _core.load_wav(wet_path)
    with _pytest.raises(ValueError, match="48000.*44100"):
        _core.compare_model_to_wet(
            di,
            di_rate,
            _identity_model(),
            wet,
            wet_rate,
            match_gain=True,
            auto_polarity=True,
            invert=False,
            max_lag_samples=0,
        )


def test_load_identity_nam_file():
    loaded = _core.load_nam(_resource_path("models/identity/linear_1.nam"))
    assert loaded.architecture == "Linear"
    assert loaded.sample_rate == _pytest.approx(48_000)
    audio = _np.linspace(-0.3, 0.3, 64)
    rendered = _core.render_audio(loaded.model, audio)
    assert _np.allclose(rendered, audio, atol=1e-6)


def test_multipart_parser_keeps_binary_wav():
    boundary = "----bound"
    wav = b"RIFF" + b"\r\n\r\n" + bytes(range(32))
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="mode"\r\n\r\n'
            "two_models\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="di"; filename="in.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode()
        + wav
        + f"\r\n--{boundary}--\r\n".encode()
    )
    fields, files = _server._parse_multipart(
        f"multipart/form-data; boundary={boundary}", body
    )
    assert fields["mode"] == "two_models"
    assert files["di"].filename == "in.wav"
    assert files["di"].data == wav


def _multipart(boundary, fields, files):
    """Build a multipart body. ``files`` maps name to (filename, bytes)."""
    chunks = []
    for name, value in fields.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode()
        )
    for name, (filename, data) in files.items():
        chunks.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n"
            ).encode()
            + data
            + b"\r\n"
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks)


def test_page_nulls_an_identity_model(tmp_path: _Path):
    di = (_noise(2400, seed=9) * 0.2).astype(_np.float64)
    di_path = tmp_path / "input.wav"
    wet_path = tmp_path / "output.wav"
    _np_to_wav(di, di_path, rate=48_000, sampwidth=2)
    _np_to_wav(di, wet_path, rate=48_000, sampwidth=2)
    nam_bytes = _resource_path("models/identity/linear_1.nam").read_bytes()
    boundary = "----nam"
    body = _multipart(
        boundary,
        {
            "mode": "model_vs_wet",
            "match_gain": "1",
            "auto_polarity": "1",
            "invert": "0",
            "max_lag_ms": "20",
        },
        {
            "di": ("input.wav", di_path.read_bytes()),
            "wet": ("output.wav", wet_path.read_bytes()),
            "model": ("linear_1.nam", nam_bytes),
        },
    )
    server = _server.make_server("127.0.0.1", 0)
    thread = _threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    try:
        page = _request.urlopen(f"http://127.0.0.1:{port}/")
        html = page.read().decode("utf-8")
        assert "Null test" in html
        assert "Run null test" in html
        request = _request.Request(
            f"http://127.0.0.1:{port}/api/null-test",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with _request.urlopen(request) as response:
            payload = _json.load(response)
        assert payload["esr"] < 1e-8
        assert payload["lag_samples"] == 0
        assert payload["reference_name"] == "linear_1.nam"
        assert payload["other_name"] == "output.wav"
        assert payload["lufs"]["residual"] is None
        assert any("400 ms" in note for note in payload["notes"])
        audio = _request.urlopen(
            f"http://127.0.0.1:{port}/api/audio/{payload['job_id']}/residual"
        )
        assert audio.headers["Content-Type"] == "audio/wav"
        residual = audio.read()
        assert residual[:4] == b"RIFF"
        missing = _request.Request(
            f"http://127.0.0.1:{port}/api/null-test",
            data=_multipart(boundary, {"mode": "two_models"}, {}),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with _pytest.raises(_request.HTTPError) as caught:
            _request.urlopen(missing)
        error = _json.load(caught.value)
        assert "DI" in error["error"]
    finally:
        server.shutdown()
        server.server_close()
