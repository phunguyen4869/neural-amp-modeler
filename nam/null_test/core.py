"""Render NAM models and null-test the resulting audio.

A null test subtracts one aligned signal from another. The residual is what
the two signals do not share. ESR is the residual power divided by the power
of the reference, the same ratio ``nam.models.losses.esr`` reports when the
signals are already aligned and the same level.

Integrated loudness is ITU-R BS.1770-4 LUFS of those aligned signals.
"""

import json as _json
import math as _math
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import List as _List
from typing import Optional as _Optional
from typing import Sequence as _Sequence
from typing import Tuple as _Tuple

import librosa as _librosa
import numpy as _np
import torch as _torch
from scipy.signal import correlate as _correlate
from scipy.signal import correlation_lags as _correlation_lags
from scipy.signal import lfilter as _lfilter

from nam.data import wav_to_np as _wav_to_np
from nam.models._from_nam import init_from_nam as _init_from_nam

_DEFAULT_CHUNK_SAMPLES = 15 * 48_000
# BS.1770-4: 400 ms blocks, 75% overlap, absolute gate, then −10 LU relative gate.
_LUFS_BLOCK_SECONDS = 0.4
_LUFS_ABSOLUTE_GATE_LUFS = -70.0
_LUFS_OFFSET_DB = -0.691


@_dataclass
class LoadedNam:
    """A .nam file loaded into a torch module."""

    model: _torch.nn.Module
    name: str
    architecture: str
    sample_rate: _Optional[float]


@_dataclass
class NullResult:
    """Aligned signals and the numbers that describe their difference."""

    esr: float
    residual_db: _Optional[float]
    lag_samples: int
    gain: float
    polarity: str
    correlation: float
    sample_rate: int
    overlap_samples: int
    rms_dbfs_a: _Optional[float]
    rms_dbfs_b: _Optional[float]
    rms_dbfs_residual: _Optional[float]
    lufs_a: _Optional[float]
    lufs_b: _Optional[float]
    lufs_residual: _Optional[float]
    reference_name: str
    other_name: str
    notes: _List[str]
    aligned_reference: _np.ndarray
    aligned_other: _np.ndarray
    residual: _np.ndarray

    def preview(self, buckets: int = 900) -> dict:
        """Min/max envelope of the aligned signals, shared vertical scale."""
        return {
            "a": _envelope(self.aligned_reference, buckets),
            "b": _envelope(self.aligned_other, buckets),
            "residual": _envelope(self.residual, buckets),
        }

    def public_dict(self, job_id: str) -> dict:
        """JSON-safe view. Audio arrays stay on the server."""
        return {
            "job_id": job_id,
            "esr": float(self.esr),
            "residual_db": _finite(self.residual_db),
            "lag_samples": int(self.lag_samples),
            "lag_ms": float(self.lag_samples) / float(self.sample_rate) * 1000.0,
            "gain": float(self.gain),
            "gain_db": _finite(_db20(abs(self.gain)) if self.gain != 0 else None),
            "polarity": self.polarity,
            "correlation": float(self.correlation),
            "sample_rate": int(self.sample_rate),
            "overlap_samples": int(self.overlap_samples),
            "rms_dbfs": {
                "a": _finite(self.rms_dbfs_a),
                "b": _finite(self.rms_dbfs_b),
                "residual": _finite(self.rms_dbfs_residual),
            },
            "lufs": {
                "a": _finite(self.lufs_a),
                "b": _finite(self.lufs_b),
                "residual": _finite(self.lufs_residual),
            },
            "reference_name": self.reference_name,
            "other_name": self.other_name,
            "notes": list(self.notes),
            "preview": self.preview(),
        }


def load_wav(path: _Path, name: _Optional[str] = None) -> _Tuple[_np.ndarray, int]:
    """Load a mono WAV as float64 samples in about [-1, 1], plus the rate.

    ``name`` is the name shown in errors. The server passes the uploaded
    filename because ``path`` is a temporary file.
    """
    display_name = name or path.name
    channels = _pcm_wav_channels(path)
    if channels is not None and channels != 1:
        raise ValueError(
            f"{display_name} must be a mono WAV. This file has {channels} channels."
        )
    try:
        audio, info = _wav_to_np(path, info=True)
    except (AssertionError, NotImplementedError) as exc:
        if _is_not_mono(exc):
            raise ValueError(f"{display_name} must be a mono WAV.") from exc
        raise ValueError(f"Could not read {display_name}: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"Could not read {display_name}: {exc}") from exc
    audio = _np.asarray(audio, dtype=_np.float64)
    if audio.ndim != 1 or len(audio) == 0:
        raise ValueError(f"{display_name} has no mono audio.")
    return audio, int(info.rate)


def load_nam(path: _Path, name: _Optional[str] = None) -> LoadedNam:
    """Load a .nam JSON file into a model ready for inference."""
    display_name = name or path.name
    try:
        with open(path, "r", encoding="utf-8") as handle:
            config = _json.load(handle)
    except (OSError, UnicodeError, _json.JSONDecodeError) as exc:
        raise ValueError(f"{display_name} is not valid .nam JSON.") from exc
    architecture = config.get("architecture")
    try:
        model = _init_from_nam(config)
    except KeyError as exc:
        raise ValueError(
            f"Architecture '{architecture}' in {display_name} is not supported."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not load {display_name}: {exc}") from exc
    model.eval()
    rate = config.get("sample_rate")
    if rate is not None:
        rate = float(rate)
    return LoadedNam(
        model=model,
        name=display_name,
        architecture=str(architecture),
        sample_rate=rate,
    )


def render_audio(
    model: _torch.nn.Module,
    audio: _np.ndarray,
    *,
    chunk_samples: int = _DEFAULT_CHUNK_SAMPLES,
) -> _np.ndarray:
    """Run ``model`` with start padding so the output length matches ``audio``.

    Long files are rendered in chunks. Each chunk after the first keeps
    ``receptive_field - 1`` samples of input history, which is what a causal
    NAM needs in order to match a single forward pass.
    """
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    audio32 = _np.ascontiguousarray(audio, dtype=_np.float32)
    if len(audio32) == 0:
        raise ValueError("The audio is empty.")
    receptive_field = int(model.receptive_field)
    if receptive_field < 1:
        raise ValueError(f"Invalid receptive field: {receptive_field}.")
    if chunk_samples is None or len(audio32) <= chunk_samples:
        return _forward_numpy(model, audio32, device, pad_start=True)

    chunk_samples = max(int(chunk_samples), receptive_field)
    pieces = []
    position = 0
    first = True
    while position < len(audio32):
        if first:
            end = min(len(audio32), chunk_samples)
            pieces.append(_forward_numpy(model, audio32[:end], device, pad_start=True))
            position = end
            first = False
            continue
        history = receptive_field - 1
        start = position - history
        end = min(len(audio32), position + chunk_samples)
        pieces.append(
            _forward_numpy(model, audio32[start:end], device, pad_start=False)
        )
        position = end
    rendered = _np.concatenate(pieces)
    if len(rendered) != len(audio32):
        raise RuntimeError(
            f"The model returned {len(rendered)} samples for {len(audio32)} samples of input."
        )
    return rendered


def null_signals(
    reference: _np.ndarray,
    other: _np.ndarray,
    *,
    max_lag: int,
    match_gain: bool,
    auto_polarity: bool,
    invert: bool,
) -> _Tuple[_np.ndarray, _np.ndarray, _np.ndarray, int, float, float]:
    """Align ``other`` to ``reference`` and subtract.

    ``lag_samples`` is positive when ``other`` is later than ``reference``.
    The returned ``other`` and residual are the signals after the chosen gain
    and polarity, so their difference is the null.

    :return: reference overlap, scaled other, residual, lag, gain, correlation
    """
    if max_lag < 0:
        raise ValueError("The alignment window must be zero or positive.")
    reference = _np.asarray(reference, dtype=_np.float64)
    other = _np.asarray(other, dtype=_np.float64)
    shortest = min(len(reference), len(other))
    if shortest - max_lag < 8:
        raise ValueError(
            "The alignment window is longer than the audio. Use fewer milliseconds."
        )
    lag = _best_delay(
        reference,
        other,
        max_lag=max_lag,
        prefer_abs=match_gain or auto_polarity,
        invert=invert,
    )
    aligned_reference, aligned_other = _slices(reference, other, lag)
    correlation = _cosine(aligned_reference, aligned_other)
    gain = _choose_gain(
        aligned_reference,
        aligned_other,
        match_gain=match_gain,
        auto_polarity=auto_polarity,
        invert=invert,
    )
    scaled_other = gain * aligned_other
    residual = aligned_reference - scaled_other
    return aligned_reference, scaled_other, residual, lag, gain, correlation


def compare_model_to_wet(
    di: _np.ndarray,
    sample_rate: int,
    model: LoadedNam,
    wet: _np.ndarray,
    wet_rate: int,
    *,
    wet_name: str = "reamp",
    match_gain: bool,
    auto_polarity: bool,
    invert: bool,
    max_lag_samples: int,
) -> NullResult:
    """Null a model's render of ``di`` against a reamped recording."""
    if wet_rate != sample_rate:
        raise ValueError(
            f"The DI is {sample_rate} Hz but the reamp is {wet_rate} Hz. "
            "The two WAV files must share a sample rate."
        )
    rendered, notes = _render_on_di(model, di, sample_rate)
    return _finish(
        rendered,
        wet,
        sample_rate=sample_rate,
        reference_name=model.name,
        other_name=wet_name,
        notes=notes,
        match_gain=match_gain,
        auto_polarity=auto_polarity,
        invert=invert,
        max_lag_samples=max_lag_samples,
    )


def compare_models(
    di: _np.ndarray,
    sample_rate: int,
    model_a: LoadedNam,
    model_b: LoadedNam,
    *,
    match_gain: bool,
    auto_polarity: bool,
    invert: bool,
    max_lag_samples: int,
) -> NullResult:
    """Null two models rendered on the same DI."""
    rendered_a, notes_a = _render_on_di(model_a, di, sample_rate)
    rendered_b, notes_b = _render_on_di(model_b, di, sample_rate)
    return _finish(
        rendered_a,
        rendered_b,
        sample_rate=sample_rate,
        reference_name=model_a.name,
        other_name=model_b.name,
        notes=notes_a + notes_b,
        match_gain=match_gain,
        auto_polarity=auto_polarity,
        invert=invert,
        max_lag_samples=max_lag_samples,
    )


def _finish(
    reference: _np.ndarray,
    other: _np.ndarray,
    *,
    sample_rate: int,
    reference_name: str,
    other_name: str,
    notes: _Sequence[str],
    match_gain: bool,
    auto_polarity: bool,
    invert: bool,
    max_lag_samples: int,
) -> NullResult:
    aligned_reference, scaled_other, residual, lag, gain, correlation = null_signals(
        reference,
        other,
        max_lag=max_lag_samples,
        match_gain=match_gain,
        auto_polarity=auto_polarity,
        invert=invert,
    )
    # RMS and LUFS of B are before gain, so the readout shows the original level.
    raw_other = scaled_other / gain if gain != 0 else scaled_other
    esr = _esr(scaled_other, aligned_reference)
    finished_notes = list(notes)
    if match_gain:
        finished_notes.append(
            "ESR is measured after matching B to A with least squares."
        )
    elif auto_polarity:
        finished_notes.append(
            "ESR is measured after choosing B's polarity automatically."
        )
    lufs_a = integrated_lufs(aligned_reference, sample_rate)
    lufs_b = integrated_lufs(raw_other, sample_rate)
    lufs_residual = integrated_lufs(residual, sample_rate)
    if _long_enough_for_lufs(len(aligned_reference), sample_rate):
        finished_notes.append(
            "LUFS-I follows ITU-R BS.1770. A and the residual are measured "
            "after alignment; B is measured before level matching."
        )
    else:
        finished_notes.append(
            "The overlap is shorter than 400 ms, so LUFS-I was not measured."
        )
    return NullResult(
        esr=esr,
        residual_db=_db10(esr),
        lag_samples=lag,
        gain=float(gain),
        polarity="inverted" if gain < 0 else "same",
        correlation=correlation,
        sample_rate=sample_rate,
        overlap_samples=int(len(aligned_reference)),
        rms_dbfs_a=_rms_dbfs(aligned_reference),
        rms_dbfs_b=_rms_dbfs(raw_other),
        rms_dbfs_residual=_rms_dbfs(residual),
        lufs_a=lufs_a,
        lufs_b=lufs_b,
        lufs_residual=lufs_residual,
        reference_name=reference_name,
        other_name=other_name,
        notes=finished_notes,
        aligned_reference=aligned_reference,
        aligned_other=scaled_other,
        residual=residual,
    )


def _render_on_di(
    loaded: LoadedNam, di: _np.ndarray, di_rate: int
) -> _Tuple[_np.ndarray, _List[str]]:
    notes: _List[str] = []
    model_rate = loaded.sample_rate
    if model_rate is None:
        notes.append(
            f"{loaded.name} does not store a sample rate. The null test runs "
            f"the model at {di_rate} Hz. If the model was trained at a different "
            "rate, the result is not meaningful."
        )
        return render_audio(loaded.model, di), notes
    if abs(float(model_rate) - di_rate) < 1.0:
        return render_audio(loaded.model, di), notes
    target = int(round(float(model_rate)))
    notes.append(
        f"{loaded.name}: resampled the DI from {di_rate} Hz to {target} Hz "
        f"to match the model, then resampled the output back to {di_rate} Hz."
    )
    upsampled = _librosa.resample(
        _np.ascontiguousarray(di, dtype=_np.float32),
        orig_sr=di_rate,
        target_sr=target,
    )
    rendered = render_audio(loaded.model, upsampled)
    downsampled = _librosa.resample(
        _np.ascontiguousarray(rendered, dtype=_np.float32),
        orig_sr=target,
        target_sr=di_rate,
    )
    return _match_length(downsampled, len(di)).astype(_np.float64), notes


def _match_length(audio: _np.ndarray, length: int) -> _np.ndarray:
    if len(audio) == length:
        return audio
    if len(audio) > length:
        return audio[:length]
    return _np.pad(audio, (0, length - len(audio)))


def _forward_numpy(model, audio32: _np.ndarray, device, pad_start: bool) -> _np.ndarray:
    tensor = _torch.as_tensor(audio32, device=device)
    try:
        with _torch.inference_mode():
            rendered = model(tensor, pad_start=pad_start)
    except ValueError as exc:
        if "too few" in str(exc):
            raise ValueError(
                f"The audio is shorter than the model receptive field "
                f"({int(model.receptive_field)} samples)."
            ) from exc
        raise
    array = rendered.detach().float().cpu().numpy()
    array = _np.squeeze(array)
    if array.ndim != 1:
        raise RuntimeError(
            f"The model returned shape {array.shape}; one-dimensional audio is required."
        )
    return array.astype(_np.float64, copy=False)


def _best_delay(
    reference: _np.ndarray,
    other: _np.ndarray,
    *,
    max_lag: int,
    prefer_abs: bool,
    invert: bool,
) -> int:
    """Positive return value means ``other`` is later than ``reference``."""
    if max_lag == 0:
        return 0
    correlation = _correlate(reference, other, mode="full", method="fft")
    lags = _correlation_lags(len(reference), len(other), mode="full")
    zero = int(_np.flatnonzero(lags == 0)[0])
    window = correlation[zero - max_lag : zero + max_lag + 1]
    scipy_lags = lags[zero - max_lag : zero + max_lag + 1]
    reference_energy = _np.cumsum(reference * reference)
    other_energy = _np.cumsum(other * other)

    best_rank = None
    best_delay = 0
    for scipy_lag, value in zip(scipy_lags, window):
        # scipy lag k pairs reference[n] with other[n-k]. A positive delay of
        # ``other`` (the same event happens later in ``other``) is k = -delay.
        delay = -int(scipy_lag)
        ref_start, other_start, count = _overlap_bounds(
            len(reference), len(other), delay
        )
        if count < 8:
            continue
        denom = _math.sqrt(
            _window_energy(reference_energy, ref_start, count)
            * _window_energy(other_energy, other_start, count)
        )
        if denom < 1e-12:
            continue
        score = float(value) / denom
        rank = abs(score) if prefer_abs else (-score if invert else score)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_delay = delay
    if best_rank is None:
        raise ValueError("Could not align the signals. One of them is nearly silent.")
    return best_delay


def _overlap_bounds(
    n_reference: int, n_other: int, delay: int
) -> _Tuple[int, int, int]:
    if delay >= 0:
        count = min(n_reference, n_other - delay)
        return 0, delay, count
    advance = -delay
    count = min(n_reference - advance, n_other)
    return advance, 0, count


def _slices(
    reference: _np.ndarray, other: _np.ndarray, delay: int
) -> _Tuple[_np.ndarray, _np.ndarray]:
    ref_start, other_start, count = _overlap_bounds(len(reference), len(other), delay)
    if count <= 0:
        raise ValueError("No overlapping audio remains after alignment.")
    return (
        reference[ref_start : ref_start + count],
        other[other_start : other_start + count],
    )


def _window_energy(cumulative: _np.ndarray, start: int, count: int) -> float:
    end = start + count - 1
    total = float(cumulative[end])
    if start > 0:
        total -= float(cumulative[start - 1])
    return total


def _choose_gain(
    reference: _np.ndarray,
    other: _np.ndarray,
    *,
    match_gain: bool,
    auto_polarity: bool,
    invert: bool,
) -> float:
    dot = float(_np.dot(reference, other))
    if match_gain:
        energy = float(_np.dot(other, other))
        if energy < 1e-20:
            raise ValueError(
                "Signal B is nearly silent, so its level cannot be matched."
            )
        return dot / energy
    if auto_polarity:
        return -1.0 if dot < 0 else 1.0
    return -1.0 if invert else 1.0


def _cosine(reference: _np.ndarray, other: _np.ndarray) -> float:
    denom = _math.sqrt(float(_np.dot(reference, reference) * _np.dot(other, other)))
    if denom < 1e-20:
        return 0.0
    return float(_np.dot(reference, other) / denom)


def _esr(prediction: _np.ndarray, target: _np.ndarray) -> float:
    """Same ratio as ``nam.models.losses.esr`` for one pair of 1D signals."""
    target_energy = float(_np.mean(target * target))
    if target_energy < 1e-20:
        raise ValueError("The reference is nearly silent, so ESR cannot be computed.")
    residual = prediction - target
    return float(_np.mean(residual * residual) / target_energy)


def _db10(power_ratio: float) -> _Optional[float]:
    if power_ratio <= 0:
        return None
    return 10.0 * _math.log10(power_ratio)


def _db20(amplitude: float) -> _Optional[float]:
    if amplitude <= 0:
        return None
    return 20.0 * _math.log10(amplitude)


def _rms_dbfs(audio: _np.ndarray) -> _Optional[float]:
    rms = _math.sqrt(float(_np.mean(audio * audio)))
    return _db20(rms)


def _pcm_wav_channels(path: _Path) -> _Optional[int]:
    """Read the channel count from a PCM WAV header, without decoding samples.

    Returns None when the file is not a simple little-endian WAV. Callers
    then fall through to the full reader.
    """
    try:
        with open(path, "rb") as handle:
            header = handle.read(12)
            if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
                return None
            scanned = 12
            while scanned < 1024 * 1024:
                chunk = handle.read(8)
                if len(chunk) < 8:
                    return None
                chunk_id = chunk[:4]
                size = int.from_bytes(chunk[4:8], "little")
                if chunk_id == b"fmt ":
                    fmt = handle.read(4)
                    if len(fmt) < 4:
                        return None
                    return int.from_bytes(fmt[2:4], "little")
                step = size + (size & 1)
                handle.seek(step, 1)
                scanned += 8 + step
    except OSError:
        return None
    return None


def _is_not_mono(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "mono" in text or "channel" in text


def _long_enough_for_lufs(n_samples: int, sample_rate: int) -> bool:
    return sample_rate > 0 and n_samples / float(sample_rate) >= _LUFS_BLOCK_SECONDS


def integrated_lufs(audio: _np.ndarray, sample_rate: int) -> _Optional[float]:
    """ITU-R BS.1770-4 integrated loudness of one channel, in LUFS.

    None means the signal is shorter than one 400 ms block, silent, or
    entirely below the absolute gate of −70 LUFS.
    """
    samples = _np.ascontiguousarray(audio, dtype=_np.float64)
    if samples.ndim != 1:
        raise ValueError("LUFS needs mono audio.")
    if not _long_enough_for_lufs(len(samples), sample_rate):
        return None
    rate = float(sample_rate)
    duration = len(samples) / rate
    step = 0.25
    num_blocks = int(
        round(((duration - _LUFS_BLOCK_SECONDS) / (_LUFS_BLOCK_SECONDS * step)) + 1)
    )
    if num_blocks < 1:
        return None
    weighted = _k_weight(samples, rate)
    powers = _np.empty(num_blocks, dtype=_np.float64)
    for index in range(num_blocks):
        lower = int(_LUFS_BLOCK_SECONDS * (index * step) * rate)
        upper = int(_LUFS_BLOCK_SECONDS * (index * step + 1) * rate)
        # Divide by the nominal block length, matching BS.1770, not by upper-lower.
        powers[index] = float(_np.sum(weighted[lower:upper] ** 2)) / (
            _LUFS_BLOCK_SECONDS * rate
        )
    return _gated_lufs(powers)


def _gated_lufs(powers: _np.ndarray) -> _Optional[float]:
    with _np.errstate(divide="ignore", invalid="ignore"):
        block_lufs = _LUFS_OFFSET_DB + 10.0 * _np.log10(powers)
    above_absolute = block_lufs >= _LUFS_ABSOLUTE_GATE_LUFS
    if not _np.any(above_absolute):
        return None
    absolute_power = float(_np.mean(powers[above_absolute]))
    if absolute_power <= 0.0:
        return None
    relative_gate = _LUFS_OFFSET_DB + 10.0 * _math.log10(absolute_power) - 10.0
    keep = (block_lufs > relative_gate) & (block_lufs > _LUFS_ABSOLUTE_GATE_LUFS)
    if not _np.any(keep):
        return None
    gated_power = float(_np.mean(powers[keep]))
    if gated_power <= 0.0:
        return None
    loudness = _LUFS_OFFSET_DB + 10.0 * _math.log10(gated_power)
    if not _math.isfinite(loudness):
        return None
    return float(loudness)


def _k_weight(audio: _np.ndarray, rate: float) -> _np.ndarray:
    """BS.1770 K-weighting. DeMan biquads match the ITU coefficients."""
    shelf_b, shelf_a = _deman_high_shelf(rate)
    highpass_b, highpass_a = _deman_high_pass(rate)
    shelved = _lfilter(shelf_b, shelf_a, audio)
    return _lfilter(highpass_b, highpass_a, shelved)


def _deman_high_shelf(rate: float) -> _Tuple[_np.ndarray, _np.ndarray]:
    gain_db = 3.99984385397
    quality = 0.7071752369554193
    frequency = 1681.9744509555319
    k = _math.tan(_math.pi * frequency / rate)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh**0.499666774155
    a0 = 1.0 + k / quality + k * k
    b0 = (vh + vb * k / quality + k * k) / a0
    b1 = 2.0 * (k * k - vh) / a0
    b2 = (vh - vb * k / quality + k * k) / a0
    a1 = 2.0 * (k * k - 1.0) / a0
    a2 = (1.0 - k / quality + k * k) / a0
    return _np.array([b0, b1, b2]), _np.array([1.0, a1, a2])


def _deman_high_pass(rate: float) -> _Tuple[_np.ndarray, _np.ndarray]:
    quality = 0.5003270373253953
    frequency = 38.13547087613982
    k = _math.tan(_math.pi * frequency / rate)
    denom = 1.0 + k / quality + k * k
    a1 = 2.0 * (k * k - 1.0) / denom
    a2 = (1.0 - k / quality + k * k) / denom
    return _np.array([1.0, -2.0, 1.0]), _np.array([1.0, a1, a2])


def _finite(value: _Optional[float]) -> _Optional[float]:
    if value is None or not _math.isfinite(value):
        return None
    return float(value)


def _envelope(audio: _np.ndarray, buckets: int) -> dict:
    count = max(1, min(int(buckets), len(audio)))
    edges = _np.linspace(0, len(audio), count + 1, dtype=int)
    minima = []
    maxima = []
    for index in range(count):
        segment = audio[edges[index] : edges[index + 1]]
        if len(segment) == 0:
            minima.append(0.0)
            maxima.append(0.0)
            continue
        minima.append(round(float(segment.min()), 5))
        maxima.append(round(float(segment.max()), 5))
    return {"min": minima, "max": maxima}
