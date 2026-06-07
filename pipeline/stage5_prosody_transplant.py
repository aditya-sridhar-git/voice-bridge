"""
Stage 5 — Prosody Transplant
==============================
Warps the synthesized audio's prosodic features (pitch, energy, duration)
to match the original speaker's emotional prosody extracted in Stage 2.

This is the core emotion-preservation mechanism.

Three sub-steps:
  5a. F0 contour transplant   — pyworld WORLD vocoder
  5b. Energy envelope warp    — RMS gain scaling
  5c. Duration rate-matching  — librosa phase vocoder time-stretch

Output: WAV file at 22050 Hz, mono.
"""

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

TARGET_SR = 22050
MIN_GAIN  = 0.4
MAX_GAIN  = 2.5


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _load_audio(path: str, target_sr: int = TARGET_SR) -> tuple:
    """Load audio, resample to target_sr, return (audio_float32, sr)."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            pass  # proceed at native sr if librosa unavailable
    return audio, target_sr


def _smooth(arr: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply simple moving-average smoothing to reduce artifacts."""
    if window < 2:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


# ---------------------------------------------------------------------------
# 5a. F0 contour transplant
# ---------------------------------------------------------------------------

def _transplant_f0(
    synth_audio: np.ndarray,
    sr: int,
    source_f0: List[float],
    source_frame_shift_ms: int = 10,
) -> np.ndarray:
    """
    Warp the synthesized audio's F0 contour to match the source speaker's
    relative pitch shape while preserving the synthesized mean F0.

    Strategy:
      1. Decompose synthesized audio with WORLD vocoder (F0, SP, AP)
      2. Normalize source F0 shape to zero-mean, unit-variance
      3. Re-scale normalized shape to synthesized mean/std
      4. Re-synthesize with transplanted F0
    """
    try:
        import pyworld as pw

        audio_f64 = synth_audio.astype(np.float64)
        frame_period = float(source_frame_shift_ms)

        # WORLD analysis
        f0_synth, sp, ap = pw.wav2world(audio_f64, sr, frame_period=frame_period)

        # Build source F0 array resampled to match synthesized length
        src_f0 = np.array(source_f0, dtype=np.float64)
        n_synth = len(f0_synth)

        if len(src_f0) != n_synth:
            # Resample source F0 to match synthesized frame count
            indices = np.linspace(0, len(src_f0) - 1, n_synth)
            src_f0 = np.interp(indices, np.arange(len(src_f0)), src_f0)

        # Work only on voiced frames (F0 > 0 in both)
        voiced_synth = f0_synth > 0
        voiced_src   = src_f0 > 0
        voiced_both  = voiced_synth & voiced_src

        if voiced_both.sum() < 5:
            logger.warning("Too few voiced frames for F0 transplant. Skipping.")
            return synth_audio

        # Normalize source contour → re-scale to synthesized distribution
        src_voiced = src_f0[voiced_both]
        syn_voiced = f0_synth[voiced_both]

        src_norm = (src_voiced - src_voiced.mean()) / (src_voiced.std() + 1e-8)
        f0_transplanted = f0_synth.copy()
        f0_transplanted[voiced_both] = (
            src_norm * syn_voiced.std() + syn_voiced.mean()
        )

        # Clip to physiologically valid range
        f0_transplanted = np.clip(f0_transplanted, 0.0, 600.0)

        # Smooth to reduce artifacts at voiced/unvoiced boundaries
        f0_transplanted = _smooth(f0_transplanted, window=3)
        f0_transplanted[~voiced_synth] = 0.0  # restore unvoiced

        # Re-synthesize
        output = pw.synthesize(f0_transplanted, sp, ap, sr, frame_period=frame_period)
        return output.astype(np.float32)

    except ImportError:
        logger.warning("pyworld not installed. Skipping F0 transplant.")
        return synth_audio
    except Exception as e:
        logger.warning(f"F0 transplant failed ({e}). Returning unchanged audio.")
        return synth_audio


# ---------------------------------------------------------------------------
# 5b. Energy envelope transplant
# ---------------------------------------------------------------------------

def _transplant_energy(
    audio: np.ndarray,
    source_energy_db: List[float],
    frame_shift_ms: int = 10,
    sr: int = TARGET_SR,
) -> np.ndarray:
    """
    Scale audio gain frame-by-frame to match source energy envelope.
    """
    if not source_energy_db:
        return audio

    frame_samples = int(sr * frame_shift_ms / 1000.0)
    src_energy = np.array(source_energy_db, dtype=np.float32)

    # Convert source dB → linear amplitude
    src_linear = 10.0 ** (src_energy / 20.0)

    # Compute synthesized energy per frame
    n_frames = len(audio) // frame_samples
    if n_frames == 0:
        return audio

    # Resample source energy to match number of synthesized frames
    if len(src_linear) != n_frames:
        indices = np.linspace(0, len(src_linear) - 1, n_frames)
        src_resampled = np.interp(indices, np.arange(len(src_linear)), src_linear)
    else:
        src_resampled = src_linear

    output = audio.copy()
    for i in range(n_frames):
        start = i * frame_samples
        end   = start + frame_samples
        frame = audio[start:end]

        rms_synth = np.sqrt(np.mean(frame ** 2)) + 1e-8
        rms_target = src_resampled[i]

        gain = float(np.clip(rms_target / rms_synth, MIN_GAIN, MAX_GAIN))
        output[start:end] = frame * gain

    # Handle remainder
    remainder = audio[n_frames * frame_samples:]
    if len(remainder) > 0:
        output[n_frames * frame_samples:] = remainder

    return output


# ---------------------------------------------------------------------------
# 5c. Duration / rate matching
# ---------------------------------------------------------------------------

def _match_duration(
    audio: np.ndarray,
    source_duration_s: float,
    sr: int = TARGET_SR,
    tolerance: float = 0.05,
) -> np.ndarray:
    """
    Time-stretch audio to match source duration using librosa phase vocoder.
    Only applied if duration mismatch exceeds `tolerance` (default 5%).
    """
    synth_duration = len(audio) / sr
    ratio = source_duration_s / max(synth_duration, 0.01)

    if abs(ratio - 1.0) < tolerance:
        logger.debug(f"Duration mismatch {ratio:.3f}x within tolerance. Skipping stretch.")
        return audio

    logger.info(f"  Duration stretch: {synth_duration:.2f}s → {source_duration_s:.2f}s (rate={ratio:.3f})")

    try:
        import librosa
        # phase vocoder: rate > 1 speeds up (shortens), rate < 1 slows down
        stretched = librosa.effects.time_stretch(audio, rate=ratio)
        return stretched
    except ImportError:
        logger.warning("librosa not installed. Skipping duration matching.")
        return audio
    except Exception as e:
        logger.warning(f"Duration matching failed ({e}).")
        return audio


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def transplant_prosody(
    synth_audio_path: str,
    source_audio_path: str,
    source_f0: List[float],
    source_energy_db: List[float],
    frame_shift_ms: int = 10,
    output_path: Optional[str] = None,
    match_duration: bool = True,
) -> str:
    """
    Apply prosody transplant to synthesized audio.

    Args:
        synth_audio_path:  Path to Stage 4 synthesized audio.
        source_audio_path: Path to original input audio (for duration ref).
        source_f0:         F0 contour from Stage 2 (list of floats, Hz).
        source_energy_db:  Energy contour from Stage 2 (list of floats, dB).
        frame_shift_ms:    Frame shift used in Stage 2 extraction.
        output_path:       Where to save the final output WAV.
        match_duration:    If True, apply phase-vocoder duration matching.

    Returns:
        Path to output WAV.
    """
    logger.info("[Stage 5] Prosody transplant")

    # Load audios
    synth_audio, sr = _load_audio(synth_audio_path)
    src_audio, _   = _load_audio(source_audio_path)
    source_duration = len(src_audio) / sr

    # 5a — F0 transplant
    logger.info("  5a. F0 contour transplant …")
    audio = _transplant_f0(synth_audio, sr, source_f0, frame_shift_ms)

    # 5b — Energy transplant
    logger.info("  5b. Energy envelope warp …")
    audio = _transplant_energy(audio, source_energy_db, frame_shift_ms, sr)

    # 5c — Duration matching (optional)
    if match_duration:
        logger.info("  5c. Duration rate-matching …")
        audio = _match_duration(audio, source_duration, sr)

    # Normalize to prevent clipping
    peak = np.abs(audio).max()
    if peak > 0.98:
        audio = audio / peak * 0.98

    # Save
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name

    sf.write(output_path, audio, sr)
    logger.info(f"  Final audio saved to {output_path} ({len(audio)/sr:.2f}s)")

    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Stage 5: Prosody transplant")
    parser.add_argument("synth_audio",  help="Stage 4 synthesized audio")
    parser.add_argument("source_audio", help="Original source audio")
    parser.add_argument("features_json", help="Stage 2 FeatureBundle JSON")
    parser.add_argument("--output", default="stage5_output.wav")
    parser.add_argument("--no-duration", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.features_json) as f:
        features = json.load(f)

    prosody = features["prosody"]
    out = transplant_prosody(
        synth_audio_path=args.synth_audio,
        source_audio_path=args.source_audio,
        source_f0=prosody["f0_hz"],
        source_energy_db=prosody["energy_db"],
        frame_shift_ms=prosody.get("frame_shift_ms", 10),
        output_path=args.output,
        match_duration=not args.no_duration,
    )
    print(f"\nFinal output: {out}")
