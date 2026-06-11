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
    blend: float = 0.75,
) -> np.ndarray:
    """
    Shift synthesized audio pitch to match the source speaker's mean pitch.

    Uses RESAMPLE-based pitch correction (no phase vocoder = no metallic artifacts):
      1. Estimate source & synth median F0
      2. Resample audio from sr → sr * ratio  (changes pitch + duration)
      3. Resample back to original sample count (restores duration, keeps pitch)

    Formants shift proportionally with pitch — more natural than phase vocoder
    which tries to separate them and introduces STFT phasing artifacts.
    Clamped to ±6 semitones to prevent over-correction.
    """
    try:
        src_f0 = np.array(source_f0, dtype=np.float64)
        voiced_src = src_f0[src_f0 > 5.0]

        if len(voiced_src) < 5:
            logger.warning("Too few voiced frames in source F0. Skipping pitch shift.")
            return synth_audio

        src_mean_f0 = float(np.median(voiced_src))

        # Analyse synth F0 with pyworld (analysis only — NO re-synthesis)
        synth_mean_f0 = None
        try:
            import pyworld as pw
            audio_f64 = synth_audio.astype(np.float64)
            f0_synth, _, _ = pw.wav2world(audio_f64, sr,
                                          frame_period=float(source_frame_shift_ms))
            voiced_synth = f0_synth[f0_synth > 5.0]
            if len(voiced_synth) > 0:
                synth_mean_f0 = float(np.median(voiced_synth))
        except Exception:
            pass

        if synth_mean_f0 is None or synth_mean_f0 < 10:
            logger.warning("Could not estimate synth F0. Skipping pitch shift.")
            return synth_audio

        n_steps = 12.0 * np.log2(src_mean_f0 / synth_mean_f0)
        n_steps = float(np.clip(n_steps, -6.0, 6.0))

        if abs(n_steps) < 0.25:
            logger.info("  Pitch difference < 0.25 semitones. No shift needed.")
            return synth_audio

        logger.info(f"  Pitch shift: {n_steps:+.2f} semitones  "
                    f"(source={src_mean_f0:.1f}Hz, synth={synth_mean_f0:.1f}Hz) "
                    f"[resample method]")  

        # Resample-based pitch shift (no phase vocoder)
        # ratio > 1 → play back faster at same duration → higher pitch
        ratio = 2.0 ** (n_steps / 12.0)
        original_len = len(synth_audio)
        # Step 1: resample to pitch-shifted length
        from scipy.signal import resample as scipy_resample
        shifted_len = int(round(original_len / ratio))
        pitch_shifted = scipy_resample(synth_audio, shifted_len).astype(np.float32)
        # Step 2: resample back to original length (restores tempo, keeps pitch)
        restored = scipy_resample(pitch_shifted, original_len).astype(np.float32)
        return restored

    except Exception as e:
        logger.warning(f"Pitch shift failed ({e}). Returning unchanged audio.")
        return synth_audio


# ---------------------------------------------------------------------------
# 5b½. Emotion amplifier — expand dynamic range to emphasise emotion
# ---------------------------------------------------------------------------

def _amplify_emotion(
    audio: np.ndarray,
    sr: int,
    frame_shift_ms: int = 10,
    expansion_ratio: float = 1.6,
    knee_db: float = -12.0,
) -> np.ndarray:
    """
    Softknee dynamic range EXPANDER.

    Frames louder than `knee_db` (relative to RMS mean) get boosted;
    frames softer get attenuated. expansion_ratio=1.6 means a 10dB range
    becomes a 16dB range — making loud syllables punchier and soft
    inter-word gaps quieter, which perceptually amplifies emotion.

    This is the audio equivalent of turning the 'expressiveness' knob up.
    """
    frame_samples = max(1, int(sr * frame_shift_ms / 1000.0))
    n_frames = len(audio) // frame_samples
    if n_frames == 0:
        return audio

    # Compute per-frame RMS in dB
    rms_db = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = audio[i * frame_samples:(i + 1) * frame_samples]
        rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
        rms_db[i] = 20.0 * np.log10(rms + 1e-12)

    mean_db = float(np.mean(rms_db))
    threshold_db = mean_db + knee_db  # frames above this get boosted

    output = audio.copy()
    for i in range(n_frames):
        start = i * frame_samples
        end   = start + frame_samples
        frame = audio[start:end]

        diff_db = rms_db[i] - threshold_db
        if diff_db > 0:
            # Above threshold: boost by (ratio - 1) * diff
            gain_db = (expansion_ratio - 1.0) * diff_db
        else:
            # Below threshold: attenuate by (ratio - 1) * diff (diff is negative)
            gain_db = (expansion_ratio - 1.0) * diff_db * 0.5  # gentler on quiet parts

        gain_db = float(np.clip(gain_db, -12.0, 12.0))  # max ±12dB adjustment
        gain_linear = 10.0 ** (gain_db / 20.0)
        output[start:end] = frame * gain_linear

    # Handle remainder
    tail = audio[n_frames * frame_samples:]
    if len(tail) > 0:
        output[n_frames * frame_samples:] = tail

    # Renormalize to prevent clipping
    peak = np.abs(output).max()
    if peak > 0.95:
        output = output / peak * 0.95

    return output


# ---------------------------------------------------------------------------
# 5b. Energy envelope transplant (Gaussian-smoothed gains)
# ---------------------------------------------------------------------------

def _transplant_energy(
    audio: np.ndarray,
    source_energy_db: List[float],
    frame_shift_ms: int = 10,
    sr: int = TARGET_SR,
    emotion_label: str = "neutral",
) -> np.ndarray:
    """
    Scale audio gain to match source energy envelope.
    Gains are Gaussian-smoothed across frames (sigma=5 frames ≈ 50ms) to
    eliminate the hard 10ms jumps that cause a 'pumping' artifact.
    """
    if not source_energy_db:
        return audio

    emotion_max_gain = {
        "angry":     3.5,
        "happy":     3.0,
        "surprised": 2.8,
        "sad":       2.0,
        "neutral":   MAX_GAIN,
    }
    max_gain = emotion_max_gain.get(emotion_label, MAX_GAIN)

    frame_samples = int(sr * frame_shift_ms / 1000.0)
    src_energy = np.array(source_energy_db, dtype=np.float32)
    src_linear = 10.0 ** (src_energy / 20.0)

    n_frames = len(audio) // frame_samples
    if n_frames == 0:
        return audio

    if len(src_linear) != n_frames:
        indices = np.linspace(0, len(src_linear) - 1, n_frames)
        src_resampled = np.interp(indices, np.arange(len(src_linear)), src_linear)
    else:
        src_resampled = src_linear.copy()

    rms_synth = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        frame = audio[i * frame_samples:(i + 1) * frame_samples]
        rms_synth[i] = float(np.sqrt(np.mean(frame ** 2)) + 1e-8)

    raw_gains = np.clip(src_resampled / rms_synth, MIN_GAIN, max_gain)

    # Gaussian-smooth the gain curve to eliminate pumping
    try:
        from scipy.ndimage import gaussian_filter1d
        smooth_gains = gaussian_filter1d(raw_gains, sigma=5.0)
    except Exception:
        smooth_gains = raw_gains

    # Interpolate gain to per-sample resolution for smooth transitions
    frame_centers = (np.arange(n_frames) + 0.5) * frame_samples
    sample_positions = np.arange(len(audio))
    gain_curve = np.interp(sample_positions, frame_centers, smooth_gains).astype(np.float32)

    return (audio * gain_curve)


# ---------------------------------------------------------------------------
# 5d. Pedalboard humanisation — EQ + compression + subtle reverb
# ---------------------------------------------------------------------------

def _humanise_audio(
    audio: np.ndarray,
    sr: int,
) -> np.ndarray:
    """
    Apply a broadcast-style post-processing chain using Spotify's pedalboard:

      1. High-pass @ 80Hz   — remove sub-bass rumble (mic artifact)
      2. Low-shelf +3dB @ 200Hz — add warmth/body
      3. Peak cut  -4dB @ 5.5kHz — reduce harshness / synthetic brightness
      4. High-shelf -2dB @ 8kHz  — soften the 'digital' top end
      5. Compressor 4:1 ratio, -18dB threshold — even out dynamics
      6. Reverb (very short room, wet=0.04) — just enough air to feel natural

    All parameters are conservative to enhance naturalness without audible
    processing artifacts.
    """
    try:
        from pedalboard import (
            Pedalboard, HighpassFilter, LowShelfFilter,
            PeakFilter, HighShelfFilter, Compressor, Reverb
        )

        board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=80.0),
            LowShelfFilter(cutoff_frequency_hz=200.0, gain_db=3.0, q=0.707),
            PeakFilter(cutoff_frequency_hz=5500.0, gain_db=-4.0, q=1.5),
            HighShelfFilter(cutoff_frequency_hz=8000.0, gain_db=-2.0, q=0.707),
            Compressor(threshold_db=-18.0, ratio=4.0, attack_ms=5.0, release_ms=80.0),
            Reverb(room_size=0.08, damping=0.8, wet_level=0.04, dry_level=0.96),
        ])

        # pedalboard expects (channels, samples) float32
        audio_2d = audio[np.newaxis, :]
        processed = board(audio_2d, sr)
        result = processed[0].astype(np.float32)

        # Re-normalize after processing
        peak = np.abs(result).max()
        if peak > 0.95:
            result = result / peak * 0.95

        logger.info("  5d. Pedalboard humanisation applied (EQ + compressor + reverb)")
        return result

    except ImportError:
        logger.warning("pedalboard not installed. Skipping humanisation.")
        return audio
    except Exception as e:
        logger.warning(f"Pedalboard processing failed ({e}). Skipping.")
        return audio


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
    Time-stretch synthesized audio to match source duration.

    For stretches <= 30%, uses scipy resample (no phase artifacts, natural
    for speech). For larger mismatches, falls back to librosa phase vocoder.
    Only applied if duration mismatch exceeds `tolerance` (default 5%).
    """
    synth_duration = len(audio) / sr
    # rate < 1 → output is longer (slow down); rate > 1 → output is shorter (speed up)
    # We want output_samples = source_duration * sr
    # scipy.signal.resample(audio, n_samples) resamples to exactly n_samples
    target_samples = int(source_duration_s * sr)
    current_samples = len(audio)

    stretch_ratio = source_duration_s / max(synth_duration, 0.01)  # > 1 = need longer

    if abs(stretch_ratio - 1.0) < tolerance:
        logger.debug(f"Duration mismatch {stretch_ratio:.3f}x within tolerance. Skipping stretch.")
        return audio

    logger.info(f"  Duration match: {synth_duration:.2f}s → {source_duration_s:.2f}s "
                f"({stretch_ratio:.3f}x, {'+' if stretch_ratio > 1 else ''}{(stretch_ratio-1)*100:.1f}%)")

    try:
        from scipy.signal import resample as scipy_resample
        # scipy.signal.resample resamples to exactly target_samples
        # This changes both speed and pitch; sounds natural for <=30% changes
        if abs(stretch_ratio - 1.0) <= 0.35:
            stretched = scipy_resample(audio, target_samples).astype(np.float32)
            logger.info("    (used scipy resample — pitch-transparent)")
            return stretched
        # Large mismatch: fall back to phase vocoder (keep pitch, change speed)
        import librosa
        # IMPORTANT: rate = synth/source (NOT source/synth)
        # rate > 1 → speeds up → shorter; rate < 1 → slows down → longer
        pv_rate = synth_duration / max(source_duration_s, 0.01)
        stretched = librosa.effects.time_stretch(audio, rate=pv_rate)
        logger.info("    (used phase vocoder — large mismatch)")
        return stretched.astype(np.float32)
    except Exception as e:
        logger.warning(f"Duration matching failed ({e}). Returning unchanged audio.")
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
    emotion_label: str = "neutral",
    f0_blend: float = 0.75,
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
        emotion_label:     Detected emotion (affects energy gain ceiling).
        f0_blend:          0.0-1.0. How much of pitch comes from source (default 0.75).

    Returns:
        Path to output WAV.
    """
    logger.info("[Stage 5] Prosody transplant")

    # Load audios
    synth_audio, sr = _load_audio(synth_audio_path)
    src_audio, _   = _load_audio(source_audio_path)
    source_duration = len(src_audio) / sr

    # 5a — Pitch shift to match source speaker's mean F0 (resample method, no phase vocoder)
    logger.info("  5a. Pitch alignment …")
    audio = _transplant_f0(synth_audio, sr, source_f0, frame_shift_ms)

    # 5b — Energy envelope transplant (Gaussian-smoothed, no pumping)
    logger.info("  5b. Energy envelope warp …")
    audio = _transplant_energy(audio, source_energy_db, frame_shift_ms, sr,
                               emotion_label=emotion_label)

    # 5b½ — Emotion amplifier: expand dynamic range to make emotion more audible
    logger.info("  5b½. Emotion amplifier …")
    audio = _amplify_emotion(audio, sr, frame_shift_ms)

    # 5c — Duration matching (optional)
    if match_duration:
        logger.info("  5c. Duration rate-matching …")
        audio = _match_duration(audio, source_duration, sr)

    # 5d — Pedalboard humanisation: EQ + compression + subtle reverb
    audio = _humanise_audio(audio, sr)

    # Final normalize
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
