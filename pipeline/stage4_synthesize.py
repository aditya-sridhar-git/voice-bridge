"""
Stage 4 — Voice Synthesis (OpenVoice v2)
==========================================
Synthesizes speech in the target accent using:
  - MeloTTS as the base TTS (multi-accent: EN-US, EN-BR, EN-AU, EN-IN)
  - OpenVoice v2 ToneColorConverter to clone the source speaker's timbre

The rewritten transcript from Stage 3 is fed into MeloTTS, and the
speaker x-vector (from Stage 2) is used to drive tone color conversion.

Output: WAV file at 22050 Hz, mono.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Accent pair → MeloTTS language/speaker ID
ACCENT_TO_MELOTTS = {
    "indian_american": ("EN-US", 0),
    "indian_british":  ("EN-BR", 0),
}

# OpenVoice v2 checkpoint path (relative to repo root)
OPENVOICE_CKPT = Path(__file__).parent.parent / "checkpoints_v2" / "converter"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_speaking_rate(word_durations: dict, target_words: list) -> float:
    """
    Compute speaking rate scalar relative to 1.0 (normal speed).
    Compares source word durations to an average reference.
    """
    if not word_durations:
        return 1.0
    durations = list(word_durations.values())
    avg_dur = sum(durations) / len(durations)
    # Reference: average English syllable duration ~0.2s
    rate = 0.2 / max(avg_dur, 0.05)
    return float(np.clip(rate, 0.6, 1.8))


# ---------------------------------------------------------------------------
# Base TTS with MeloTTS
# ---------------------------------------------------------------------------

def _synthesize_base(
    text: str,
    accent_pair: str = "indian_american",
    speed: float = 1.0,
    output_path: str = None,
    device: str = "cpu",
) -> str:
    """
    Synthesize text using MeloTTS with the target accent.
    Returns path to synthesized WAV.
    """
    try:
        from melo.api import TTS

        lang_key, speaker_idx = ACCENT_TO_MELOTTS.get(accent_pair, ("EN-US", 0))

        logger.info(f"[Stage 4] MeloTTS: lang={lang_key}, speed={speed:.2f}")
        model = TTS(language="EN", device=device)
        speaker_ids = model.hps.data.spk2id

        # Pick the correct speaker ID
        speaker_id = list(speaker_ids.values())[speaker_idx]

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name

        model.tts_to_file(
            text=text,
            speaker_id=speaker_id,
            output_path=output_path,
            speed=speed,
            quiet=True,
        )
        logger.info(f"  Base TTS written to {output_path}")
        return output_path

    except ImportError:
        raise ImportError(
            "MeloTTS not installed. Install with: pip install melo-tts"
        )
    except Exception as e:
        raise RuntimeError(f"MeloTTS synthesis failed: {e}") from e


# ---------------------------------------------------------------------------
# Tone color conversion (speaker voice cloning)
# ---------------------------------------------------------------------------

def _apply_tone_color_conversion(
    tts_audio_path: str,
    source_audio_path: str,
    output_path: str,
    device: str = "cpu",
) -> str:
    """
    Apply OpenVoice v2 ToneColorConverter to clone source speaker timbre
    onto the TTS-synthesized audio.

    Args:
        tts_audio_path:   Path to MeloTTS synthesized audio (target accent).
        source_audio_path: Path to original source speaker audio.
        output_path:      Path to write the converted audio.
        device:           "cuda" or "cpu".

    Returns:
        output_path
    """
    try:
        import torch
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter

        ckpt = str(OPENVOICE_CKPT)
        if not Path(ckpt).exists():
            logger.warning(
                f"OpenVoice v2 checkpoint not found at {ckpt}. "
                "Skipping tone color conversion — output is raw MeloTTS audio."
            )
            import shutil
            shutil.copy(tts_audio_path, output_path)
            return output_path

        logger.info("[Stage 4] Loading OpenVoice ToneColorConverter …")
        converter = ToneColorConverter(
            f"{ckpt}/config.json", device=device
        )
        converter.load_ckpt(f"{ckpt}/checkpoint.pth")

        # Extract speaker style embedding from source audio
        logger.info("  Extracting source speaker style embedding …")
        target_se, _ = se_extractor.get_se(
            source_audio_path, converter, vad=True
        )

        # Extract base speaker embedding from TTS audio
        logger.info("  Extracting base speaker style embedding …")
        src_se, _ = se_extractor.get_se(
            tts_audio_path, converter, vad=False
        )

        logger.info("  Running tone color conversion …")
        converter.convert(
            audio_src_path=tts_audio_path,
            src_se=src_se,
            tgt_se=target_se,
            output_path=output_path,
            message="@voice-bridge",
        )

        logger.info(f"  Converted audio saved to {output_path}")
        return output_path

    except ImportError:
        logger.warning(
            "OpenVoice not installed. Skipping tone color conversion."
        )
        import shutil
        shutil.copy(tts_audio_path, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Tone color conversion failed: {e}. Using raw TTS output.")
        import shutil
        shutil.copy(tts_audio_path, output_path)
        return output_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def synthesize(
    transcript: str,
    source_audio_path: str,
    accent_pair: str = "indian_american",
    word_durations: Optional[dict] = None,
    output_path: Optional[str] = None,
    device: str = "auto",
) -> str:
    """
    Full Stage 4 synthesis: MeloTTS → ToneColorConverter.

    Args:
        transcript:        Plain-text transcript (from Stage 3).
        source_audio_path: Original input audio (for voice cloning).
        accent_pair:       Target accent pair key.
        word_durations:    Word duration map from Stage 2 (for speed matching).
        output_path:       Where to save the final synthesized WAV.
        device:            "auto", "cuda", or "cpu".

    Returns:
        Path to synthesized output WAV.
    """
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    logger.info(f"[Stage 4] Voice synthesis (device={device})")

    # Determine speaking rate from source prosody
    words = transcript.split()
    speed = _get_speaking_rate(word_durations or {}, words)
    logger.info(f"  Speaking rate: {speed:.2f}x")

    # Step 1: MeloTTS base synthesis
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tts_path = tmp.name

    tts_path = _synthesize_base(
        text=transcript,
        accent_pair=accent_pair,
        speed=speed,
        output_path=tts_path,
        device=device,
    )

    # Step 2: Tone color conversion
    if output_path is None:
        output_path = tts_path.replace(".wav", "_converted.wav")

    final_path = _apply_tone_color_conversion(
        tts_audio_path=tts_path,
        source_audio_path=source_audio_path,
        output_path=output_path,
        device=device,
    )

    # Cleanup temp TTS file if different from output
    if tts_path != final_path and os.path.exists(tts_path):
        os.unlink(tts_path)

    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4: Voice synthesis")
    parser.add_argument("transcript", help="Text or path to Stage 1/3 JSON")
    parser.add_argument("source_audio", help="Original speaker audio for voice cloning")
    parser.add_argument("--accent-pair", default="indian_american")
    parser.add_argument("--output", default="stage4_output.wav")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    text = args.transcript
    if text.endswith(".json") and Path(text).exists():
        import sys, json
        with open(text) as f:
            data = json.load(f)
        text = data.get("target_transcript", data.get("transcript", ""))

    out = synthesize(
        transcript=text,
        source_audio_path=args.source_audio,
        accent_pair=args.accent_pair,
        output_path=args.output,
        device=args.device,
    )
    print(f"\nSynthesized audio: {out}")
