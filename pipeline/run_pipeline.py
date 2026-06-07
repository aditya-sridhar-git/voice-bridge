"""
Pipeline Orchestrator
======================
Runs all 5 stages end-to-end for emotion-preserving accent conversion.

Usage:
    python pipeline/run_pipeline.py input.wav --accent indian_american --output output.wav

Or programmatically:
    from pipeline.run_pipeline import run_pipeline
    result = run_pipeline("input.wav", accent_pair="indian_american")
"""

import sys
from pathlib import Path as _Path
# Ensure repo root is on sys.path so `pipeline.*` is importable
# whether this file is run as `python pipeline/run_pipeline.py`
# or imported as `from pipeline.run_pipeline import run_pipeline`.
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).parent.parent / "runs"


# ---------------------------------------------------------------------------
# Run context — intermediate files saved to runs/<uuid>/
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    run_id: str
    run_dir: str
    input_audio: str
    output_audio: str
    accent_pair: str
    transcript: str
    emotion_label: str
    emotion_score: float
    rewrites_count: int
    duration_s: float

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def _make_run_dir(run_id: str) -> Path:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    input_audio: str,
    accent_pair: str = "indian_american",
    output_path: Optional[str] = None,
    whisper_model: str = "large-v3",
    apply_vad: bool = True,
    match_duration: bool = True,
    device: str = "auto",
    run_id: Optional[str] = None,
    verbose: bool = True,
) -> PipelineResult:
    """
    Run the full emotion-preserving accent conversion pipeline.

    Args:
        input_audio:    Path to input audio file (WAV/MP3/FLAC).
        accent_pair:    "indian_american" or "indian_british".
        output_path:    Output WAV path. If None, saved to runs/<id>/output.wav.
        whisper_model:  Whisper model size (tiny/base/small/medium/large-v3).
        apply_vad:      Strip silence before Whisper transcription.
        match_duration: Apply duration rate-matching in Stage 5.
        device:         "auto", "cuda", or "cpu".
        run_id:         Reuse an existing run directory for debugging.
        verbose:        Enable INFO logging.

    Returns:
        PipelineResult with paths to all intermediate outputs.
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # ---- Setup run directory ----
    run_id = run_id or str(uuid.uuid4())[:8]
    run_dir = _make_run_dir(run_id)
    logger.info(f"=== voice-bridge pipeline | run_id={run_id} ===")
    logger.info(f"Run directory: {run_dir}")

    t_start = time.time()

    # ========================================================================
    # Stage 1 — Transcription
    # ========================================================================
    logger.info("\n── Stage 1: Transcription ──")
    from pipeline.stage1_transcribe import transcribe  # noqa: E402

    transcript_path = str(run_dir / "stage1_transcript.json")
    transcript_result = transcribe(
        audio_path=input_audio,
        model_size=whisper_model,
        apply_vad=apply_vad,
        device=device,
        output_path=transcript_path,
    )
    logger.info(f"Transcript: \"{transcript_result.transcript[:80]}…\"")

    word_timestamps = [
        {"word": w.word, "start": w.start, "end": w.end}
        for w in transcript_result.words
    ]

    # ========================================================================
    # Stage 2 — Feature extraction
    # ========================================================================
    logger.info("\n── Stage 2: Feature Extraction ──")
    from pipeline.stage2_features import extract_features  # noqa: E402

    features_path = str(run_dir / "stage2_features.json")
    feature_bundle = extract_features(
        audio_path=input_audio,
        word_timestamps=word_timestamps,
        device=device,
        output_path=features_path,
    )

    # ========================================================================
    # Stage 3 — Phonetic rewriting
    # ========================================================================
    logger.info("\n── Stage 3: Phonetic Rewriting ──")
    from pipeline.stage3_phonetic_rewrite import rewrite_phonetics  # noqa: E402

    phonetic_path = str(run_dir / "stage3_phonetic.json")
    phonetic_result = rewrite_phonetics(
        transcript=transcript_result.transcript,
        accent_pair=accent_pair,
        output_path=phonetic_path,
    )

    # ========================================================================
    # Stage 4 — Voice synthesis
    # ========================================================================
    logger.info("\n── Stage 4: Voice Synthesis ──")
    from pipeline.stage4_synthesize import synthesize  # noqa: E402

    synth_path = str(run_dir / "stage4_synthesized.wav")
    synthesize(
        transcript=phonetic_result.target_transcript,
        source_audio_path=input_audio,
        accent_pair=accent_pair,
        word_durations=feature_bundle.prosody.word_durations_s,
        output_path=synth_path,
        device=device,
    )

    # ========================================================================
    # Stage 5 — Prosody transplant
    # ========================================================================
    logger.info("\n── Stage 5: Prosody Transplant ──")
    from pipeline.stage5_prosody_transplant import transplant_prosody  # noqa: E402

    if output_path is None:
        output_path = str(run_dir / "output.wav")

    transplant_prosody(
        synth_audio_path=synth_path,
        source_audio_path=input_audio,
        source_f0=feature_bundle.prosody.f0_hz,
        source_energy_db=feature_bundle.prosody.energy_db,
        frame_shift_ms=feature_bundle.prosody.frame_shift_ms,
        output_path=output_path,
        match_duration=match_duration,
    )

    # ========================================================================
    # Wrap up
    # ========================================================================
    elapsed = time.time() - t_start

    result = PipelineResult(
        run_id=run_id,
        run_dir=str(run_dir),
        input_audio=str(input_audio),
        output_audio=str(output_path),
        accent_pair=accent_pair,
        transcript=transcript_result.transcript,
        emotion_label=feature_bundle.emotion.label,
        emotion_score=feature_bundle.emotion.score,
        rewrites_count=len(phonetic_result.rewrite_map),
        duration_s=round(elapsed, 2),
    )

    result.save(str(run_dir / "pipeline_result.json"))

    logger.info(f"\n{'='*60}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"  Input:   {input_audio}")
    logger.info(f"  Output:  {output_path}")
    logger.info(f"  Emotion: {result.emotion_label} ({result.emotion_score:.2f})")
    logger.info(f"  Accent:  {accent_pair}")
    logger.info(f"  Rewrites: {result.rewrites_count} words")
    logger.info(f"  Run dir: {run_dir}")
    logger.info(f"{'='*60}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="voice-bridge: Emotion-preserving accent conversion"
    )
    parser.add_argument("input_audio", help="Input audio file (WAV/MP3/FLAC)")
    parser.add_argument(
        "--accent",
        default="indian_american",
        choices=["indian_american", "indian_british"],
        help="Target accent pair (default: indian_american)",
    )
    parser.add_argument("--output", default=None, help="Output WAV path")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model size")
    parser.add_argument("--no-vad", action="store_true", help="Skip VAD pre-filter")
    parser.add_argument("--no-duration", action="store_true", help="Skip duration matching")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    parser.add_argument("--run-id", default=None, help="Custom run ID (for debugging)")
    args = parser.parse_args()

    result = run_pipeline(
        input_audio=args.input_audio,
        accent_pair=args.accent,
        output_path=args.output,
        whisper_model=args.whisper_model,
        apply_vad=not args.no_vad,
        match_duration=not args.no_duration,
        device=args.device,
        run_id=args.run_id,
        verbose=True,
    )

    print(f"\n✓ Output saved to: {result.output_audio}")
