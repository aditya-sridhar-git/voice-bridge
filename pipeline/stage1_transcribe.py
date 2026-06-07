"""
Stage 1 — Transcription
========================
Uses faster-whisper (large-v3) with silero-VAD pre-filtering to produce
a word-level timestamped transcript from raw input audio.

Output contract:
{
  "transcript": str,
  "words": [{"word": str, "start": float, "end": float}, ...],
  "language": str,
  "segments": [{"text": str, "start": float, "end": float, "confidence": float}, ...]
}
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float


@dataclass
class Segment:
    text: str
    start: float
    end: float
    confidence: float


@dataclass
class TranscriptResult:
    transcript: str
    words: List[WordTimestamp]
    language: str
    segments: List[Segment]

    def to_dict(self):
        return {
            "transcript": self.transcript,
            "words": [asdict(w) for w in self.words],
            "language": self.language,
            "segments": [asdict(s) for s in self.segments],
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TranscriptResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            transcript=data["transcript"],
            words=[WordTimestamp(**w) for w in data["words"]],
            language=data["language"],
            segments=[Segment(**s) for s in data["segments"]],
        )


# ---------------------------------------------------------------------------
# VAD pre-filtering
# ---------------------------------------------------------------------------

def _apply_vad(audio: np.ndarray, sr: int, threshold: float = 0.5) -> np.ndarray:
    """
    Strip leading/trailing silence using silero-VAD.
    Falls back to raw audio if silero-VAD is not installed.
    """
    try:
        import torch
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        (get_speech_timestamps, _, read_audio, *_) = utils

        audio_tensor = torch.from_numpy(audio).float()
        if sr != 16000:
            import torchaudio
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
            sr = 16000

        timestamps = get_speech_timestamps(
            audio_tensor, vad_model, threshold=threshold, sampling_rate=sr
        )

        if not timestamps:
            logger.warning("VAD found no speech — returning full audio.")
            return audio

        start = timestamps[0]["start"]
        end = timestamps[-1]["end"]
        return audio[start:end]

    except Exception as e:
        logger.warning(f"silero-VAD unavailable ({e}), skipping VAD.")
        return audio


# ---------------------------------------------------------------------------
# Core transcription
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: str,
    model_size: str = "large-v3",
    language: str = "en",
    apply_vad: bool = True,
    device: str = "auto",
    output_path: Optional[str] = None,
) -> TranscriptResult:
    """
    Transcribe audio with word-level timestamps using faster-whisper.

    Args:
        audio_path:   Path to input audio file (WAV/MP3/FLAC).
        model_size:   Whisper model variant. Options: tiny, base, small,
                      medium, large-v2, large-v3.
        language:     Force language (default: "en" to avoid mis-detection
                      on accented English).
        apply_vad:    If True, strip silence with silero-VAD before
                      transcription.
        device:       "auto", "cuda", or "cpu".
        output_path:  If provided, save TranscriptResult JSON here.

    Returns:
        TranscriptResult with transcript, word timestamps, and segments.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper not installed. Run: pip install faster-whisper"
        )

    # ---- resolve device ----
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    compute_type = "float16" if device == "cuda" else "int8"

    logger.info(f"Loading Whisper model '{model_size}' on {device} ({compute_type})")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # ---- load + optionally VAD ----
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # stereo → mono

    if apply_vad:
        logger.info("Applying silero-VAD pre-filter …")
        audio = _apply_vad(audio, sr)

    # Write VAD-filtered audio to a temp file for faster-whisper
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio, sr)

    # ---- transcribe ----
    logger.info("Running Whisper transcription …")
    segments_gen, info = model.transcribe(
        tmp_path,
        language=language,
        word_timestamps=True,
        vad_filter=False,  # we already did VAD above
        initial_prompt="The following is clear English speech.",
    )

    os.unlink(tmp_path)

    # ---- collect results ----
    all_words: List[WordTimestamp] = []
    all_segments: List[Segment] = []
    full_transcript_parts = []

    for seg in segments_gen:
        all_segments.append(
            Segment(
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
                confidence=float(np.exp(seg.avg_logprob)),
            )
        )
        full_transcript_parts.append(seg.text.strip())

        if seg.words:
            for w in seg.words:
                all_words.append(
                    WordTimestamp(
                        word=w.word.strip(),
                        start=w.start,
                        end=w.end,
                    )
                )

    result = TranscriptResult(
        transcript=" ".join(full_transcript_parts),
        words=all_words,
        language=info.language,
        segments=all_segments,
    )

    if output_path:
        result.save(output_path)
        logger.info(f"Transcript saved to {output_path}")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 1: Whisper transcription")
    parser.add_argument("audio", help="Input audio file")
    parser.add_argument("--model", default="large-v3", help="Whisper model size")
    parser.add_argument("--language", default="en", help="Force language code")
    parser.add_argument("--no-vad", action="store_true", help="Skip VAD pre-filter")
    parser.add_argument("--output", default=None, help="Save JSON output to path")
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    result = transcribe(
        audio_path=args.audio,
        model_size=args.model,
        language=args.language,
        apply_vad=not args.no_vad,
        device=args.device,
        output_path=args.output,
    )

    print(f"\nTranscript: {result.transcript}")
    print(f"Language detected: {result.language}")
    print(f"Word count: {len(result.words)}")
    print(f"Segments: {len(result.segments)}")
