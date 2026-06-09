"""
Stage 2 — Feature Extraction
==============================
Extracts three parallel feature streams from the raw input audio:

  2a. Emotion embedding  — SpeechBrain wav2vec2 IEMOCAP classifier
  2b. Speaker embedding  — SpeechBrain ECAPA-TDNN x-vector
  2c. Prosody contours   — parselmouth (Praat): F0, energy, duration

Output contract:
{
  "emotion": {
    "label": str,            # neutral | happy | sad | angry | surprised
    "score": float,          # confidence 0-1
    "embedding": [float]     # float32[256]
  },
  "speaker_embedding": [float],   # float32[192] ECAPA x-vector
  "prosody": {
    "f0_hz": [float],        # per-frame F0 (0.0 = unvoiced)
    "energy_db": [float],    # per-frame RMS energy in dB
    "frame_shift_ms": int,   # always 10
    "word_durations_s": {str: float}  # word → duration in seconds
  }
}
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "surprised"]


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class EmotionFeatures:
    label: str
    score: float
    embedding: List[float]


@dataclass
class ProsodyFeatures:
    f0_hz: List[float]
    energy_db: List[float]
    frame_shift_ms: int = 10
    word_durations_s: Dict[str, float] = field(default_factory=dict)


@dataclass
class FeatureBundle:
    emotion: EmotionFeatures
    speaker_embedding: List[float]
    prosody: ProsodyFeatures

    def to_dict(self):
        return {
            "emotion": asdict(self.emotion),
            "speaker_embedding": self.speaker_embedding,
            "prosody": asdict(self.prosody),
        }

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FeatureBundle":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            emotion=EmotionFeatures(**d["emotion"]),
            speaker_embedding=d["speaker_embedding"],
            prosody=ProsodyFeatures(**d["prosody"]),
        )


# ---------------------------------------------------------------------------
# 2a. Emotion extraction
# ---------------------------------------------------------------------------

def _extract_emotion(audio_path: str, device: str = "cpu", transcript: str = "") -> EmotionFeatures:
    """
    Detect emotion using a transformers text classifier on the transcript.
    Falls back to neutral if model unavailable or transcript is empty.

    Model: j-hartmann/emotion-english-distilroberta-base
    Labels: anger, disgust, fear, joy, neutral, sadness, surprise
    """
    if not transcript or not transcript.strip():
        logger.warning("No transcript for emotion detection. Using neutral fallback.")
        return EmotionFeatures(label="neutral", score=1.0, embedding=[0.0] * 256)

    try:
        from transformers import pipeline as hf_pipeline

        logger.info("Loading emotion classifier (j-hartmann/emotion-english-distilroberta-base) …")
        classifier = hf_pipeline(
            task="text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1,
            device=0 if device == "cuda" else -1,
        )

        # Truncate transcript to model max length (512 tokens ~ 400 words)
        text = " ".join(transcript.split()[:400])
        result = classifier(text)

        # result is [[{label, score}]] with top_k=1
        top = result[0][0] if isinstance(result[0], list) else result[0]
        label = top["label"].lower()
        score = float(top["score"])

        # Map to our internal label set
        label_map = {
            "joy": "happy",
            "surprise": "surprised",
            "anger": "angry",
            "sadness": "sad",
            "disgust": "angry",   # merge into angry
            "fear": "sad",        # merge into sad
            "neutral": "neutral",
        }
        label = label_map.get(label, label)

        logger.info(f"  Emotion detected from transcript: {label} ({score:.3f})")
        return EmotionFeatures(
            label=label,
            score=score,
            embedding=[0.0] * 256,  # text-based model has no audio embedding
        )

    except Exception as e:
        logger.warning(f"Emotion extraction failed ({e}). Using neutral fallback.")
        return EmotionFeatures(label="neutral", score=1.0, embedding=[0.0] * 256)


# ---------------------------------------------------------------------------
# 2b. Speaker embedding (ECAPA-TDNN)
# ---------------------------------------------------------------------------

def _extract_speaker_embedding(audio_path: str, device: str = "cpu") -> List[float]:
    """
    Extract ECAPA-TDNN x-vector from SpeechBrain.
    Falls back to zeros if SpeechBrain is unavailable.
    """
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier as SpeakerEncoder

        logger.info("Loading SpeechBrain speaker encoder (ECAPA-TDNN) …")
        encoder = SpeakerEncoder.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/speechbrain_speaker",
            run_opts={"device": device},
        )
        embedding = encoder.encode_file(audio_path)
        return embedding.squeeze().cpu().numpy().tolist()

    except Exception as e:
        logger.warning(f"Speaker embedding failed ({e}). Using zero fallback.")
        return [0.0] * 192


# ---------------------------------------------------------------------------
# 2c. Prosody extraction
# ---------------------------------------------------------------------------

def _extract_prosody(
    audio_path: str,
    word_timestamps: Optional[List[dict]] = None,
    frame_shift_ms: int = 10,
) -> ProsodyFeatures:
    """
    Extract F0 and energy contours using parselmouth (Praat).

    Args:
        audio_path:       Path to input WAV.
        word_timestamps:  Word-level timestamps from Stage 1 output
                          (list of {"word": str, "start": float, "end": float}).
        frame_shift_ms:   Analysis frame hop in milliseconds.

    Returns:
        ProsodyFeatures with f0_hz, energy_db, and word_durations_s.
    """
    try:
        import parselmouth
        from parselmouth.praat import call

        logger.info("Extracting prosody with parselmouth …")
        sound = parselmouth.Sound(audio_path)
        duration = sound.duration

        # ---- F0 (pitch) ----
        pitch = sound.to_pitch(
            time_step=frame_shift_ms / 1000.0,
            pitch_floor=75.0,
            pitch_ceiling=600.0,
        )
        n_frames = pitch.n_frames
        f0_values = []
        for i in range(1, n_frames + 1):
            v = pitch.get_value_in_frame(i)
            f0_values.append(float(v) if v == v else 0.0)  # NaN → 0.0

        # ---- Energy (RMS intensity) ----
        intensity = sound.to_intensity(
            minimum_pitch=75.0,
            time_step=frame_shift_ms / 1000.0,
        )
        energy_values = []
        n_int_frames = intensity.n_frames
        for i in range(1, n_int_frames + 1):
            v = intensity.get_value(intensity.frame_number_to_time(i))
            energy_values.append(float(v) if v == v else -80.0)

        # ---- Word durations ----
        word_durations: Dict[str, float] = {}
        if word_timestamps:
            for w in word_timestamps:
                dur = w["end"] - w["start"]
                key = w["word"]
                # handle repeated words by appending index
                if key in word_durations:
                    idx = sum(1 for k in word_durations if k.startswith(key))
                    key = f"{key}_{idx}"
                word_durations[key] = round(dur, 4)

        return ProsodyFeatures(
            f0_hz=f0_values,
            energy_db=energy_values,
            frame_shift_ms=frame_shift_ms,
            word_durations_s=word_durations,
        )

    except Exception as e:
        logger.warning(f"Prosody extraction failed ({e}). Returning empty contours.")
        audio, sr = sf.read(audio_path, dtype="float32")
        n_frames = int(len(audio) / sr / (frame_shift_ms / 1000.0))
        # Even without parselmouth, we can still compute word durations from timestamps
        fallback_word_durations: Dict[str, float] = {}
        if word_timestamps:
            for w in word_timestamps:
                dur = w["end"] - w["start"]
                key = w["word"]
                if key in fallback_word_durations:
                    idx = sum(1 for k in fallback_word_durations if k.startswith(key))
                    key = f"{key}_{idx}"
                fallback_word_durations[key] = round(dur, 4)
        return ProsodyFeatures(
            f0_hz=[0.0] * n_frames,
            energy_db=[-80.0] * n_frames,
            frame_shift_ms=frame_shift_ms,
            word_durations_s=fallback_word_durations,
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_features(
    audio_path: str,
    word_timestamps: Optional[List[dict]] = None,
    device: str = "auto",
    output_path: Optional[str] = None,
    transcript: str = "",
) -> FeatureBundle:
    """
    Run all three feature extraction sub-stages on the input audio.

    Args:
        audio_path:       Path to input audio file.
        word_timestamps:  Word-level timestamps from Stage 1 (for duration map).
        device:           "auto", "cuda", or "cpu".
        output_path:      If provided, save FeatureBundle JSON here.
        transcript:       Plain-text transcript from Stage 1 (used for emotion).

    Returns:
        FeatureBundle with emotion, speaker embedding, and prosody.
    """
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    logger.info(f"[Stage 2] Feature extraction on device={device}")

    emotion = _extract_emotion(audio_path, device=device, transcript=transcript)
    logger.info(f"  Emotion: {emotion.label} (score={emotion.score:.3f})")

    speaker_emb = _extract_speaker_embedding(audio_path, device=device)
    logger.info(f"  Speaker embedding: dim={len(speaker_emb)}")

    prosody = _extract_prosody(audio_path, word_timestamps=word_timestamps)
    logger.info(
        f"  Prosody: {len(prosody.f0_hz)} F0 frames, "
        f"{len(prosody.word_durations_s)} word durations"
    )

    bundle = FeatureBundle(
        emotion=emotion,
        speaker_embedding=speaker_emb,
        prosody=prosody,
    )

    if output_path:
        bundle.save(output_path)
        logger.info(f"  FeatureBundle saved to {output_path}")

    return bundle


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 2: Feature extraction")
    parser.add_argument("audio", help="Input audio file")
    parser.add_argument("--timestamps", default=None, help="Stage 1 JSON output path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default=None, help="Save FeatureBundle JSON here")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    words = None
    if args.timestamps:
        from stage1_transcribe import TranscriptResult
        tr = TranscriptResult.load(args.timestamps)
        words = [{"word": w.word, "start": w.start, "end": w.end} for w in tr.words]

    bundle = extract_features(
        audio_path=args.audio,
        word_timestamps=words,
        device=args.device,
        output_path=args.output,
    )

    print(f"\nEmotion: {bundle.emotion.label} ({bundle.emotion.score:.3f})")
    print(f"Speaker embedding dim: {len(bundle.speaker_embedding)}")
    print(f"F0 frames: {len(bundle.prosody.f0_hz)}")
    print(f"Word durations: {bundle.prosody.word_durations_s}")
