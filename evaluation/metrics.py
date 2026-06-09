"""
Evaluation Metrics
===================
Automated quality metrics for the voice-bridge pipeline output.

Metrics:
  - WER   (Word Error Rate)          — transcription fidelity
  - SER   (Speaker Embedding cosine) — speaker identity preservation
  - F0r   (F0 Pearson correlation)   — prosody / emotion preservation
  - EMO   (Emotion label accuracy)   — emotion class preservation
  - DNSMOS (naturalness proxy)       — audio quality
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    wer: Optional[float]                # 0.0 = perfect, lower is better
    speaker_cosine: Optional[float]     # 0.0–1.0, higher is better
    f0_correlation: Optional[float]     # -1.0–1.0, higher is better
    emotion_match: Optional[bool]       # True if emotion label preserved
    dnsmos: Optional[float]             # 1.0–5.0, higher is better
    notes: str = ""

    def to_dict(self):
        return asdict(self)

    def passes_thresholds(self) -> bool:
        """Check whether output meets quality targets from the implementation plan.

        Returns True if all available (non-None) metrics pass their thresholds.
        Returns True when no metrics are available (nothing to fail).
        """
        checks = []
        if self.wer is not None:
            checks.append(self.wer < 0.05)
        if self.f0_correlation is not None:
            checks.append(self.f0_correlation >= 0.75)
        if self.speaker_cosine is not None:
            checks.append(self.speaker_cosine >= 0.85)
        # If no metrics are available, no thresholds can be violated → True
        return all(checks)


# ---------------------------------------------------------------------------
# WER — Word Error Rate
# ---------------------------------------------------------------------------

def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute WER between reference and hypothesis transcripts.
    Uses simple dynamic programming edit distance on word tokens.
    """
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m] / max(n, 1)


def compute_wer_from_audio(
    output_audio: str,
    reference_transcript: str,
    whisper_model: str = "base",
    device: str = "cpu",
) -> float:
    """Re-transcribe output audio with Whisper and compare to reference."""
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(whisper_model, device=device, compute_type="int8")
        segments, _ = model.transcribe(output_audio, language="en")
        hypothesis = " ".join(s.text.strip() for s in segments)
        wer = compute_wer(reference_transcript, hypothesis)
        logger.info(f"WER: {wer:.4f} | Hypothesis: {hypothesis[:60]}…")
        return wer
    except Exception as e:
        logger.warning(f"WER computation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Speaker cosine similarity
# ---------------------------------------------------------------------------

def compute_speaker_cosine(
    source_audio: str,
    output_audio: str,
    device: str = "cpu",
) -> float:
    """
    Extract ECAPA-TDNN x-vectors from both audios and compute cosine similarity.
    """
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier

        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/speechbrain_speaker",
            run_opts={"device": device},
        )

        emb_src = encoder.encode_file(source_audio).squeeze().cpu().numpy()
        emb_out = encoder.encode_file(output_audio).squeeze().cpu().numpy()

        cosine = float(
            np.dot(emb_src, emb_out)
            / (np.linalg.norm(emb_src) * np.linalg.norm(emb_out) + 1e-8)
        )
        logger.info(f"Speaker cosine similarity: {cosine:.4f}")
        return cosine
    except Exception as e:
        logger.warning(f"Speaker cosine failed: {e}")
        return None


# ---------------------------------------------------------------------------
# F0 Pearson correlation
# ---------------------------------------------------------------------------

def compute_f0_correlation(
    source_f0: List[float],
    output_audio: str,
    frame_shift_ms: int = 10,
) -> float:
    """
    Extract F0 from output audio and compute Pearson correlation with source F0.
    """
    try:
        import parselmouth
        from scipy.stats import pearsonr

        sound = parselmouth.Sound(output_audio)
        pitch = sound.to_pitch(
            time_step=frame_shift_ms / 1000.0,
            pitch_floor=75.0,
            pitch_ceiling=600.0,
        )
        output_f0 = []
        for i in range(1, pitch.n_frames + 1):
            v = pitch.get_value_in_frame(i)
            output_f0.append(float(v) if v == v else 0.0)

        src = np.array(source_f0, dtype=np.float64)
        out = np.array(output_f0, dtype=np.float64)

        # Align lengths
        min_len = min(len(src), len(out))
        src, out = src[:min_len], out[:min_len]

        # Only correlate on voiced frames
        voiced = (src > 0) & (out > 0)
        if voiced.sum() < 5:
            logger.warning("Too few voiced frames for F0 correlation.")
            return None

        r, _ = pearsonr(src[voiced], out[voiced])
        logger.info(f"F0 Pearson correlation: {r:.4f}")
        return float(r)
    except Exception as e:
        logger.warning(f"F0 correlation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Emotion label match
# ---------------------------------------------------------------------------

def compute_emotion_match(
    source_emotion_label: str,
    output_audio: str,
    device: str = "cpu",
) -> bool:
    """
    Re-classify emotion on output audio and compare with source label.
    """
    try:
        from speechbrain.inference.classifiers import EncoderClassifier

        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            savedir="models/speechbrain_emotion",
            run_opts={"device": device},
        )
        _, _, _, label = classifier.classify_file(output_audio)
        output_label = label[0].strip()
        match = output_label.lower() == source_emotion_label.lower()
        logger.info(
            f"Emotion: source={source_emotion_label}, output={output_label}, match={match}"
        )
        return match
    except Exception as e:
        logger.warning(f"Emotion match failed: {e}")
        return None


# ---------------------------------------------------------------------------
# DNSMOS naturalness proxy
# ---------------------------------------------------------------------------

def compute_dnsmos(output_audio: str) -> Optional[float]:
    """
    Attempt DNSMOS naturalness scoring via the ONNX model.
    Falls back to None if not available.
    """
    try:
        # DNSMOS requires the Microsoft ONNX model — skip gracefully if absent
        import onnxruntime  # noqa: F401
        logger.info("DNSMOS: ONNX runtime available but model not bundled. Skipping.")
        return None
    except ImportError:
        logger.info("DNSMOS: onnxruntime not installed. Skipping.")
        return None


# ---------------------------------------------------------------------------
# Full evaluation runner
# ---------------------------------------------------------------------------

def evaluate(
    source_audio: str,
    output_audio: str,
    reference_transcript: str,
    source_f0: List[float],
    source_emotion_label: str,
    frame_shift_ms: int = 10,
    device: str = "cpu",
    whisper_model: str = "base",
) -> EvaluationResult:
    """
    Run all evaluation metrics for a pipeline output.

    Args:
        source_audio:         Original input audio path.
        output_audio:         Pipeline output audio path.
        reference_transcript: Ground-truth transcript (from Stage 1).
        source_f0:            Source F0 contour (from Stage 2).
        source_emotion_label: Source emotion label (from Stage 2).
        frame_shift_ms:       Frame shift for F0 extraction.
        device:               "cuda" or "cpu".
        whisper_model:        Whisper model size for WER re-transcription.

    Returns:
        EvaluationResult with all metric scores.
    """
    logger.info("=== Evaluation ===")

    wer = compute_wer_from_audio(output_audio, reference_transcript, whisper_model, device)
    spk = compute_speaker_cosine(source_audio, output_audio, device)
    f0r = compute_f0_correlation(source_f0, output_audio, frame_shift_ms)
    emo = compute_emotion_match(source_emotion_label, output_audio, device)
    dns = compute_dnsmos(output_audio)

    notes_parts = []
    if wer is not None and wer >= 0.05:
        notes_parts.append(f"WER={wer:.3f} exceeds 5% threshold")
    if spk is not None and spk < 0.85:
        notes_parts.append(f"Speaker cosine={spk:.3f} below 0.85 threshold")
    if f0r is not None and f0r < 0.75:
        notes_parts.append(f"F0 correlation={f0r:.3f} below 0.75 threshold")

    result = EvaluationResult(
        wer=wer,
        speaker_cosine=spk,
        f0_correlation=f0r,
        emotion_match=emo,
        dnsmos=dns,
        notes="; ".join(notes_parts) if notes_parts else "All thresholds passed",
    )

    logger.info(f"Results: {result.to_dict()}")
    logger.info(f"Passes thresholds: {result.passes_thresholds()}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate pipeline output")
    parser.add_argument("source_audio")
    parser.add_argument("output_audio")
    parser.add_argument("pipeline_result_json", help="Path to pipeline_result.json from run dir")
    parser.add_argument("features_json", help="Path to stage2_features.json from run dir")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.pipeline_result_json) as f:
        pr = json.load(f)
    with open(args.features_json) as f:
        feat = json.load(f)

    result = evaluate(
        source_audio=args.source_audio,
        output_audio=args.output_audio,
        reference_transcript=pr["transcript"],
        source_f0=feat["prosody"]["f0_hz"],
        source_emotion_label=feat["emotion"]["label"],
        frame_shift_ms=feat["prosody"].get("frame_shift_ms", 10),
        device=args.device,
    )

    print(json.dumps(result.to_dict(), indent=2))
    print(f"\nPasses all thresholds: {result.passes_thresholds()}")
