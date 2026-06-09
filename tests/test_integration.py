"""
Integration Tests
==================
End-to-end smoke tests for the full pipeline.
These use mocked heavy models (Whisper, SpeechBrain, OpenVoice) so they
run fast without GPU or downloaded checkpoints.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf


SR = 22050


def _make_dummy_wav(duration_s: float = 3.0, sr: int = SR) -> str:
    """Create a sine-wave WAV file for testing."""
    t = np.linspace(0, duration_s, int(sr * duration_s))
    audio = (np.sin(2 * np.pi * 220 * t) * 0.1).astype(np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return tmp.name


class TestProsodyTransplant(unittest.TestCase):
    """Stage 5 can be tested without any ML models."""

    def setUp(self):
        self.wav = _make_dummy_wav()
        n_frames = 300
        self.f0 = [200.0 if i % 10 < 6 else 0.0 for i in range(n_frames)]
        self.energy = [-12.0] * n_frames

    def tearDown(self):
        os.unlink(self.wav)

    def test_transplant_returns_wav(self):
        from pipeline.stage5_prosody_transplant import transplant_prosody
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            result = transplant_prosody(
                synth_audio_path=self.wav,
                source_audio_path=self.wav,
                source_f0=self.f0,
                source_energy_db=self.energy,
                output_path=out,
                match_duration=False,
            )
            self.assertTrue(Path(result).exists())
            audio, sr = sf.read(result)
            self.assertGreater(len(audio), 0)
        finally:
            if os.path.exists(out):
                os.unlink(out)

    def test_output_normalized(self):
        """Output audio should not clip (peak < 1.0)."""
        from pipeline.stage5_prosody_transplant import transplant_prosody
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        try:
            transplant_prosody(
                synth_audio_path=self.wav,
                source_audio_path=self.wav,
                source_f0=self.f0,
                source_energy_db=self.energy,
                output_path=out,
                match_duration=False,
            )
            audio, _ = sf.read(out, dtype="float32")
            self.assertLessEqual(np.abs(audio).max(), 1.0)
        finally:
            if os.path.exists(out):
                os.unlink(out)


class TestEvaluationThresholds(unittest.TestCase):
    """Test EvaluationResult threshold logic."""

    def test_passes_all(self):
        from evaluation.metrics import EvaluationResult
        r = EvaluationResult(
            wer=0.02,
            speaker_cosine=0.92,
            f0_correlation=0.80,
            emotion_match=True,
            dnsmos=4.2,
        )
        self.assertTrue(r.passes_thresholds())

    def test_fails_wer(self):
        from evaluation.metrics import EvaluationResult
        r = EvaluationResult(
            wer=0.10,
            speaker_cosine=0.92,
            f0_correlation=0.80,
            emotion_match=True,
            dnsmos=None,
        )
        self.assertFalse(r.passes_thresholds())

    def test_fails_f0(self):
        from evaluation.metrics import EvaluationResult
        r = EvaluationResult(
            wer=0.02,
            speaker_cosine=0.92,
            f0_correlation=0.50,
            emotion_match=True,
            dnsmos=None,
        )
        self.assertFalse(r.passes_thresholds())

    def test_none_metrics_ignored(self):
        from evaluation.metrics import EvaluationResult
        r = EvaluationResult(
            wer=None,
            speaker_cosine=None,
            f0_correlation=None,
            emotion_match=None,
            dnsmos=None,
        )
        # All None — no thresholds to check, should not fail
        self.assertTrue(r.passes_thresholds())


class TestWERMetric(unittest.TestCase):

    def test_wer_symmetry(self):
        from evaluation.metrics import compute_wer
        wer_ab = compute_wer("hello world foo", "hello bar foo")
        wer_ba = compute_wer("hello bar foo", "hello world foo")
        self.assertAlmostEqual(wer_ab, wer_ba, places=5)

    def test_wer_insertion(self):
        from evaluation.metrics import compute_wer
        wer = compute_wer("hello world", "hello beautiful world")
        self.assertGreater(wer, 0.0)


if __name__ == "__main__":
    unittest.main()
