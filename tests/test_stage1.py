"""Tests for Stage 1: Transcription"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf


def _make_dummy_wav(duration_s: float = 2.0, sr: int = 16000) -> str:
    """Create a silent WAV file for testing and return its path."""
    audio = np.zeros(int(sr * duration_s), dtype=np.float32)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return tmp.name


class TestTranscriptResult(unittest.TestCase):

    def test_to_dict(self):
        from pipeline.stage1_transcribe import TranscriptResult, WordTimestamp, Segment

        result = TranscriptResult(
            transcript="Hello world",
            words=[WordTimestamp("Hello", 0.0, 0.4), WordTimestamp("world", 0.5, 0.9)],
            language="en",
            segments=[Segment("Hello world", 0.0, 0.9, 0.95)],
        )
        d = result.to_dict()
        self.assertEqual(d["transcript"], "Hello world")
        self.assertEqual(len(d["words"]), 2)
        self.assertEqual(d["words"][0]["word"], "Hello")

    def test_save_and_load(self):
        from pipeline.stage1_transcribe import TranscriptResult, WordTimestamp, Segment

        result = TranscriptResult(
            transcript="Test sentence",
            words=[WordTimestamp("Test", 0.0, 0.3), WordTimestamp("sentence", 0.4, 0.9)],
            language="en",
            segments=[Segment("Test sentence", 0.0, 0.9, 0.9)],
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            result.save(path)
            loaded = TranscriptResult.load(path)
            self.assertEqual(loaded.transcript, result.transcript)
            self.assertEqual(len(loaded.words), 2)
            self.assertEqual(loaded.language, "en")
        finally:
            os.unlink(path)


class TestWERComputation(unittest.TestCase):
    """Test the WER metric (lives in evaluation but tests transcription accuracy)."""

    def test_perfect_match(self):
        from evaluation.metrics import compute_wer
        self.assertEqual(compute_wer("hello world", "hello world"), 0.0)

    def test_one_substitution(self):
        from evaluation.metrics import compute_wer
        wer = compute_wer("hello world", "hello there")
        self.assertAlmostEqual(wer, 0.5, places=2)

    def test_empty_reference(self):
        from evaluation.metrics import compute_wer
        # Should not crash
        wer = compute_wer("", "hello")
        self.assertGreaterEqual(wer, 0.0)

    def test_complete_mismatch(self):
        from evaluation.metrics import compute_wer
        wer = compute_wer("one two three", "four five six")
        self.assertEqual(wer, 1.0)


class TestVADFallback(unittest.TestCase):

    def test_vad_fallback_returns_audio(self):
        """If silero-VAD is unavailable, should return original audio unchanged."""
        from pipeline.stage1_transcribe import _apply_vad
        audio = np.ones(16000, dtype=np.float32)
        result = _apply_vad(audio, sr=16000)
        # Result should be a numpy array of similar length
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
