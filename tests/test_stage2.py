"""Tests for Stage 2: Feature Extraction"""

import json
import os
import tempfile
import unittest

import numpy as np
import soundfile as sf


def _make_dummy_wav(duration_s: float = 2.0, sr: int = 16000) -> str:
    audio = np.random.randn(int(sr * duration_s)).astype(np.float32) * 0.01
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sr)
    return tmp.name


class TestFeatureBundle(unittest.TestCase):

    def test_to_dict_and_load(self):
        from pipeline.stage2_features import (
            FeatureBundle, EmotionFeatures, ProsodyFeatures
        )

        bundle = FeatureBundle(
            emotion=EmotionFeatures(label="happy", score=0.85, embedding=[0.1] * 256),
            speaker_embedding=[0.2] * 192,
            prosody=ProsodyFeatures(
                f0_hz=[100.0, 120.0, 0.0],
                energy_db=[-10.0, -12.0, -15.0],
                frame_shift_ms=10,
                word_durations_s={"hello": 0.4, "world": 0.3},
            ),
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            bundle.save(path)
            loaded = FeatureBundle.load(path)
            self.assertEqual(loaded.emotion.label, "happy")
            self.assertAlmostEqual(loaded.emotion.score, 0.85)
            self.assertEqual(len(loaded.speaker_embedding), 192)
            self.assertEqual(len(loaded.prosody.f0_hz), 3)
            self.assertEqual(loaded.prosody.word_durations_s["hello"], 0.4)
        finally:
            os.unlink(path)


class TestProsodyExtraction(unittest.TestCase):

    def test_prosody_fallback(self):
        """Prosody extraction should return empty contours gracefully if parselmouth fails."""
        from pipeline.stage2_features import _extract_prosody
        wav_path = _make_dummy_wav()
        try:
            prosody = _extract_prosody(wav_path, word_timestamps=None)
            self.assertIsInstance(prosody.f0_hz, list)
            self.assertIsInstance(prosody.energy_db, list)
            self.assertEqual(prosody.frame_shift_ms, 10)
        finally:
            os.unlink(wav_path)

    def test_word_durations_populated(self):
        from pipeline.stage2_features import _extract_prosody
        wav_path = _make_dummy_wav()
        word_timestamps = [
            {"word": "Hello", "start": 0.0, "end": 0.4},
            {"word": "world", "start": 0.5, "end": 0.8},
        ]
        try:
            prosody = _extract_prosody(wav_path, word_timestamps=word_timestamps)
            # Word durations should be populated
            self.assertIn("Hello", prosody.word_durations_s)
            self.assertIn("world", prosody.word_durations_s)
            self.assertAlmostEqual(prosody.word_durations_s["Hello"], 0.4, places=3)
        finally:
            os.unlink(wav_path)


class TestEmotionFallback(unittest.TestCase):

    def test_emotion_fallback_returns_neutral(self):
        """When SpeechBrain is unavailable, should return neutral with zero embedding."""
        from pipeline.stage2_features import _extract_emotion
        wav_path = _make_dummy_wav()
        try:
            # Monkeypatch: force SpeechBrain import to fail
            import unittest.mock as mock
            with mock.patch.dict("sys.modules", {"speechbrain": None}):
                emotion = _extract_emotion(wav_path)
            self.assertEqual(emotion.label, "neutral")
            self.assertEqual(len(emotion.embedding), 256)
        finally:
            os.unlink(wav_path)


if __name__ == "__main__":
    unittest.main()
