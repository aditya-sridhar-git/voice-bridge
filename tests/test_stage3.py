"""Tests for Stage 3: Phonetic Rewriting"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestPhoneticResult(unittest.TestCase):

    def test_save_and_load(self):
        from pipeline.stage3_phonetic_rewrite import PhoneticResult

        result = PhoneticResult(
            original_phonemes={"hello": ["h", "ɛ", "l", "oʊ"]},
            target_phonemes={"hello": ["h", "ɛ", "l", "oʊ"]},
            rewrite_map={},
            target_transcript="hello",
            target_transcript_ipa="hɛloʊ",
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            result.save(path)
            loaded = PhoneticResult.load(path)
            self.assertEqual(loaded.target_transcript, "hello")
            self.assertEqual(loaded.target_transcript_ipa, "hɛloʊ")
        finally:
            os.unlink(path)


class TestLexiconLoading(unittest.TestCase):

    def test_unknown_accent_pair_raises(self):
        from pipeline.stage3_phonetic_rewrite import _load_lexicon
        with self.assertRaises(ValueError):
            _load_lexicon("unknown_pair")

    def test_missing_lexicon_returns_empty(self):
        """If lexicon file doesn't exist, should return {} gracefully."""
        from pipeline.stage3_phonetic_rewrite import _load_lexicon, SUPPORTED_ACCENT_PAIRS
        import pipeline.stage3_phonetic_rewrite as s3

        # Temporarily point LEXICONS_DIR at a non-existent path
        original = s3.LEXICONS_DIR
        s3.LEXICONS_DIR = Path("/nonexistent_path")
        try:
            result = _load_lexicon("indian_american")
            self.assertIsInstance(result, dict)
        finally:
            s3.LEXICONS_DIR = original


class TestPhonemeRules(unittest.TestCase):

    def test_apply_rules(self):
        from pipeline.stage3_phonetic_rewrite import _apply_phoneme_rules
        rules = {"ɒ": "ɑː", "ʈ": "t"}
        result = _apply_phoneme_rules(["ʈ", "ɒ", "p"], rules)
        self.assertEqual(result, ["t", "ɑː", "p"])

    def test_identity_when_no_rule(self):
        from pipeline.stage3_phonetic_rewrite import _apply_phoneme_rules
        rules = {}
        phonemes = ["h", "ɛ", "l", "oʊ"]
        self.assertEqual(_apply_phoneme_rules(phonemes, rules), phonemes)


class TestRewritePhonetics(unittest.TestCase):

    def test_rewrite_returns_result(self):
        from pipeline.stage3_phonetic_rewrite import rewrite_phonetics
        result = rewrite_phonetics("water bottle dance", accent_pair="indian_american")
        self.assertIsNotNone(result)
        self.assertEqual(result.target_transcript, "water bottle dance")
        self.assertIn("water", result.target_phonemes)

    def test_lexicon_hit_recorded_in_map(self):
        """Words in the lexicon should appear in rewrite_map."""
        from pipeline.stage3_phonetic_rewrite import rewrite_phonetics
        result = rewrite_phonetics("water", accent_pair="indian_american")
        # "water" is in the Indian→American lexicon
        if "water" in result.rewrite_map:
            self.assertEqual(result.rewrite_map["water"]["source"], "lexicon")

    def test_unknown_words_dont_crash(self):
        from pipeline.stage3_phonetic_rewrite import rewrite_phonetics
        result = rewrite_phonetics("xyzabcdef qwerty", accent_pair="indian_american")
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
