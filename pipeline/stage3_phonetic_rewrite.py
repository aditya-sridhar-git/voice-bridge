"""
Stage 3 — Phonetic Rewriting
==============================
Rewrites the transcript's pronunciation from source accent to target accent
using a two-level strategy:

  Level 1: Word-level lexicon lookup  (indian_to_american.json)
  Level 2: Phoneme-level substitution rules (phoneme_rules.py) — fallback

Output contract:
{
  "original_phonemes": {str: [str]},     # word → source IPA phonemes
  "target_phonemes":   {str: [str]},     # word → target IPA phonemes
  "rewrite_map": {str: {"from": str, "to": str}},
  "target_transcript": str,              # plain text (same words, accent-neutral)
  "target_transcript_ipa": str           # full IPA string for target accent
}
"""

import json
import logging
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

LEXICONS_DIR = Path(__file__).parent.parent / "lexicons"

SUPPORTED_ACCENT_PAIRS = {
    "indian_american": "indian_to_american.json",
    "indian_british":  "indian_to_british.json",
}


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class PhoneticResult:
    original_phonemes: Dict[str, List[str]]
    target_phonemes:   Dict[str, List[str]]
    rewrite_map:       Dict[str, Dict[str, str]]
    target_transcript: str
    target_transcript_ipa: str

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "PhoneticResult":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# G2P (Grapheme-to-Phoneme) conversion
# ---------------------------------------------------------------------------

def _g2p_word(word: str, lang: str = "en-us") -> List[str]:
    """
    Convert a single word to IPA phonemes using gruut.
    Falls back to eng_to_ipa if gruut is unavailable.
    """
    clean = re.sub(r"[^\w']", "", word.lower())
    if not clean:
        return []

    # --- Try gruut first ---
    try:
        from gruut import sentences as gruut_sentences
        for sent in gruut_sentences(clean, lang=lang):
            for token in sent:
                if token.phonemes:
                    return list(token.phonemes)
        return []
    except Exception:
        pass

    # --- Fallback: eng_to_ipa ---
    try:
        import eng_to_ipa as ipa
        result = ipa.convert(clean)
        # eng_to_ipa returns the word unchanged if unknown
        if result != clean and result:
            return list(result)
        return list(clean)
    except Exception:
        return list(clean)


def _g2p_transcript(transcript: str, lang: str = "en-us") -> Dict[str, List[str]]:
    """Return {word: [phonemes]} for every word in the transcript."""
    words = transcript.split()
    return {word: _g2p_word(word, lang=lang) for word in words}


# ---------------------------------------------------------------------------
# Lexicon loading
# ---------------------------------------------------------------------------

def _load_lexicon(accent_pair: str) -> Dict[str, Dict[str, str]]:
    """
    Load the pronunciation diff lexicon for the given accent pair.
    Returns {word_lower: {"from_ipa": str, "to_ipa": str}}.
    """
    filename = SUPPORTED_ACCENT_PAIRS.get(accent_pair)
    if not filename:
        raise ValueError(
            f"Unknown accent pair '{accent_pair}'. "
            f"Supported: {list(SUPPORTED_ACCENT_PAIRS)}"
        )

    lexicon_path = LEXICONS_DIR / filename
    if not lexicon_path.exists():
        logger.warning(f"Lexicon not found at {lexicon_path}. Returning empty lexicon.")
        return {}

    with open(lexicon_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Phoneme-level rule-based substitution (fallback)
# ---------------------------------------------------------------------------

def _apply_phoneme_rules(phonemes: List[str], rules: Dict) -> List[str]:
    """
    Apply sequential phoneme substitution rules.
    Rules format: {source_phoneme: target_phoneme}
    """
    return [rules.get(p, p) for p in phonemes]


def _load_phoneme_rules(accent_pair: str) -> Dict[str, str]:
    """Load phoneme substitution rules from lexicons/phoneme_rules.py."""
    try:
        import importlib.util
        rules_path = LEXICONS_DIR / "phoneme_rules.py"
        spec = importlib.util.spec_from_file_location("phoneme_rules", rules_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rules_map = getattr(module, "RULES", {})
        return rules_map.get(accent_pair, {})
    except Exception as e:
        logger.warning(f"Could not load phoneme rules ({e}). No rule fallback.")
        return {}


# ---------------------------------------------------------------------------
# Core rewrite logic
# ---------------------------------------------------------------------------

def rewrite_phonetics(
    transcript: str,
    accent_pair: str = "indian_american",
    output_path: Optional[str] = None,
) -> PhoneticResult:
    """
    Rewrite the phonetics of a transcript for a target accent.

    Args:
        transcript:   Plain-text transcript (from Stage 1).
        accent_pair:  One of "indian_american" | "indian_british".
        output_path:  If provided, save PhoneticResult JSON here.

    Returns:
        PhoneticResult with original/target phonemes and IPA strings.
    """
    logger.info(f"[Stage 3] Phonetic rewriting for accent pair: {accent_pair}")

    lexicon = _load_lexicon(accent_pair)
    phoneme_rules = _load_phoneme_rules(accent_pair)

    # Source G2P
    source_lang = "en-us"  # assume Indian English maps phonetically to en-us baseline
    original_phonemes = _g2p_transcript(transcript, lang=source_lang)

    target_phonemes: Dict[str, List[str]] = {}
    rewrite_map: Dict[str, Dict[str, str]] = {}

    words = transcript.split()
    for word in words:
        word_lower = word.lower().rstrip(".,!?;:")
        src_phones = original_phonemes.get(word, [])

        if word_lower in lexicon:
            # Level 1: lexicon lookup
            entry = lexicon[word_lower]
            tgt_phones = list(entry.get("to_ipa", ""))
            rewrite_map[word] = {
                "from": entry.get("from_ipa", "".join(src_phones)),
                "to":   entry.get("to_ipa",   "".join(tgt_phones)),
                "source": "lexicon",
            }
        else:
            # Level 2: rule-based fallback
            tgt_phones = _apply_phoneme_rules(src_phones, phoneme_rules)
            if tgt_phones != src_phones:
                rewrite_map[word] = {
                    "from": "".join(src_phones),
                    "to":   "".join(tgt_phones),
                    "source": "rules",
                }

        target_phonemes[word] = tgt_phones

    # Build full IPA string
    target_ipa_parts = []
    for word in words:
        phones = target_phonemes.get(word, [])
        target_ipa_parts.append("".join(phones))
    target_ipa = " ".join(target_ipa_parts)

    result = PhoneticResult(
        original_phonemes=original_phonemes,
        target_phonemes=target_phonemes,
        rewrite_map=rewrite_map,
        target_transcript=transcript,         # plain text unchanged
        target_transcript_ipa=target_ipa,
    )

    if output_path:
        result.save(output_path)
        logger.info(f"  PhoneticResult saved to {output_path}")

    rewritten_count = len(rewrite_map)
    logger.info(
        f"  {rewritten_count}/{len(words)} words rewritten "
        f"({rewritten_count/max(len(words),1)*100:.1f}%)"
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3: Phonetic rewriting")
    parser.add_argument("transcript", help="Transcript string or path to Stage 1 JSON")
    parser.add_argument(
        "--accent-pair",
        default="indian_american",
        choices=list(SUPPORTED_ACCENT_PAIRS),
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    text = args.transcript
    if text.endswith(".json") and Path(text).exists():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from stage1_transcribe import TranscriptResult
        text = TranscriptResult.load(text).transcript

    result = rewrite_phonetics(text, accent_pair=args.accent_pair, output_path=args.output)

    print(f"\nOriginal:  {text}")
    print(f"Target IPA: {result.target_transcript_ipa}")
    print(f"\nRewrites ({len(result.rewrite_map)}):")
    for word, change in result.rewrite_map.items():
        print(f"  {word}: /{change['from']}/ → /{change['to']}/ [{change['source']}]")
