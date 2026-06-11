"""
Stage 3 — Phonetic Rewriting (LLM-powered)
============================================
Rewrites the transcript from Indian English to the target accent using
GPT-4o mini for context-aware, vocabulary-level rewriting.

Strategy (priority order):
  1. GPT-4o mini  — context-aware, handles proper nouns, idioms, vocab gaps
                     (requires OPENAI_API_KEY in env / .env file)
  2. Static lexicon fallback  — ~50-word JSON pronunciation diff dict
  3. Phoneme-rule fallback     — broad IPA substitution rules

Output contract:
{
  "original_phonemes":   {str: [str]},
  "target_phonemes":     {str: [str]},
  "rewrite_map":         {str: {"from": str, "to": str, "source": str}},
  "target_transcript":   str,   ← THIS is what Stage 4 / MeloTTS receives
  "target_transcript_ipa": str
}
"""

import json
import logging
import os
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

LEXICONS_DIR = Path(__file__).parent.parent / "lexicons"

SUPPORTED_ACCENT_PAIRS = {
    "indian_american": "indian_to_american.json",
    "indian_british":  "indian_to_british.json",
}

_TARGET_ACCENT_NAME = {
    "indian_american": "American English",
    "indian_british":  "British English",
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
# LLM rewriting (GPT-4o mini)
# ---------------------------------------------------------------------------

def _load_env():
    """Load .env file if present (so OPENAI_API_KEY is available)."""
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        load_dotenv(dotenv_path=env_path, override=False)
    except ImportError:
        pass  # python-dotenv not installed; rely on shell env


def _llm_available() -> bool:
    """Return True if an OpenAI API key is configured."""
    _load_env()
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _llm_rewrite(transcript: str, accent_pair: str, model: str = "gpt-4o-mini") -> str:
    """
    Use GPT-4o mini to rewrite the transcript for the target accent.

    The LLM:
    - Replaces Indian English vocabulary with target-accent equivalents
      (e.g. 'prepone' → 'move earlier', 'revert' → 'get back to you')
    - Applies stress/pronunciation-aware respelling for words whose
      standard spelling will be mispronounced by American TTS
    - Leaves unchanged any word that is standard English with the same
      pronunciation in both accents

    Returns the rewritten transcript as a plain string.
    """
    _load_env()
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    client = OpenAI(api_key=api_key)
    target = _TARGET_ACCENT_NAME.get(accent_pair, "American English")

    system_prompt = f"""You are a computational linguist specialising in accent conversion from Indian English to {target}.

Your job is to rewrite a speech transcript so that when it is read aloud by a {target} text-to-speech system, the output sounds natural in {target}.

Instructions:
1. Replace Indian English vocabulary and idioms with {target} equivalents.
   Examples: "prepone" → "move to an earlier time", "revert back to me" → "get back to me",
             "do the needful" → "do what is needed", "out of station" → "out of town".
2. For words whose spelling will cause a {target} TTS to mispronounce them, use a more
   phonetically transparent respelling or a synonym the TTS will handle correctly.
3. Preserve the full meaning and natural sentence structure.
4. Do NOT add any explanation, commentary, or formatting — return ONLY the rewritten transcript.
5. If the transcript is already natural {target}, return it unchanged."""

    user_prompt = f"Rewrite the following Indian English transcript for {target} TTS:\n\n{transcript}"

    logger.info(f"[Stage 3] Calling GPT-4o mini for accent rewriting ({len(transcript.split())} words)...")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=max(256, len(transcript.split()) * 4),
    )

    rewritten = response.choices[0].message.content.strip()
    logger.info(f"[Stage 3] LLM rewrite complete.")
    return rewritten


def _build_rewrite_map(original: str, rewritten: str) -> Dict[str, Dict[str, str]]:
    """
    Build a word-level diff map between original and rewritten transcripts.
    Words that changed are recorded; unchanged words are skipped.
    """
    orig_words = original.split()
    new_words  = rewritten.split()
    rewrite_map = {}

    # Simple aligned diff for same-length output
    if len(orig_words) == len(new_words):
        for o, n in zip(orig_words, new_words):
            if o.lower().rstrip(".,!?;:") != n.lower().rstrip(".,!?;:"):
                rewrite_map[o] = {"from": o, "to": n, "source": "llm"}
    else:
        # Different lengths — record at sentence level
        rewrite_map["<transcript>"] = {
            "from": original,
            "to": rewritten,
            "source": "llm",
        }

    return rewrite_map


# ---------------------------------------------------------------------------
# G2P (Grapheme-to-Phoneme) — used for IPA metadata only
# ---------------------------------------------------------------------------

def _g2p_word(word: str, lang: str = "en-us") -> List[str]:
    clean = re.sub(r"[^\w']", "", word.lower())
    if not clean:
        return []

    try:
        from gruut import sentences as gruut_sentences
        for sent in gruut_sentences(clean, lang=lang):
            for token in sent:
                if token.phonemes:
                    return list(token.phonemes)
        return []
    except Exception:
        pass

    try:
        import eng_to_ipa as ipa
        result = ipa.convert(clean)
        if result != clean and result:
            return list(result)
        return list(clean)
    except Exception:
        return list(clean)


def _g2p_transcript(transcript: str, lang: str = "en-us") -> Dict[str, List[str]]:
    words = transcript.split()
    return {word: _g2p_word(word, lang=lang) for word in words}


# ---------------------------------------------------------------------------
# Lexicon fallback (existing static approach)
# ---------------------------------------------------------------------------

def _load_lexicon(accent_pair: str) -> Dict[str, Dict[str, str]]:
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


def _load_phoneme_rules(accent_pair: str) -> Dict[str, str]:
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


def _apply_phoneme_rules(phonemes: List[str], rules: Dict) -> List[str]:
    return [rules.get(p, p) for p in phonemes]


def _lexicon_rewrite(transcript: str, accent_pair: str):
    """Original lexicon + phoneme-rule approach (fallback when no API key)."""
    lexicon       = _load_lexicon(accent_pair)
    phoneme_rules = _load_phoneme_rules(accent_pair)
    original_phonemes = _g2p_transcript(transcript)

    target_phonemes: Dict[str, List[str]] = {}
    rewrite_map: Dict[str, Dict[str, str]] = {}
    words = transcript.split()

    for word in words:
        word_lower  = word.lower().rstrip(".,!?;:")
        src_phones  = original_phonemes.get(word, [])

        if word_lower in lexicon:
            entry      = lexicon[word_lower]
            tgt_phones = list(entry.get("to_ipa", ""))
            rewrite_map[word] = {
                "from":   entry.get("from_ipa", "".join(src_phones)),
                "to":     entry.get("to_ipa",   "".join(tgt_phones)),
                "source": "lexicon",
            }
        else:
            tgt_phones = _apply_phoneme_rules(src_phones, phoneme_rules)
            if tgt_phones != src_phones:
                rewrite_map[word] = {
                    "from":   "".join(src_phones),
                    "to":     "".join(tgt_phones),
                    "source": "rules",
                }

        target_phonemes[word] = tgt_phones

    target_ipa = " ".join("".join(target_phonemes.get(w, [])) for w in words)
    return transcript, original_phonemes, target_phonemes, rewrite_map, target_ipa


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rewrite_phonetics(
    transcript: str,
    accent_pair: str = "indian_american",
    output_path: Optional[str] = None,
    use_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
) -> PhoneticResult:
    """
    Rewrite transcript phonetics / vocabulary for the target accent.

    Priority:
      1. GPT-4o mini  (if use_llm=True and OPENAI_API_KEY is set)
      2. Static lexicon + phoneme rules  (always available fallback)

    Args:
        transcript:   Plain-text transcript from Stage 1.
        accent_pair:  "indian_american" or "indian_british".
        output_path:  Optional path to save PhoneticResult JSON.
        use_llm:      Set False to force lexicon-only mode.
        llm_model:    OpenAI model name (default: gpt-4o-mini).

    Returns:
        PhoneticResult
    """
    logger.info(f"[Stage 3] Phonetic rewriting | accent={accent_pair} | llm={use_llm and _llm_available()}")

    if use_llm and _llm_available():
        try:
            rewritten = _llm_rewrite(transcript, accent_pair, model=llm_model)
            original_phonemes = _g2p_transcript(transcript)
            target_phonemes   = _g2p_transcript(rewritten)
            rewrite_map       = _build_rewrite_map(transcript, rewritten)
            target_ipa        = " ".join(
                "".join(target_phonemes.get(w, [])) for w in rewritten.split()
            )
            source = "llm"
        except Exception as e:
            logger.warning(f"[Stage 3] LLM rewrite failed ({e}). Falling back to lexicon.")
            rewritten, original_phonemes, target_phonemes, rewrite_map, target_ipa = \
                _lexicon_rewrite(transcript, accent_pair)
            source = "lexicon_fallback"
    else:
        if use_llm:
            logger.warning("[Stage 3] OPENAI_API_KEY not set — using lexicon fallback.")
        rewritten, original_phonemes, target_phonemes, rewrite_map, target_ipa = \
            _lexicon_rewrite(transcript, accent_pair)
        source = "lexicon"

    result = PhoneticResult(
        original_phonemes=original_phonemes,
        target_phonemes=target_phonemes,
        rewrite_map=rewrite_map,
        target_transcript=rewritten,
        target_transcript_ipa=target_ipa,
    )

    if output_path:
        result.save(output_path)
        logger.info(f"  PhoneticResult saved to {output_path}")

    n_words    = len(transcript.split())
    n_rewrites = len(rewrite_map)
    logger.info(
        f"  [{source}] {n_rewrites}/{n_words} words rewritten "
        f"({n_rewrites / max(n_words, 1) * 100:.1f}%)"
    )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3: Phonetic rewriting (LLM-powered)")
    parser.add_argument("transcript", help="Transcript string or path to Stage 1 JSON")
    parser.add_argument("--accent-pair", default="indian_american", choices=list(SUPPORTED_ACCENT_PAIRS))
    parser.add_argument("--output",      default=None)
    parser.add_argument("--no-llm",      action="store_true", help="Force lexicon-only mode")
    parser.add_argument("--model",       default="gpt-4o-mini", help="OpenAI model name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    text = args.transcript
    if text.endswith(".json") and Path(text).exists():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from stage1_transcribe import TranscriptResult
        text = TranscriptResult.load(text).transcript

    result = rewrite_phonetics(
        text,
        accent_pair=args.accent_pair,
        output_path=args.output,
        use_llm=not args.no_llm,
        llm_model=args.model,
    )

    print(f"\nOriginal:  {text}")
    print(f"Rewritten: {result.target_transcript}")
    print(f"\nRewrites ({len(result.rewrite_map)}):")
    for word, change in result.rewrite_map.items():
        print(f"  {word!r}: {change['from']!r} -> {change['to']!r}  [{change['source']}]")
