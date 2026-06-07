"""
Phoneme substitution rules for accent conversion.
Used as a fallback when a word is not found in the word-level lexicon.

Format:
    RULES = {
        "<accent_pair>": {
            "<source_phoneme>": "<target_phoneme>",
            ...
        }
    }

IPA symbols used follow the gruut/espeak-ng convention.
"""

RULES = {

    # -----------------------------------------------------------------------
    # Indian English → American English
    # Key phonological differences:
    #   1. Trap-bath split: Indian /ɑː/ → American /æ/ in TRAP words
    #   2. Flapping: intervocalic /t/, /d/ → tap /ɾ/ (handled via lexicon)
    #   3. Rhoticity: Indian non-rhotic /ə/ → rhotic /ɚ/ in unstressed coda
    #   4. LOT vowel: Indian /ɒ/ → American /ɑː/
    #   5. Strut vowel: generally preserved
    # -----------------------------------------------------------------------
    "indian_american": {
        # LOT–PALM merger: British/Indian /ɒ/ → American /ɑː/
        "ɒ":  "ɑː",
        # Non-rhotic schwa → rhotic (rough approximation)
        "ər": "ɚ",
        # Trap-bath: long /ɑː/ in BATH words → /æ/ (only safe in known words)
        # NOTE: broad rule; lexicon is more precise
        # Dental stops (some Indian speakers use retroflex): approximate
        "ʈ":  "t",
        "ɖ":  "d",
        "ɳ":  "n",
        "ɽ":  "ɾ",
    },

    # -----------------------------------------------------------------------
    # Indian English → British (RP) English
    # Key differences from Indian:
    #   1. Non-rhoticity preserved (RP is also non-rhotic)
    #   2. TRAP-BATH split: RP keeps /ɑː/ in BATH words (same as Indian)
    #   3. LOT vowel: /ɒ/ preserved in RP
    #   4. STRUT: /ʌ/ used in RP (similar to Indian)
    #   5. Dental stops: same fix as above
    # -----------------------------------------------------------------------
    "indian_british": {
        # Dental/retroflex → standard alveolar
        "ʈ":  "t",
        "ɖ":  "d",
        "ɳ":  "n",
        "ɽ":  "ɾ",
        # Some Indian speakers front the GOOSE vowel /uː/ slightly
        # RP uses a more back /uː/ — no simple substitution needed
    },
}
