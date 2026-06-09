"""
Phoneme substitution rules for accent conversion.
Used as a fallback when a word is not found in the word-level lexicon.

These rules operate on IPA phonemes produced by gruut (en-us) and cover
the systematic phonological differences between Indian English and
American/British English. Every word in the transcript gets G2P'd via
gruut and then these rules are applied — so 100% of words are processed,
not just the ~150 in the lexicon (the lexicon provides high-precision
overrides for ambiguous cases).

Format:
    RULES = {
        "<accent_pair>": {
            "<source_phoneme>": "<target_phoneme>",
            ...
        }
    }

IPA symbols follow the gruut/espeak-ng convention.
"""

RULES = {

    # -------------------------------------------------------------------------
    # Indian English → American English
    #
    # Key phonological differences (Indian → American):
    #
    # 1. TRAP-BATH split:
    #    Indian /ɑː/ in BATH-class words → American /æ/
    #    (handled by lexicon for known words; rules handle the rest)
    #
    # 2. Intervocalic flapping:
    #    /t/ between vowels → tap /ɾ/ (e.g. "water", "butter", "city")
    #    (Context-sensitive — lexicon handles known cases precisely)
    #
    # 3. Rhoticity:
    #    Indian English is weakly rhotic; American is strongly rhotic.
    #    /ɚ/ preserved, non-rhotic schwa + r → /ɚ/
    #
    # 4. LOT-PALM merger:
    #    Indian/British /ɒ/ → American /ɑː/
    #
    # 5. GOAT vowel:
    #    Indian /əʊ/ → American /oʊ/
    #
    # 6. Dental/retroflex consonants (regional Indian English):
    #    /ʈ/ → /t/, /ɖ/ → /d/, /ɳ/ → /n/, /ɽ/ → /ɾ/
    #
    # 7. STRUT vowel — generally preserved between dialects
    #
    # 8. Unstressed vowel reduction:
    #    Indian /ɪ/ in unstressed syllables → American /ə/
    #    (context-dependent; conservative rules applied)
    # -------------------------------------------------------------------------
    "indian_american": {

        # --- Vowel rules ---

        # LOT-PALM merger: /ɒ/ → /ɑː/  (British "lot" → American "lot/palm")
        "ɒ":   "ɑː",

        # GOAT vowel: /əʊ/ → /oʊ/  (Indian/British → American)
        "əʊ":  "oʊ",

        # THOUGHT vowel: /ɔː/ mostly preserved, but some environments
        # American reduces it — keep as is for safety
        # "ɔː": "ɑː",  # too aggressive — disabled

        # Schwa + r: non-rhotic coda → rhotic
        "ər":  "ɚ",
        "ɜr":  "ɝ",

        # Some Indian speakers use /eː/ for FACE vowel → American /eɪ/
        "eː":  "eɪ",

        # Indian /iː/ in unstressed positions sometimes → /ɪ/ in American
        # (too context-sensitive for blanket rule — leave as is)

        # PRICE vowel — mostly shared
        # "aɪ": "aɪ",  # no change

        # --- Consonant rules ---

        # Dental/retroflex stops → alveolar
        "ʈ":   "t",
        "ɖ":   "d",
        "ɳ":   "n",
        "ɽ":   "ɾ",

        # Indian dental fricatives (some speakers): /θ̪/ → /θ/, /ð̪/ → /ð/
        "θ̪":  "θ",
        "ð̪":  "ð",

        # Aspirated stops (some Indian speakers): drop aspiration marker
        "pʰ":  "p",
        "tʰ":  "t",
        "kʰ":  "k",
        "bʰ":  "b",
        "dʰ":  "d",
        "ɡʰ":  "ɡ",

        # Labiodental /ʋ/ (Indian) → American /v/
        "ʋ":   "v",

        # Tapped /ɾ/ — already American-style, preserve
        # "ɾ": "ɾ",

    },

    # -------------------------------------------------------------------------
    # Indian English → British (RP) English
    #
    # Key differences from Indian:
    #
    # 1. Non-rhoticity: RP is non-rhotic like Indian (easier conversion)
    # 2. LOT vowel: /ɒ/ preserved in RP (unlike American merger)
    # 3. TRAP-BATH split: RP keeps /ɑː/ in BATH words (same as Indian)
    # 4. GOAT vowel: RP /əʊ/ matches Indian English (same)
    # 5. STRUT: /ʌ/ used in RP (similar to Indian)
    # 6. Dental/retroflex: same fixes as American
    # -------------------------------------------------------------------------
    "indian_british": {

        # Dental/retroflex → alveolar (same as American fix)
        "ʈ":   "t",
        "ɖ":   "d",
        "ɳ":   "n",
        "ɽ":   "ɾ",

        # Labiodental /ʋ/ → /v/
        "ʋ":   "v",

        # Aspirated stops — drop aspiration
        "pʰ":  "p",
        "tʰ":  "t",
        "kʰ":  "k",

        # Some Indian regional accents use /ɛː/ for FACE vowel → RP /eɪ/
        "ɛː":  "eɪ",

        # Dental fricatives
        "θ̪":  "θ",
        "ð̪":  "ð",
    },
}
