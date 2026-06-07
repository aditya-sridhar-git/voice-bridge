"""
voice-bridge pipeline package
Emotion-preserving accent conversion pipeline.
"""

from .stage1_transcribe import transcribe
from .stage2_features import extract_features
from .stage3_phonetic_rewrite import rewrite_phonetics
from .stage4_synthesize import synthesize
from .stage5_prosody_transplant import transplant_prosody

__all__ = [
    "transcribe",
    "extract_features",
    "rewrite_phonetics",
    "synthesize",
    "transplant_prosody",
]
