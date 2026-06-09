# Emotion-Preserving Accent Conversion — Implementation Plan

## Overview

Convert a speaker's accent to a target accent while preserving their **vocal identity**, **emotional expressiveness**, and **prosodic contours** (pitch, energy, rhythm). The system is designed as a modular offline pipeline with clear data contracts between each stage.

---

## Target Accents (Initial Scope)

- Indian English → American English
- Indian English → British (RP) English
- *(Extensible to other accent pairs via new phoneme lexicons)*

---

## Full Pipeline Architecture

```
Input Audio (WAV/MP3, 16kHz mono)
        │
        ▼
┌─────────────────────────────────────┐
│  Stage 1: Transcription             │
│  Whisper (large-v3)                 │
│  → Transcript + word timestamps     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Stage 2: Feature Extraction        │
│  SpeechBrain + pyannote.audio       │
│  → Emotion embedding (vector)       │
│  → Speaker embedding (x-vector/     │
│    ECAPA-TDNN)                      │
│  → Prosody contours                 │
│    (F0, energy, duration per word)  │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Stage 3: Phonetic Rewriting        │
│  G2P + Accent Lexicon Substitution  │
│  → Accent-rewritten phoneme string  │
│  → Aligned phoneme durations        │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Stage 4: Voice Synthesis           │
│  OpenVoice v2                       │
│  → Synthesized audio in target      │
│    accent with cloned speaker voice │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│  Stage 5: Prosody Transplant        │
│  PSOLA / World Vocoder              │
│  → Warp F0 contour from source      │
│  → Match energy envelope            │
│  → Adjust speaking rate             │
└──────────────────┬──────────────────┘
                   │
                   ▼
Output Audio (WAV, 22kHz)
(Target Accent + Preserved Emotion + Same Speaker Identity)
```

---

## Stage-by-Stage Implementation

---

### Stage 1 — Transcription (Whisper)

**Goal**: Extract transcript with word-level timestamps.

**Model**: `openai/whisper-large-v3`

**Library**: `faster-whisper` (CTranslate2 backend, 4× faster than original)

**Output contract**:
```json
{
  "transcript": "Hello, how are you doing today?",
  "words": [
    { "word": "Hello",  "start": 0.00, "end": 0.42 },
    { "word": "how",    "start": 0.50, "end": 0.68 },
    ...
  ],
  "language": "en"
}
```

**Implementation notes**:
- Force language to `en` to avoid mis-detection on accented speech
- Use `word_timestamps=True` — needed for prosody alignment downstream
- Segment-level confidence scores: flag low-confidence segments for review
- VAD pre-filter using `silero-vad` to strip silence before passing to Whisper

**Key file**: `pipeline/stage1_transcribe.py`

---

### Stage 2 — Feature Extraction (SpeechBrain + pyannote)

**Goal**: Extract three independent feature streams from the raw input audio.

#### 2a. Emotion Embedding

**Model**: SpeechBrain `emotion-recognition-wav2vec2-IEMOCAP`

**Output**: 
- Discrete label: `{neutral, happy, sad, angry, surprised}`
- Continuous embedding vector: `float32[256]` — used for re-injection later

```python
from speechbrain.pretrained import EmotionRecognition
classifier = EmotionRecognition.from_hparams("speechbrain/emotion-recognition-wav2vec2-IEMOCAP")
label, score, embedding = classifier.classify_file(audio_path)
```

**Note**: Run on the **full utterance**, not per-word, to get a global emotion state. For longer clips (>10s), use a sliding window and majority-vote the label.

#### 2b. Speaker Embedding

**Model**: SpeechBrain `spkrec-ecapa-voxceleb` (ECAPA-TDNN)

**Output**: `float32[192]` x-vector — passed directly to OpenVoice v2 as the speaker identity anchor.

#### 2c. Prosody Contour Extraction

**Library**: `parselmouth` (Praat Python bindings)

Extract per-frame and word-aligned:
- **F0 (pitch)**: 75–600 Hz range, frame shift 10ms
- **Energy (RMS)**: per-frame energy envelope
- **Duration**: word-level timestamps → phoneme-level via MFA alignment

```python
import parselmouth
sound = parselmouth.Sound(audio_path)
pitch = sound.to_pitch()          # F0 contour
intensity = sound.to_intensity()  # Energy envelope
```

**Output contract**:
```json
{
  "emotion": {
    "label": "happy",
    "embedding": [0.12, -0.34, ...]
  },
  "speaker_embedding": [0.05, 0.91, ...],
  "prosody": {
    "f0_hz": [180.2, 183.5, ...],
    "energy_db": [-12.1, -11.8, ...],
    "frame_shift_ms": 10,
    "word_durations_s": { "Hello": 0.42, "how": 0.18, ... }
  }
}
```

**Key file**: `pipeline/stage2_features.py`

---

### Stage 3 — Phonetic Rewriting

**Goal**: Rewrite the transcript's pronunciation to match the target accent's phoneme patterns.

**Strategy**: Two-level substitution — word-level lexicon lookup, then phoneme-level rule application as fallback.

#### 3a. G2P Conversion

**Library**: `gruut` (multilingual G2P, IPA output)

```python
from gruut import sentences
for sent in sentences("Hello how are you", lang="en-us"):
    for word in sent:
        print(word.phonemes)  # ['h', 'ə', 'l', 'oʊ']
```

#### 3b. Accent Lexicon

Maintain a **pronunciation difference dictionary** per accent pair:

| Word | Indian English IPA | American English IPA |
|---|---|---|
| water | /ˈwɔːtər/ | /ˈwɑːɾər/ |
| butter | /ˈbʌtər/ | /ˈbʌɾər/ |
| schedule | /ˈʃɛdjuːl/ | /ˈskɛdʒuːl/ |
| privacy | /ˈprɪvəsi/ | /ˈpraɪvəsi/ |

**Source**: CMU Pronouncing Dictionary + manual curated diff list

#### 3c. Phoneme Substitution Rules (Fallback)

For words not in the lexicon, apply rule-based phoneme transforms:

```python
INDIAN_TO_AMERICAN_RULES = {
    # Flapping: /t/ and /d/ between vowels → tap /ɾ/
    ("t", "V_V"): "ɾ",
    # Rhoticity: preserve /r/ in coda position (Indian is non-rhotic in some variants)
    ("ə", "coda"): "ɚ",
    # Trap-bath split
    ("aː", "*"): "æ",
}
```

**Output contract**:
```json
{
  "original_phonemes": ["h","ɛ","l","oʊ"],
  "target_phonemes":   ["h","ɛ","l","oʊ"],
  "rewrite_map": {
    "water": { "from": "wɔːtər", "to": "wɑːɾər" }
  },
  "target_transcript_ipa": "həloʊ haʊ ɑːr juː"
}
```

**Key file**: `pipeline/stage3_phonetic_rewrite.py`

---

### Stage 4 — Voice Synthesis (OpenVoice v2)

**Goal**: Synthesize speech in the target accent using the cloned speaker voice.

**Model**: `myshell-ai/OpenVoice` (v2, MIT license)

**How OpenVoice v2 works**:
- Base TTS synthesizes speech from text using a **base speaker** (American/British native)
- **Tone Color Converter** then transfers the *timbre* from source speaker to the synthesized audio
- The speaker embedding (x-vector from Stage 2b) drives this transfer

```python
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

# Load converter
converter = ToneColorConverter(ckpt_converter, device="cuda")

# Extract target speaker style from original audio
target_se, _ = se_extractor.get_se(source_audio, converter, vad=True)

# Synthesize with base TTS (MeloTTS, American accent)
tts_model.tts_to_file(
    text=rewritten_transcript,
    speaker_ids={"EN-US": 0},
    output_path=tts_output_path,
    speed=1.0
)

# Apply tone color conversion
converter.convert(
    audio_src_path=tts_output_path,
    src_se=base_speaker_se,
    tgt_se=target_se,
    output_path=converted_path,
)
```

**Key decisions**:
- Use `MeloTTS` as the base TTS within OpenVoice v2 — it supports multi-accent natively (`EN-US`, `EN-BR`, `EN-AU`, `EN-IN`)
- Pass the **rewritten IPA/text** from Stage 3, NOT the original transcript
- `speed` parameter should be set to match original speaking rate from prosody extraction

**Key file**: `pipeline/stage4_synthesize.py`

---

### Stage 5 — Prosody Transplant

**Goal**: Warp the synthesized audio's pitch, energy, and timing to match the original speaker's emotional prosody.

This is the **core emotion preservation mechanism** — the step that connects emotion extraction to the final output.

#### 5a. F0 Contour Transplant

Use **PSOLA (Pitch-Synchronous Overlap-Add)** via `pyworld` or `praat-parselmouth`:

```python
import pyworld as pw
import numpy as np

# Extract from synthesized audio
f0_synth, sp_synth, ap_synth = pw.wav2world(synth_audio, sr)

# Build scaled F0: preserve shape but warp to source contour
# Normalize both to [0,1], then re-scale to synthesized mean/std
f0_source_norm = (f0_source - f0_source.mean()) / f0_source.std()
f0_transplanted = f0_source_norm * f0_synth.std() + f0_synth.mean()

# Resynthesize
output = pw.synthesize(f0_transplanted, sp_synth, ap_synth, sr)
```

**Normalization strategy**: Don't copy F0 verbatim (different phonemes have different natural pitch). Instead:
- Match **relative shape** (rises, falls, contour slope)
- Preserve **absolute mean F0** of synthesized version (respects target phoneme naturalness)

#### 5b. Energy Envelope Transplant

```python
from scipy.signal import resample

# Resample source energy contour to match synthesized length
energy_resampled = resample(energy_source, len(synth_audio))

# Apply gain scaling
gain = energy_resampled / (energy_synth + 1e-8)
output_audio = output * np.clip(gain, 0.5, 2.0)  # clip to prevent artifacts
```

#### 5c. Duration / Rate Adjustment

After synthesizing, if the output duration differs significantly from the original:

```python
# Time-stretch using phase vocoder (librosa)
import librosa
rate = len(source_audio) / len(synth_audio)
output_stretched = librosa.effects.time_stretch(output_audio, rate=rate)
```

Apply this **after** F0 transplant to avoid pitch drift from stretching.

**Output contract**:
- WAV file, 22050 Hz, mono
- Duration ≈ original ±5%
- F0 contour correlation with source ≥ 0.75 (validated by evaluation module)

**Key file**: `pipeline/stage5_prosody_transplant.py`

---

## Project File Structure

```
voice-bridge/
├── pipeline/
│   ├── __init__.py
│   ├── stage1_transcribe.py          # Whisper + VAD
│   ├── stage2_features.py            # SpeechBrain + pyannote + parselmouth
│   ├── stage3_phonetic_rewrite.py    # G2P + accent lexicons
│   ├── stage4_synthesize.py          # OpenVoice v2 TTS + tone color conversion
│   ├── stage5_prosody_transplant.py  # PSOLA / pyworld F0+energy warp
│   └── run_pipeline.py               # Orchestrator: runs all stages end-to-end
├── lexicons/
│   ├── indian_to_american.json       # Word-level pronunciation diff dict
│   ├── indian_to_british.json
│   └── phoneme_rules.py              # Fallback phoneme substitution rules
├── models/
│   │                                 # Downloaded model checkpoints (gitignored)
│   └── .gitkeep
├── evaluation/
│   ├── metrics.py                    # MOS proxy, F0 correlation, WER, SER
│   └── eval_pipeline.py             # Run evaluation on test set
├── tests/
│   ├── test_stage1.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   └── test_integration.py
├── notebooks/
│   └── explore_prosody.ipynb         # Prosody analysis scratch pad
├── requirements.txt
├── setup.py
└── README.md
```

---

## Dependencies

```txt
# requirements.txt

# Stage 1
faster-whisper>=1.0.0
silero-vad

# Stage 2
speechbrain>=1.0.0
pyannote.audio>=3.1.0
praat-parselmouth>=0.4.3

# Stage 3
gruut>=2.3.0

# Stage 4
# OpenVoice v2 — install from GitHub
# git clone https://github.com/myshell-ai/OpenVoice
# pip install -e OpenVoice/
melo-tts

# Stage 5
pyworld>=0.3.4
librosa>=0.10.0
scipy>=1.11.0
numpy>=1.24.0

# Utilities
torch>=2.1.0
torchaudio>=2.1.0
soundfile>=0.12.0
pydub>=0.25.0
```

---

## Data Contracts Between Stages

| Stage | Output Type | Consumed By |
|---|---|---|
| Stage 1 | `TranscriptResult` (JSON) | Stage 2, Stage 3 |
| Stage 2 | `FeatureBundle` (JSON + arrays) | Stage 4, Stage 5 |
| Stage 3 | `PhoneticResult` (JSON) | Stage 4 |
| Stage 4 | WAV file (synthesized) | Stage 5 |
| Stage 5 | WAV file (final output) | User / downstream |

All intermediate outputs are saved to disk in a run directory (`runs/<uuid>/`) for reproducibility and debugging.

---

## Evaluation Metrics

| Metric | What It Measures | Tool |
|---|---|---|
| **WER** (Word Error Rate) | Transcription fidelity after conversion | Whisper re-transcription |
| **SER** (Speaker Error Rate) | Speaker identity preservation | Cosine sim of x-vectors |
| **F0 Pearson Correlation** | Emotion/prosody preservation | scipy.stats |
| **Accent Accuracy** | % of target-accent phonemes correct | CMU dict lookup |
| **MOS Proxy** (DNSMOS) | Naturalness / audio quality | Microsoft DNSMOS API |
| **Emotion Accuracy** | Emotion label match: source vs output | SpeechBrain re-classify |

**Target thresholds**:
- WER < 5% vs original transcript
- F0 correlation ≥ 0.75
- Emotion label match ≥ 80% on test set
- Speaker cosine similarity ≥ 0.85

---

## Open Questions

> [!IMPORTANT]
> **Q1**: Should the pipeline support real-time / streaming operation, or is batch-only acceptable for v1?
> *Impacts*: Whisper model size choice, VAD strategy, whether Stage 5 can be applied online.

> [!IMPORTANT]
> **Q2**: What is the primary target accent pair for v1? (Indian → American assumed above)
> *Impacts*: Which MeloTTS speaker ID to use, which lexicon to build first.

> [!WARNING]
> **Q3**: OpenVoice v2's tone color converter may partially re-impose original accent characteristics during voice cloning. We need to empirically test whether the phonetic rewriting in Stage 3 survives the conversion step. If it doesn't, Stage 3 may need to move inside the TTS inference loop (i.e., phoneme-level control in MeloTTS directly).

> [!NOTE]
> **Q4**: For the emotion embedding re-injection — the current plan uses prosody transplant as the mechanism. Should we also condition the TTS speed/speaking rate on the detected emotion? (e.g., fast + high energy → excited, slow + low energy → sad)

> [!NOTE]
> **Q5**: Do we need multi-speaker support (diarization for conversations), or is single-speaker per audio file the assumed input for v1?

---

## Implementation Phases

### Phase 1 — Skeleton + Stage 1 & 2 (Week 1)
- [ ] Set up repo structure and `requirements.txt`
- [ ] Implement `stage1_transcribe.py` with Whisper + VAD
- [ ] Implement `stage2_features.py` with emotion, speaker, prosody extraction
- [ ] Define and validate data contracts (JSON schemas)
- [ ] Write unit tests for Stage 1 & 2

### Phase 2 — Phonetic Rewriting (Week 2)
- [ ] Build `indian_to_american.json` lexicon (top 2000 divergent words)
- [ ] Implement `stage3_phonetic_rewrite.py` with G2P + lexicon + rule fallback
- [ ] Validate rewrite quality on 50-word sample manually

### Phase 3 — Synthesis (Week 2–3)
- [ ] Set up OpenVoice v2 environment + MeloTTS
- [ ] Implement `stage4_synthesize.py`
- [ ] Test tone color conversion on 5 sample speakers
- [ ] Validate that phoneme rewrites survive tone color conversion

### Phase 4 — Prosody Transplant (Week 3)
- [ ] Implement F0 contour normalization + PSOLA warp in `stage5_prosody_transplant.py`
- [ ] Implement energy envelope transplant
- [ ] Implement duration rate-matching
- [ ] Tune clipping thresholds to avoid artifacts

### Phase 5 — Evaluation + Polish (Week 4)
- [ ] Implement `evaluation/metrics.py`
- [ ] Run pipeline on 20-utterance test set (5 speakers × 4 sentences)
- [ ] Tune prosody transplant parameters to meet F0 correlation threshold
- [ ] Write `README.md` with usage instructions

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Phoneme rewrites not preserved through OpenVoice | Medium | Test early; fall back to phoneme-level TTS control |
| F0 transplant causing unnatural artifacts | Medium | Clip gain, smooth transitions, perceptual evaluation |
| Emotion classifier unreliable on accented speech | Low–Medium | Fine-tune SpeechBrain on accented emotion datasets |
| OpenVoice v2 speaker similarity degradation | Low | Tune `tau` parameter in tone color conversion |
| Whisper hallucinations on heavily accented input | Low | Use `initial_prompt` with domain context |
