# voice-bridge — Emotion-Preserving Accent Conversion

> **Convert a speaker's accent while preserving their vocal identity, emotion, and prosody.**
> Built on top of [OpenVoice v2](https://github.com/myshell-ai/OpenVoice) by MyShell AI.

---

## What This Project Does

`voice-bridge` is a 5-stage audio ML pipeline that takes speech in one accent (e.g. Indian English) and converts it to a target accent (e.g. American English) while keeping:

- **Who** is speaking (voice identity / timbre)
- **How** they feel (emotion — pitch contour, energy, rhythm)
- **What** they said (transcript fidelity)

---

## Pipeline Architecture

```
Input Audio (WAV/MP3)
        │
        ▼
┌─────────────────────────────────┐
│  Stage 1: Transcription         │  faster-whisper (large-v3)
│  → Transcript + word timestamps │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 2: Feature Extraction    │  SpeechBrain + parselmouth
│  → Emotion embedding            │
│  → Speaker x-vector (ECAPA)     │
│  → F0 / energy / duration       │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 3: Phonetic Rewriting    │  gruut G2P + accent lexicons
│  → Accent-rewritten phonemes    │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 4: Voice Synthesis       │  MeloTTS + OpenVoice v2
│  → TTS in target accent         │
│  → Tone color conversion        │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  Stage 5: Prosody Transplant    │  pyworld WORLD vocoder
│  → F0 contour warp              │
│  → Energy envelope match        │
│  → Duration rate-matching       │
└──────────────┬──────────────────┘
               │
               ▼
Output Audio (WAV, 22kHz)
(Target Accent + Same Emotion + Same Speaker)
```

---

## Project Structure

```
voice-bridge/
├── pipeline/
│   ├── stage1_transcribe.py        # Whisper + VAD
│   ├── stage2_features.py          # Emotion, speaker, prosody extraction
│   ├── stage3_phonetic_rewrite.py  # G2P + lexicon substitution
│   ├── stage4_synthesize.py        # MeloTTS + OpenVoice ToneColorConverter
│   ├── stage5_prosody_transplant.py# WORLD vocoder F0/energy/duration warp
│   └── run_pipeline.py             # End-to-end orchestrator
├── lexicons/
│   ├── indian_to_american.json     # 50-word pronunciation diff dict
│   ├── indian_to_british.json      # British RP variant
│   └── phoneme_rules.py            # Phoneme-level substitution rules
├── evaluation/
│   ├── metrics.py                  # WER, speaker cosine, F0 corr, emotion match
│   └── eval_pipeline.py            # Batch evaluation runner
├── tests/
│   ├── test_stage1.py
│   ├── test_stage2.py
│   ├── test_stage3.py
│   └── test_integration.py
├── openvoice/                      # OpenVoice v2 source (from MyShell AI)
├── docs/
│   ├── IMPLEMENTATION.md           # Full architecture design doc
│   ├── USAGE.md                    # OpenVoice usage guide
│   └── QA.md                       # Common issues
├── resources/                      # Sample audio files
├── runs/                           # Per-run intermediate outputs (gitignored)
└── models/                         # Downloaded checkpoints (gitignored)
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/voice-bridge.git
cd voice-bridge
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# MeloTTS must be installed from GitHub
pip install git+https://github.com/myshell-ai/MeloTTS.git --no-deps

# Download required NLTK + unidic data (one-time setup)
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')"
python -m unidic download
```

### 3. Run the pipeline

```bash
python pipeline/run_pipeline.py your_audio.wav \
    --accent indian_american \
    --output result.wav \
    --whisper-model base
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--accent` | `indian_american` | Target accent: `indian_american` or `indian_british` |
| `--output` | `runs/<id>/output.wav` | Output WAV path |
| `--whisper-model` | `large-v3` | Whisper size: `tiny/base/small/medium/large-v3` |
| `--no-vad` | off | Skip silero-VAD pre-filter |
| `--no-duration` | off | Skip phase-vocoder duration matching |
| `--device` | `auto` | `cuda` or `cpu` |

### 4. Run tests

```bash
python -m pytest tests/ -v
```

---

## What Works Right Now

| Component | Status | Notes |
|---|---|---|
| Stage 1 — Whisper transcription | ✅ **Working** | Word-level timestamps, VAD pre-filter |
| Stage 2 — Prosody extraction | ✅ **Working** | F0, energy, word durations via parselmouth |
| Stage 2 — Emotion extraction | ⚠️ **Fallback** | Returns "neutral" — SpeechBrain not installed yet |
| Stage 2 — Speaker embedding | ⚠️ **Fallback** | Returns zeros — SpeechBrain not installed yet |
| Stage 3 — Phonetic rewriting | ✅ **Working** | G2P via gruut, lexicon + rule fallback |
| Stage 4 — MeloTTS synthesis | ✅ **Working** | EN-US accent TTS at matched speaking rate |
| Stage 4 — OpenVoice voice cloning | ⚠️ **Skipped** | Checkpoints not downloaded yet (see below) |
| Stage 5 — F0 contour transplant | ✅ **Working** | WORLD vocoder pitch warp |
| Stage 5 — Energy envelope warp | ✅ **Working** | Frame-level RMS gain scaling |
| Stage 5 — Duration matching | ✅ **Working** | librosa phase-vocoder time-stretch |
| End-to-end pipeline | ✅ **Working** | Tested on CPU, ~160s for 60s of audio |
| Evaluation suite | ✅ **Written** | Not yet run on a proper test set |

---

## What Needs To Be Done

### Priority 1 — Critical for full quality output

#### 1.1 Download OpenVoice v2 Checkpoints
Without this, Stage 4 skips voice cloning entirely and outputs raw MeloTTS audio (correct accent but **wrong speaker voice**).

```bash
# Option A: Download from HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='myshell-ai/OpenVoiceV2', local_dir='checkpoints_v2')
"
```

The checkpoint directory must be at `checkpoints_v2/converter/` relative to the repo root (already configured in `stage4_synthesize.py`).

#### 1.2 Install SpeechBrain
Without this, emotion and speaker embeddings are dummy zeros — the pipeline still runs but emotion preservation is not active.

```bash
pip install speechbrain
```

After installing, Stage 2 will automatically use the real models:
- `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` — emotion label + embedding
- `speechbrain/spkrec-ecapa-voxceleb` — ECAPA-TDNN speaker x-vector

---

### Priority 2 — Quality improvements

#### 2.1 Expand the accent lexicon
Currently `lexicons/indian_to_american.json` has ~50 words. Needs to be expanded to cover the most frequent divergent pronunciations.

**Approach**: Use CMU Pronouncing Dictionary + manual curation of top 2000 divergent words. The JSON format is:
```json
{
  "word": { "from_ipa": "source IPA", "to_ipa": "target IPA" }
}
```

#### 2.2 Validate phoneme rewrites survive OpenVoice tone color conversion
This is the biggest open research question (Q3 in the implementation plan). Once OpenVoice checkpoints are set up, run:
```bash
python pipeline/stage4_synthesize.py "The water bottle is on the table" resources/example_reference.mp3 --output test_s4.wav
```
Then manually listen and check if "water" sounds American (/wɑːɾər/) or Indian (/wɔːtər/).

If rewrites are overwritten by the tone color converter, the fix is to pass phoneme-level input directly to MeloTTS's internal VITS model instead of going through the text cleaner.

#### 2.3 Run the evaluation suite
```bash
python evaluation/eval_pipeline.py \
    --test-dir path/to/your/test/wavs/ \
    --output eval_report.json \
    --accent indian_american
```

Target thresholds (from `evaluation/metrics.py`):
- WER < 5%
- F0 Pearson correlation ≥ 0.75
- Speaker cosine similarity ≥ 0.85
- Emotion label match ≥ 80%

#### 2.4 Tune prosody transplant parameters
The F0 normalization strategy in Stage 5 uses a simple z-score normalization. If the output sounds unnatural, tune:
- `MIN_GAIN` / `MAX_GAIN` clipping in `stage5_prosody_transplant.py`
- The F0 contour smoothing window size
- The duration tolerance threshold (default ±5%)

---

### Priority 3 — Future work

| Item | Description |
|---|---|
| **GPU support** | All stages support `--device cuda`. Should give ~10x speedup on Stage 1 and 4 |
| **Multi-speaker input** | Add pyannote diarization before Stage 1 for conversation audio |
| **British accent lexicon** | `lexicons/indian_to_british.json` has 33 entries — needs expansion |
| **Streaming mode** | Currently batch-only. Real-time would require chunk-wise Whisper + streaming TTS |
| **Web UI** | Gradio/Streamlit interface wrapping `run_pipeline.py` |
| **Fine-tune emotion classifier** | SpeechBrain model is trained on IEMOCAP (acted speech). May perform poorly on accented natural speech — fine-tune on accented data |
| **Speaking rate ↔ emotion conditioning** | Detected emotion should also modulate TTS speed (fast = excited, slow = sad). See Q4 in `docs/IMPLEMENTATION.md` |

---

## Key Design Decisions

- **Prosody transplant = emotion preservation**: We don't re-inject emotion as a discrete label. Instead, the raw pitch contour (F0), energy envelope, and speaking rate from the original audio are warped onto the synthesized output. This is what actually carries emotional information.

- **Two-level phonetic rewriting**: Word-level lexicon lookup first (precise), phoneme-level rules as fallback (broad). This means lexicon quality directly determines accent conversion quality — the lexicon is the most impactful thing to improve.

- **Graceful fallbacks everywhere**: Every stage degrades gracefully if a model is missing. The pipeline always produces audio even without SpeechBrain or OpenVoice.

- **Run directory per execution**: All intermediate outputs (transcript, features, synthesized WAV) are saved to `runs/<uuid>/` for reproducibility and debugging.

---

## Architecture Reference

See [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) for the full technical design doc including data contracts between stages, open questions, and the 4-week implementation phases.

---

## Dependencies Summary

| Package | Purpose |
|---|---|
| `faster-whisper` | Stage 1 transcription |
| `praat-parselmouth` | Stage 2 F0/energy extraction |
| `speechbrain` | Stage 2 emotion + speaker (install separately) |
| `gruut` | Stage 3 G2P phoneme conversion |
| `MeloTTS` (GitHub) | Stage 4 multi-accent TTS |
| `openvoice` (this repo) | Stage 4 tone color conversion |
| `pyworld` | Stage 5 WORLD vocoder F0 warp |
| `librosa` | Stage 5 time-stretch |

---

## Original OpenVoice Credits

This repo is built on top of [OpenVoice](https://github.com/myshell-ai/OpenVoice) by MyShell AI.

```
@article{qin2023openvoice,
  title={OpenVoice: Versatile Instant Voice Cloning},
  author={Qin, Zengyi and Zhao, Wenliang and Yu, Xumin and Sun, Xin},
  journal={arXiv preprint arXiv:2312.01479},
  year={2023}
}
```

OpenVoice V1 and V2 are MIT Licensed. Free for both commercial and research use.
