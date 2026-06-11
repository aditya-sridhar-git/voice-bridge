"""
app.py — voice-bridge Gradio Web UI
=====================================
Drag-and-drop audio → accent-converted audio, all in a beautiful interface.

Usage:
    python app.py

Then open http://localhost:7860 in your browser.
"""

import sys
import os
import json
import tempfile
import logging
from pathlib import Path

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env so OPENAI_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline runner (lazy import so Gradio loads fast even if deps missing)
# ---------------------------------------------------------------------------

def _run_pipeline(
    audio_path: str,
    accent_pair: str,
    whisper_model: str,
    device: str,
    apply_vad: bool,
    match_duration: bool,
    use_llm: bool,
) -> tuple:
    """Wrapper around run_pipeline that returns (output_wav_path, summary_text)."""
    try:
        from pipeline.run_pipeline import run_pipeline
    except ImportError as e:
        return None, f"Import error: {e}\n\nMake sure you've installed all requirements:\n  pip install -r requirements.txt"

    try:
        result = run_pipeline(
            input_audio=audio_path,
            accent_pair=accent_pair,
            whisper_model=whisper_model,
            apply_vad=apply_vad,
            match_duration=match_duration,
            device=device,
            use_llm=use_llm,
            verbose=True,
        )

        llm_label = "GPT-4o mini" if use_llm and os.environ.get("OPENAI_API_KEY") else "Lexicon fallback"
        summary = f"""Pipeline completed in {result.duration_s:.1f}s

Transcript:
{result.transcript}

Emotion detected: {result.emotion_label} (score: {result.emotion_score:.2f})
Words rewritten (accent): {result.rewrites_count}
Phonetic rewriter: {llm_label}
Accent: {result.accent_pair}
Run ID: {result.run_id}

Output saved to:
{result.output_audio}

All intermediate files:
{result.run_dir}
"""
        return result.output_audio, summary

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return None, f"Pipeline failed:\n\n{str(e)}\n\nTraceback:\n{tb}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    min-height: 100vh;
}

.gradio-container {
    max-width: 980px !important;
    margin: 0 auto !important;
}

/* Header */
#header-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.25) 0%, rgba(168,85,247,0.15) 100%);
    border: 1px solid rgba(99,102,241,0.4);
    border-radius: 20px;
    padding: 36px 40px 28px;
    margin-bottom: 28px;
    text-align: center;
    backdrop-filter: blur(12px);
}

#header-box h1 {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
}

#header-box p {
    color: rgba(200,200,220,0.85);
    font-size: 1.05rem;
    margin: 0;
    font-weight: 300;
}

/* Labels */
label span, .label-wrap span {
    color: rgba(200,200,220,0.9) !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px;
}

/* Run button */
#run-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(99,102,241,0.4) !important;
    letter-spacing: 0.3px;
    width: 100%;
}

#run-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.55) !important;
}

#run-btn:active {
    transform: translateY(0) !important;
}

/* Clear button */
#clear-btn {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: rgba(200,200,220,0.9) !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    width: 100%;
}

/* Result textbox */
#result-text textarea {
    background: rgba(0,0,0,0.3) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    color: #c4cad6 !important;
    border-radius: 10px !important;
    font-family: 'Inter', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
}

/* Pipeline info box */
#pipeline-info {
    background: rgba(0,0,0,0.25);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 20px 24px;
    margin-top: 8px;
}

#pipeline-info p {
    color: rgba(170,180,200,0.85);
    font-size: 0.88rem;
    line-height: 1.7;
    margin: 0;
}

/* Section titles */
.section-title {
    color: rgba(200,200,220,0.9);
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin: 16px 0 8px;
    opacity: 0.75;
}

/* LLM badge */
#llm-badge {
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 10px;
    padding: 10px 16px;
    margin: 8px 0 4px;
    font-size: 0.85rem;
    color: rgba(180,185,210,0.9);
}
"""


def build_ui():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    ACCENT_CHOICES = {
        "Indian -> American English": "indian_american",
        "Indian -> British English":  "indian_british",
    }

    WHISPER_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]
    DEVICE_CHOICES  = ["auto", "cpu", "cuda"]

    has_api_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    api_status  = (
        "GPT-4o mini connected" if has_api_key
        else "No API key — using lexicon fallback"
    )
    api_color = "#4ade80" if has_api_key else "#f87171"

    with gr.Blocks(
        title="voice-bridge — Emotion-Preserving Accent Conversion",
        css=custom_css,
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        with gr.Column(elem_id="header-box"):
            gr.HTML(f"""
            <h1>voice-bridge</h1>
            <p>Emotion-preserving accent conversion &nbsp;·&nbsp;
               Indian English &rarr; American or British English<br>
               Keeping your voice identity, emotion, and prosody intact.</p>
            <p style="margin-top:10px;font-size:0.85rem">
              <span style="color:{api_color};font-weight:600">&#9679;</span>
              &nbsp;Stage 3: {api_status}
            </p>
            """)

        # ── Pipeline info ────────────────────────────────────────────────────
        with gr.Accordion("How it works (pipeline overview)", open=False):
            gr.HTML("""
            <div id="pipeline-info">
            <p>
            <strong style="color:#a78bfa">Stage 1 — Transcription</strong><br>
            faster-whisper with silero-VAD silence stripping → word-level timestamps<br><br>

            <strong style="color:#60a5fa">Stage 2 — Feature Extraction</strong><br>
            SpeechBrain ECAPA-TDNN speaker x-vector · wav2vec2 emotion classifier ·
            parselmouth (Praat) F0/energy/duration contours<br><br>

            <strong style="color:#34d399">Stage 3 — Phonetic Rewriting (GPT-4o mini)</strong><br>
            Context-aware vocabulary + pronunciation rewriting via LLM.
            Handles proper nouns, Indian English idioms, and stress patterns.
            Falls back to static lexicon if no API key is set.<br><br>

            <strong style="color:#f472b6">Stage 4 — Voice Synthesis</strong><br>
            MeloTTS accent TTS → OpenVoice v2 ToneColorConverter (voice cloning)<br><br>

            <strong style="color:#fbbf24">Stage 5 — Prosody Transplant</strong><br>
            pyworld WORLD vocoder F0 warp · RMS energy envelope matching ·
            librosa phase-vocoder duration stretch
            </p>
            </div>
            """)

        # ── Main UI ─────────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left column — inputs
            with gr.Column(scale=5):
                gr.HTML('<p class="section-title">Input Audio</p>')
                audio_input = gr.Audio(
                    label="Upload or record your audio",
                    type="filepath",
                    source="upload",   # Gradio 3.x: use 'source' not 'sources'
                )

                gr.HTML('<p class="section-title">Settings</p>')

                accent_choice = gr.Radio(
                    label="Target accent",
                    choices=list(ACCENT_CHOICES.keys()),
                    value=list(ACCENT_CHOICES.keys())[0],
                )

                with gr.Row():
                    whisper_choice = gr.Dropdown(
                        label="Whisper model",
                        choices=WHISPER_CHOICES,
                        value="base",
                    )
                    device_choice = gr.Dropdown(
                        label="Device",
                        choices=DEVICE_CHOICES,
                        value="auto",
                    )

                with gr.Row():
                    vad_check = gr.Checkbox(
                        label="Apply VAD silence stripping",
                        value=True,
                    )
                    dur_check = gr.Checkbox(
                        label="Match speaking duration",
                        value=True,
                    )

                llm_check = gr.Checkbox(
                    label=f"Use GPT-4o mini for phonetic rewriting (Stage 3)",
                    value=has_api_key,
                    interactive=has_api_key,
                )

                gr.HTML('<div style="height:12px"></div>')
                with gr.Row():
                    run_btn   = gr.Button("Convert Accent", variant="primary",   elem_id="run-btn")
                    clear_btn = gr.Button("Clear",          variant="secondary",  elem_id="clear-btn")

            # Right column — outputs
            with gr.Column(scale=5):
                gr.HTML('<p class="section-title">Output Audio</p>')
                audio_output = gr.Audio(
                    label="Converted audio",
                    type="filepath",
                    interactive=False,
                )

                gr.HTML('<p class="section-title">Pipeline Summary</p>')
                result_text = gr.Textbox(
                    label="",
                    lines=14,
                    interactive=False,
                    placeholder="Pipeline results will appear here after conversion...",
                    elem_id="result-text",
                )

        # ── Sample files ─────────────────────────────────────────────────────
        sample_files = sorted(Path("resources").glob("*.mp3"))
        if sample_files:
            gr.HTML('<p class="section-title" style="margin-top:16px">Sample Audio Files</p>')
            gr.Examples(
                examples=[[str(f)] for f in sample_files[:3]],
                inputs=[audio_input],
                label="Click a sample to load it",
            )

        # ── Footer ───────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;margin-top:28px;padding:16px;
                    color:rgba(150,160,180,0.6);font-size:0.82rem;">
            Built on <a href="https://github.com/myshell-ai/OpenVoice" target="_blank"
                style="color:#a78bfa;text-decoration:none">OpenVoice v2</a> &middot;
            Whisper &middot; MeloTTS &middot; SpeechBrain &middot; WORLD vocoder &middot; GPT-4o mini
        </div>
        """)

        # ── Event handlers ───────────────────────────────────────────────────
        def on_run(audio, accent_label, whisper_model, device, use_vad, use_dur, use_llm):
            if audio is None:
                return None, "Please upload or record audio first."
            accent_pair = ACCENT_CHOICES.get(accent_label, "indian_american")
            out_wav, summary = _run_pipeline(
                audio_path=audio,
                accent_pair=accent_pair,
                whisper_model=whisper_model,
                device=device,
                apply_vad=use_vad,
                match_duration=use_dur,
                use_llm=use_llm,
            )
            return out_wav, summary

        def on_clear():
            return None, None, "base", "auto", True, True, has_api_key, ""

        run_btn.click(
            fn=on_run,
            inputs=[audio_input, accent_choice, whisper_choice, device_choice,
                    vad_check, dur_check, llm_check],
            outputs=[audio_output, result_text],
        )

        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[audio_input, audio_output, whisper_choice, device_choice,
                     vad_check, dur_check, llm_check, result_text],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="voice-bridge Gradio Web UI")
    parser.add_argument("--host",  default="0.0.0.0",  help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port",  default=7860, type=int, help="Port (default: 7860)")
    parser.add_argument("--share", action="store_true",  help="Create a public Gradio share link")
    parser.add_argument("--debug", action="store_true",  help="Enable Gradio debug mode")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )
