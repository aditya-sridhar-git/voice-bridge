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
) -> tuple:
    """Wrapper around run_pipeline that returns (output_wav_path, summary_text)."""
    try:
        from pipeline.run_pipeline import run_pipeline
    except ImportError as e:
        return None, f"❌ Import error: {e}\n\nMake sure you've installed all requirements:\n  pip install -r requirements.txt"

    try:
        result = run_pipeline(
            input_audio=audio_path,
            accent_pair=accent_pair,
            whisper_model=whisper_model,
            apply_vad=apply_vad,
            match_duration=match_duration,
            device=device,
            verbose=True,
        )

        summary = f"""✅ Pipeline completed in {result.duration_s:.1f}s

📝 Transcript:
{result.transcript}

🎭 Emotion detected: {result.emotion_label} (score: {result.emotion_score:.2f})
🔤 Words rewritten (accent): {result.rewrites_count}
🎯 Accent: {result.accent_pair}
📁 Run ID: {result.run_id}

💾 Output saved to:
{result.output_audio}

📂 All intermediate files:
{result.run_dir}
"""
        return result.output_audio, summary

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return None, f"❌ Pipeline failed:\n\n{str(e)}\n\nTraceback:\n{tb}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def build_ui():
    try:
        import gradio as gr
    except ImportError:
        print("❌ Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    ACCENT_CHOICES = {
        "Indian → American English": "indian_american",
        "Indian → British English":  "indian_british",
    }

    WHISPER_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]

    DEVICE_CHOICES = ["auto", "cpu", "cuda"]

    # ---- CSS ----
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { box-sizing: border-box; }

    body, .gradio-container {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        min-height: 100vh;
    }

    .gradio-container {
        max-width: 960px !important;
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

    /* Cards */
    .panel-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(8px);
    }

    /* Labels */
    label span {
        color: rgba(200,200,220,0.9) !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px;
    }

    /* Dropdowns & radios */
    .gr-dropdown select, .gr-radio input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(99,102,241,0.35) !important;
        color: #e2e8f0 !important;
        border-radius: 10px !important;
    }

    /* Audio component */
    .gr-audio {
        border: 1px solid rgba(99,102,241,0.3) !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.03) !important;
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
    }

    /* Result textbox */
    #result-text textarea {
        background: rgba(0,0,0,0.3) !important;
        border: 1px solid rgba(99,102,241,0.2) !important;
        color: #c4cad6 !important;
        border-radius: 10px !important;
        font-family: 'Inter', monospace !important;
        font-size: 0.9rem !important;
        line-height: 1.6 !important;
    }

    /* Pipeline diagram box */
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

    /* Tabs */
    .tab-nav button {
        color: rgba(200,200,220,0.7) !important;
        font-weight: 500 !important;
        border-radius: 8px 8px 0 0 !important;
    }

    .tab-nav button.selected {
        color: #a78bfa !important;
        border-bottom: 2px solid #a78bfa !important;
        background: rgba(99,102,241,0.1) !important;
    }

    /* Section titles */
    .section-title {
        color: rgba(200,200,220,0.9);
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 12px;
        opacity: 0.75;
    }

    /* Checkboxes */
    .gr-checkbox input[type=checkbox] {
        accent-color: #6366f1;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    """

    with gr.Blocks(
        title="voice-bridge — Emotion-Preserving Accent Conversion",
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        with gr.Column(elem_id="header-box"):
            gr.HTML("""
            <h1>🎙️ voice-bridge</h1>
            <p>Emotion-preserving accent conversion &nbsp;·&nbsp;
               Convert Indian English → American or British English<br>
               while keeping your voice identity, emotion, and prosody intact.</p>
            """)

        # ── Pipeline info ────────────────────────────────────────────────────
        with gr.Accordion("📐 How it works (pipeline overview)", open=False):
            gr.HTML("""
            <div id="pipeline-info">
            <p>
            <strong style="color:#a78bfa">Stage 1 — Transcription</strong><br>
            faster-whisper (large-v3) with silero-VAD silence stripping → word-level timestamps<br><br>

            <strong style="color:#60a5fa">Stage 2 — Feature Extraction</strong><br>
            SpeechBrain ECAPA-TDNN speaker x-vector · wav2vec2 emotion classifier ·
            parselmouth (Praat) F0/energy/duration contours<br><br>

            <strong style="color:#34d399">Stage 3 — Phonetic Rewriting</strong><br>
            gruut G2P → word-level lexicon lookup → phoneme-level rules fallback<br><br>

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
                gr.HTML('<p class="section-title">🎤 Input Audio</p>')
                audio_input = gr.Audio(
                    label="Upload or record your audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )

                gr.HTML('<p class="section-title" style="margin-top:20px">⚙️ Settings</p>')

                accent_choice = gr.Radio(
                    label="Target accent",
                    choices=list(ACCENT_CHOICES.keys()),
                    value=list(ACCENT_CHOICES.keys())[0],
                )

                with gr.Row():
                    whisper_choice = gr.Dropdown(
                        label="Whisper model size",
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
                        label="Match speaking duration (Stage 5)",
                        value=True,
                    )

                with gr.Row():
                    run_btn   = gr.Button("▶  Convert Accent", variant="primary", elem_id="run-btn")
                    clear_btn = gr.Button("✕  Clear", variant="secondary", elem_id="clear-btn")

            # Right column — outputs
            with gr.Column(scale=5):
                gr.HTML('<p class="section-title">🔊 Output Audio</p>')
                audio_output = gr.Audio(
                    label="Converted audio",
                    type="filepath",
                    interactive=False,
                )

                gr.HTML('<p class="section-title" style="margin-top:20px">📊 Pipeline Summary</p>')
                result_text = gr.Textbox(
                    label="",
                    lines=12,
                    interactive=False,
                    placeholder="Pipeline results will appear here after conversion...",
                    elem_id="result-text",
                )

        # ── Examples ────────────────────────────────────────────────────────
        sample_files = sorted(Path("resources").glob("*.mp3"))
        if sample_files:
            gr.HTML('<p class="section-title" style="margin-top:16px">🎵 Sample Audio Files</p>')
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
                style="color:#a78bfa;text-decoration:none">OpenVoice v2</a> by MyShell AI ·
            Whisper · MeloTTS · SpeechBrain · WORLD vocoder
        </div>
        """)

        # ── Event handlers ───────────────────────────────────────────────────
        def on_run(audio, accent_label, whisper_model, device, use_vad, use_dur):
            if audio is None:
                return None, "⚠️  Please upload or record audio first."

            accent_pair = ACCENT_CHOICES.get(accent_label, "indian_american")
            out_wav, summary = _run_pipeline(
                audio_path=audio,
                accent_pair=accent_pair,
                whisper_model=whisper_model,
                device=device,
                apply_vad=use_vad,
                match_duration=use_dur,
            )
            return out_wav, summary

        def on_clear():
            return None, None, "base", "auto", True, True, ""

        run_btn.click(
            fn=on_run,
            inputs=[audio_input, accent_choice, whisper_choice, device_choice, vad_check, dur_check],
            outputs=[audio_output, result_text],
        )

        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[audio_input, audio_output, whisper_choice, device_choice, vad_check, dur_check, result_text],
        )

    return demo, custom_css, gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="voice-bridge Gradio Web UI")
    parser.add_argument("--host",   default="0.0.0.0",  help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port",   default=7860, type=int, help="Port (default: 7860)")
    parser.add_argument("--share",  action="store_true",  help="Create a public Gradio share link")
    parser.add_argument("--debug",  action="store_true",  help="Enable Gradio debug mode")
    args = parser.parse_args()

    demo, css, theme = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
        css=css,
        theme=theme,
    )
