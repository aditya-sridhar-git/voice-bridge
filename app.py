"""
app.py - voice-bridge Gradio Web UI
=====================================
Premium glassmorphism UI for emotion-preserving accent conversion.

Usage:
    python app.py
    python app.py --share   # public link
"""

import sys
import os
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline(audio_path, accent_pair, whisper_model, device, apply_vad, match_duration):
    try:
        from pipeline.run_pipeline import run_pipeline
    except ImportError as e:
        return None, None, f"Import error: {e}\n\nRun: pip install -r requirements.txt"

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

        stats_html = f"""
        <div class="stats-grid">
          <div class="stat-card">
            <div class="stat-icon">🎭</div>
            <div class="stat-label">Emotion</div>
            <div class="stat-value">{result.emotion_label.title()}</div>
            <div class="stat-sub">{result.emotion_score:.0%} confidence</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon">⏱️</div>
            <div class="stat-label">Pipeline Time</div>
            <div class="stat-value">{result.duration_s:.1f}s</div>
            <div class="stat-sub">end-to-end</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon">🔤</div>
            <div class="stat-label">Rewrites</div>
            <div class="stat-value">{result.rewrites_count}</div>
            <div class="stat-sub">words adapted</div>
          </div>
          <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-label">Run ID</div>
            <div class="stat-value" style="font-size:0.9rem">{result.run_id}</div>
            <div class="stat-sub">saved to runs/</div>
          </div>
        </div>
        <div class="transcript-box">
          <div class="transcript-label">Transcript</div>
          <div class="transcript-text">{result.transcript}</div>
        </div>
        """
        return result.output_audio, stats_html, ""

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return None, None, f"Pipeline failed:\n\n{str(e)}\n\n{tb}"


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & Base ─────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #080810 !important;
    min-height: 100vh;
    color: #e2e8f0;
}

.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 0 16px 60px !important;
}

/* ── Animated background ──────────────────────────────────────────── */
.gradio-container::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background:
        radial-gradient(ellipse at 20% 20%, rgba(99,102,241,0.12) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 10%, rgba(168,85,247,0.10) 0%, transparent 45%),
        radial-gradient(ellipse at 60% 80%, rgba(236,72,153,0.07) 0%, transparent 40%);
    z-index: -1;
    animation: bgPulse 12s ease-in-out infinite alternate;
    pointer-events: none;
}

@keyframes bgPulse {
    0%   { opacity: 0.6; transform: scale(1) rotate(0deg); }
    100% { opacity: 1;   transform: scale(1.08) rotate(2deg); }
}

/* ── Header ───────────────────────────────────────────────────────── */
#vb-header {
    text-align: center;
    padding: 56px 32px 44px;
    position: relative;
}

#vb-header .badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 99px;
    padding: 5px 16px;
    font-size: 0.75rem;
    font-weight: 600;
    color: #a78bfa;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 22px;
    animation: fadeSlideDown 0.6s ease both;
}

#vb-header h1 {
    font-size: clamp(2.6rem, 6vw, 4rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #c4b5fd 0%, #818cf8 30%, #38bdf8 60%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    background-size: 300% 300%;
    animation: gradientShift 6s ease infinite, fadeSlideDown 0.7s ease 0.1s both;
    margin-bottom: 16px;
}

@keyframes gradientShift {
    0%,100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
}

#vb-header p {
    color: rgba(180,190,215,0.75);
    font-size: 1.05rem;
    font-weight: 300;
    line-height: 1.7;
    max-width: 560px;
    margin: 0 auto;
    animation: fadeSlideDown 0.8s ease 0.2s both;
}

@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-14px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Panels ───────────────────────────────────────────────────────── */
.vb-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 28px;
    backdrop-filter: blur(16px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.vb-panel:hover {
    border-color: rgba(99,102,241,0.25);
    box-shadow: 0 0 40px rgba(99,102,241,0.06);
}

/* ── Section labels ───────────────────────────────────────────────── */
.vb-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: rgba(148,163,184,0.7);
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.vb-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
}

/* ── Gradio component overrides ───────────────────────────────────── */
label span, .label-wrap span {
    color: rgba(180,190,215,0.85) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Audio component */
.gr-audio, [data-testid="audio"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 14px !important;
}

/* Dropdowns */
select {
    background: rgba(15,12,40,0.8) !important;
    border: 1px solid rgba(99,102,241,0.25) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}

/* Radio buttons */
.gr-radio-row {
    gap: 10px !important;
}

/* Checkboxes */
input[type="checkbox"] {
    accent-color: #6366f1 !important;
    width: 15px !important;
    height: 15px !important;
}

/* Accordion */
.gr-accordion {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
}

/* ── Convert button ───────────────────────────────────────────────── */
#run-btn {
    position: relative;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%) !important;
    border: none !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    padding: 16px 28px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 30px rgba(99,102,241,0.45), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    letter-spacing: 0.4px;
    width: 100%;
    overflow: hidden;
}

#run-btn::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    transition: left 0.5s ease;
}

#run-btn:hover::before { left: 100%; }

#run-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 40px rgba(99,102,241,0.6), inset 0 1px 0 rgba(255,255,255,0.2) !important;
}

#run-btn:active {
    transform: translateY(0px) scale(0.99) !important;
}

/* ── Clear button ─────────────────────────────────────────────────── */
#clear-btn {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: rgba(180,190,215,0.8) !important;
    border-radius: 14px !important;
    font-weight: 500 !important;
    width: 100%;
    transition: all 0.2s ease !important;
}

#clear-btn:hover {
    background: rgba(255,255,255,0.09) !important;
    border-color: rgba(255,255,255,0.18) !important;
}

/* ── Pipeline stages ──────────────────────────────────────────────── */
#pipeline-stages {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 28px 24px;
    flex-wrap: wrap;
}

.stage-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    flex: 1;
    min-width: 80px;
    position: relative;
}

.stage-item:not(:last-child)::after {
    content: '';
    position: absolute;
    top: 18px;
    right: -50%;
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, rgba(99,102,241,0.5), rgba(99,102,241,0.15));
    z-index: 0;
}

.stage-dot {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.95rem;
    position: relative;
    z-index: 1;
    border: 1px solid rgba(255,255,255,0.12);
    transition: all 0.3s ease;
}

.stage-dot.s1 { background: rgba(167,139,250,0.2); box-shadow: 0 0 16px rgba(167,139,250,0.3); }
.stage-dot.s2 { background: rgba(96,165,250,0.2);  box-shadow: 0 0 16px rgba(96,165,250,0.3); }
.stage-dot.s3 { background: rgba(52,211,153,0.2);  box-shadow: 0 0 16px rgba(52,211,153,0.3); }
.stage-dot.s4 { background: rgba(244,114,182,0.2); box-shadow: 0 0 16px rgba(244,114,182,0.3); }
.stage-dot.s5 { background: rgba(251,191,36,0.2);  box-shadow: 0 0 16px rgba(251,191,36,0.3); }

.stage-name {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: rgba(150,160,185,0.75);
    text-align: center;
}

/* ── Stats grid ───────────────────────────────────────────────────── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}

.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 16px 14px;
    text-align: center;
    backdrop-filter: blur(8px);
    transition: all 0.2s ease;
    animation: fadeIn 0.5s ease both;
}

.stat-card:hover {
    background: rgba(99,102,241,0.08);
    border-color: rgba(99,102,241,0.25);
    transform: translateY(-2px);
}

.stat-icon { font-size: 1.3rem; margin-bottom: 6px; }
.stat-label { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; color: rgba(148,163,184,0.6); margin-bottom: 4px; }
.stat-value { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; }
.stat-sub   { font-size: 0.68rem; color: rgba(148,163,184,0.55); margin-top: 2px; }

/* ── Transcript box ───────────────────────────────────────────────── */
.transcript-box {
    background: rgba(0,0,0,0.3);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 12px;
    padding: 16px 18px;
    animation: fadeIn 0.6s ease both;
}

.transcript-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: rgba(99,102,241,0.8);
    margin-bottom: 8px;
}

.transcript-text {
    font-size: 0.9rem;
    color: rgba(200,210,230,0.85);
    line-height: 1.7;
    font-style: italic;
}

/* ── Error box ────────────────────────────────────────────────────── */
#error-box textarea {
    background: rgba(239,68,68,0.06) !important;
    border: 1px solid rgba(239,68,68,0.25) !important;
    color: rgba(252,165,165,0.9) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    line-height: 1.6 !important;
}

/* ── Accent selector ──────────────────────────────────────────────── */
.accent-btn-row {
    display: flex;
    gap: 10px;
    margin: 8px 0;
}

/* ── Processing overlay ───────────────────────────────────────────── */
#processing-bar {
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899, #6366f1);
    background-size: 200% 100%;
    border-radius: 2px;
    animation: shimmer 1.5s infinite linear;
    display: none;
    margin-bottom: 12px;
}

@keyframes shimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Info accordion ───────────────────────────────────────────────── */
.gr-accordion > .label-wrap {
    color: rgba(180,190,215,0.7) !important;
    font-size: 0.88rem !important;
}

/* ── Divider ──────────────────────────────────────────────────────── */
.vb-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.2), rgba(168,85,247,0.15), transparent);
    margin: 24px 0;
}

/* ── Footer ───────────────────────────────────────────────────────── */
#vb-footer {
    text-align: center;
    padding: 20px;
    color: rgba(100,116,139,0.6);
    font-size: 0.78rem;
    line-height: 1.8;
}

#vb-footer a {
    color: rgba(167,139,250,0.7);
    text-decoration: none;
    transition: color 0.2s;
}
#vb-footer a:hover { color: #a78bfa; }

/* ── Responsive ───────────────────────────────────────────────────── */
@media (max-width: 720px) {
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
    #pipeline-stages { gap: 4px; }
    .stage-item::after { display: none; }
}
"""


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

def build_ui():
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        sys.exit(1)

    ACCENT_CHOICES = {
        "Indian  ->  American English": "indian_american",
        "Indian  ->  British English":  "indian_british",
    }
    WHISPER_CHOICES = ["tiny", "base", "small", "medium", "large-v3"]
    DEVICE_CHOICES  = ["auto", "cpu", "cuda"]

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    ).set(
        body_background_fill="#080810",
        body_text_color="#e2e8f0",
        block_background_fill="rgba(255,255,255,0.03)",
        block_border_color="rgba(255,255,255,0.08)",
        block_border_width="1px",
        block_radius="16px",
        input_background_fill="rgba(15,12,40,0.7)",
        input_border_color="rgba(99,102,241,0.25)",
        button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
        button_primary_text_color="white",
    )

    with gr.Blocks(title="voice-bridge — Accent Conversion", css=CSS, theme=theme) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="vb-header">
          <div class="badge">&#127897; AI Voice Lab</div>
          <h1>voice-bridge</h1>
          <p>Emotion-preserving accent conversion &mdash; transform Indian English
             into American or British English while keeping your voice identity,
             emotion, and prosody completely intact.</p>
        </div>
        """)

        # ── Pipeline stages ──────────────────────────────────────────────────
        gr.HTML("""
        <div id="pipeline-stages">
          <div class="stage-item">
            <div class="stage-dot s1">&#128226;</div>
            <div class="stage-name">Transcribe</div>
          </div>
          <div class="stage-item">
            <div class="stage-dot s2">&#128202;</div>
            <div class="stage-name">Features</div>
          </div>
          <div class="stage-item">
            <div class="stage-dot s3">&#128289;</div>
            <div class="stage-name">Phonetics</div>
          </div>
          <div class="stage-item">
            <div class="stage-dot s4">&#127908;</div>
            <div class="stage-name">Synthesis</div>
          </div>
          <div class="stage-item">
            <div class="stage-dot s5">&#127911;</div>
            <div class="stage-name">Prosody</div>
          </div>
        </div>
        <div class="vb-divider"></div>
        """)

        # ── Main layout ──────────────────────────────────────────────────────
        with gr.Row(equal_height=False, variant="panel"):

            # Left — Input & Settings
            with gr.Column(scale=5, min_width=320):
                gr.HTML('<div class="vb-label">Input Audio</div>')
                audio_input = gr.Audio(
                    label="Upload or record audio",
                    type="filepath",
                    source="upload",
                    show_label=True,
                )

                gr.HTML('<div class="vb-label" style="margin-top:20px">Target Accent</div>')
                accent_choice = gr.Radio(
                    choices=list(ACCENT_CHOICES.keys()),
                    value=list(ACCENT_CHOICES.keys())[0],
                    label="",
                    show_label=False,
                )

                gr.HTML('<div class="vb-label" style="margin-top:16px">Model Settings</div>')
                with gr.Row():
                    whisper_choice = gr.Dropdown(
                        label="Whisper model",
                        choices=WHISPER_CHOICES,
                        value="base",
                        scale=3,
                    )
                    device_choice = gr.Dropdown(
                        label="Device",
                        choices=DEVICE_CHOICES,
                        value="auto",
                        scale=2,
                    )

                with gr.Row():
                    vad_check = gr.Checkbox(label="Silence stripping (VAD)", value=True)
                    dur_check = gr.Checkbox(label="Duration matching", value=True)

                gr.HTML('<div style="height:8px"></div>')
                with gr.Row():
                    run_btn   = gr.Button("Convert Accent", variant="primary",   elem_id="run-btn")
                    clear_btn = gr.Button("Clear",          variant="secondary",  elem_id="clear-btn")

                # Sample files
                sample_files = sorted(Path("resources").glob("*.mp3"))
                if sample_files:
                    gr.HTML('<div class="vb-label" style="margin-top:20px">Sample Files</div>')
                    gr.Examples(
                        examples=[[str(f)] for f in sample_files[:3]],
                        inputs=[audio_input],
                        label="",
                    )

            # Right — Outputs
            with gr.Column(scale=5, min_width=320):
                gr.HTML('<div class="vb-label">Output Audio</div>')
                audio_output = gr.Audio(
                    label="Converted audio",
                    type="filepath",
                    interactive=False,
                    show_label=True,
                )

                gr.HTML('<div class="vb-label" style="margin-top:20px">Results</div>')
                stats_output = gr.HTML(
                    value="""
                    <div style="text-align:center;padding:32px 16px;color:rgba(148,163,184,0.4)">
                      <div style="font-size:2rem;margin-bottom:12px">&#128222;</div>
                      <div style="font-size:0.88rem;line-height:1.7">
                        Upload audio and click <strong style="color:rgba(167,139,250,0.7)">Convert Accent</strong><br>
                        to see results here.
                      </div>
                    </div>
                    """,
                )

                error_box = gr.Textbox(
                    label="Error log",
                    lines=5,
                    visible=False,
                    interactive=False,
                    elem_id="error-box",
                )

        # ── How it works ─────────────────────────────────────────────────────
        gr.HTML('<div class="vb-divider" style="margin-top:32px"></div>')
        with gr.Accordion("How it works", open=False):
            gr.HTML("""
            <div style="padding:20px 8px;display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:16px">
              <div style="background:rgba(167,139,250,0.07);border:1px solid rgba(167,139,250,0.2);border-radius:12px;padding:16px">
                <div style="color:#a78bfa;font-weight:700;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Stage 1</div>
                <div style="color:#c4b5fd;font-weight:600;margin-bottom:6px">Transcription</div>
                <div style="color:rgba(167,139,250,0.65);font-size:0.82rem;line-height:1.6">faster-whisper with silero-VAD &rarr; word-level timestamps</div>
              </div>
              <div style="background:rgba(96,165,250,0.07);border:1px solid rgba(96,165,250,0.2);border-radius:12px;padding:16px">
                <div style="color:#60a5fa;font-weight:700;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Stage 2</div>
                <div style="color:#93c5fd;font-weight:600;margin-bottom:6px">Feature Extraction</div>
                <div style="color:rgba(96,165,250,0.65);font-size:0.82rem;line-height:1.6">SpeechBrain emotion + speaker &middot; parselmouth F0 / energy / duration</div>
              </div>
              <div style="background:rgba(52,211,153,0.07);border:1px solid rgba(52,211,153,0.2);border-radius:12px;padding:16px">
                <div style="color:#34d399;font-weight:700;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Stage 3</div>
                <div style="color:#6ee7b7;font-weight:600;margin-bottom:6px">Phonetic Rewriting</div>
                <div style="color:rgba(52,211,153,0.65);font-size:0.82rem;line-height:1.6">Accent lexicon + phoneme substitution rules</div>
              </div>
              <div style="background:rgba(244,114,182,0.07);border:1px solid rgba(244,114,182,0.2);border-radius:12px;padding:16px">
                <div style="color:#f472b6;font-weight:700;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Stage 4</div>
                <div style="color:#f9a8d4;font-weight:600;margin-bottom:6px">Voice Synthesis</div>
                <div style="color:rgba(244,114,182,0.65);font-size:0.82rem;line-height:1.6">MeloTTS EN-US accent &rarr; OpenVoice v2 tone color conversion</div>
              </div>
              <div style="background:rgba(251,191,36,0.07);border:1px solid rgba(251,191,36,0.2);border-radius:12px;padding:16px">
                <div style="color:#fbbf24;font-weight:700;font-size:0.82rem;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:8px">Stage 5</div>
                <div style="color:#fde68a;font-weight:600;margin-bottom:6px">Prosody Transplant</div>
                <div style="color:rgba(251,191,36,0.65);font-size:0.82rem;line-height:1.6">WORLD vocoder F0 warp &middot; energy envelope &middot; duration stretch</div>
              </div>
            </div>
            """)

        # ── Footer ───────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="vb-footer">
          Built on <a href="https://github.com/myshell-ai/OpenVoice" target="_blank">OpenVoice v2</a>
          by MyShell AI &nbsp;&middot;&nbsp;
          <a href="https://github.com/guillaumekln/faster-whisper" target="_blank">faster-whisper</a>
          &nbsp;&middot;&nbsp;
          <a href="https://github.com/myshell-ai/MeloTTS" target="_blank">MeloTTS</a>
          &nbsp;&middot;&nbsp;
          <a href="https://speechbrain.github.io" target="_blank">SpeechBrain</a>
          &nbsp;&middot;&nbsp;
          WORLD vocoder
        </div>
        """)

        # ── Event handlers ────────────────────────────────────────────────────
        def on_run(audio, accent_label, whisper_model, device, use_vad, use_dur):
            if audio is None:
                return (
                    None,
                    """<div style="text-align:center;padding:24px;color:rgba(251,191,36,0.8);font-size:0.9rem">
                       Please upload or record audio first.</div>""",
                    gr.update(visible=False),
                )
            accent_pair = ACCENT_CHOICES.get(accent_label, "indian_american")
            out_wav, stats_html, error = _run_pipeline(
                audio_path=audio,
                accent_pair=accent_pair,
                whisper_model=whisper_model,
                device=device,
                apply_vad=use_vad,
                match_duration=use_dur,
            )
            if error:
                return (
                    None,
                    """<div style="text-align:center;padding:24px;color:rgba(239,68,68,0.8);font-size:0.9rem">
                       Pipeline failed. See error log below.</div>""",
                    gr.update(value=error, visible=True),
                )
            return out_wav, stats_html, gr.update(visible=False)

        def on_clear():
            empty_stats = """
            <div style="text-align:center;padding:32px 16px;color:rgba(148,163,184,0.4)">
              <div style="font-size:2rem;margin-bottom:12px">&#128222;</div>
              <div style="font-size:0.88rem;line-height:1.7">
                Upload audio and click <strong style="color:rgba(167,139,250,0.7)">Convert Accent</strong><br>
                to see results here.
              </div>
            </div>
            """
            return (
                None,               # audio_input
                None,               # audio_output
                "base",             # whisper_choice
                "auto",             # device_choice
                True,               # vad_check
                True,               # dur_check
                empty_stats,        # stats_output
                gr.update(visible=False),  # error_box
            )

        run_btn.click(
            fn=on_run,
            inputs=[audio_input, accent_choice, whisper_choice, device_choice, vad_check, dur_check],
            outputs=[audio_output, stats_output, error_box],
        )

        clear_btn.click(
            fn=on_clear,
            inputs=[],
            outputs=[audio_input, audio_output, whisper_choice, device_choice,
                     vad_check, dur_check, stats_output, error_box],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="voice-bridge Web UI")
    parser.add_argument("--host",  default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port",  default=7860, type=int)
    parser.add_argument("--share", action="store_true", help="Public Gradio link")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )
