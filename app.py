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
import random
import logging
from pathlib import Path

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
# Animated SVG waveform generator
# ---------------------------------------------------------------------------

def _make_waveform_svg(n: int = 56, width: int = 460, height: int = 54, seed: int = 7) -> str:
    """Generate an animated bar-style audio waveform SVG."""
    rng = random.Random(seed)
    spacing = width / n
    bar_w = max(2, spacing * 0.55)
    bars = []

    for i in range(n):
        x = i * spacing + (spacing - bar_w) / 2
        # Envelope: taller in the middle
        t = abs(i - n / 2) / (n / 2)
        env = 1.0 - t * 0.65
        h_max = rng.uniform(18, 46) * env + 3
        h_min = rng.uniform(1.5, 6)
        h_s = rng.uniform(h_min, h_max)
        y_s = (height - h_s) / 2
        dur = f"{rng.uniform(0.45, 1.85):.2f}s"
        delay = f"{rng.uniform(0, 2.2):.2f}s"
        bars.append(
            f'<rect x="{x:.1f}" y="{y_s:.1f}" width="{bar_w:.1f}" height="{h_s:.1f}" rx="2">'
            f'<animate attributeName="height" values="{h_min:.1f};{h_max:.1f};{h_min:.1f}" '
            f'dur="{dur}" begin="{delay}" repeatCount="indefinite"/>'
            f'<animate attributeName="y" '
            f'values="{(height-h_min)/2:.1f};{(height-h_max)/2:.1f};{(height-h_min)/2:.1f}" '
            f'dur="{dur}" begin="{delay}" repeatCount="indefinite"/>'
            f'</rect>'
        )

    gradient = (
        '<defs>'
        '<linearGradient id="wg" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0%"   stop-color="rgba(167,139,250,0)"/>'
        '<stop offset="10%"  stop-color="rgba(167,139,250,0.9)"/>'
        '<stop offset="38%"  stop-color="rgba(99,102,241,1)"/>'
        '<stop offset="62%"  stop-color="rgba(96,165,250,1)"/>'
        '<stop offset="88%"  stop-color="rgba(244,114,182,0.9)"/>'
        '<stop offset="100%" stop-color="rgba(244,114,182,0)"/>'
        '</linearGradient>'
        '</defs>'
    )

    return (
        f'<svg style="width:{width}px;height:{height}px;display:block;'
        f'margin:20px auto 0;opacity:0.78" '
        f'viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        + gradient
        + f'<g fill="url(#wg)">{"".join(bars)}</g>'
        + '</svg>'
    )


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

    # ── Theme ────────────────────────────────────────────────────────────────
    theme = gr.themes.Base(
        primary_hue="violet",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    )

    # ── CSS ──────────────────────────────────────────────────────────────────
    css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --av: rgba(167,139,250,VAL);
        --ab: rgba(96,165,250,VAL);
        --ap: rgba(244,114,182,VAL);
        --ag: rgba(52,211,153,VAL);
        --ay: rgba(251,191,36,VAL);
    }

    *, *::before, *::after { box-sizing: border-box; }

    body, .gradio-container {
        font-family: 'Inter', sans-serif !important;
        background: #07050f !important;
        min-height: 100vh;
    }

    .gradio-container {
        max-width: 1040px !important;
        margin: 0 auto !important;
        padding-bottom: 64px !important;
        position: relative;
        z-index: 1;
    }

    /* ── HEADER ─────────────────────────────────────────────────────────── */
    #vb-header {
        position: relative;
        overflow: hidden;
        border-radius: 26px;
        padding: 52px 52px 46px;
        margin-bottom: 18px;
        text-align: center;
        background:
            linear-gradient(135deg,
                rgba(99,102,241,0.22) 0%,
                rgba(139,92,246,0.14) 40%,
                rgba(20,14,50,0.35)   100%);
        border: 1px solid rgba(167,139,250,0.3);
        backdrop-filter: blur(28px);
        box-shadow:
            0 0 120px rgba(99,102,241,0.12),
            0 0 240px rgba(168,85,247,0.06),
            inset 0 1px 0 rgba(255,255,255,0.09),
            inset 0 -1px 0 rgba(0,0,0,0.25);
    }

    /* Rotating glow blob behind header */
    #vb-header::before {
        content: '';
        position: absolute;
        top: -60%; left: -30%;
        width: 160%; height: 220%;
        background: conic-gradient(
            from 0deg at 50% 50%,
            transparent 0deg,
            rgba(99,102,241,0.07) 60deg,
            transparent 120deg,
            rgba(168,85,247,0.05) 200deg,
            transparent 260deg
        );
        animation: hdrRotate 18s linear infinite;
        pointer-events: none;
        will-change: transform;
    }

    @keyframes hdrRotate {
        from { transform: rotate(0deg); }
        to   { transform: rotate(360deg); }
    }

    /* Soft radial spots */
    #vb-header::after {
        content: '';
        position: absolute;
        inset: 0;
        background:
            radial-gradient(ellipse at 20% 10%, rgba(99,102,241,0.18) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 90%, rgba(168,85,247,0.14) 0%, transparent 50%);
        pointer-events: none;
    }

    .vb-inner { position: relative; z-index: 2; }

    .vb-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.4rem, 5vw, 3.6rem);
        font-weight: 700;
        letter-spacing: -1.5px;
        line-height: 1;
        background: linear-gradient(90deg,
            #a78bfa 0%, #818cf8 22%, #60a5fa 44%,
            #f472b6 66%, #c084fc 88%, #a78bfa 100%);
        background-size: 300% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: titleFlow 6s linear infinite;
        display: block;
        margin-bottom: 4px;
    }

    @keyframes titleFlow {
        0%   { background-position: 0% center; }
        100% { background-position: 300% center; }
    }

    .vb-subtitle {
        color: rgba(200,212,235,0.78);
        font-size: 1.02rem;
        font-weight: 300;
        line-height: 1.7;
        margin: 14px auto 0;
        max-width: 600px;
        display: block;
    }

    .vb-subtitle strong {
        color: rgba(225,230,248,0.95);
        font-weight: 600;
    }

    .vb-badges {
        display: flex;
        gap: 8px;
        justify-content: center;
        flex-wrap: wrap;
        margin-top: 22px;
    }

    .vb-badge {
        padding: 5px 15px;
        border-radius: 999px;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        border: 1px solid;
        animation: badgePop 0.55s cubic-bezier(.34,1.56,.64,1) both;
    }

    .vb-badge:nth-child(1) { animation-delay: 0.05s; }
    .vb-badge:nth-child(2) { animation-delay: 0.12s; }
    .vb-badge:nth-child(3) { animation-delay: 0.19s; }
    .vb-badge:nth-child(4) { animation-delay: 0.26s; }
    .vb-badge:nth-child(5) { animation-delay: 0.33s; }

    @keyframes badgePop {
        from { opacity: 0; transform: scale(0.6) translateY(10px); }
        to   { opacity: 1; transform: scale(1) translateY(0); }
    }

    .vb-bv { background: rgba(167,139,250,0.14); border-color: rgba(167,139,250,0.45); color: #c4b5fd; }
    .vb-bb { background: rgba(96,165,250,0.12);  border-color: rgba(96,165,250,0.42);  color: #93c5fd; }
    .vb-bp { background: rgba(244,114,182,0.12); border-color: rgba(244,114,182,0.42); color: #f9a8d4; }
    .vb-bg { background: rgba(52,211,153,0.11);  border-color: rgba(52,211,153,0.4);  color: #6ee7b7; }
    .vb-by { background: rgba(251,191,36,0.11);  border-color: rgba(251,191,36,0.4);  color: #fcd34d; }

    /* ── SECTION LABELS ──────────────────────────────────────────────────── */
    .sec-lbl {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.73rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.4px;
        color: rgba(167,139,250,0.75);
        margin-bottom: 10px;
    }

    .sec-lbl::after {
        content: '';
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(167,139,250,0.28), transparent);
    }

    /* ── FORM ELEMENTS ───────────────────────────────────────────────────── */
    label span {
        color: rgba(210,218,238,0.88) !important;
        font-size: 0.875rem !important;
        font-weight: 500 !important;
    }

    textarea, select, input[type=text], input[type=number] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(167,139,250,0.22) !important;
        color: #dde5f5 !important;
        border-radius: 11px !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    textarea:focus, select:focus, input[type=text]:focus {
        border-color: rgba(167,139,250,0.55) !important;
        box-shadow: 0 0 0 3px rgba(167,139,250,0.1) !important;
        outline: none !important;
    }

    input[type=checkbox] { accent-color: #8b5cf6 !important; }

    /* Gradio 3.x Audio */
    .gr-audio, .audio-component, [data-testid="audio"] {
        border: 1px solid rgba(167,139,250,0.22) !important;
        border-radius: 14px !important;
        background: rgba(255,255,255,0.025) !important;
        transition: border-color 0.25s !important;
    }

    .gr-audio:hover, [data-testid="audio"]:hover {
        border-color: rgba(167,139,250,0.45) !important;
    }

    /* ── ACCORDION ───────────────────────────────────────────────────────── */
    .gr-accordion, .accordion, details {
        border: 1px solid rgba(255,255,255,0.07) !important;
        border-radius: 18px !important;
        background: rgba(255,255,255,0.025) !important;
        overflow: hidden !important;
        margin-bottom: 16px;
    }

    /* ── RUN BUTTON ──────────────────────────────────────────────────────── */
    #run-btn {
        position: relative !important;
        overflow: hidden !important;
        background: linear-gradient(135deg, #4f46e5, #7c3aed, #9333ea) !important;
        border: 1px solid rgba(167,139,250,0.5) !important;
        color: #fff !important;
        font-family: 'Space Grotesk', 'Inter', sans-serif !important;
        font-size: 1.0rem !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        letter-spacing: 0.6px !important;
        transition: transform 0.2s ease, box-shadow 0.25s ease !important;
        animation: runGlow 2.8s ease-in-out infinite !important;
    }

    @keyframes runGlow {
        0%,100% { box-shadow: 0 0 28px rgba(99,102,241,0.4), 0 4px 16px rgba(0,0,0,0.4); }
        50%      { box-shadow: 0 0 48px rgba(139,92,246,0.58), 0 4px 24px rgba(0,0,0,0.45); }
    }

    /* Shimmer sweep */
    #run-btn::before {
        content: '' !important;
        position: absolute !important;
        inset: 0 !important;
        background: linear-gradient(
            105deg,
            transparent 30%,
            rgba(255,255,255,0.18) 50%,
            transparent 70%
        ) !important;
        background-size: 200% 100% !important;
        animation: runShine 2.6s linear infinite !important;
    }

    @keyframes runShine {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    #run-btn:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow: 0 0 60px rgba(99,102,241,0.58), 0 8px 28px rgba(0,0,0,0.5) !important;
        animation: none !important;
    }

    #run-btn:active {
        transform: translateY(0) scale(0.98) !important;
        box-shadow: 0 0 20px rgba(99,102,241,0.35) !important;
        animation: none !important;
    }

    /* ── CLEAR BUTTON ────────────────────────────────────────────────────── */
    #clear-btn {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: rgba(195,208,232,0.72) !important;
        border-radius: 14px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }

    #clear-btn:hover {
        background: rgba(255,255,255,0.08) !important;
        border-color: rgba(255,255,255,0.2) !important;
        color: rgba(220,230,248,0.95) !important;
    }

    /* ── RESULT TEXTBOX ──────────────────────────────────────────────────── */
    #result-text textarea {
        background: rgba(0,0,0,0.45) !important;
        border: 1px solid rgba(167,139,250,0.16) !important;
        color: #aab6cc !important;
        border-radius: 12px !important;
        font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Courier New', monospace !important;
        font-size: 0.83rem !important;
        line-height: 1.75 !important;
    }

    /* ── PIPELINE GRID ───────────────────────────────────────────────────── */
    .pipe-wrap {
        padding: 16px 12px 12px;
    }

    .pipe-grid {
        display: grid;
        grid-template-columns: 1fr 22px 1fr 22px 1fr 22px 1fr 22px 1fr;
        gap: 6px;
        align-items: center;
    }

    .pipe-arrow {
        text-align: center;
        color: rgba(167,139,250,0.45);
        font-size: 1.1rem;
        line-height: 1;
    }

    .pipe-stage {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 16px 10px 14px;
        text-align: center;
        transition: border-color 0.28s, box-shadow 0.28s, transform 0.22s;
        cursor: default;
    }

    .pipe-stage:hover { transform: translateY(-4px); }

    .pipe-icon {
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.35rem;
        margin: 0 auto 10px;
    }

    .pipe-name {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 6px;
    }

    .pipe-desc {
        font-size: 0.67rem;
        color: rgba(155,168,192,0.7);
        line-height: 1.45;
    }

    .p1 { border-color: rgba(167,139,250,0.28); }
    .p1:hover { border-color: rgba(167,139,250,0.55); box-shadow: 0 8px 28px rgba(99,102,241,0.18); }
    .p1 .pipe-icon { background: rgba(167,139,250,0.14); }
    .p1 .pipe-name { color: #c4b5fd; }

    .p2 { border-color: rgba(96,165,250,0.28); }
    .p2:hover { border-color: rgba(96,165,250,0.55); box-shadow: 0 8px 28px rgba(59,130,246,0.18); }
    .p2 .pipe-icon { background: rgba(96,165,250,0.14); }
    .p2 .pipe-name { color: #93c5fd; }

    .p3 { border-color: rgba(52,211,153,0.28); }
    .p3:hover { border-color: rgba(52,211,153,0.55); box-shadow: 0 8px 28px rgba(16,185,129,0.18); }
    .p3 .pipe-icon { background: rgba(52,211,153,0.14); }
    .p3 .pipe-name { color: #6ee7b7; }

    .p4 { border-color: rgba(244,114,182,0.28); }
    .p4:hover { border-color: rgba(244,114,182,0.55); box-shadow: 0 8px 28px rgba(236,72,153,0.18); }
    .p4 .pipe-icon { background: rgba(244,114,182,0.14); }
    .p4 .pipe-name { color: #f9a8d4; }

    .p5 { border-color: rgba(251,191,36,0.28); }
    .p5:hover { border-color: rgba(251,191,36,0.55); box-shadow: 0 8px 28px rgba(245,158,11,0.18); }
    .p5 .pipe-icon { background: rgba(251,191,36,0.14); }
    .p5 .pipe-name { color: #fcd34d; }

    @media (max-width: 700px) {
        .pipe-grid { grid-template-columns: 1fr 1fr; }
        .pipe-arrow { display: none; }
        .pipe-stage { padding: 12px 8px; }
    }

    /* ── FOOTER ──────────────────────────────────────────────────────────── */
    #vb-footer {
        text-align: center;
        padding: 22px 16px;
        color: rgba(110,122,148,0.55);
        font-size: 0.8rem;
        line-height: 1.65;
    }

    #vb-footer a {
        color: rgba(167,139,250,0.72);
        text-decoration: none;
        transition: color 0.18s;
    }

    #vb-footer a:hover { color: #a78bfa; text-decoration: underline; }

    /* ── SCROLLBAR ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
    ::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.28); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(167,139,250,0.48); }
    """

    waveform_svg = _make_waveform_svg()

    # ── Build Blocks ─────────────────────────────────────────────────────────
    with gr.Blocks(
        title="voice-bridge — Emotion-Preserving Accent Conversion",
        css=css,
        theme=theme,
    ) as demo:

        # ── Particle / network background ──────────────────────────────────
        gr.HTML("""
        <script>
        (function () {
            function boot() {
                if (!document.body) { return setTimeout(boot, 60); }
                var cv = document.createElement('canvas');
                cv.style.cssText = [
                    'position:fixed', 'top:0', 'left:0',
                    'width:100vw', 'height:100vh',
                    'pointer-events:none', 'z-index:0', 'opacity:0.42'
                ].join(';');
                document.body.insertBefore(cv, document.body.firstChild);

                var ctx = cv.getContext('2d');
                var W, H, pts = [];
                var cols = ['167,139,250', '96,165,250', '244,114,182'];

                function resize() { W = cv.width = window.innerWidth; H = cv.height = window.innerHeight; }

                function Pt() {
                    this.x  = Math.random() * W;
                    this.y  = Math.random() * H;
                    this.vx = (Math.random() - 0.5) * 0.32;
                    this.vy = (Math.random() - 0.5) * 0.32;
                    this.r  = Math.random() * 1.1 + 0.4;
                    this.a  = Math.random() * 0.32 + 0.06;
                    this.c  = cols[Math.floor(Math.random() * cols.length)];
                }

                Pt.prototype.step = function () {
                    this.x += this.vx; this.y += this.vy;
                    if (this.x < 0 || this.x > W) this.vx *= -1;
                    if (this.y < 0 || this.y > H) this.vy *= -1;
                };

                resize();
                window.addEventListener('resize', resize);
                for (var i = 0; i < 72; i++) pts.push(new Pt());

                function frame() {
                    ctx.clearRect(0, 0, W, H);
                    for (var i = 0; i < pts.length; i++) {
                        pts[i].step();
                        var p = pts[i];
                        ctx.beginPath();
                        ctx.arc(p.x, p.y, p.r, 0, 6.2832);
                        ctx.fillStyle = 'rgba(' + p.c + ',' + p.a + ')';
                        ctx.fill();
                        for (var j = i + 1; j < pts.length; j++) {
                            var q = pts[j];
                            var dx = p.x - q.x, dy = p.y - q.y;
                            var d = Math.sqrt(dx * dx + dy * dy);
                            if (d < 128) {
                                ctx.beginPath();
                                ctx.moveTo(p.x, p.y);
                                ctx.lineTo(q.x, q.y);
                                ctx.strokeStyle = 'rgba(167,139,250,' + (0.075 * (1 - d / 128)) + ')';
                                ctx.lineWidth = 0.55;
                                ctx.stroke();
                            }
                        }
                    }
                    requestAnimationFrame(frame);
                }
                frame();
            }
            boot();
        })();
        </script>
        """)

        # ── Hero header ─────────────────────────────────────────────────────
        gr.HTML(f"""
        <div id="vb-header">
          <div class="vb-inner">
            <span class="vb-title">🎙️ voice-bridge</span>
            <span class="vb-subtitle">
              <strong>Emotion-preserving accent conversion</strong> &nbsp;—&nbsp;
              Transform Indian English into American or British English<br>
              while keeping your voice identity, emotion &amp; natural prosody intact.
            </span>
            {waveform_svg}
            <div class="vb-badges">
              <span class="vb-badge vb-bv">faster-whisper</span>
              <span class="vb-badge vb-bb">MeloTTS</span>
              <span class="vb-badge vb-bp">OpenVoice v2</span>
              <span class="vb-badge vb-bg">SpeechBrain</span>
              <span class="vb-badge vb-by">WORLD Vocoder</span>
            </div>
          </div>
        </div>
        """)

        # ── Pipeline overview ────────────────────────────────────────────────
        with gr.Accordion("📐  How it works — 5-stage pipeline", open=False):
            gr.HTML("""
            <div class="pipe-wrap">
              <div class="pipe-grid">

                <div class="pipe-stage p1">
                  <div class="pipe-icon">🎵</div>
                  <div class="pipe-name">Transcribe</div>
                  <div class="pipe-desc">faster-whisper<br>+ silero-VAD<br>word timestamps</div>
                </div>
                <div class="pipe-arrow">›</div>

                <div class="pipe-stage p2">
                  <div class="pipe-icon">📊</div>
                  <div class="pipe-name">Features</div>
                  <div class="pipe-desc">ECAPA-TDNN<br>wav2vec2 emotion<br>F0 / energy</div>
                </div>
                <div class="pipe-arrow">›</div>

                <div class="pipe-stage p3">
                  <div class="pipe-icon">🔤</div>
                  <div class="pipe-name">Phonetics</div>
                  <div class="pipe-desc">gruut G2P<br>accent lexicon<br>phoneme rules</div>
                </div>
                <div class="pipe-arrow">›</div>

                <div class="pipe-stage p4">
                  <div class="pipe-icon">🗣️</div>
                  <div class="pipe-name">Synthesize</div>
                  <div class="pipe-desc">MeloTTS TTS<br>OpenVoice v2<br>tone cloning</div>
                </div>
                <div class="pipe-arrow">›</div>

                <div class="pipe-stage p5">
                  <div class="pipe-icon">🎯</div>
                  <div class="pipe-name">Prosody</div>
                  <div class="pipe-desc">WORLD F0 warp<br>RMS envelope<br>duration stretch</div>
                </div>

              </div>
            </div>
            """)

        # ── Main 2-column layout ─────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left — inputs
            with gr.Column(scale=5):
                gr.HTML('<div class="sec-lbl">🎤 Input Audio</div>')
                audio_input = gr.Audio(
                    label="Upload or record your audio",
                    type="filepath",
                    sources=["upload", "microphone"],
                )

                gr.HTML('<div class="sec-lbl" style="margin-top:20px">⚙️ Settings</div>')

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
                    run_btn   = gr.Button("▶  Convert Accent", variant="primary",   elem_id="run-btn")
                    clear_btn = gr.Button("✕  Clear",          variant="secondary", elem_id="clear-btn")

            # Right — outputs
            with gr.Column(scale=5):
                gr.HTML('<div class="sec-lbl">🔊 Output Audio</div>')
                audio_output = gr.Audio(
                    label="Converted audio",
                    type="filepath",
                    interactive=False,
                )

                gr.HTML('<div class="sec-lbl" style="margin-top:20px">📊 Pipeline Summary</div>')
                result_text = gr.Textbox(
                    label="",
                    lines=12,
                    interactive=False,
                    placeholder="Pipeline results will appear here after conversion…",
                    elem_id="result-text",
                )

        # ── Sample audio files ───────────────────────────────────────────────
        sample_files = sorted(Path("resources").glob("*.mp3"))
        if sample_files:
            gr.HTML('<div class="sec-lbl" style="margin-top:16px">🎵 Sample Files</div>')
            gr.Examples(
                examples=[[str(f)] for f in sample_files[:3]],
                inputs=[audio_input],
                label="Click to load a sample",
            )

        # ── Footer ───────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="vb-footer">
          Built with
          <a href="https://github.com/myshell-ai/OpenVoice" target="_blank">OpenVoice v2</a>
          &nbsp;·&nbsp; faster-whisper &nbsp;·&nbsp; MeloTTS &nbsp;·&nbsp;
          SpeechBrain &nbsp;·&nbsp; WORLD vocoder<br>
          <span style="opacity:0.5;font-size:0.76rem">Made with ♥ for the hackathon</span>
        </div>
        """)

        # ── Event handlers ───────────────────────────────────────────────────
        def on_run(audio, accent_label, whisper_model, device, use_vad, use_dur):
            if audio is None:
                return None, "⚠️  Please upload or record audio first."
            accent_pair = ACCENT_CHOICES.get(accent_label, "indian_american")
            return _run_pipeline(
                audio_path=audio,
                accent_pair=accent_pair,
                whisper_model=whisper_model,
                device=device,
                apply_vad=use_vad,
                match_duration=use_dur,
            )

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

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="voice-bridge Gradio Web UI")
    parser.add_argument("--host",  default="0.0.0.0",  help="Host to bind")
    parser.add_argument("--port",  default=7860, type=int, help="Port")
    parser.add_argument("--share", action="store_true",   help="Public Gradio share link")
    parser.add_argument("--debug", action="store_true",   help="Gradio debug mode")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True,
    )
