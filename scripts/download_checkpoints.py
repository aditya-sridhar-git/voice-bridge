"""
download_checkpoints.py
========================
One-command script to download OpenVoice v2 checkpoints from HuggingFace
into checkpoints_v2/converter/ (the path expected by stage4_synthesize.py).

Usage:
    python scripts/download_checkpoints.py

Requirements:
    pip install huggingface_hub
"""

import sys
import os
from pathlib import Path

# Ensure repo root is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_DIR = REPO_ROOT / "checkpoints_v2"
CONVERTER_DIR  = CHECKPOINT_DIR / "converter"

REPO_ID = "myshell-ai/OpenVoiceV2"


def download_openvoice_v2():
    """Download OpenVoice v2 checkpoints from HuggingFace."""

    # Check huggingface_hub
    try:
        from huggingface_hub import snapshot_download, hf_hub_download
    except ImportError:
        print("❌  huggingface_hub not installed.")
        print("    Run:  pip install huggingface_hub")
        sys.exit(1)

    print(f"📥  Downloading OpenVoice v2 from HuggingFace: {REPO_ID}")
    print(f"    Target directory: {CHECKPOINT_DIR}\n")

    try:
        local_dir = snapshot_download(
            repo_id=REPO_ID,
            local_dir=str(CHECKPOINT_DIR),
            ignore_patterns=["*.md", "*.txt", "*.png", "*.jpg"],
        )
        print(f"\n✅  Download complete!")
        print(f"    Checkpoints saved to: {local_dir}")
    except Exception as e:
        print(f"\n❌  Download failed: {e}")
        print("\nManual alternative:")
        print(f"  1. Visit https://huggingface.co/{REPO_ID}")
        print(f"  2. Download the 'converter/' folder")
        print(f"  3. Place it at: {CONVERTER_DIR}")
        sys.exit(1)

    # Verify the converter/ subdirectory exists
    if CONVERTER_DIR.exists():
        files = list(CONVERTER_DIR.iterdir())
        print(f"\n    Converter directory contains {len(files)} files:")
        for f in sorted(files):
            print(f"      - {f.name}")
    else:
        print(f"\n⚠️   Converter subdirectory not found at {CONVERTER_DIR}")
        print("    Check the HuggingFace repo structure manually.")

    # Check for config.json and checkpoint.pth
    config_ok = (CONVERTER_DIR / "config.json").exists()
    ckpt_ok   = (CONVERTER_DIR / "checkpoint.pth").exists()

    print()
    print(f"    config.json   : {'✅' if config_ok else '❌ MISSING'}")
    print(f"    checkpoint.pth: {'✅' if ckpt_ok   else '❌ MISSING'}")

    if config_ok and ckpt_ok:
        print("\n🎉  OpenVoice v2 is ready! Voice cloning is now active in Stage 4.")
    else:
        print("\n⚠️   One or more checkpoint files are missing.")
        print("    Voice cloning will be skipped (Stage 4 will output raw MeloTTS).")


if __name__ == "__main__":
    download_openvoice_v2()
