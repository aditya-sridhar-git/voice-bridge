"""
scripts/fix_melotts_windows.py
================================
Patches the installed MeloTTS japanese.py to make MeCab optional,
so MeloTTS imports correctly on Windows where the MeCab binary
is not available.

Run this once after installing or reinstalling MeloTTS:
    python scripts/fix_melotts_windows.py

Background:
    MeloTTS's japanese.py hard-crashes at import time if MeCab isn't installed.
    On Windows, mecab-python3 installs the Python wrapper but the MeCab binary
    (and its dictionary mecabrc) may not be available, causing a RuntimeError
    when MeCab.Tagger() is called at module level.

    voice-bridge only uses English, so Japanese support is not needed.
    This patch makes MeCab optional: if it fails to load, Japanese text
    processing is disabled but everything else continues to work.
"""

import sys
import re
from pathlib import Path


def find_japanese_py() -> Path:
    """Find the MeloTTS japanese.py file in the current Python environment."""
    import importlib.util
    spec = importlib.util.find_spec("melo")
    if spec is None:
        raise RuntimeError("MeloTTS (melo) is not installed.")
    melo_dir = Path(spec.origin).parent
    jp_file = melo_dir / "text" / "japanese.py"
    if not jp_file.exists():
        raise FileNotFoundError(f"japanese.py not found at: {jp_file}")
    return jp_file


def patch_japanese_py(jp_file: Path) -> bool:
    """
    Apply patches to make MeCab optional.
    Returns True if patched, False if already patched.
    """
    content = jp_file.read_text(encoding="utf-8")

    # Check if already patched
    if "_MECAB_AVAILABLE" in content:
        print(f"  ✅ Already patched: {jp_file}")
        return False

    # Patch 1: Make MeCab import optional
    old_import = (
        "try:\n"
        "    import MeCab\n"
        "except ImportError as e:\n"
        "    raise ImportError(\"Japanese requires mecab-python3 and unidic-lite.\") from e"
    )
    new_import = (
        "try:\n"
        "    import MeCab\n"
        "    _MECAB_AVAILABLE = True\n"
        "except ImportError:\n"
        "    _MECAB_AVAILABLE = False\n"
        "    MeCab = None"
    )

    if old_import not in content:
        print("  ⚠️  Import block not found in expected format. Manual check needed.")
        return False

    content = content.replace(old_import, new_import)

    # Patch 2: Guard _TAGGER initialization
    old_tagger = "_TAGGER = MeCab.Tagger()"
    new_tagger = (
        "try:\n"
        "    _TAGGER = MeCab.Tagger() if _MECAB_AVAILABLE else None\n"
        "except Exception:\n"
        "    _MECAB_AVAILABLE = False\n"
        "    _TAGGER = None"
    )

    if old_tagger in content:
        content = content.replace(old_tagger, new_tagger)
    else:
        print("  ⚠️  _TAGGER line not found. It may already be patched or changed upstream.")

    jp_file.write_text(content, encoding="utf-8")
    print(f"  ✅ Patched: {jp_file}")
    return True


def verify_import() -> bool:
    """Verify that MeloTTS imports cleanly after patching."""
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", "from melo.api import TTS; print('OK')"],
        capture_output=True, text=True, timeout=60
    )
    if "OK" in result.stdout:
        print("  ✅ MeloTTS imports successfully.")
        return True
    else:
        print(f"  ❌ Import still failing:\n{result.stderr[-800:]}")
        return False


if __name__ == "__main__":
    print("🔧 Patching MeloTTS for Windows compatibility...\n")

    try:
        jp_file = find_japanese_py()
        print(f"  Found: {jp_file}")
    except Exception as e:
        print(f"  ❌ {e}")
        sys.exit(1)

    patched = patch_japanese_py(jp_file)

    print("\n🧪 Verifying import...")
    ok = verify_import()

    if ok:
        print("\n🎉 Done! MeloTTS is ready for English TTS on Windows.")
        print("   Japanese TTS is disabled (MeCab not available), but English/Chinese/Korean work.")
    else:
        print("\n❌ Import still fails. Check the error above.")
        sys.exit(1)
