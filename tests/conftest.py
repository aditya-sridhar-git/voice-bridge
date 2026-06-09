"""
pytest configuration — adds repo root to sys.path so that
`from pipeline.stage1_transcribe import ...` works regardless
of how pytest is invoked.
"""
import sys
from pathlib import Path

# Insert repo root (the directory containing pipeline/, evaluation/, etc.)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
