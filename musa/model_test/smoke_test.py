from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
subprocess.check_call([sys.executable, str(REPO_ROOT / "musa/samples/list_ep_devices.py")])
