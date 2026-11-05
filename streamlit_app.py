"""Root launcher for RetenSight AI."""

from __future__ import annotations

import runpy
from pathlib import Path

# Run app/Home.py as main entrypoint
home_script = Path(__file__).resolve().parent / "app" / "Home.py"
runpy.run_path(str(home_script), run_name="__main__")

