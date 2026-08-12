"""Switch only this agent process to the bundle-local Python environment."""

import os
import sys
from pathlib import Path


agent_dir = Path(__file__).resolve().parent
if os.name == "nt":
    python = agent_dir / ".venv" / "Scripts" / "python.exe"
else:
    python = agent_dir / ".venv" / "bin" / "python"

if not python.is_file():
    raise FileNotFoundError(f"Run setup_venv.sh first; agent Python was not found: {python}")

main = agent_dir / "main.py"
os.execv(str(python), [str(python), str(main), *sys.argv[1:]])
