from __future__ import annotations

import subprocess
from pathlib import Path


def run_command(command: list[str], *, cwd: Path, dry_run: bool = False) -> list[str]:
    print("[command] " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, check=True)
    return command
