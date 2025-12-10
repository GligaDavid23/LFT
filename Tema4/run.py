#!/usr/bin/env python3
from __future__ import annotations

import os, subprocess, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEMA3_DIR = SCRIPT_DIR.parent / "Tema3"


def run_bison_tables() -> None:
    build_script = TEMA3_DIR / "build_all_lr1.py"
    if not build_script.exists():
        print("Lipseste Tema3/build_all_lr1.py; sar peste generare.")
        return
    env = os.environ.copy()
    env["SKIP_LR1"] = "1"
    print(">>> Rulez Bison prin Tema3/build_all_lr1.py ...")
    try:
        subprocess.run([sys.executable, str(build_script)], check=True, cwd=TEMA3_DIR, env=env)
    except Exception as exc:
        print(f"Eroare la generarea tabelelor: {exc}")


def main() -> None:
    run_bison_tables()
    import translator

    translator.main()


if __name__ == "__main__":
    main()
