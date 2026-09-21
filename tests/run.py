#!/usr/bin/env python3
"""Run every test module. No pytest dependency, no network, no API spend."""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
MODULES = ["test_chunking.py", "test_routing.py", "test_ingest.py", "test_app.py"]

failed = []
for name in MODULES:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    result = subprocess.run([sys.executable, str(ROOT / name)])
    if result.returncode:
        failed.append(name)

print(f"\n{'=' * 60}")
print("FAILED: " + ", ".join(failed) if failed else "all modules passed")
sys.exit(1 if failed else 0)
