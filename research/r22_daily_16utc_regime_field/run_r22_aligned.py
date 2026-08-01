#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
source_path = ROOT / "run_r22.py"
source = source_path.read_text(encoding="utf-8")

old = '''    events = daily[
        daily["physics_selected"] & daily["target_available"]
    ].copy()
    expected = set(frozen["timestamp"])
'''
new = '''    evaluation_start = frozen["timestamp"].min()
    evaluation_end = frozen["timestamp"].max()
    events = daily[
        daily["physics_selected"]
        & daily["target_available"]
        & daily["open_time"].between(evaluation_start, evaluation_end)
    ].copy()
    expected = set(frozen["timestamp"])
'''

if old not in source:
    raise RuntimeError("R22 alignment patch target was not found")
patched = source.replace(old, new, 1)
namespace = {
    "__name__": "__main__",
    "__file__": str(source_path),
}
exec(compile(patched, str(source_path), "exec"), namespace, namespace)
