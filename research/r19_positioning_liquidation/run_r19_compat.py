#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import runpy
from pathlib import Path
from typing import Any

_original_dumps = json.dumps


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _safe_dumps(obj: Any, *args: Any, **kwargs: Any) -> str:
    # Rendering-only compatibility: preserve all finite scientific results and
    # encode mathematically undefined values as JSON null rather than NaN.
    return _original_dumps(_json_safe(obj), *args, **kwargs)


json.dumps = _safe_dumps  # type: ignore[assignment]
payload = Path(__file__).resolve().parent / "run_r19.py"
runpy.run_path(str(payload), run_name="__main__")
