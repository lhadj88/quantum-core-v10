#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

# Compatibility shim only: pandas 2.3 removed Index.sum, while the frozen
# R18 payload may temporarily expose a numeric Index during path slicing.
# Converting that Index to an ndarray preserves the exact intended sum.
if not hasattr(pd.Index, "sum"):
    def _index_sum(self, *args, **kwargs):
        return np.asarray(self).sum(*args, **kwargs)
    pd.Index.sum = _index_sum  # type: ignore[attr-defined]

payload = Path(__file__).resolve().parent / "run_r18.py"
runpy.run_path(str(payload), run_name="__main__")
