#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Compatibility shim only: pandas 2.3 removed Index.sum, while the frozen
# R18 payload may temporarily expose a numeric Index during path slicing.
# Converting that Index to an ndarray preserves the exact intended sum.
if not hasattr(pd.Index, "sum"):
    def _index_sum(self, *args, **kwargs):
        return np.asarray(self).sum(*args, **kwargs)
    pd.Index.sum = _index_sum  # type: ignore[attr-defined]

# Fold-local missing-value compatibility layer.
# Some cross-price features are mathematically undefined when their source
# series is absent or constant. Values are imputed using medians learned only
# from the current training fold. A column that is entirely undefined in that
# training fold receives 0.0. The learned fills are then reused unchanged for
# validation/test rows, preventing target or future leakage.
_original_fit = Pipeline.fit
_original_predict = Pipeline.predict
_original_predict_proba = getattr(Pipeline, "predict_proba", None)
_original_decision_function = getattr(Pipeline, "decision_function", None)
_original_score = Pipeline.score


def _coerce_matrix(X: Any) -> tuple[np.ndarray, tuple[Any, ...]]:
    if isinstance(X, pd.DataFrame):
        arr = X.to_numpy(dtype=float, copy=True)
        return arr, ("dataframe", X.index, X.columns)
    arr = np.asarray(X, dtype=float).copy()
    was_vector = arr.ndim == 1
    if was_vector:
        arr = arr.reshape(-1, 1)
    return arr, ("array", was_vector)


def _restore_matrix(arr: np.ndarray, template: tuple[Any, ...]) -> Any:
    if template[0] == "dataframe":
        return pd.DataFrame(arr, index=template[1], columns=template[2])
    if template[1]:
        return arr.reshape(-1)
    return arr


def _learn_and_apply_fill(X: Any) -> tuple[Any, np.ndarray]:
    arr, template = _coerce_matrix(X)
    arr[~np.isfinite(arr)] = np.nan
    fills = np.zeros(arr.shape[1], dtype=float)
    for col in range(arr.shape[1]):
        finite = arr[np.isfinite(arr[:, col]), col]
        fills[col] = float(np.median(finite)) if finite.size else 0.0
        missing = ~np.isfinite(arr[:, col])
        if missing.any():
            arr[missing, col] = fills[col]
    return _restore_matrix(arr, template), fills


def _apply_fill(X: Any, fills: np.ndarray) -> Any:
    arr, template = _coerce_matrix(X)
    if arr.shape[1] != len(fills):
        raise ValueError(
            f"R18 compatibility fill width mismatch: X has {arr.shape[1]} columns, "
            f"training fold has {len(fills)}"
        )
    for col in range(arr.shape[1]):
        missing = ~np.isfinite(arr[:, col])
        if missing.any():
            arr[missing, col] = fills[col]
    return _restore_matrix(arr, template)


def _pipeline_fit(self: Pipeline, X: Any, y: Any = None, **params: Any) -> Pipeline:
    X_filled, fills = _learn_and_apply_fill(X)
    self._r18_fold_fill_values = fills
    return _original_fit(self, X_filled, y, **params)


def _pipeline_predict(self: Pipeline, X: Any, **params: Any) -> Any:
    fills = getattr(self, "_r18_fold_fill_values", None)
    X_filled = _apply_fill(X, fills) if fills is not None else X
    return _original_predict(self, X_filled, **params)


def _pipeline_predict_proba(self: Pipeline, X: Any, **params: Any) -> Any:
    fills = getattr(self, "_r18_fold_fill_values", None)
    X_filled = _apply_fill(X, fills) if fills is not None else X
    assert _original_predict_proba is not None
    return _original_predict_proba(self, X_filled, **params)


def _pipeline_decision_function(self: Pipeline, X: Any, **params: Any) -> Any:
    fills = getattr(self, "_r18_fold_fill_values", None)
    X_filled = _apply_fill(X, fills) if fills is not None else X
    assert _original_decision_function is not None
    return _original_decision_function(self, X_filled, **params)


def _pipeline_score(self: Pipeline, X: Any, y: Any = None, sample_weight: Any = None) -> float:
    fills = getattr(self, "_r18_fold_fill_values", None)
    X_filled = _apply_fill(X, fills) if fills is not None else X
    return _original_score(self, X_filled, y, sample_weight=sample_weight)


Pipeline.fit = _pipeline_fit  # type: ignore[assignment]
Pipeline.predict = _pipeline_predict  # type: ignore[assignment]
if _original_predict_proba is not None:
    Pipeline.predict_proba = _pipeline_predict_proba  # type: ignore[assignment]
if _original_decision_function is not None:
    Pipeline.decision_function = _pipeline_decision_function  # type: ignore[assignment]
Pipeline.score = _pipeline_score  # type: ignore[assignment]

payload = Path(__file__).resolve().parent / "run_r18.py"
runpy.run_path(str(payload), run_name="__main__")
