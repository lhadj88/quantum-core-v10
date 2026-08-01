from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from r22_state import BASE_FEATURES, DAILY_STATE, EVENT_STATE

THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


def feature_sets() -> dict[str, list[str]]:
    return {
        "base": BASE_FEATURES,
        "daily_state": BASE_FEATURES + DAILY_STATE,
        "event_state": BASE_FEATURES + EVENT_STATE,
        "full_state": BASE_FEATURES + DAILY_STATE + EVENT_STATE,
    }


def fill_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_values = train[columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    test_values = test[columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    train_values[~np.isfinite(train_values)] = np.nan
    test_values[~np.isfinite(test_values)] = np.nan
    for column in range(train_values.shape[1]):
        finite = train_values[
            np.isfinite(train_values[:, column]), column
        ]
        replacement = float(np.median(finite)) if len(finite) else 0.0
        train_values[
            ~np.isfinite(train_values[:, column]), column
        ] = replacement
        test_values[
            ~np.isfinite(test_values[:, column]), column
        ] = replacement
    return train_values, test_values


def classifier() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.5,
                    class_weight="balanced",
                    max_iter=4000,
                    random_state=20260731,
                ),
            ),
        ]
    )


def supervised_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    period: str,
    model_name: str,
) -> list[dict]:
    train_x, test_x = fill_train_test(train, test, columns)
    train_y = train["snapback_h1"].astype(int).to_numpy()
    fitted = classifier()
    fitted.fit(train_x, train_y)
    probabilities = fitted.predict_proba(test_x)[:, 1]
    rows: list[dict] = []
    for position, (_, row) in enumerate(test.iterrows()):
        probability = float(probabilities[position])
        opposite = int(probability >= 0.5)
        event_sign = int(row["event_sign"])
        predicted_direction = int(
            (-event_sign if opposite else event_sign) > 0
        )
        target_direction = int(row["target_dir_h1"])
        baseline_direction = int(-event_sign > 0)
        rows.append(
            {
                "period": period,
                "model": model_name,
                "timestamp": row["open_time"],
                "year": int(row["open_time"].year),
                "predicted_dir": predicted_direction,
                "target_dir": target_direction,
                "correct": int(predicted_direction == target_direction),
                "confidence": 0.5 + abs(probability - 0.5),
                "baseline_correct": int(
                    baseline_direction == target_direction
                ),
                "prob_snapback": probability,
            }
        )
    return rows


def kmeans_predictions(
    train_daily: pd.DataFrame,
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    period: str,
) -> list[dict]:
    daily_x, _ = fill_train_test(
        train_daily,
        train_daily,
        DAILY_STATE,
    )
    fitted = KMeans(
        n_clusters=4,
        random_state=20260731,
        n_init=20,
    ).fit(daily_x)
    _, event_train_x = fill_train_test(
        train_daily,
        train_events,
        DAILY_STATE,
    )
    train_clusters = fitted.predict(event_train_x)
    state_rates: dict[int, float] = {}
    for state in range(4):
        values = train_events["snapback_h1"].to_numpy()[
            train_clusters == state
        ]
        state_rates[state] = float(np.mean(values)) if len(values) else 0.5
    _, event_test_x = fill_train_test(
        train_daily,
        test_events,
        DAILY_STATE,
    )
    test_clusters = fitted.predict(event_test_x)
    rows: list[dict] = []
    for state, (_, row) in zip(test_clusters, test_events.iterrows()):
        probability = state_rates[int(state)]
        opposite = int(probability >= 0.5)
        event_sign = int(row["event_sign"])
        predicted_direction = int(
            (-event_sign if opposite else event_sign) > 0
        )
        target_direction = int(row["target_dir_h1"])
        baseline_direction = int(-event_sign > 0)
        rows.append(
            {
                "period": period,
                "model": "kmeans_daily_state",
                "timestamp": row["open_time"],
                "year": int(row["open_time"].year),
                "predicted_dir": predicted_direction,
                "target_dir": target_direction,
                "correct": int(predicted_direction == target_direction),
                "confidence": 0.5 + abs(probability - 0.5),
                "baseline_correct": int(
                    baseline_direction == target_direction
                ),
                "prob_snapback": probability,
                "cluster": int(state),
            }
        )
    return rows


def evaluate(group: pd.DataFrame) -> dict:
    target = group["target_dir"].astype(int)
    prediction = group["predicted_dir"].astype(int)
    balanced = (
        float(balanced_accuracy_score(target, prediction))
        if target.nunique() > 1
        else None
    )
    return {
        "n": len(group),
        "successes": int(group["correct"].sum()),
        "accuracy": float(group["correct"].mean()),
        "balanced_accuracy": balanced,
    }


def paired_test(group: pd.DataFrame) -> dict:
    model_only = int(
        ((group["correct"] == 1) & (group["baseline_correct"] == 0)).sum()
    )
    baseline_only = int(
        ((group["correct"] == 0) & (group["baseline_correct"] == 1)).sum()
    )
    probability = (
        float(
            binomtest(
                min(model_only, baseline_only),
                n=model_only + baseline_only,
                p=0.5,
            ).pvalue
        )
        if model_only + baseline_only
        else 1.0
    )
    return {
        "n": len(group),
        "model_accuracy": float(group["correct"].mean()),
        "baseline_accuracy": float(group["baseline_correct"].mean()),
        "paired_difference": float(
            group["correct"].mean()
            - group["baseline_correct"].mean()
        ),
        "mcnemar_b": model_only,
        "mcnemar_c": baseline_only,
        "mcnemar_p": probability,
    }


def safe_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value
