#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from r22_data import build_daily_field, load_venue
from r22_models import (
    THRESHOLDS,
    evaluate,
    feature_sets,
    kmeans_predictions,
    paired_test,
    safe_json,
    supervised_predictions,
)
from r22_state import construct_regime_field

ROOT = Path(__file__).resolve().parent
OUT = Path("artifacts/r22_daily_16utc_regime_field_v0_1")
OUT.mkdir(parents=True, exist_ok=True)


def execute_predictions(
    daily: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    sets = feature_sets()
    rows: list[dict] = []
    years = sorted(events["open_time"].dt.year.unique())
    for year in years:
        train = events[events["open_time"].dt.year != year]
        test = events[events["open_time"].dt.year == year]
        for name, columns in sets.items():
            rows.extend(
                supervised_predictions(
                    train,
                    test,
                    columns,
                    "loyo",
                    name,
                )
            )
        train_daily = daily[daily["open_time"].dt.year != year]
        rows.extend(
            kmeans_predictions(
                train_daily,
                train,
                test,
                "loyo",
            )
        )

    for position in range(60, len(events)):
        train = events.iloc[:position]
        test = events.iloc[position : position + 1]
        cutoff = test.iloc[0]["open_time"]
        for name, columns in sets.items():
            rows.extend(
                supervised_predictions(
                    train,
                    test,
                    columns,
                    "prequential",
                    name,
                )
            )
        rows.extend(
            kmeans_predictions(
                daily[daily["open_time"] < cutoff],
                train,
                test,
                "prequential",
            )
        )
    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame):
    model_rows: list[dict] = []
    selective_rows: list[dict] = []
    yearly_rows: list[dict] = []
    paired_rows: list[dict] = []

    for (period, model), group in predictions.groupby(["period", "model"]):
        summary = evaluate(group)
        summary.update(
            {
                "period": period,
                "model": model,
                "universe": len(group),
            }
        )
        model_rows.append(summary)

        for year, yearly in group.groupby("year"):
            yearly_summary = evaluate(yearly)
            yearly_summary.update(
                {
                    "period": period,
                    "model": model,
                    "year": int(year),
                }
            )
            yearly_rows.append(yearly_summary)

        for threshold in THRESHOLDS:
            selected = group[group["confidence"] >= threshold]
            if len(selected):
                selected_summary = evaluate(selected)
                yearly_balanced: list[float] = []
                for _, subset in selected.groupby("year"):
                    value = evaluate(subset).get("balanced_accuracy")
                    if value is not None and np.isfinite(value):
                        yearly_balanced.append(value)
                paired = paired_test(selected)
            else:
                selected_summary = {
                    "n": 0,
                    "successes": 0,
                    "accuracy": None,
                    "balanced_accuracy": None,
                }
                yearly_balanced = []
                paired = {
                    "n": 0,
                    "model_accuracy": None,
                    "baseline_accuracy": None,
                    "paired_difference": None,
                    "mcnemar_b": 0,
                    "mcnemar_c": 0,
                    "mcnemar_p": 1.0,
                }
            selected_summary.update(
                {
                    "period": period,
                    "model": model,
                    "threshold": threshold,
                    "coverage": len(selected) / len(group),
                    "years": selected["year"].nunique(),
                    "min_year_ba": (
                        min(yearly_balanced) if yearly_balanced else None
                    ),
                }
            )
            selective_rows.append(selected_summary)
            paired.update(
                {
                    "period": period,
                    "model": model,
                    "threshold": threshold,
                }
            )
            paired_rows.append(paired)

    return (
        pd.DataFrame(model_rows),
        pd.DataFrame(selective_rows),
        pd.DataFrame(yearly_rows),
        pd.DataFrame(paired_rows),
    )


def candidate_gate(selective: pd.DataFrame) -> dict:
    passing: list[dict] = []
    for _, row in selective.iterrows():
        values = [
            row["accuracy"],
            row["balanced_accuracy"],
            row["n"],
            row["coverage"],
            row["years"],
            row["min_year_ba"],
        ]
        if not all(pd.notna(value) for value in values):
            continue
        if (
            row["accuracy"] >= 0.80
            and row["balanced_accuracy"] >= 0.80
            and row["n"] >= 60
            and row["coverage"] >= 0.15
            and row["years"] >= 4
            and row["min_year_ba"] >= 0.65
        ):
            passing.append(row.to_dict())
    return {
        "candidate_family": "R22_DAILY_16UTC_REGIME_FIELD_v0_1",
        "status": (
            "CANDIDATE_FOUND_DEVELOPMENT_ONLY"
            if passing
            else "NO_CANDIDATE_GATE_FAILED"
        ),
        "passing_rows": passing,
        "gate": {
            "accuracy_min": 0.80,
            "balanced_accuracy_min": 0.80,
            "minimum_predictions": 60,
            "minimum_coverage": 0.15,
            "minimum_years": 4,
            "minimum_year_balanced_accuracy": 0.65,
        },
        "governance": {
            "through_2026_07_29_exposed": True,
            "trading_authorized": False,
        },
    }


def main() -> None:
    source_ledger: list[dict] = []
    spot = load_venue("spot", source_ledger)
    perp = load_venue("perp", source_ledger)
    daily = construct_regime_field(build_daily_field(spot, perp))

    frozen = pd.read_csv(ROOT / "01_EVENT_TARGETS.csv")
    frozen["timestamp"] = pd.to_datetime(frozen["timestamp"], utc=True)
    events = daily[
        daily["physics_selected"] & daily["target_available"]
    ].copy()
    expected = set(frozen["timestamp"])
    reconstructed = set(events["open_time"])
    audit = {
        "expected_events": len(expected),
        "reconstructed_events": len(reconstructed),
        "exact_calendar": expected == reconstructed,
        "missing": [str(item) for item in sorted(expected - reconstructed)],
        "extra": [str(item) for item in sorted(reconstructed - expected)],
    }
    if not audit["exact_calendar"]:
        raise RuntimeError(f"R22 physical calendar mismatch: {audit}")

    predictions = execute_predictions(daily, events)
    model_summary, selective, yearly, paired = summarize_predictions(predictions)
    gate = candidate_gate(selective)
    best = selective.sort_values(
        ["balanced_accuracy", "accuracy", "n"],
        ascending=False,
        na_position="last",
    ).head(30)
    result = {
        "id": "R22_DAILY_16UTC_REGIME_FIELD_v0_1",
        "status": gate["status"],
        "audit": audit,
        "daily_opportunities": len(daily),
        "physical_events": len(events),
        "feature_sets": {
            name: len(columns)
            for name, columns in feature_sets().items()
        },
        "model_summary": model_summary.to_dict("records"),
        "best_selective": best.to_dict("records"),
        "gate": gate,
    }

    daily.to_csv(OUT / "01_ALL_DAILY_16UTC_STATE.csv", index=False)
    events.to_csv(OUT / "02_PHYSICAL_EVENTS_WITH_STATE.csv", index=False)
    predictions.to_csv(OUT / "03_OUT_OF_SAMPLE_PREDICTIONS.csv", index=False)
    model_summary.to_csv(OUT / "04_MODEL_SUMMARY.csv", index=False)
    selective.to_csv(OUT / "05_SELECTIVE_CURVES.csv", index=False)
    yearly.to_csv(OUT / "06_YEARLY_RESULTS.csv", index=False)
    paired.to_csv(OUT / "07_PAIRED_TESTS.csv", index=False)
    (OUT / "08_SOURCE_LEDGER.json").write_text(
        json.dumps(safe_json(source_ledger), indent=2),
        encoding="utf-8",
    )
    (OUT / "09_CALENDAR_AUDIT.json").write_text(
        json.dumps(safe_json(audit), indent=2),
        encoding="utf-8",
    )
    (OUT / "10_CANDIDATE_GATE_DECISION.json").write_text(
        json.dumps(safe_json(gate), indent=2),
        encoding="utf-8",
    )
    (OUT / "00_RESULT.json").write_text(
        json.dumps(safe_json(result), indent=2),
        encoding="utf-8",
    )
    report = [
        "# R22 — Daily 16UTC Regime Field",
        "",
        f"Status: **{gate['status']}**",
        "",
        f"- daily opportunities: {len(daily)}",
        f"- physical events: {len(events)}",
        f"- exact frozen calendar: {audit['exact_calendar']}",
        "",
        "## Out-of-sample models",
        "",
        model_summary.to_markdown(index=False),
        "",
        "## Decision",
        "The daily latent-regime field is development-only. All observations through 2026-07-29 are exposed; no output authorizes trading.",
    ]
    (OUT / "11_MASTER_REPORT.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    checksums = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
            )
    (OUT / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            safe_json(
                {
                    "status": gate["status"],
                    "audit": audit,
                    "models": model_summary.to_dict("records"),
                    "best": best.head(10).to_dict("records"),
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
