from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "event_sign",
    "shock_close",
    "aligned_taker_gap",
    "gap_prev_days",
    "repeat3",
]
DAILY_STATE = [
    "all_rate_7",
    "all_rate_14",
    "all_rate_30",
    "all_rate_60",
    "all_count_7",
    "all_count_14",
    "all_count_30",
    "all_count_60",
    "daily_ret_mean_7",
    "daily_ret_mean_30",
    "daily_ret_vol_7",
    "daily_ret_vol_30",
    "daily_shock_mean_7",
    "daily_shock_mean_30",
    "daily_gap_mean_7",
    "daily_gap_mean_30",
    "pretrend_6",
    "pretrend_24",
    "prevol_24",
    "pre_activity_ratio",
]
EVENT_STATE = [
    "phys_rate_3",
    "phys_rate_5",
    "phys_rate_10",
    "phys_rate_20",
    "phys_count_7d",
    "phys_count_14d",
    "phys_count_30d",
    "days_since_phys",
    "last_phys_success",
    "phys_streak_success",
    "polarity_rate_10",
    "polarity_count_10",
]


def add_daily_state(daily: pd.DataFrame) -> pd.DataFrame:
    output = daily.copy()
    for length in (7, 14, 30, 60):
        minimum = max(3, length // 3)
        output[f"all_rate_{length}"] = (
            output["snapback_h1"].shift(1).rolling(
                length, min_periods=minimum
            ).mean()
        )
        output[f"all_count_{length}"] = (
            output["snapback_h1"].shift(1).rolling(
                length, min_periods=1
            ).count()
        )
    for length in (7, 30):
        minimum = max(3, length // 3)
        output[f"daily_ret_mean_{length}"] = (
            output["ret"].shift(1).rolling(
                length, min_periods=minimum
            ).mean()
        )
        output[f"daily_ret_vol_{length}"] = (
            output["ret"].shift(1).rolling(
                length, min_periods=minimum
            ).std(ddof=1)
        )
        output[f"daily_shock_mean_{length}"] = (
            output["shock_close"].shift(1).rolling(
                length, min_periods=minimum
            ).mean()
        )
        output[f"daily_gap_mean_{length}"] = (
            output["aligned_taker_gap"].shift(1).rolling(
                length, min_periods=minimum
            ).mean()
        )
    return output


def add_event_memory(daily: pd.DataFrame) -> pd.DataFrame:
    output = daily.copy()
    physical_history: list[dict] = []
    last_physical_time: pd.Timestamp | None = None
    success_streak = 0

    for index, row in output.iterrows():
        if last_physical_time is None:
            days_since = np.nan
        else:
            days_since = (
                row["open_time"] - last_physical_time
            ).total_seconds() / 86400
        output.loc[index, "days_since_phys"] = days_since
        output.loc[index, "last_phys_success"] = (
            physical_history[-1]["success"] if physical_history else np.nan
        )
        output.loc[index, "phys_streak_success"] = success_streak

        for count in (3, 5, 10, 20):
            values = [
                item["success"] for item in physical_history[-count:]
            ]
            output.loc[index, f"phys_rate_{count}"] = (
                float(np.mean(values)) if values else np.nan
            )

        for days in (7, 14, 30):
            count = sum(
                (
                    row["open_time"] - item["time"]
                ).total_seconds()
                / 86400
                <= days
                for item in physical_history
            )
            output.loc[index, f"phys_count_{days}d"] = count

        polarity_values = [
            item["success"]
            for item in physical_history
            if item["sign"] == int(row["event_sign"])
        ][-10:]
        output.loc[index, "polarity_rate_10"] = (
            float(np.mean(polarity_values))
            if polarity_values
            else np.nan
        )
        output.loc[index, "polarity_count_10"] = len(polarity_values)

        if bool(row["physics_selected"]) and bool(row["target_available"]):
            success = int(row["snapback_h1"])
            physical_history.append(
                {
                    "time": row["open_time"],
                    "success": success,
                    "sign": int(row["event_sign"]),
                }
            )
            last_physical_time = row["open_time"]
            success_streak = success_streak + 1 if success else 0

    return output


def construct_regime_field(daily: pd.DataFrame) -> pd.DataFrame:
    return add_event_memory(add_daily_state(daily))
