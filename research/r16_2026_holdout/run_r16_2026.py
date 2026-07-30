#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SPOT_URL = "https://api.binance.com/api/v3/klines"
PERP_URL = "https://fapi.binance.com/fapi/v1/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "4h"
STEP_MS = 4 * 60 * 60 * 1000
FETCH_START = pd.Timestamp("2024-01-01T00:00:00Z")
TEST_START = pd.Timestamp("2026-01-01T00:00:00Z")
CUTOFF = pd.Timestamp("2026-07-30T20:00:00Z")
SHOCK_THRESHOLD = 0.49053003411162827
GAP_THRESHOLD = -0.0031137634157309413
EXPECTED_2025_PHYSICAL = 51
EXPECTED_2025_ACTIONED = 10
EXPECTED_2025_SUCCESS = 8


def ts_ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def fetch_klines(url: str, start: pd.Timestamp, end: pd.Timestamp, limit: int) -> list[list]:
    cursor = ts_ms(start)
    end_ms = ts_ms(end) - 1
    rows: list[list] = []
    while cursor <= end_ms:
        params = {"symbol": SYMBOL, "interval": INTERVAL, "startTime": cursor, "endTime": end_ms, "limit": limit}
        req = urllib.request.Request(f"{url}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "SBC-GANN-R16-2026-HOLDOUT/1.0"})
        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    batch = json.loads(response.read().decode("utf-8"))
                if isinstance(batch, dict):
                    raise RuntimeError(f"Binance API error: {batch}")
                break
            except Exception as exc:
                last_exc = exc
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(str(last_exc))
        if not batch:
            break
        rows.extend(batch)
        next_cursor = int(batch[-1][0]) + STEP_MS
        if next_cursor <= cursor:
            raise RuntimeError("Pagination did not advance")
        cursor = next_cursor
        if len(batch) < limit:
            break
        time.sleep(0.08)
    dedup = {int(r[0]): r for r in rows}
    return [dedup[k] for k in sorted(dedup)]


def frame(rows: list[list], venue: str) -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    df["trades"] = pd.to_numeric(df["trades"], errors="raise").astype("int64")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    expected = df["open_time"].diff().dropna().eq(pd.Timedelta(hours=4))
    if not bool(expected.all()):
        bad = df.loc[~df["open_time"].diff().eq(pd.Timedelta(hours=4)), "open_time"].head().tolist()
        raise RuntimeError(f"{venue} discontinuities: {bad}")
    if (df["close_time"] >= CUTOFF).any():
        df = df[df["close_time"] < CUTOFF].copy()
    return df


def rolling_rank_last(values: np.ndarray) -> float:
    if len(values) == 0 or not np.isfinite(values[-1]):
        return np.nan
    return float(np.sum(values <= values[-1]) / len(values))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recalls = []
    for cls in (0, 1):
        mask = y_true == cls
        if mask.any():
            recalls.append(float(np.mean(y_pred[mask] == cls)))
    return float(np.mean(recalls)) if recalls else float("nan")


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z*z/n
    center = (p + z*z/(2*n)) / den
    half = z * math.sqrt((p*(1-p) + z*z/(4*n))/n) / den
    return center-half, center+half


def compute(spot: pd.DataFrame, perp: pd.DataFrame) -> pd.DataFrame:
    s = spot.copy()
    s["spot_ret"] = np.log(s["close"]).diff()
    s["v6"] = s["spot_ret"].rolling(6, min_periods=6).std(ddof=1)
    s["v24"] = s["spot_ret"].rolling(24, min_periods=24).std(ddof=1)
    denom = s["high"] - s["low"]
    s["cloc"] = np.where(denom > 0, (2*s["close"] - s["high"] - s["low"]) / denom, 0.0)
    s["event_sign"] = np.sign(s["spot_ret"]).astype("float64")
    s["shock_norm"] = s["spot_ret"].abs() / s["v6"]
    s["shock_rank720"] = s["shock_norm"].rolling(720, min_periods=720).apply(rolling_rank_last, raw=True)
    s["shock_close"] = np.maximum(s["event_sign"] * s["cloc"], 0.0) * s["shock_rank720"]
    s["spot_taker_share"] = s["taker_buy_base"] / s["volume"]
    s["future_open_time"] = s["open_time"].shift(-1)
    s["future_close_time"] = s["close_time"].shift(-1)
    s["future_ret_h1"] = s["spot_ret"].shift(-1)

    p = perp[["open_time", "close", "volume", "trades", "taker_buy_base"]].copy()
    p = p.rename(columns={"close": "perp_close", "volume": "perp_volume", "trades": "perp_trades", "taker_buy_base": "perp_taker_buy_base"})
    p["perp_taker_share"] = p["perp_taker_buy_base"] / p["perp_volume"]
    x = s.merge(p[["open_time", "perp_close", "perp_volume", "perp_trades", "perp_taker_share"]], on="open_time", how="left", validate="one_to_one")
    if x["perp_taker_share"].isna().any():
        raise RuntimeError("Missing perp bars after join")
    x["aligned_taker_gap"] = x["event_sign"] * (x["perp_taker_share"] - x["spot_taker_share"])
    x["is_16utc"] = x["open_time"].dt.hour.eq(16)
    x["physics_selected"] = x["is_16utc"] & x["shock_close"].ge(SHOCK_THRESHOLD) & x["aligned_taker_gap"].le(GAP_THRESHOLD)
    phys = x[x["physics_selected"]].copy().sort_values("open_time")
    phys["gap_prev_days"] = phys["open_time"].diff().dt.total_seconds() / 86400.0
    phys["repeat3"] = phys["gap_prev_days"].le(3.0)
    phys["predicted_dir"] = np.where(phys["event_sign"] < 0, 1, 0).astype("int64")
    phys["target_available"] = phys["future_open_time"].eq(phys["open_time"] + pd.Timedelta(hours=4)) & phys["future_close_time"].lt(CUTOFF) & phys["future_ret_h1"].notna() & phys["future_ret_h1"].ne(0)
    phys["target_dir"] = np.where(phys["future_ret_h1"] > 0, 1, 0).astype("int64")
    phys["correct"] = phys["predicted_dir"].eq(phys["target_dir"])
    return phys


def summarize(df: pd.DataFrame) -> dict:
    evaluated = df[df["target_available"]].copy()
    n = len(evaluated)
    k = int(evaluated["correct"].sum())
    acc = k/n if n else float("nan")
    ba = balanced_accuracy(evaluated["target_dir"].to_numpy(), evaluated["predicted_dir"].to_numpy()) if n else float("nan")
    lo, hi = wilson(k, n)
    return {"n": n, "successes": k, "accuracy": acc, "balanced_accuracy": ba, "wilson_95": [lo, hi], "pending": int((~df["target_available"]).sum())}


def main() -> None:
    out = Path("artifacts/r16_2026_holdout_v0_1")
    out.mkdir(parents=True, exist_ok=True)
    spot_rows = fetch_klines(SPOT_URL, FETCH_START, CUTOFF, 1000)
    perp_rows = fetch_klines(PERP_URL, FETCH_START, CUTOFF, 1500)
    spot = frame(spot_rows, "spot")
    perp = frame(perp_rows, "perp")
    phys = compute(spot, perp)

    hist2025 = phys[(phys["open_time"] >= pd.Timestamp("2025-01-01T00:00:00Z")) & (phys["open_time"] < TEST_START)]
    hist2025_actioned = hist2025[hist2025["repeat3"] & hist2025["target_available"]]
    reproduction = {
        "physical_count": int(len(hist2025)),
        "actioned_count": int(len(hist2025_actioned)),
        "successes": int(hist2025_actioned["correct"].sum()),
        "expected": {"physical_count": EXPECTED_2025_PHYSICAL, "actioned_count": EXPECTED_2025_ACTIONED, "successes": EXPECTED_2025_SUCCESS},
    }
    reproduction["exact"] = reproduction["physical_count"] == EXPECTED_2025_PHYSICAL and reproduction["actioned_count"] == EXPECTED_2025_ACTIONED and reproduction["successes"] == EXPECTED_2025_SUCCESS

    test = phys[(phys["open_time"] >= TEST_START) & (phys["open_time"] < CUTOFF)].copy()
    r16 = test[test["repeat3"]].copy()
    primary = summarize(r16)
    secondary = summarize(test)
    monthly_rows = []
    if not r16.empty:
        r16["month"] = r16["open_time"].dt.strftime("%Y-%m")
        for month, g in r16.groupby("month", sort=True):
            monthly_rows.append({"month": month, **summarize(g)})

    result = {
        "candidate_id": "R16_PHYSICS_REPEAT_3D_CONTRARIAN_H1_v0_1",
        "status": "2026_YTD_BLIND_HOLDOUT_OPENED",
        "frozen_rule": {"clock": "16:00 UTC H4 open / decision at 20:00 UTC close", "shock_close_min": SHOCK_THRESHOLD, "aligned_taker_gap_max": GAP_THRESHOLD, "repeat_window_days": 3.0, "prediction": "opposite sign of event H4 candle", "horizon": "next H4 candle"},
        "data_window": {"fetch_start": str(FETCH_START), "test_start": str(TEST_START), "cutoff_exclusive_close_time": str(CUTOFF), "last_scored_event_must_have_target_close_before": str(CUTOFF)},
        "reproduction_2025": reproduction,
        "primary_r16_2026_ytd": primary,
        "secondary_all_physical_2026_ytd": secondary,
        "monthly_primary": monthly_rows,
        "interpretation_gate": {"accuracy_threshold": 0.80, "balanced_accuracy_threshold": 0.80, "minimum_effective_n": 60, "passes_metric_thresholds": bool(primary["n"] > 0 and primary["accuracy"] >= 0.80 and primary["balanced_accuracy"] >= 0.80), "passes_full_solution_gate": bool(primary["n"] >= 60 and primary["accuracy"] >= 0.80 and primary["balanced_accuracy"] >= 0.80)},
        "governance": {"2026_is_now_exposed": True, "not_strictly_prospective_after_freeze": True, "may_not_be_reused_as_holdout_for_rule_changes": True, "trading_authorized": False},
    }

    cols = ["open_time", "event_sign", "shock_close", "aligned_taker_gap", "gap_prev_days", "repeat3", "predicted_dir", "future_ret_h1", "target_dir", "target_available", "correct", "spot_taker_share", "perp_taker_share", "open", "high", "low", "close", "perp_close"]
    test[cols].to_csv(out / "01_ALL_2026_PHYSICAL_EVENTS.csv", index=False)
    r16[cols].to_csv(out / "02_R16_2026_SIGNALS.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(out / "03_MONTHLY_RESULTS.csv", index=False)
    (out / "00_RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")

    report = ["# R16 — TEST 2026 YTD", "", f"Cutoff: {CUTOFF.isoformat()}.", f"Reproduction 2025 exact: **{reproduction['exact']}** ({reproduction['physical_count']} physical, {reproduction['actioned_count']} actioned, {reproduction['successes']} successes).", "", "## Primary frozen R16", f"- evaluated: {primary['n']}", f"- successes: {primary['successes']}", f"- accuracy: {primary['accuracy']:.4%}" if primary['n'] else "- accuracy: n/a", f"- balanced accuracy: {primary['balanced_accuracy']:.4%}" if primary['n'] else "- balanced accuracy: n/a", f"- pending target: {primary['pending']}", "", "## Governance", "2026 is now exposed. This is a one-time blind historical holdout, not a strictly prospective post-freeze validation. No rule change may use 2026 and still call it holdout evidence."]
    (out / "04_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    hashes = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            hashes.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    if not reproduction["exact"]:
        raise SystemExit("Historical 2025 reproduction failed; 2026 result withheld")
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
