#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import io
import json
import math
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data"
SYMBOL = "BTCUSDT"
INTERVAL = "4h"
FETCH_START = pd.Timestamp("2024-08-01T00:00:00Z")
TEST_START = pd.Timestamp("2026-01-01T00:00:00Z")
CUTOFF = pd.Timestamp("2026-07-30T00:00:00Z")
SHOCK_THRESHOLD = 0.49053003411162827
GAP_THRESHOLD = -0.0031137634157309413
EXPECTED_2025_PHYSICAL = 51
EXPECTED_2025_ACTIONED = 25
EXPECTED_2025_SUCCESS = 20
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def download(url: str, retries: int = 6) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "SBC-GANN-R16-2026-HOLDOUT/1.1"})
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except Exception as exc:
            last = exc
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(str(last))


def archive_specs(venue: str) -> list[tuple[str, str]]:
    root = "spot" if venue == "spot" else "futures/um"
    specs: list[tuple[str, str]] = []
    for period in pd.period_range("2024-08", "2026-06", freq="M"):
        stamp = str(period)
        name = f"{SYMBOL}-{INTERVAL}-{stamp}.zip"
        specs.append((stamp, f"{BASE}/{root}/monthly/klines/{SYMBOL}/{INTERVAL}/{name}"))
    for day in pd.date_range("2026-07-01", "2026-07-29", freq="D"):
        stamp = day.strftime("%Y-%m-%d")
        name = f"{SYMBOL}-{INTERVAL}-{stamp}.zip"
        specs.append((stamp, f"{BASE}/{root}/daily/klines/{SYMBOL}/{INTERVAL}/{name}"))
    return specs


def parse_archive(data: bytes, venue: str, label: str) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if len(names) != 1:
            raise RuntimeError(f"Unexpected archive members for {venue} {label}: {names}")
        raw = pd.read_csv(zf.open(names[0]), header=None)
    if raw.shape[1] < 12:
        raise RuntimeError(f"Unexpected columns for {venue} {label}: {raw.shape}")
    raw = raw.iloc[:, :12]
    raw.columns = COLS
    if not pd.to_numeric(raw["open_time"], errors="coerce").notna().all():
        raw = raw[pd.to_numeric(raw["open_time"], errors="coerce").notna()].copy()
    return raw


def load_venue(venue: str) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    ledger = []
    for label, url in archive_specs(venue):
        data = download(url)
        digest = hashlib.sha256(data).hexdigest()
        checksum_text = download(url + ".CHECKSUM").decode("utf-8", errors="replace").strip()
        expected = checksum_text.split()[0].lower()
        if digest.lower() != expected:
            raise RuntimeError(f"Checksum mismatch: {url}")
        frame = parse_archive(data, venue, label)
        frames.append(frame)
        ledger.append({"venue": venue, "label": label, "url": url, "sha256": digest, "rows": int(len(frame))})
    df = pd.concat(frames, ignore_index=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="raise")
    df["trades"] = pd.to_numeric(df["trades"], errors="raise").astype("int64")
    open_raw = pd.to_numeric(df["open_time"], errors="raise")
    close_raw = pd.to_numeric(df["close_time"], errors="raise")
    def mixed_datetime(values: pd.Series) -> pd.Series:
        mask_ms = values < 1e14
        result = pd.Series(pd.NaT, index=values.index, dtype="datetime64[ns, UTC]")
        result.loc[mask_ms] = pd.to_datetime(values.loc[mask_ms], unit="ms", utc=True)
        result.loc[~mask_ms] = pd.to_datetime(values.loc[~mask_ms], unit="us", utc=True)
        return result
    df["open_time"] = mixed_datetime(open_raw)
    df["close_time"] = mixed_datetime(close_raw)
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    df = df[(df["open_time"] >= FETCH_START) & (df["close_time"] < CUTOFF)].copy()
    expected_gap = df["open_time"].diff().dropna().eq(pd.Timedelta(hours=4))
    if not bool(expected_gap.all()):
        gaps = df.loc[~df["open_time"].diff().eq(pd.Timedelta(hours=4)), "open_time"].iloc[1:6].astype(str).tolist()
        raise RuntimeError(f"{venue} discontinuities: {gaps}")
    return df, ledger


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
    if n == 0:
        return {"n": 0, "successes": 0, "accuracy": None, "balanced_accuracy": None, "wilson_95": [None, None], "pending": int((~df["target_available"]).sum())}
    acc = k/n
    ba = balanced_accuracy(evaluated["target_dir"].to_numpy(), evaluated["predicted_dir"].to_numpy())
    lo, hi = wilson(k, n)
    return {"n": n, "successes": k, "accuracy": acc, "balanced_accuracy": ba, "wilson_95": [lo, hi], "pending": int((~df["target_available"]).sum())}


def main() -> None:
    out = Path("artifacts/r16_2026_holdout_v0_2")
    out.mkdir(parents=True, exist_ok=True)
    spot, spot_ledger = load_venue("spot")
    perp, perp_ledger = load_venue("perp")
    phys = compute(spot, perp)

    hist2025 = phys[(phys["open_time"] >= pd.Timestamp("2025-01-01T00:00:00Z")) & (phys["open_time"] < TEST_START)]
    hist2025_actioned = hist2025[hist2025["repeat3"] & hist2025["target_available"]]
    reproduction = {"physical_count": int(len(hist2025)), "actioned_count": int(len(hist2025_actioned)), "successes": int(hist2025_actioned["correct"].sum()), "expected": {"physical_count": EXPECTED_2025_PHYSICAL, "actioned_count": EXPECTED_2025_ACTIONED, "successes": EXPECTED_2025_SUCCESS}}
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

    passes_metrics = bool(primary["n"] > 0 and primary["accuracy"] is not None and primary["accuracy"] >= 0.80 and primary["balanced_accuracy"] >= 0.80)
    result = {
        "candidate_id": "R16_PHYSICS_REPEAT_3D_CONTRARIAN_H1_v0_1",
        "status": "2026_YTD_BLIND_HOLDOUT_OPENED",
        "frozen_rule": {"clock": "16:00 UTC H4 open / decision at 20:00 UTC close", "shock_close_min": SHOCK_THRESHOLD, "aligned_taker_gap_max": GAP_THRESHOLD, "repeat_window_days": 3.0, "prediction": "opposite sign of event H4 candle", "horizon": "next H4 candle"},
        "data_window": {"fetch_start": str(FETCH_START), "test_start": str(TEST_START), "cutoff_exclusive": str(CUTOFF), "last_included_day": "2026-07-29"},
        "reproduction_2025": reproduction,
        "primary_r16_2026_ytd": primary,
        "secondary_all_physical_2026_ytd": secondary,
        "monthly_primary": monthly_rows,
        "interpretation_gate": {"accuracy_threshold": 0.80, "balanced_accuracy_threshold": 0.80, "minimum_effective_n": 60, "passes_metric_thresholds": passes_metrics, "passes_full_solution_gate": bool(primary["n"] >= 60 and passes_metrics)},
        "governance": {"2026_is_now_exposed": True, "not_strictly_prospective_after_freeze": True, "may_not_be_reused_as_holdout_for_rule_changes": True, "trading_authorized": False},
    }

    cols = ["open_time", "event_sign", "shock_close", "aligned_taker_gap", "gap_prev_days", "repeat3", "predicted_dir", "future_ret_h1", "target_dir", "target_available", "correct", "spot_taker_share", "perp_taker_share", "open", "high", "low", "close", "perp_close"]
    test[cols].to_csv(out / "01_ALL_2026_PHYSICAL_EVENTS.csv", index=False)
    r16[cols].to_csv(out / "02_R16_2026_SIGNALS.csv", index=False)
    pd.DataFrame(monthly_rows).to_csv(out / "03_MONTHLY_RESULTS.csv", index=False)
    (out / "00_RESULT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "05_SOURCE_LEDGER.json").write_text(json.dumps(spot_ledger + perp_ledger, indent=2), encoding="utf-8")

    report = ["# R16 — TEST AVEUGLE 2026 YTD", "", "Fenêtre testée : 1 janvier au 29 juillet 2026 inclus.", f"Reproduction 2025 exacte : **{reproduction['exact']}** ({reproduction['physical_count']} événements physiques, {reproduction['actioned_count']} actionnés, {reproduction['successes']} succès).", "", "## R16 gelé", f"- signaux évalués : {primary['n']}", f"- succès : {primary['successes']}", f"- accuracy : {primary['accuracy']:.4%}" if primary['accuracy'] is not None else "- accuracy : n/a", f"- balanced accuracy : {primary['balanced_accuracy']:.4%}" if primary['balanced_accuracy'] is not None else "- balanced accuracy : n/a", f"- cibles en attente : {primary['pending']}", "", "## Gouvernance", "2026 est désormais exposé. Ce test constitue un holdout historique aveugle à ouverture unique, pas une validation prospective postérieure au gel. Toute modification ultérieure de la règle ne pourra plus utiliser 2026 comme holdout."]
    (out / "04_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    hashes = []
    for p in sorted(out.iterdir()):
        if p.is_file() and p.name != "SHA256SUMS.txt":
            hashes.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (out / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    if not reproduction["exact"]:
        raise SystemExit("Historical 2025 reproduction failed; 2026 result withheld")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
