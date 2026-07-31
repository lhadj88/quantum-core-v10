from __future__ import annotations

import csv
import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd

BASE_URL = "https://data.binance.vision/data"
SYMBOL = "BTCUSDT"
INTERVAL = "4h"
FETCH_START = pd.Timestamp("2020-08-01T00:00:00Z")
TEST_END = pd.Timestamp("2026-07-30T00:00:00Z")
SHOCK_THRESHOLD = 0.49053003411162827
GAP_THRESHOLD = -0.0031137634157309413


def get_bytes(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(6):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "SBC-GANN-R22/1.0"},
            )
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except Exception as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise
            if attempt == 5:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(str(last))


def verified_archive(url: str, ledger: list[dict]) -> bytes:
    blob = get_bytes(url)
    checksum_text = get_bytes(url + ".CHECKSUM").decode(
        "utf-8", errors="replace"
    ).strip()
    expected = checksum_text.split()[0].lower()
    actual = hashlib.sha256(blob).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch: {url}")
    ledger.append(
        {
            "url": url,
            "sha256": actual,
            "size_bytes": len(blob),
            "checksum_match": True,
        }
    )
    return blob


def decode_timestamp(values: pd.Series) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce")
    units = np.where(raw.abs() >= 1e15, "us", "ms")
    output = pd.Series(
        pd.NaT,
        index=values.index,
        dtype="datetime64[ns, UTC]",
    )
    for unit in ("ms", "us"):
        mask = (units == unit) & raw.notna()
        if mask.any():
            output.loc[mask] = pd.to_datetime(
                raw.loc[mask],
                unit=unit,
                utc=True,
                errors="coerce",
            )
    return output


def parse_kline_archive(blob: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        csv_name = next(
            name for name in archive.namelist() if name.lower().endswith(".csv")
        )
        text = archive.read(csv_name).decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if rows and rows[0] and not rows[0][0].replace(".", "", 1).isdigit():
        rows = rows[1:]
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    frame = pd.DataFrame(
        [row[:12] for row in rows if len(row) >= 12],
        columns=columns,
    )
    frame["open_time"] = decode_timestamp(frame["open_time"])
    frame["close_time"] = decode_timestamp(frame["close_time"])
    numeric = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "taker_buy_base",
        "taker_buy_quote",
        "trades",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.dropna(
            subset=["open_time", "open", "high", "low", "close", "volume"]
        )
        .sort_values("open_time")
        .drop_duplicates("open_time")
    )


def load_venue(venue: str, ledger: list[dict]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    root = "spot" if venue == "spot" else "futures/um"
    for period in pd.period_range("2020-08", "2026-06", freq="M"):
        month = str(period)
        filename = f"{SYMBOL}-{INTERVAL}-{month}.zip"
        url = (
            f"{BASE_URL}/{root}/monthly/klines/"
            f"{SYMBOL}/{INTERVAL}/{filename}"
        )
        frames.append(parse_kline_archive(verified_archive(url, ledger)))
    for day in pd.date_range("2026-07-01", "2026-07-30", freq="D"):
        date_text = day.strftime("%Y-%m-%d")
        filename = f"{SYMBOL}-{INTERVAL}-{date_text}.zip"
        url = (
            f"{BASE_URL}/{root}/daily/klines/"
            f"{SYMBOL}/{INTERVAL}/{filename}"
        )
        try:
            frames.append(parse_kline_archive(verified_archive(url, ledger)))
        except urllib.error.HTTPError as exc:
            if exc.code == 404 and day >= pd.Timestamp("2026-07-30"):
                continue
            raise
    frame = (
        pd.concat(frames, ignore_index=True)
        .sort_values("open_time")
        .drop_duplicates("open_time")
    )
    frame = frame[
        (frame["open_time"] >= FETCH_START)
        & (frame["open_time"] < TEST_END)
    ].reset_index(drop=True)
    differences = frame["open_time"].diff().dropna()
    bad = differences[differences != pd.Timedelta(hours=4)]
    if len(bad):
        raise RuntimeError(f"{venue} discontinuities: {bad.head().to_dict()}")
    return frame


def rank_last(values: np.ndarray) -> float:
    if not len(values) or not np.isfinite(values[-1]):
        return np.nan
    return float(np.sum(values <= values[-1]) / len(values))


def build_daily_field(spot: pd.DataFrame, perp: pd.DataFrame) -> pd.DataFrame:
    source = spot.copy()
    source["ret"] = np.log(source["close"]).diff()
    source["v6"] = source["ret"].rolling(6, min_periods=6).std(ddof=1)
    source["shock_norm"] = source["ret"].abs() / source["v6"]
    source["rank720"] = source["shock_norm"].rolling(
        720, min_periods=720
    ).apply(rank_last, raw=True)
    candle_range = source["high"] - source["low"]
    source["cloc"] = np.where(
        candle_range > 0,
        (2 * source["close"] - source["high"] - source["low"])
        / candle_range,
        0.0,
    )
    source["event_sign"] = np.sign(source["ret"])
    source["shock_close"] = np.maximum(
        source["event_sign"] * source["cloc"], 0.0
    ) * source["rank720"]
    source["spot_taker_share"] = source["taker_buy_base"] / source["volume"]
    source["future_ret_h1"] = source["ret"].shift(-1)
    source["future_open_time"] = source["open_time"].shift(-1)
    source["pretrend_6"] = np.log(
        source["open"] / source["close"].shift(6)
    )
    source["pretrend_24"] = np.log(
        source["open"] / source["close"].shift(24)
    )
    source["prevol_24"] = source["ret"].shift(1).rolling(
        24, min_periods=24
    ).std(ddof=1)
    source["pre_activity_ratio"] = (
        source["volume"].shift(1).rolling(6, min_periods=6).mean()
        / source["volume"].shift(1).rolling(42, min_periods=42).mean()
    )

    derivative = perp[["open_time", "close", "volume", "taker_buy_base"]].copy()
    derivative = derivative.rename(
        columns={
            "close": "perp_close",
            "volume": "perp_volume",
            "taker_buy_base": "perp_taker_buy_base",
        }
    )
    derivative["perp_taker_share"] = (
        derivative["perp_taker_buy_base"] / derivative["perp_volume"]
    )
    merged = source.merge(
        derivative[["open_time", "perp_close", "perp_taker_share"]],
        on="open_time",
        how="inner",
        validate="one_to_one",
    )
    merged["aligned_taker_gap"] = merged["event_sign"] * (
        merged["perp_taker_share"] - merged["spot_taker_share"]
    )
    daily = merged[merged["open_time"].dt.hour.eq(16)].copy()
    daily = daily.sort_values("open_time").reset_index(drop=True)
    daily["target_dir_h1"] = (daily["future_ret_h1"] > 0).astype(int)
    daily["snapback_h1"] = (
        np.sign(daily["future_ret_h1"]) == -daily["event_sign"]
    ).astype(int)
    daily["target_available"] = (
        daily["future_open_time"].eq(
            daily["open_time"] + pd.Timedelta(hours=4)
        )
        & daily["future_ret_h1"].notna()
        & daily["future_ret_h1"].ne(0)
    )
    daily["physics_selected"] = (
        daily["shock_close"].ge(SHOCK_THRESHOLD)
        & daily["aligned_taker_gap"].le(GAP_THRESHOLD)
    )
    physical_times = daily.loc[daily["physics_selected"], "open_time"]
    gaps = (physical_times - physical_times.shift(1)).dt.total_seconds() / 86400
    daily["gap_prev_days"] = np.nan
    daily.loc[daily["physics_selected"], "gap_prev_days"] = gaps.to_numpy()
    daily["repeat3"] = daily["gap_prev_days"].le(3).fillna(False).astype(int)
    return daily
