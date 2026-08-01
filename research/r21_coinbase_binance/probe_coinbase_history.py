#!/usr/bin/env python3
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("artifacts/r21_coinbase_history_probe_v0_1")
OUT.mkdir(parents=True, exist_ok=True)
ENDPOINT = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
EVENTS = [
    "2021-02-23T16:00:00Z",
    "2023-03-03T16:00:00Z",
    "2025-01-07T16:00:00Z",
    "2026-01-15T16:00:00Z",
]


def fetch(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SBC-GANN-R21-PROBE/1.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()
        except Exception as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(str(last))


rows = []
for event_text in EVENTS:
    event = dt.datetime.fromisoformat(event_text.replace("Z", "+00:00"))
    start = event - dt.timedelta(hours=4)
    end = event + dt.timedelta(hours=4)
    params = urllib.parse.urlencode({
        "granularity": 300,
        "start": start.isoformat().replace("+00:00", "Z"),
        "end": end.isoformat().replace("+00:00", "Z"),
    })
    url = ENDPOINT + "?" + params
    rec = {"event": event_text, "start": start.isoformat(), "end": end.isoformat(), "url": url}
    try:
        blob = fetch(url)
        payload = json.loads(blob.decode("utf-8"))
        if not isinstance(payload, list):
            raise RuntimeError(f"Unexpected payload: {payload}")
        candles = sorted(payload, key=lambda x: int(x[0]))
        rec.update({
            "status": "available",
            "response_sha256": hashlib.sha256(blob).hexdigest(),
            "candles": len(candles),
            "first_timestamp": int(candles[0][0]) if candles else None,
            "last_timestamp": int(candles[-1][0]) if candles else None,
            "sample_first": candles[0] if candles else None,
            "sample_last": candles[-1] if candles else None,
        })
    except Exception as exc:
        rec.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    rows.append(rec)
    print(event_text, rec["status"], rec.get("candles"))
    time.sleep(0.15)

(OUT / "00_PROBE.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
with (OUT / "01_STATUS.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["event", "status", "candles", "first_timestamp", "last_timestamp", "response_sha256", "error"], extrasaction="ignore")
    writer.writeheader(); writer.writerows(rows)
report = ["# R21 Coinbase historical candles probe", ""]
for rec in rows:
    report += [f"## {rec['event']}", f"- status: {rec['status']}", f"- candles: {rec.get('candles')}", f"- sha256: `{rec.get('response_sha256')}`", ""]
(OUT / "02_REPORT.md").write_text("\n".join(report), encoding="utf-8")
if not all(r["status"] == "available" and int(r.get("candles") or 0) >= 90 for r in rows):
    raise SystemExit("Coinbase historical probe gate failed")
