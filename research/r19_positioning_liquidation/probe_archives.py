#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://data.binance.vision/data/futures/um/daily"
SYMBOL = "BTCUSDT"
DATES = ["2021-02-23", "2023-03-03", "2025-01-07", "2026-01-15"]
TYPES = ["metrics", "bookDepth", "liquidationSnapshot"]
OUT = Path("artifacts/r19_archive_probe_v0_1")
OUT.mkdir(parents=True, exist_ok=True)


def get(url: str) -> bytes:
    last: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SBC-GANN-R19-PROBE/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception as exc:
            last = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code == 404:
                raise
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError(str(last))


rows: list[dict] = []
for dtype in TYPES:
    for date in DATES:
        filename = f"{SYMBOL}-{dtype}-{date}.zip"
        url = f"{BASE}/{dtype}/{SYMBOL}/{filename}"
        rec = {"data_type": dtype, "date": date, "url": url, "status": None}
        try:
            blob = get(url)
            rec["status"] = "available"
            rec["size_bytes"] = len(blob)
            rec["sha256"] = hashlib.sha256(blob).hexdigest()
            try:
                check = get(url + ".CHECKSUM").decode("utf-8", errors="replace").strip()
                rec["checksum_text"] = check
                rec["checksum_match"] = check.split()[0].lower() == rec["sha256"]
            except Exception as exc:
                rec["checksum_text"] = None
                rec["checksum_match"] = None
                rec["checksum_error"] = f"{type(exc).__name__}: {exc}"
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                names = zf.namelist()
                rec["members"] = names
                csv_name = next((n for n in names if n.lower().endswith(".csv")), names[0])
                raw = zf.read(csv_name).decode("utf-8-sig", errors="replace")
                reader = csv.reader(io.StringIO(raw))
                sample = []
                for _, line in zip(range(4), reader):
                    sample.append(line)
                rec["sample_rows"] = sample
                rec["header"] = sample[0] if sample else []
        except urllib.error.HTTPError as exc:
            rec["status"] = f"http_{exc.code}"
            rec["error"] = str(exc)
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(rec)
        print(dtype, date, rec["status"])

(OUT / "00_PROBE.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
with (OUT / "01_STATUS.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["data_type", "date", "status", "size_bytes", "sha256", "checksum_match", "error"], extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

report = ["# R19 archive probe", ""]
for rec in rows:
    report.append(f"## {rec['data_type']} — {rec['date']}")
    report.append(f"- status: {rec['status']}")
    if rec.get("header"):
        report.append(f"- header: `{rec['header']}`")
    if rec.get("checksum_match") is not None:
        report.append(f"- checksum match: {rec['checksum_match']}")
    report.append("")
(OUT / "02_REPORT.md").write_text("\n".join(report), encoding="utf-8")
