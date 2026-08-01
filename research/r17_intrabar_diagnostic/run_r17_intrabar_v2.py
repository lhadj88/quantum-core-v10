#!/usr/bin/env python3
import base64, gzip, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent
bootstrap = (ROOT / "run_r17_intrabar.py").read_text(encoding="utf-8")
parts = re.findall(r'b64decode\("([A-Za-z0-9+/=]+)"\)', bootstrap)
if len(parts) < 2:
    raise RuntimeError("Unable to extract embedded R17 source and events")
events_b64, source_b64 = parts[0], parts[1]
(ROOT / "events_2021_2026.csv").write_bytes(gzip.decompress(base64.b64decode(events_b64)))
source = gzip.decompress(base64.b64decode(source_b64)).decode("utf-8")
source = source.replace("import urllib.request\n", "import urllib.request\nimport urllib.error\n")
source = source.replace(
    "def parse_archive(blob: bytes, venue: str) -> pd.DataFrame:\n",
    '''def daily_archive_url(venue: str, day: str) -> str:\n    if venue == "spot":\n        return f"{BASE}/spot/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{day}.zip"\n    return f"{BASE}/futures/um/daily/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{day}.zip"\n\n\ndef parse_archive(blob: bytes, venue: str) -> pd.DataFrame:\n''',
)
old = '''            url = archive_url(venue, month)\n            checksum_url = url + ".CHECKSUM"\n            blob = get_bytes(url)\n            checksum_text = get_bytes(checksum_url).decode("utf-8", errors="replace").strip()\n            expected = checksum_text.split()[0].lower()\n            actual = sha256_bytes(blob)\n            if expected != actual:\n                raise RuntimeError(f"Checksum mismatch {url}: {expected} != {actual}")\n            f = parse_archive(blob, venue)\n            frames[venue].append(f)\n            ledger.append({"venue": venue, "month": month, "url": url, "sha256": actual, "rows": int(len(f)), "first": str(f["open_time"].min()), "last": str(f["open_time"].max())})\n'''
new = '''            url = archive_url(venue, month)\n            try:\n                blob = get_bytes(url)\n                checksum_text = get_bytes(url + ".CHECKSUM").decode("utf-8", errors="replace").strip()\n                expected = checksum_text.split()[0].lower()\n                actual = sha256_bytes(blob)\n                if expected != actual:\n                    raise RuntimeError(f"Checksum mismatch {url}: {expected} != {actual}")\n                f = parse_archive(blob, venue)\n                frames[venue].append(f)\n                ledger.append({"venue": venue, "month": month, "granularity": "monthly", "url": url, "sha256": actual, "rows": int(len(f)), "first": str(f["open_time"].min()), "last": str(f["open_time"].max())})\n            except urllib.error.HTTPError as exc:\n                if exc.code != 404:\n                    raise\n                days = sorted(events.loc[events["timestamp"].dt.strftime("%Y-%m") == month, "timestamp"].dt.strftime("%Y-%m-%d").unique())\n                for day in days:\n                    durl = daily_archive_url(venue, day)\n                    dblob = get_bytes(durl)\n                    checksum_text = get_bytes(durl + ".CHECKSUM").decode("utf-8", errors="replace").strip()\n                    expected = checksum_text.split()[0].lower()\n                    actual = sha256_bytes(dblob)\n                    if expected != actual:\n                        raise RuntimeError(f"Checksum mismatch {durl}: {expected} != {actual}")\n                    f = parse_archive(dblob, venue)\n                    frames[venue].append(f)\n                    ledger.append({"venue": venue, "month": month, "day": day, "granularity": "daily_fallback", "url": durl, "sha256": actual, "rows": int(len(f)), "first": str(f["open_time"].min()), "last": str(f["open_time"].max())})\n'''
if old not in source:
    raise RuntimeError("Monthly loader block not found for patching")
source = source.replace(old, new)
g = {"__name__": "__main__", "__file__": str(__file__)}
exec(compile(source, str(__file__), "exec"), g, g)
