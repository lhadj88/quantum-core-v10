#!/usr/bin/env python3
import csv, io, json, math, time, hashlib, urllib.request, zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

# Frozen COMPETITION_SECOND_025 event-disjoint transport rows.
# Fields: event_id, year, event_sign, side, landmark_t.
ROWS = [["2021-08-07T16:00:00Z",2021,1,"ALIGNED",10],["2021-09-09T16:00:00Z",2021,-1,"ALIGNED",325],["2021-10-30T16:00:00Z",2021,1,"ALIGNED",60],["2021-12-09T16:00:00Z",2021,-1,"ALIGNED",80],["2021-12-16T16:00:00Z",2021,-1,"ALIGNED",130],["2021-12-19T16:00:00Z",2021,-1,"ALIGNED",20],["2022-01-30T16:00:00Z",2022,-1,"ALIGNED",15],["2022-02-10T16:00:00Z",2022,-1,"ALIGNED",25],["2022-03-31T16:00:00Z",2022,-1,"ALIGNED",230],["2022-05-11T16:00:00Z",2022,-1,"ALIGNED",270],["2022-06-18T16:00:00Z",2022,-1,"ALIGNED",85],["2022-06-21T16:00:00Z",2022,-1,"ALIGNED",120],["2023-08-12T16:00:00Z",2023,-1,"ALIGNED",110],["2024-01-28T16:00:00Z",2024,-1,"ALIGNED",165],["2024-02-27T16:00:00Z",2024,1,"ALIGNED",700],["2024-03-04T16:00:00Z",2024,1,"ALIGNED",40],["2024-05-05T16:00:00Z",2024,-1,"ALIGNED",205],["2024-05-27T16:00:00Z",2024,-1,"ALIGNED",350],["2024-06-06T16:00:00Z",2024,-1,"ALIGNED",40],["2024-07-03T16:00:00Z",2024,-1,"ALIGNED",65],["2024-07-24T16:00:00Z",2024,-1,"ALIGNED",95],["2024-09-22T16:00:00Z",2024,1,"ALIGNED",120],["2024-10-23T16:00:00Z",2024,1,"ALIGNED",35],["2024-10-24T16:00:00Z",2024,1,"ALIGNED",75],["2024-11-03T16:00:00Z",2024,1,"ALIGNED",40],["2024-11-26T16:00:00Z",2024,-1,"ALIGNED",20],["2024-12-11T16:00:00Z",2024,1,"ALIGNED",70],["2025-01-09T16:00:00Z",2025,-1,"ALIGNED",85],["2025-01-12T16:00:00Z",2025,-1,"ALIGNED",75],["2025-06-21T16:00:00Z",2025,-1,"ALIGNED",85],["2025-06-28T16:00:00Z",2025,-1,"ALIGNED",95],["2025-07-01T16:00:00Z",2025,-1,"ALIGNED",15],["2025-08-02T16:00:00Z",2025,-1,"ALIGNED",280],["2025-11-16T16:00:00Z",2025,-1,"ALIGNED",70],["2025-12-28T16:00:00Z",2025,-1,"ALIGNED",70],["2026-03-22T16:00:00Z",2026,-1,"ALIGNED",70],["2026-04-01T16:00:00Z",2026,-1,"ALIGNED",310],["2026-05-20T16:00:00Z",2026,1,"ALIGNED",15],["2026-07-08T16:00:00Z",2026,1,"ALIGNED",310]]

# Target-blind causal R28B state for rows where an R28B decision exists.
# event_id -> first_side, decision (0=CLEAR,1=ALERT), decision_time minutes.
R28 = {"2023-08-12T16:00:00Z":["COUNTER",0,55],"2024-01-28T16:00:00Z":["COUNTER",0,85],"2024-05-05T16:00:00Z":["ALIGNED",0,65],"2024-05-27T16:00:00Z":["COUNTER",0,70],"2024-07-03T16:00:00Z":["COUNTER",0,55],"2024-07-24T16:00:00Z":["COUNTER",0,50],"2024-09-22T16:00:00Z":["ALIGNED",0,70],"2024-10-23T16:00:00Z":["COUNTER",1,20],"2024-11-03T16:00:00Z":["COUNTER",1,35],"2025-01-12T16:00:00Z":["COUNTER",0,55],"2025-06-21T16:00:00Z":["COUNTER",0,80],"2025-06-28T16:00:00Z":["ALIGNED",0,60],"2025-07-01T16:00:00Z":["ALIGNED",1,10],"2025-08-02T16:00:00Z":["COUNTER",1,25],"2026-04-01T16:00:00Z":["COUNTER",0,135],"2026-07-08T16:00:00Z":["COUNTER",0,100]}

BASE = "https://data.binance.vision/data"
CACHE = Path("cache")
OUT = Path("out")
CACHE.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
HASHES = {}


def parse_iso(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def ms(dt):
    return int(dt.timestamp() * 1000)


def normalize_ts(x):
    v = int(float(x))
    # Spot archive timestamps from 2025 onward may be microseconds.
    if v > 10**14:
        v //= 1000
    return v


def daterange(d0, d1):
    d = d0
    while d <= d1:
        yield d
        d += timedelta(days=1)


def archive_url(market, day):
    ds = day.strftime("%Y-%m-%d")
    if market == "spot":
        return f"{BASE}/spot/daily/klines/BTCUSDT/30m/BTCUSDT-30m-{ds}.zip"
    return f"{BASE}/futures/um/daily/klines/BTCUSDT/30m/BTCUSDT-30m-{ds}.zip"


def download(url):
    name = url.rsplit("/", 1)[-1]
    market = "spot" if "/spot/" in url else "um"
    p = CACHE / f"{market}_{name}"
    if not p.exists():
        last = None
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"User-Agent":"r34f13-research/1.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    b = r.read()
                p.write_bytes(b)
                break
            except Exception as e:
                last = e
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"download failed {url}: {last}")
    b = p.read_bytes()
    HASHES[url] = hashlib.sha256(b).hexdigest()
    return b


def read_day(market, day):
    url = archive_url(market, day)
    raw = download(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        members = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise RuntimeError(f"unexpected zip members {url}: {z.namelist()}")
        text = z.read(members[0]).decode("utf-8")
    rows = []
    for r in csv.reader(io.StringIO(text)):
        if not r:
            continue
        try:
            ot = normalize_ts(r[0]); ct = normalize_ts(r[6])
        except Exception:
            continue
        rows.append({
            "open_time":ot, "open":float(r[1]), "close":float(r[4]),
            "volume":float(r[5]), "close_time":ct, "taker_buy_base":float(r[9])
        })
    return rows


def load_window(market, start_ms, end_ms):
    d0 = datetime.fromtimestamp(start_ms/1000, tz=timezone.utc).date()
    d1 = datetime.fromtimestamp(end_ms/1000, tz=timezone.utc).date()
    allrows = []
    for d in daterange(d0, d1):
        allrows.extend(read_day(market, d))
    by_ot = {r["open_time"]:r for r in allrows if r["open_time"] >= start_ms and r["close_time"] <= end_ms}
    return [by_ot[k] for k in sorted(by_ot)]


def dct_basis(T, d):
    n = np.arange(T, dtype=float)
    cols = []
    for k in range(d):
        v = np.cos(np.pi * (n + 0.5) * k / T)
        v *= (1/np.sqrt(T)) if k == 0 else np.sqrt(2/T)
        cols.append(v)
    return np.stack(cols, axis=1)


def eff_rank(x):
    x = np.asarray(x, float)
    x = x[x > 0]
    if len(x) == 0:
        return 0.0
    ss = float(np.square(x).sum())
    return 0.0 if ss == 0 else float(x.sum()**2 / ss)


def cert(delta, d):
    delta = np.asarray(delta, float)
    phi = dct_basis(len(delta), d)
    H = phi.T @ (delta[:,None] * phi)
    H = (H + H.T) / 2
    eig = np.linalg.eigvalsh(H)
    scale = float(np.max(np.abs(eig))) if len(eig) else 0.0
    tol = 100*np.finfo(float).eps*max(1,len(eig))*scale if scale else 0.0
    score = eff_rank(eig[eig > tol]) - eff_rank(-eig[eig < -tol])
    return int(np.sign(score)), float(score)


def r34_panel(delta):
    cs = [cert(delta, d) for d in (3,4,6)]
    signs = [x[0] for x in cs]
    pred = 1 if all(s == 1 for s in signs) else (-1 if all(s == -1 for s in signs) else 0)
    return pred, [x[1] for x in cs]


def side_rel(s):
    return 1 if s == "ALIGNED" else -1


def r28_direct(event_id, event_sign, landmark_t):
    if event_id not in R28:
        return 0
    first_side, decision, decision_time = R28[event_id]
    if decision_time > landmark_t:
        return 0
    rel = side_rel(first_side)
    support_rel = rel if decision == 0 else -rel
    return int(support_rel * event_sign)


def process_row(row):
    event_id, year, event_sign, side, landmark_t = row
    event_dt = parse_iso(event_id)
    landmark = event_dt + timedelta(hours=4, minutes=landmark_t)
    # Last close strictly before exact landmark.
    floormin = (landmark.minute // 30) * 30
    floor = landmark.replace(minute=floormin, second=0, microsecond=0)
    end_dt = floor - timedelta(milliseconds=1)
    start_ms = ms(event_dt); end_ms = ms(end_dt)

    spot = load_window("spot", start_ms, end_ms)
    perp = load_window("um", start_ms, end_ms)
    smap = {r["open_time"]:r for r in spot}
    pmap = {r["open_time"]:r for r in perp}
    common = sorted(set(smap) & set(pmap))
    if len(common) < 6:
        raise RuntimeError(f"{event_id} {landmark_t}: only {len(common)} common bars")
    s = [smap[t] for t in common]; p = [pmap[t] for t in common]

    fs = np.array([2*r["taker_buy_base"]/r["volume"] - 1 if r["volume"] else 0.0 for r in s])
    fp = np.array([2*r["taker_buy_base"]/r["volume"] - 1 if r["volume"] else 0.0 for r in p])
    dominant = fs*np.abs(fs) + fp*np.abs(fp)
    flow_vote, spectral_scores = r34_panel(dominant)

    spot_ret = math.log(s[-1]["close"] / s[0]["open"])
    perp_ret = math.log(p[-1]["close"] / p[0]["open"])
    eta_s = abs(spot_ret) / (float(np.mean(np.abs(fs))) + 1e-12)
    eta_p = abs(perp_ret) / (float(np.mean(np.abs(fp))) + 1e-12)
    ps = int(np.sign(float(fs.sum()))); pp = int(np.sign(float(fp.sum())))
    if abs(eta_s - eta_p) < 1e-15:
        transmission = 0
    else:
        transmission = ps if eta_s > eta_p else pp

    current = int(side_rel(side) * event_sign)
    r28v = r28_direct(event_id, event_sign, landmark_t)
    votes = [v for v in (current, r28v, flow_vote, transmission) if v != 0]
    total = sum(votes)
    majority = int(np.sign(total))
    unanimous = votes[0] if len(votes) >= 3 and len(set(votes)) == 1 else 0

    return {
        "node":"COMPETITION_SECOND_025", "event_id":event_id, "year":year,
        "event_sign":event_sign, "side":side, "landmark_t":landmark_t,
        "bars_common":len(common), "CURRENT_SIDE":current, "R28B_DIRECT":r28v,
        "R34_FLOW_DOMINANT":flow_vote, "TRANSMISSION_LEADER":transmission,
        "MAJORITY":majority, "UNANIMOUS_3PLUS":unanimous,
        "flow_d3":spectral_scores[0], "flow_d4":spectral_scores[1], "flow_d6":spectral_scores[2],
        "spot_pressure_mass":float(fs.sum()), "perp_pressure_mass":float(fp.sum()),
        "eta_spot":eta_s, "eta_perp":eta_p, "spot_net_return":spot_ret, "perp_net_return":perp_ret,
        "start_ms":start_ms, "end_ms":end_ms
    }


def main():
    results=[]
    errors=[]
    for i,row in enumerate(ROWS,1):
        print(f"[{i}/{len(ROWS)}] {row[0]} t={row[4]}", flush=True)
        try:
            results.append(process_row(row))
        except Exception as e:
            errors.append({"event_id":row[0],"error":repr(e)})
            print("ERROR", row[0], repr(e), flush=True)
    fields = list(results[0].keys()) if results else []
    with (OUT/"predictions_targetblind.csv").open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(results)
    (OUT/"errors.json").write_text(json.dumps(errors, indent=2))
    (OUT/"archive_sha256.json").write_text(json.dumps(HASHES, indent=2, sort_keys=True))
    summary={"requested_rows":len(ROWS),"completed_rows":len(results),"errors":len(errors),"unique_archive_files":len(HASHES)}
    (OUT/"runner_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    if errors:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
