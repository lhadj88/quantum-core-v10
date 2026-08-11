from __future__ import annotations
import hashlib, io, re, time, zipfile
from pathlib import Path
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import requests

BASE = 'https://data.binance.vision/data'
CACHE = Path('cache_r34f16_plain')
CACHE.mkdir(exist_ok=True)
AUDIT = []
FRAME = {}
SESSION = requests.Session()
SESSION.headers.update({'User-Agent':'SBC-GANN-R34F16/1.0 research-only'})
KLINE_COLS = ['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']


def http_get(url, timeout=90, retries=5):
    err = None
    for i in range(retries):
        try:
            r = SESSION.get(url, timeout=timeout)
            if r.status_code == 200:
                return r
            err = RuntimeError(f'HTTP {r.status_code}')
            if r.status_code == 404:
                break
        except Exception as e:
            err = e
        time.sleep(1 + i)
    raise RuntimeError(f'GET failed {url}: {err}')


def fetch_archive(url, source, key):
    cache = CACHE / (hashlib.sha256(url.encode()).hexdigest() + '.zip')
    official = None
    try:
        if cache.exists():
            raw = cache.read_bytes()
        else:
            raw = http_get(url).content
            cache.write_bytes(raw)
        got = hashlib.sha256(raw).hexdigest()
        try:
            text = http_get(url + '.CHECKSUM', timeout=30, retries=2).text.strip()
            token = text.split()[0] if text else ''
            if re.fullmatch(r'[0-9a-fA-F]{64}', token):
                official = token.lower()
        except Exception:
            pass
        if official and official != got:
            raise RuntimeError(f'checksum mismatch {got} != {official}')
        AUDIT.append({'kind':'archive','source':source,'key':str(key),'url':url,'sha256':got,'official_sha256':official,'integrity':True,'status':'OK','error':''})
        return raw
    except Exception as e:
        AUDIT.append({'kind':'archive','source':source,'key':str(key),'url':url,'sha256':None,'official_sha256':official,'integrity':False,'status':'ERROR','error':f'{type(e).__name__}: {e}'})
        return None


def read_zip_csv(raw, header='infer'):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if not names:
            raise RuntimeError('no CSV member')
        return pd.read_csv(z.open(names[0]), header=header)


def epoch_utc(values):
    x = pd.to_numeric(values, errors='coerce')
    unit = 'us' if float(x.dropna().median()) > 1e14 else 'ms'
    return pd.to_datetime(x, unit=unit, utc=True, errors='coerce')


def source_url(source, date):
    ds = pd.Timestamp(date).strftime('%Y-%m-%d')
    if source == 'spot':
        return f'{BASE}/spot/daily/klines/BTCUSDT/5m/BTCUSDT-5m-{ds}.zip'
    names = {'mark':'markPriceKlines','index':'indexPriceKlines','premium':'premiumIndexKlines'}
    if source in names:
        name = names[source]
        return f'{BASE}/futures/um/daily/{name}/BTCUSDT/5m/BTCUSDT-5m-{ds}.zip'
    if source == 'metrics':
        return f'{BASE}/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-{ds}.zip'
    raise KeyError(source)


def parse_kline(raw):
    d = read_zip_csv(raw, header=None).iloc[:,:12].copy()
    d.columns = KLINE_COLS
    d['open_time'] = epoch_utc(d.open_time)
    d['close_time'] = epoch_utc(d.close_time)
    for c in ['open','high','low','close','volume','quote_volume','trades','taker_buy_base','taker_buy_quote']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.sort_values('close_time').drop_duplicates('close_time', keep='last')


def parse_metrics(raw):
    d = read_zip_csv(raw)
    d['ts'] = pd.to_datetime(d['create_time'], utc=True, errors='coerce')
    cols = ['sum_open_interest','sum_open_interest_value','count_toptrader_long_short_ratio','sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio']
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.sort_values('ts').drop_duplicates('ts', keep='last')


def get_daily(source, dates):
    parts = []
    parser = parse_metrics if source == 'metrics' else parse_kline
    for date in sorted(set(pd.Timestamp(x).strftime('%Y-%m-%d') for x in dates)):
        ck = (source, date)
        if ck not in FRAME:
            raw = fetch_archive(source_url(source,date), source, date)
            FRAME[ck] = parser(raw) if raw is not None else pd.DataFrame()
        if not FRAME[ck].empty:
            parts.append(FRAME[ck])
    if not parts:
        return pd.DataFrame()
    tc = 'ts' if source == 'metrics' else 'close_time'
    return pd.concat(parts, ignore_index=True).sort_values(tc).drop_duplicates(tc, keep='last')


def funding_frame(month):
    ck = ('funding', month)
    if ck in FRAME:
        return FRAME[ck]
    url = f'{BASE}/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-{month}.zip'
    raw = fetch_archive(url, 'funding', month)
    if raw is None:
        FRAME[ck] = pd.DataFrame()
        return FRAME[ck]
    d = read_zip_csv(raw)
    tc = next((c for c in d.columns if 'time' in c.lower()), d.columns[0])
    rc = next((c for c in d.columns if 'funding' in c.lower() and 'time' not in c.lower() and 'interval' not in c.lower()), d.columns[-1])
    num = pd.to_numeric(d[tc], errors='coerce')
    if num.notna().mean() > 0.8 and float(num.dropna().median()) > 1e11:
        ts = epoch_utc(num)
    else:
        ts = pd.to_datetime(d[tc], utc=True, errors='coerce')
    FRAME[ck] = pd.DataFrame({'ts':ts,'funding_rate':pd.to_numeric(d[rc],errors='coerce')}).dropna().sort_values('ts').drop_duplicates('ts',keep='last')
    return FRAME[ck]


def get_funding(landmark):
    months = sorted({(landmark-pd.DateOffset(months=1)).strftime('%Y-%m'), landmark.strftime('%Y-%m')})
    parts = [funding_frame(m) for m in months]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return np.nan, np.nan, np.nan
    d = pd.concat(parts).drop_duplicates('ts').sort_values('ts')
    x = d[d.ts <= landmark].tail(2)
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    last = float(x.iloc[-1].funding_rate)
    delta = last - float(x.iloc[-2].funding_rate)
    age = float((landmark - x.iloc[-1].ts).total_seconds()/3600)
    return last, delta, age


def coinbase_5m(landmark):
    start = (landmark-pd.Timedelta(minutes=65)).isoformat().replace('+00:00','Z')
    end = landmark.isoformat().replace('+00:00','Z')
    url = 'https://api.exchange.coinbase.com/products/BTC-USD/candles?' + urlencode({'granularity':300,'start':start,'end':end})
    try:
        r = http_get(url, timeout=60, retries=5)
        got = hashlib.sha256(r.content).hexdigest()
        obj = r.json()
        AUDIT.append({'kind':'api','source':'coinbase','key':landmark.isoformat(),'url':url,'sha256':got,'official_sha256':None,'integrity':True,'status':'OK','error':''})
    except Exception as e:
        AUDIT.append({'kind':'api','source':'coinbase','key':landmark.isoformat(),'url':url,'sha256':None,'official_sha256':None,'integrity':False,'status':'ERROR','error':f'{type(e).__name__}: {e}'})
        return pd.DataFrame()
    time.sleep(0.12)
    rows = []
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x,list) and len(x)>=6:
                rows.append({'open_time':pd.to_datetime(int(x[0]),unit='s',utc=True),'low':float(x[1]),'high':float(x[2]),'open':float(x[3]),'close':float(x[4]),'volume':float(x[5])})
    if not rows:
        return pd.DataFrame()
    d = pd.DataFrame(rows)
    d['close_time'] = d.open_time + pd.Timedelta(minutes=5)
    return d[(d.close_time<=landmark)&(d.close_time>landmark-pd.Timedelta(minutes=65))].sort_values('close_time').drop_duplicates('close_time')
