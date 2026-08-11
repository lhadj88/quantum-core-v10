from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from r34f16_fetch_plain import AUDIT, get_daily, get_funding, coinbase_5m

WINDOWS = [5,15,30,60]
METRIC_WINDOWS = [15,30,60]
OUT = Path('out_r34f16')
OUT.mkdir(exist_ok=True)


def finite(x):
    try: return bool(np.isfinite(float(x)))
    except Exception: return False


def req_mean(values):
    values = list(values)
    return float(np.mean(values)) if values and all(finite(x) for x in values) else np.nan


def slice_window(d, time_col, end, minutes):
    if d.empty:
        return d
    return d[(d[time_col] > end-pd.Timedelta(minutes=minutes)) & (d[time_col] <= end)].copy()


def return_stats(d, end, minutes):
    x = slice_window(d, 'close_time', end, minutes)
    if len(x) < max(1, minutes//5):
        return np.nan, np.nan, np.nan, np.nan
    ret = float(x.iloc[-1].close/x.iloc[0].open - 1.0)
    rng = float(x.high.max()/x.low.min() - 1.0)
    volume = float(x.volume.sum())
    efficiency = float(ret/max(rng,1e-12))
    return ret, rng, volume, efficiency


def metric_stats(d, end, minutes, col):
    x = slice_window(d, 'ts', end, minutes)[['ts',col]].dropna()
    if len(x) < max(2, int(math.floor(minutes/5*0.8))):
        return np.nan, np.nan, np.nan, np.nan
    start = float(x.iloc[0][col]); last = float(x.iloc[-1][col])
    logchg = float(math.log(last/start)) if start>0 and last>0 else np.nan
    xx = (x.ts-x.ts.iloc[0]).dt.total_seconds().to_numpy()/60.0
    yy = np.log(np.maximum(x[col].to_numpy(float),1e-18))
    slope = float(np.polyfit(xx,yy,1)[0]) if len(x)>1 and np.ptp(xx)>0 else np.nan
    return start, last, logchg, slope


def level_stats(times, values, end, minutes):
    d = pd.DataFrame({'ts':times,'v':values}).dropna()
    x = slice_window(d, 'ts', end, minutes)
    if len(x) < max(1, minutes//5):
        return np.nan, np.nan, np.nan
    start = float(x.iloc[0].v); last = float(x.iloc[-1].v)
    return start, last, last-start


def build(row):
    node,event_id,year,event_sign,side,stratum_side,landmark_t = row
    event = pd.Timestamp(event_id)
    event = event.tz_convert('UTC') if event.tzinfo else event.tz_localize('UTC')
    landmark = event + pd.Timedelta(hours=4, minutes=float(landmark_t))
    dates = [(landmark-pd.Timedelta(minutes=60)).normalize(), landmark.normalize()]

    spot = get_daily('spot',dates)
    mark = get_daily('mark',dates)
    index = get_daily('index',dates)
    premium = get_daily('premium',dates)
    metrics = get_daily('metrics',dates)
    coinbase = coinbase_5m(landmark)
    funding_rate, funding_delta, funding_age = get_funding(landmark)

    rec = {'node':node,'event_id':event_id,'year':int(year),'event_sign':int(event_sign),'side':side,'stratum_side':stratum_side,'landmark_t':float(landmark_t),'landmark_ts':landmark.isoformat()}
    side_direction = float(event_sign) * (1.0 if side=='ALIGNED' else -1.0)
    rec['side_direction'] = side_direction

    if not mark.empty and not index.empty and not spot.empty and not premium.empty:
        joined = mark[['close_time','close']].rename(columns={'close':'mark'})
        joined = joined.merge(index[['close_time','close']].rename(columns={'close':'index'}),on='close_time')
        joined = joined.merge(spot[['close_time','close']].rename(columns={'close':'spot'}),on='close_time')
        joined = joined.merge(premium[['close_time','close']].rename(columns={'close':'premium'}),on='close_time')
        joined['mark_gap'] = (joined.mark-joined['index'])/joined['index']
        joined['spot_gap'] = (joined.spot-joined['index'])/joined['index']
    else:
        joined = pd.DataFrame()

    for w in WINDOWS:
        sr, srange, svol, seff = return_stats(spot,landmark,w)
        cr, crange, cvol, ceff = return_stats(coinbase,landmark,w) if not coinbase.empty else (np.nan,)*4
        rec[f'spot_ret_{w}'] = sr
        rec[f'cb_ret_{w}'] = cr
        rec[f'cb_range_{w}'] = crange
        rec[f'cb_volume_{w}'] = cvol
        rec[f'cb_eff_{w}'] = ceff
        rec[f'cb_minus_spot_{w}'] = cr-sr if finite(cr) and finite(sr) else np.nan
        for col in ['mark_gap','spot_gap','premium']:
            if joined.empty:
                start = last = change = np.nan
            else:
                start,last,change = level_stats(joined.close_time,joined[col],landmark,w)
            rec[f'{col}_end_{w}'] = last
            rec[f'{col}_chg_{w}'] = change

    metric_cols = ['sum_open_interest','sum_open_interest_value','count_toptrader_long_short_ratio','sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio']
    for w in METRIC_WINDOWS:
        for col in metric_cols:
            start,last,logchg,slope = metric_stats(metrics,landmark,w,col) if not metrics.empty else (np.nan,)*4
            rec[f'metric_{col}_end_{w}'] = last
            rec[f'metric_{col}_logchg_{w}'] = logchg
            rec[f'metric_{col}_slope_{w}'] = slope

    rec['funding_rate'] = funding_rate
    rec['funding_delta'] = funding_delta
    rec['funding_age_hours'] = funding_age

    # Exact frozen W1-W7 definitions from run_r33_r35.py.
    rec['W1'] = req_mean(side_direction*rec[f'spot_ret_{w}']*rec[f'metric_sum_open_interest_logchg_{w}'] for w in METRIC_WINDOWS)
    rec['W2'] = req_mean(side_direction*req_mean([rec[f'metric_sum_toptrader_long_short_ratio_logchg_{w}'],rec[f'metric_count_long_short_ratio_logchg_{w}'],rec[f'metric_sum_taker_long_short_vol_ratio_logchg_{w}']]) for w in METRIC_WINDOWS)
    rec['W3'] = req_mean(side_direction*req_mean([rec[f'mark_gap_end_{w}'],rec[f'spot_gap_end_{w}'],rec[f'premium_end_{w}']]) for w in WINDOWS)
    rec['W4'] = req_mean(side_direction*req_mean([rec[f'mark_gap_chg_{w}'],rec[f'spot_gap_chg_{w}'],rec[f'premium_chg_{w}']]) for w in WINDOWS)
    rec['W5'] = req_mean(side_direction*rec[f'cb_minus_spot_{w}'] for w in WINDOWS)
    rec['W6'] = req_mean(side_direction*rec[f'cb_eff_{w}'] for w in METRIC_WINDOWS)
    rec['W7'] = side_direction*funding_delta if finite(funding_delta) else np.nan
    rec['winner_primary_complete'] = int(all(finite(rec[x]) for x in ['W1','W2','W3','W4','W5','W6']))
    return rec


def main():
    rows = json.loads(Path('r34f16_rows.json').read_text())
    output = []; errors = []
    for i,row in enumerate(rows,1):
        print(f'FEATURE {i}/{len(rows)} {row[0]} {row[1]} t={row[6]}', flush=True)
        try:
            output.append(build(row))
        except Exception as e:
            errors.append({'node':row[0],'event_id':row[1],'landmark_t':row[6],'error':f'{type(e).__name__}: {e}'})
            print('ERROR',repr(e),flush=True)
    features = pd.DataFrame(output)
    features.to_csv(OUT/'dynamic_winner_features.csv',index=False)
    audit = pd.DataFrame(AUDIT)
    audit.to_csv(OUT/'source_audit.csv',index=False)
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2))
    hashes = {str(r.url):str(r.sha256) for r in audit.itertuples() if getattr(r,'status','')=='OK' and pd.notna(getattr(r,'sha256',None))}
    (OUT/'source_sha256.json').write_text(json.dumps(hashes,indent=2,sort_keys=True))
    summary = {'requested_rows':len(rows),'completed_rows':len(features),'errors':len(errors),'winner_primary_complete':int(features.winner_primary_complete.sum()) if len(features) else 0,'audit_rows':len(audit),'integrity_ok_rows':int(audit.integrity.fillna(False).sum()) if len(audit) else 0}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    if errors:
        raise SystemExit(2)

if __name__ == '__main__':
    main()
