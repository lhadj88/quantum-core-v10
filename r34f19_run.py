from pathlib import Path
import io,json,hashlib,time,zipfile
import numpy as np,pandas as pd,requests
import r34f16_fetch_plain as fetch

# Frozen R34F19: last 60m, aligned fully closed 5m spot/perp BTCUSDT bars.
BASE='https://data.binance.vision/data'; OUT=Path('out_r34f19'); OUT.mkdir(exist_ok=True); CACHE=Path('cache_r34f19'); CACHE.mkdir(exist_ok=True); S=requests.Session(); S.headers.update({'User-Agent':'SBC-GANN-R34F19/1.0'}); DIMS=(3,4,6)
_old=fetch.parse_kline
def _pk(raw):
    d=_old(raw)
    if not d.empty:
        d=d.copy(); d['open_time']=d['open_time'].dt.floor('5min'); d['close_time']=d['open_time']+pd.Timedelta(minutes=5)
    return d
fetch.parse_kline=_pk; fetch.FRAME.clear()

def perp_day(date):
    ds=pd.Timestamp(date).strftime('%Y-%m-%d'); url=f'{BASE}/futures/um/daily/klines/BTCUSDT/5m/BTCUSDT-5m-{ds}.zip'; p=CACHE/(hashlib.sha256(url.encode()).hexdigest()+'.zip')
    if p.exists(): raw=p.read_bytes()
    else:
        raw=None
        for i in range(5):
            try:
                r=S.get(url,timeout=90)
                if r.status_code==200: raw=r.content; p.write_bytes(raw); break
            except Exception: pass
            time.sleep(1+i)
        if raw is None: raise RuntimeError(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        n=[x for x in z.namelist() if x.endswith('.csv')][0]; d=pd.read_csv(z.open(n),header=None)
    d=d.iloc[:,:12]; d.columns=['open_time','open','high','low','close','volume','close_time','qv','trades','taker_buy_base','tbq','ignore']; x=pd.to_numeric(d.open_time,errors='coerce'); unit='us' if x.dropna().median()>1e14 else 'ms'; d['open_time']=pd.to_datetime(x,unit=unit,utc=True,errors='coerce').dt.floor('5min')
    for c in ['volume','taker_buy_base']: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d.sort_values('open_time').drop_duplicates('open_time')

def basis(T,d):
    n=np.arange(T,dtype=float); cols=[]
    for k in range(d):
        v=np.cos(np.pi*(n+.5)*k/T); v*=1/np.sqrt(T) if k==0 else np.sqrt(2/T); cols.append(v)
    return np.stack(cols,axis=1)
def erank(x):
    x=np.asarray(x,float); x=x[x>1e-12]
    return 0. if len(x)==0 else float(x.sum()**2/np.square(x).sum())
def cert(x,d):
    P=basis(len(x),d); H=P.T@(x[:,None]*P); H=(H+H.T)/2; e=np.linalg.eigvalsh(H); scale=max(float(np.max(np.abs(e))),1e-300); tol=100*np.finfo(float).eps*max(1,len(e))*scale; s=erank(e[e>tol])-erank(-e[e<-tol]); return int(np.sign(s)),float(s)
def panel(x):
    x=np.asarray(x,float)
    if len(x)<6 or not np.isfinite(x).all(): return 0,[np.nan]*3
    c=[cert(x,d) for d in DIMS]; ss=[z[0] for z in c]; p=1 if all(s==1 for s in ss) else (-1 if all(s==-1 for s in ss) else 0); return p,[z[1] for z in c]

def main():
    rows=json.loads(Path('r34f16_rows.json').read_text()); cache={}; out=[]; errors=[]
    for i,row in enumerate(rows,1):
        node,eid,year,es,side,stratum,t=row; ev=pd.Timestamp(eid); ev=ev.tz_convert('UTC') if ev.tzinfo else ev.tz_localize('UTC'); lm=ev+pd.Timedelta(hours=4,minutes=float(t)); dates=sorted({(lm-pd.Timedelta(minutes=65)).strftime('%Y-%m-%d'),lm.strftime('%Y-%m-%d')})
        try:
            s=[]; p=[]
            for date in dates:
                s.append(fetch.get_daily('spot',[pd.Timestamp(date,tz='UTC')]))
                if date not in cache: cache[date]=perp_day(date)
                p.append(cache[date])
            spot=pd.concat(s).drop_duplicates('open_time'); perp=pd.concat(p).drop_duplicates('open_time'); start=lm-pd.Timedelta(minutes=60)
            spot=spot[(spot.open_time>=start)&(spot.open_time<lm)]; perp=perp[(perp.open_time>=start)&(perp.open_time<lm)]
            J=spot[['open_time','volume','taker_buy_base']].merge(perp[['open_time','volume','taker_buy_base']],on='open_time',suffixes=('_s','_p')).sort_values('open_time')
            fs=2*J.taker_buy_base_s/J.volume_s-1; fp=2*J.taker_buy_base_p/J.volume_p-1
            chans={'SPOT':fs.to_numpy(float),'PERP':fp.to_numpy(float),'CONSENSUS':(fs+fp).to_numpy(float),'DOMINANT':(fs*np.abs(fs)+fp*np.abs(fp)).to_numpy(float),'RELATIVE':(fs-fp).to_numpy(float)}; rec={'node':node,'event_id':eid,'year':year,'event_sign':es,'side':side,'landmark_t':t,'bars':len(J)}; votes=[]
            for name,x in chans.items():
                pv,sc=panel(x); rec[name+'_pred']=pv
                for dd,z in zip(DIMS,sc): rec[f'{name}_d{dd}']=z
                if name!='RELATIVE' and pv: votes.append(pv)
            rec['flow_votes_n']=len(votes); rec['FLOW_CHANNEL_MAJORITY']=int(np.sign(sum(votes))) if len(votes)>=2 else 0; cur=int(es) if side=='ALIGNED' else -int(es); rec['CURRENT_SIDE_PLUS_FLOW']=int(np.sign(cur+sum(votes))) if len(votes)>=2 else 0; rec['DOMINANT_ONLY']=rec['DOMINANT_pred']; out.append(rec)
        except Exception as e: errors.append({'node':node,'event_id':eid,'landmark_t':t,'error':repr(e)})
        print(f'{i}/{len(rows)} {node} {eid}',flush=True)
    pd.DataFrame(out).to_csv(OUT/'predictions_targetblind.csv',index=False); (OUT/'errors.json').write_text(json.dumps(errors,indent=2)); (OUT/'summary.json').write_text(json.dumps({'requested':len(rows),'completed':len(out),'errors':len(errors)},indent=2)); print(json.dumps({'completed':len(out),'errors':len(errors)}),flush=True)
    if errors: raise SystemExit(2)
if __name__=='__main__': main()
