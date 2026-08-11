from pathlib import Path
import json, math
import numpy as np, pandas as pd
import r34f16_fetch_plain as fetch

_old=fetch.parse_kline
def _pk(raw):
    d=_old(raw)
    if not d.empty:
        d=d.copy(); d['close_time']=d['open_time']+pd.Timedelta(minutes=5)
    return d
fetch.parse_kline=_pk; fetch.FRAME.clear()

DIMS=(3,4,6); OUT=Path('out_r34f17'); OUT.mkdir(exist_ok=True)
def basis(T,d):
    n=np.arange(T,dtype=float); cols=[]
    for k in range(d):
        v=np.cos(np.pi*(n+.5)*k/T); v*=1/np.sqrt(T) if k==0 else np.sqrt(2/T); cols.append(v)
    return np.stack(cols,axis=1)
def erank(x):
    x=np.asarray(x,float); x=x[x>0]
    return 0. if not len(x) else float(x.sum()**2/np.square(x).sum())
def cert(x,d):
    x=np.asarray(x,float); P=basis(len(x),d); H=P.T@(x[:,None]*P); H=(H+H.T)/2; e=np.linalg.eigvalsh(H); sc=np.max(np.abs(e)) if len(e) else 0.; tol=100*np.finfo(float).eps*max(1,len(e))*sc if sc else 0.; s=erank(e[e>tol])-erank(-e[e<-tol]); return int(np.sign(s)),float(s)
def panel(x):
    if len(x)<6 or not np.isfinite(x).all(): return 0,[np.nan]*3
    c=[cert(x,d) for d in DIMS]; ss=[z[0] for z in c]; p=1 if all(v==1 for v in ss) else (-1 if all(v==-1 for v in ss) else 0); return p,[z[1] for z in c]
def ret5(d): return np.log(d.close.astype(float).to_numpy()/d.open.astype(float).to_numpy())
def main():
    rows=json.loads(Path('r34f16_rows.json').read_text()); out=[]; errors=[]
    for i,row in enumerate(rows,1):
        node,eid,year,es,side,stratum,t=row; ev=pd.Timestamp(eid); ev=ev.tz_convert('UTC') if ev.tzinfo else ev.tz_localize('UTC'); lm=ev+pd.Timedelta(hours=4,minutes=float(t)); dates=[(lm-pd.Timedelta(minutes=65)).normalize(),lm.normalize()]
        try:
            spot=fetch.get_daily('spot',dates); mark=fetch.get_daily('mark',dates); idx=fetch.get_daily('index',dates); prem=fetch.get_daily('premium',dates); met=fetch.get_daily('metrics',dates); cb=fetch.coinbase_5m(lm)
            def w(d,tc): return d[(d[tc]>lm-pd.Timedelta(minutes=65))&(d[tc]<=lm)].copy()
            spot=w(spot,'close_time')
            S=spot[['open_time','open','high','low','close']].copy(); S['spot_ret']=ret5(S)
            M=met[(met.ts>lm-pd.Timedelta(minutes=65))&(met.ts<=lm)].copy().rename(columns={'ts':'open_time'})
            c1=pd.merge(S[['open_time','spot_ret']],M[['open_time','sum_open_interest']],on='open_time').sort_values('open_time'); c1['oi_d']=np.log(c1.sum_open_interest.astype(float)).diff(); x1=(c1.spot_ret*c1.oi_d).dropna().to_numpy(float)
            pc=['sum_toptrader_long_short_ratio','count_long_short_ratio','sum_taker_long_short_vol_ratio']; c2=M[['open_time']+pc].dropna().sort_values('open_time'); vals=[]
            if len(c2)>=7:
                ds=np.column_stack([np.log(c2[c].astype(float)).diff().to_numpy() for c in pc]); vals=np.nanmean(ds,axis=1)[1:]
            x2=np.asarray(vals,float)
            J=mark[['open_time','close']].rename(columns={'close':'mark'}).merge(idx[['open_time','close']].rename(columns={'close':'index'}),on='open_time').merge(spot[['open_time','close']].rename(columns={'close':'spot'}),on='open_time').merge(prem[['open_time','close']].rename(columns={'close':'premium'}),on='open_time'); J=J[(J.open_time>=lm-pd.Timedelta(minutes=60))&(J.open_time<lm)].sort_values('open_time'); level=((J['mark']-J['index'])/J['index']+(J['spot']-J['index'])/J['index']+J['premium'])/3; x3=level.to_numpy(float); x4=np.diff(x3)
            C=cb[['open_time','open','high','low','close']].copy(); C['cb_ret']=ret5(C); X=S[['open_time','spot_ret']].merge(C[['open_time','cb_ret','high','low']],on='open_time').sort_values('open_time'); x5=(X.cb_ret-X.spot_ret).to_numpy(float); x6=(X.cb_ret/np.maximum(X['high']/X['low']-1,1e-12)).to_numpy(float)
            chans=[x1,x2,x3,x4,x5,x6]; rec={'node':node,'event_id':eid,'year':year,'event_sign':es,'side':side,'landmark_t':t}; votes=[]
            for j,x in enumerate(chans,1):
                p,sc=panel(x); rec[f'C{j}_n']=len(x);rec[f'C{j}_pred']=p
                for dd,v in zip(DIMS,sc): rec[f'C{j}_d{dd}']=v
                if p: votes.append(p)
            rec['available_channels']=len(votes); s=sum(votes); rec['CHANNEL_MAJORITY']=int(np.sign(s)) if len(votes)>=3 else 0; rec['UNANIMOUS_3PLUS']=votes[0] if len(votes)>=3 and len(set(votes))==1 else 0; cur=int(es) if side=='ALIGNED' else -int(es); s2=cur+sum(votes); rec['CURRENT_SIDE_PLUS_CHANNELS']=int(np.sign(s2)) if len(votes)>=3 else 0; out.append(rec)
        except Exception as e: errors.append({'node':node,'event_id':eid,'landmark_t':t,'error':repr(e)})
        print(f'{i}/{len(rows)} {node} {eid}',flush=True)
    pd.DataFrame(out).to_csv(OUT/'predictions_targetblind.csv',index=False); (OUT/'errors.json').write_text(json.dumps(errors,indent=2)); (OUT/'summary.json').write_text(json.dumps({'requested':len(rows),'completed':len(out),'errors':len(errors)},indent=2))
    if errors: raise SystemExit(2)
if __name__=='__main__': main()
