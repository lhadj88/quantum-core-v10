from pathlib import Path
import json
import numpy as np,pandas as pd
import r34f16_fetch_plain as fetch
import r34f19_run as f19

OUT=Path('out_r34f20'); OUT.mkdir(exist_ok=True); DIMS=(3,4,6)
# Same 5m bar-identity repair frozen before scoring.
_old=fetch.parse_kline
def _pk(raw):
    d=_old(raw)
    if not d.empty:
        d=d.copy(); d['open_time']=d['open_time'].dt.floor('5min'); d['close_time']=d['open_time']+pd.Timedelta(minutes=5)
    return d
fetch.parse_kline=_pk; fetch.FRAME.clear()

def basis(T,d):
    n=np.arange(T,dtype=float); cols=[]
    for k in range(d):
        v=np.cos(np.pi*(n+.5)*k/T); v*=1/np.sqrt(T) if k==0 else np.sqrt(2/T); cols.append(v)
    return np.stack(cols,axis=1)
def norm(X):
    X=np.asarray(X,float).copy()
    for j in range(X.shape[1]):
        s=np.median(np.abs(X[:,j])); X[:,j]/=(s if np.isfinite(s) and s>0 else 1.)
    return X
def erank(x):
    x=np.asarray(x,float); x=x[x>1e-12]
    return 0. if len(x)==0 else float(x.sum()**2/np.square(x).sum())
def score(X,d,coupled=True):
    Z=norm(X); T,C=Z.shape; P=basis(T,d); H=np.zeros((C*d,C*d))
    for c in range(C):
        for e in range(C):
            if not coupled and c!=e: continue
            z=(Z[:,c]+Z[:,e])/2
            H[c*d:(c+1)*d,e*d:(e+1)*d]=P.T@(z[:,None]*P)
    H=(H+H.T)/2; vals=np.linalg.eigvalsh(H); sc=erank(vals[vals>1e-12])-erank(-vals[vals<-1e-12]); return int(np.sign(sc)),float(sc)
def panel(X,coupled=True):
    c=[score(X,d,coupled) for d in DIMS]; ss=[z[0] for z in c]; p=1 if all(s==1 for s in ss) else (-1 if all(s==-1 for s in ss) else 0); return p,[z[1] for z in c]
def mean_panel(X):
    Z=norm(X); return panel(np.mean(Z,axis=1)[:,None],False)

def main():
    rows=json.loads(Path('r34f16_rows.json').read_text()); pcache={}; out=[]; errors=[]
    for i,row in enumerate(rows,1):
        node,eid,year,es,side,stratum,t=row; ev=pd.Timestamp(eid); ev=ev.tz_convert('UTC') if ev.tzinfo else ev.tz_localize('UTC'); lm=ev+pd.Timedelta(hours=4,minutes=float(t)); start=lm-pd.Timedelta(minutes=60); dates=sorted({(lm-pd.Timedelta(minutes=65)).strftime('%Y-%m-%d'),lm.strftime('%Y-%m-%d')})
        try:
            def allsrc(name):
                return pd.concat([fetch.get_daily(name,[pd.Timestamp(x,tz='UTC')]) for x in dates]).drop_duplicates('open_time')
            spot=allsrc('spot'); mark=allsrc('mark'); idx=allsrc('index'); prem=allsrc('premium'); met=pd.concat([fetch.get_daily('metrics',[pd.Timestamp(x,tz='UTC')]) for x in dates]).drop_duplicates('ts'); cb=fetch.coinbase_5m(lm)
            pp=[]
            for date in dates:
                if date not in pcache: pcache[date]=f19.perp_day(date)
                pp.append(pcache[date])
            perp=pd.concat(pp).drop_duplicates('open_time')
            for z in [spot,perp,mark,idx,prem]: z.drop(z[(z.open_time<start)|(z.open_time>=lm)].index,inplace=True)
            met=met[(met.ts>=start)&(met.ts<lm)].copy().rename(columns={'ts':'open_time'}); cb=cb[(cb.open_time>=start)&(cb.open_time<lm)]
            J=spot[['open_time','open','close','volume','taker_buy_base']].rename(columns={'open':'sopen','close':'sclose','volume':'svol','taker_buy_base':'sbuy'}).merge(perp[['open_time','volume','taker_buy_base']].rename(columns={'volume':'pvol','taker_buy_base':'pbuy'}),on='open_time').merge(met[['open_time','sum_open_interest']],on='open_time').merge(mark[['open_time','close']].rename(columns={'close':'mark'}),on='open_time').merge(idx[['open_time','close']].rename(columns={'close':'index'}),on='open_time').merge(prem[['open_time','close']].rename(columns={'close':'premium'}),on='open_time').merge(cb[['open_time','open','close']].rename(columns={'open':'cbopen','close':'cbclose'}),on='open_time').sort_values('open_time')
            fs=2*J.sbuy/J.svol-1; fp=2*J.pbuy/J.pvol-1; sr=np.log(J.sclose/J.sopen); oid=np.log(J.sum_open_interest).diff(); level=((J['mark']-J['index'])/J['index']+(J.sclose-J['index'])/J['index']+J.premium)/3; dlevel=level.diff(); cross=np.log(J.cbclose/J.cbopen)-sr; X=np.column_stack([fs,fp,sr*oid,dlevel,cross])[1:,:]
            if len(X)<6 or not np.isfinite(X).all(): raise RuntimeError(f'incomplete aligned matrix n={len(X)}')
            full,sf=panel(X,True); diag,sd=panel(X,False); mean,sm=mean_panel(X); rec={'node':node,'event_id':eid,'year':year,'event_sign':es,'side':side,'landmark_t':t,'n':len(X),'COUPLED_FULL':full,'BLOCK_DIAGONAL':diag,'MEAN_CHANNEL':mean}
            for nm,ss in [('FULL',sf),('DIAG',sd),('MEAN',sm)]:
                for dd,z in zip(DIMS,ss): rec[f'{nm}_d{dd}']=z
            out.append(rec)
        except Exception as e: errors.append({'node':node,'event_id':eid,'landmark_t':t,'error':repr(e)})
        print(f'{i}/{len(rows)} {node} {eid}',flush=True)
    pd.DataFrame(out).to_csv(OUT/'predictions_targetblind.csv',index=False); (OUT/'errors.json').write_text(json.dumps(errors,indent=2)); (OUT/'summary.json').write_text(json.dumps({'requested':len(rows),'completed':len(out),'errors':len(errors)},indent=2)); print(json.dumps({'completed':len(out),'errors':len(errors)}),flush=True)
    if errors: raise SystemExit(2)
if __name__=='__main__': main()
