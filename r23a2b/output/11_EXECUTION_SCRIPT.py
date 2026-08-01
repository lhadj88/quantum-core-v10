#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, sys, time, zipfile
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests
from scipy.stats import rankdata

SEED=2302
N_PERM=10000
N_BOOT=10000
COLS=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']


def sha256_bytes(b:bytes)->str: return hashlib.sha256(b).hexdigest()
def sha256_file(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def holm(ps):
    ps=np.asarray(ps,float); m=len(ps); order=np.argsort(ps); adj=np.empty(m); running=0.0
    for k,i in enumerate(order):
        running=max(running,(m-k)*ps[i]); adj[i]=min(1.0,running)
    return adj

def spearman(x,y):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if len(x)<3 or np.nanstd(x)==0 or np.nanstd(y)==0: return np.nan
    rx=rankdata(x,method='average'); ry=rankdata(y,method='average')
    return float(np.corrcoef(rx,ry)[0,1])

def fetch(session,url,retries=6):
    last=None
    for a in range(retries):
        try:
            r=session.get(url,timeout=90)
            if r.status_code==200: return r.content
            last=RuntimeError(f'HTTP {r.status_code} {url}')
        except Exception as e: last=e
        time.sleep(min(20,2**a))
    raise last

def parse_daily_zip(blob:bytes,day:str):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names=[n for n in z.namelist() if not n.endswith('/')]
        if len(names)!=1: raise RuntimeError(f'{day}: archive members={names}')
        raw=z.read(names[0])
    df=pd.read_csv(io.BytesIO(raw),header=None,names=COLS)
    df['open_time']=pd.to_numeric(df['open_time'],errors='coerce')
    df=df[df.open_time.notna()].copy()
    for c in ['open','high','low','close','volume']:
        df[c]=pd.to_numeric(df[c],errors='raise')
    unit='microseconds' if float(df.open_time.median())>1e15 else 'milliseconds'
    if unit=='microseconds': df['open_time']=(df.open_time//1000).astype('int64')
    else: df['open_time']=df.open_time.astype('int64')
    df=df.sort_values('open_time').drop_duplicates('open_time',keep=False)
    return df,unit,len(raw)

def expected_times(start_ms): return start_ms+np.arange(144,dtype=np.int64)*300000

def path_metrics(row,path):
    sign=float(row.event_sign); p0=float(path.iloc[0].open); scale=float(row.event_scale)
    if not (p0>0 and scale>0): raise RuntimeError(f'{row.event_id}: invalid p0/scale')
    high=np.log(path.high.to_numpy(float)/p0); low=np.log(path.low.to_numpy(float)/p0); close=np.log(path.close.to_numpy(float)/p0)
    if sign>0:
        counter=np.maximum(0,-low)/scale; aligned=np.maximum(0,high)/scale
    else:
        counter=np.maximum(0,high)/scale; aligned=np.maximum(0,-low)/scale
    signed_close=sign*close
    counter_close=np.maximum(0,-signed_close)/scale
    out={'event_id':row.event_id,'event_time':row.event_time,'year':int(row.year),'cluster_month':row.cluster_month,
         'event_sign':sign,'P2_score':float(row.P2_score),'event_scale':scale,'price_20utc':p0}
    horizons={1:12,3:36,6:72,12:144}
    for h,n in horizons.items():
        out[f'max_counter_h{h}']=float(np.max(counter[:n])); out[f'max_aligned_h{h}']=float(np.max(aligned[:n]))
    out['T1_max_counter_60m']=out['max_counter_h1']
    imax=int(np.argmax(counter)); out['time_max_counter_min']=5*(imax+1)
    for b in [0.10,0.25,0.50,1.00]:
        idx=np.flatnonzero(counter>=b)
        out[f'fp_{b:.2f}_event']=int(len(idx)>0); out[f'fp_{b:.2f}_time_min']=int(5*(idx[0]+1)) if len(idx)>0 else 720
    # sign changes of nonzero signed displacement at bar closes
    s=np.sign(signed_close); s=s[s!=0]
    out['sign_change_count']=int(np.sum(s[1:]!=s[:-1])) if len(s)>1 else 0
    # trapezoidal integral, fraction-hours
    out['auc_counter_fraction_hours']=float(np.trapezoid(counter_close,dx=5/60))
    return out

def bootstrap_indices(df,rng):
    parts=[]
    for year,g in df.groupby('year',sort=True):
        months=sorted(g.cluster_month.unique()); chosen=rng.choice(months,size=len(months),replace=True)
        for j,m in enumerate(chosen):
            z=g[g.cluster_month==m].copy(); z['_boot_cluster']=f'{year}-{j}' ; parts.append(z)
    return pd.concat(parts,ignore_index=True)

def permute_within_year(df,rng):
    x=df.P2_score.to_numpy(float).copy()
    for _,idx in df.groupby('year').groups.items():
        ii=np.asarray(list(idx),int); x[ii]=rng.permutation(x[ii])
    return x

def cox_components(df,beta,xcol='P2_score',timecol='fp_0.25_time_min',eventcol='fp_0.25_event'):
    ll=score=info=0.0
    for _,g in df.groupby('year',sort=True):
        t=g[timecol].to_numpy(float); e=g[eventcol].to_numpy(int); x=g[xcol].to_numpy(float)
        for tt in np.unique(t[e==1]):
            ev=(t==tt)&(e==1); risk=t>=tt; d=int(ev.sum())
            eta=np.clip(beta*x[risk],-50,50); w=np.exp(eta); sw=w.sum()
            if sw<=0: continue
            mx=np.sum(w*x[risk])/sw; vx=np.sum(w*(x[risk]-mx)**2)/sw
            ll += beta*x[ev].sum()-d*np.log(sw); score += x[ev].sum()-d*mx; info += d*vx
    return ll,score,info

def cox_fit(df,timecol='fp_0.25_time_min',eventcol='fp_0.25_event'):
    if df[eventcol].sum()==0 or df.P2_score.nunique()<2: return np.nan,np.nan,np.nan
    beta=0.0
    for _ in range(60):
        ll,u,info=cox_components(df,beta,timecol=timecol,eventcol=eventcol)
        if not np.isfinite(info) or info<=1e-12: return np.nan,np.nan,np.nan
        step=np.clip(u/info,-2,2); beta+=step
        if abs(step)<1e-10: break
    ll,u,info=cox_components(df,beta,timecol=timecol,eventcol=eventcol)
    return float(beta),float(math.exp(np.clip(beta,-50,50))),float(info)

def cox_score_z(df,x,timecol,eventcol):
    tmp=df.copy(); tmp['P2_score']=x
    _,u,info=cox_components(tmp,0.0,timecol=timecol,eventcol=eventcol)
    return float(u/math.sqrt(info)) if info>0 else np.nan

def run_tests(df,outdir):
    rng=np.random.default_rng(SEED)
    # T1
    rho=spearman(df.P2_score,df.T1_max_counter_60m)
    yr1=[]
    for y,g in df.groupby('year'):
        r=spearman(g.P2_score,g.T1_max_counter_60m); yr1.append({'test':'T1','year':int(y),'estimate':r,'direction_pass':bool(np.isfinite(r) and r<0),'n':len(g)})
    perm=np.empty(N_PERM)
    y=df.T1_max_counter_60m.to_numpy(float)
    for i in range(N_PERM): perm[i]=spearman(permute_within_year(df,rng),y)
    p1=(1+np.sum(perm<=rho))/(N_PERM+1)
    boots=[]
    for _ in range(N_BOOT):
        b=bootstrap_indices(df,rng); r=spearman(b.P2_score,b.T1_max_counter_60m)
        if np.isfinite(r): boots.append(r)
    ci1=np.quantile(boots,[.025,.975])
    t1_years=sum(r['direction_pass'] for r in yr1)
    # T2
    beta,hr,info=cox_fit(df)
    yr2=[]
    for y,g in df.groupby('year'):
        b,h,_=cox_fit(g); yr2.append({'test':'T2','year':int(y),'estimate':h,'beta':b,'direction_pass':bool(np.isfinite(h) and h<1),'n':len(g),'events':int(g['fp_0.25_event'].sum())})
    zobs=cox_score_z(df,df.P2_score.to_numpy(float),'fp_0.25_time_min','fp_0.25_event')
    zperm=np.empty(N_PERM)
    for i in range(N_PERM): zperm[i]=cox_score_z(df,permute_within_year(df,rng),'fp_0.25_time_min','fp_0.25_event')
    p2=(1+np.sum(zperm<=zobs))/(N_PERM+1)
    bhr=[]
    for _ in range(N_BOOT):
        b=bootstrap_indices(df,rng); _,h,_=cox_fit(b)
        if np.isfinite(h): bhr.append(h)
    ci2=np.quantile(bhr,[.025,.975])
    t2_years=sum(r['direction_pass'] for r in yr2)
    adj=holm([p1,p2])
    t1_gate=bool(t1_years>=4 and ci1[1]<0 and adj[0]<.05)
    t2_gate=bool(t2_years>=4 and ci2[1]<1 and adj[1]<.05)
    primary=pd.DataFrame([
        {'test':'T1','estimate':rho,'metric':'Spearman_rho','ci_low':ci1[0],'ci_high':ci1[1],'p_one_sided':p1,'p_holm':adj[0],'years_direction_pass':t1_years,'gate_pass':t1_gate},
        {'test':'T2','estimate':hr,'metric':'hazard_ratio','beta':beta,'score_z':zobs,'ci_low':ci2[0],'ci_high':ci2[1],'p_one_sided':p2,'p_holm':adj[1],'years_direction_pass':t2_years,'gate_pass':t2_gate},
    ])
    primary.to_csv(outdir/'03_PRIMARY_TESTS.csv',index=False,float_format='%.12g')
    pd.DataFrame(yr1+yr2).to_csv(outdir/'04_YEARLY_PRIMARY_TESTS.csv',index=False,float_format='%.12g')
    # Secondary fixed barriers, separate Holm family
    sec=[]
    for barrier in [0.10,0.50,1.00]:
        tc=f'fp_{barrier:.2f}_time_min'; ec=f'fp_{barrier:.2f}_event'
        b,h,_=cox_fit(df,timecol=tc,eventcol=ec); z=cox_score_z(df,df.P2_score.to_numpy(float),tc,ec)
        zp=np.empty(N_PERM)
        for i in range(N_PERM): zp[i]=cox_score_z(df,permute_within_year(df,rng),tc,ec)
        p=(1+np.sum(zp<=z))/(N_PERM+1)
        sec.append({'barrier':barrier,'beta':b,'hazard_ratio':h,'score_z':z,'p_one_sided':p,'events':int(df[ec].sum())})
    sadj=holm([r['p_one_sided'] for r in sec])
    for r,a in zip(sec,sadj): r['p_holm_secondary']=a
    pd.DataFrame(sec).to_csv(outdir/'05_SECONDARY_BARRIER_TESTS.csv',index=False,float_format='%.12g')
    # Descriptive secondary associations
    assoc=[]
    metrics=['max_counter_h1','max_counter_h3','max_counter_h6','max_counter_h12','max_aligned_h1','max_aligned_h3','max_aligned_h6','max_aligned_h12','time_max_counter_min','sign_change_count','auc_counter_fraction_hours']
    for m in metrics: assoc.append({'metric':m,'spearman_rho':spearman(df.P2_score,df[m]),'n':len(df)})
    pd.DataFrame(assoc).to_csv(outdir/'06_SECONDARY_ASSOCIATIONS.csv',index=False,float_format='%.12g')
    if t1_gate and t2_gate: verdict='P2_TEMPORAL_TRANSITION_STATE_SUPPORTED_FOR_FUTURE_MULTIPROCESS_INTEGRATION'
    elif t1_gate or t2_gate: verdict='P2_PARTIAL_TEMPORAL_MECHANISM_NO_PREDICTIVE_INTEGRATION'
    else: verdict='P2_ROUTE_ASSOCIATION_ONLY_MECHANISM_FORMULATION_CLOSED'
    result={'protocol_id':'R23A2B_P2_TEMPORAL_BUDGET_COMPLETE_CASE_v0_1','execution_amendment':'R23A2B_COMPLETE_CASE_PROTOCOL_v0_1','seed':SEED,'n_permutations':N_PERM,'n_bootstraps':N_BOOT,'events':len(df),'T1_gate':t1_gate,'T2_gate':t2_gate,'verdict':verdict,'R23B_authorized':False,'multiprocess_guardrail':'P2 is one mechanism/state only; no universal route or global classifier is promoted.'}
    (outdir/'07_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    return primary,verdict,result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='r23a2b/input_events_202.csv'); ap.add_argument('--output',default='r23a2b/output'); ap.add_argument('--prereg',default='r23a2b/00_PREREGISTRATION.json'); ap.add_argument('--amendment',default='r23a2b/00_COMPLETE_CASE_PROTOCOL.md'); args=ap.parse_args()
    inp=Path(args.input); outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    ev=pd.read_csv(inp)
    if len(ev)!=202 or not ev['event_id'].is_unique: raise RuntimeError('Frozen input must contain 202 unique events')
    ev['event_time']=pd.to_datetime(ev['event_id'],utc=True)
    ev['path_start']=pd.to_datetime(ev['path_start'],utc=True)
    expected_hash={}
    for r in ev.itertuples(index=False):
        d=r.path_start.strftime('%Y-%m-%d')
        h=str(r.event_day_sha256)
        if d in expected_hash and expected_hash[d]!=h: raise RuntimeError(f'{d}: conflicting expected hashes')
        expected_hash[d]=h
    required_days=set(expected_hash)
    required_days.update((pd.Timestamp(d)+pd.Timedelta(days=1)).strftime('%Y-%m-%d') for d in list(expected_hash))
    integrity=[]; days={}; session=requests.Session(); session.headers['User-Agent']='SBC-GANN-R23A2-research/0.1'
    for day in sorted(required_days):
        url=f'https://data.binance.vision/data/spot/daily/klines/BTCUSDT/5m/BTCUSDT-5m-{day}.zip'
        expected=expected_hash.get(day,'')
        blob=fetch(session,url); actual=sha256_bytes(blob); status='OK'
        if expected and actual.lower()!=expected.lower(): status='HASH_MISMATCH'
        df,unit,raw_bytes=parse_daily_zip(blob,day)
        if status!='OK': raise RuntimeError(f'{day}: {status}')
        days[day]=df
        dif=np.diff(df.open_time.to_numpy(np.int64)); full_cont=bool(len(dif)==0 or np.all(dif==300000))
        role='EVENT_DAY' if day in expected_hash else 'NEXT_DAY'
        if day in expected_hash and any((pd.Timestamp(x)+pd.Timedelta(days=1)).strftime('%Y-%m-%d')==day for x in expected_hash): role='EVENT_AND_NEXT_DAY'
        integrity.append({'day':day,'role':role,'url':url,'expected_sha256':expected,'actual_sha256':actual,'zip_bytes':len(blob),'csv_bytes':raw_bytes,'rows':len(df),'timestamp_unit_source':unit,'full_day_continuity':full_cont,'status':status})
    pd.DataFrame(integrity).to_csv(outdir/'01_ARCHIVE_INTEGRITY.csv',index=False)
    records=[]
    for r in ev.itertuples(index=False):
        d0=r.path_start.strftime('%Y-%m-%d'); d1=(r.path_start+pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        allbars=pd.concat([days[d0],days[d1]],ignore_index=True).sort_values('open_time').drop_duplicates('open_time',keep=False)
        start_ms=int(r.path_start.timestamp()*1000); exp=expected_times(start_ms)
        path=allbars[allbars.open_time.isin(exp)].sort_values('open_time')
        got=path.open_time.to_numpy(np.int64)
        if len(path)!=144 or not np.array_equal(got,exp):
            missing=sorted(set(exp)-set(got)); extra=sorted(set(got)-set(exp)); raise RuntimeError(f'{r.event_id}: discontinuous path rows={len(path)} missing={missing[:5]} extra={extra[:5]}')
        records.append(path_metrics(r,path))
    paths=pd.DataFrame(records)
    paths.to_csv(outdir/'02_EVENT_PATH_LEDGER.csv',index=False,float_format='%.17g')
    primary,verdict,result=run_tests(paths,outdir)
    report=f"""# R23A2B — Rapport maître

**Verdict :** `{verdict}`  
**Événements :** {len(paths)}  
**Usage :** recherche uniquement ; aucune autorisation de trading.

## Résultats primaires

{primary.to_markdown(index=False)}

## Interprétation canonique

R23A2B ne teste qu’un mécanisme partiel : le budget de réversion déjà consommé avant 20:00 UTC. Même si les deux tests passent, P2 n’est ni une route dominante ni une loi universelle. Il devient seulement un état de transition admissible dans une future phénoménologie causale multi-processus.

Les mécanismes d’épuisement, d’absorption, de continuation, de libération retardée et d’ambiguïté restent distincts. Leur représentation devra être recherchée séparément avec des observables causaux adaptés.

`R23B_authorized = false` : aucune procédure directionnelle globale n’est promue par ce tribunal.
"""
    (outdir/'08_MASTER_REPORT.md').write_text(report)
    # Copy protocol/amendment into output for immutability
    for src,name in [(Path(args.prereg),'09_PREREGISTRATION.json'),(Path(args.amendment),'10_EXECUTION_AMENDMENT.md')]:
        (outdir/name).write_bytes(src.read_bytes())
    sums=[]
    for p in sorted(outdir.iterdir()):
        if p.is_file() and p.name!='SHA256SUMS.txt': sums.append(f'{sha256_file(p)}  {p.name}')
    (outdir/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
