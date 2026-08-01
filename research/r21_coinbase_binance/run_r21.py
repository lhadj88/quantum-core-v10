#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
OUT=Path('artifacts/r21_coinbase_binance_leadership_v0_1'); OUT.mkdir(parents=True,exist_ok=True)
ENDPOINT='https://api.exchange.coinbase.com/products/BTC-USD/candles'
HORIZONS=[1,3,6,12]
THRESHOLDS=[.50,.55,.60,.65,.70,.75,.80,.85,.90]
BASE=['event_sign','shock_close','aligned_taker_gap','gap_prev_days','repeat3']
MARKET=['retreat_from_extreme','time_since_extreme_min','extreme_time_frac','extreme_reconquest_runs','aligned_ret_60','aligned_ret_30','aligned_ret_15','late_price_acceleration','spot_counter_eff_60','spot_absorption_60','perp_counter_eff_60','perp_absorption_60','taker_gap_slope_60','basis_reconv_60','premium_reconv_60','counterflow_peak_time_frac']


def fetch(url:str)->bytes:
    last=None
    for attempt in range(7):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'SBC-GANN-R21/1.0','Accept':'application/json'})
            with urllib.request.urlopen(req,timeout=60) as r: return r.read()
        except Exception as exc:
            last=exc
            if isinstance(exc,urllib.error.HTTPError) and exc.code not in (429,500,502,503,504): raise
            if attempt==6: raise
            time.sleep(2**attempt)
    raise RuntimeError(str(last))


def coinbase_window(ts:pd.Timestamp)->tuple[pd.DataFrame,dict]:
    start=ts-pd.Timedelta(hours=4); decision=ts+pd.Timedelta(hours=4)
    params=urllib.parse.urlencode({'granularity':300,'start':start.isoformat().replace('+00:00','Z'),'end':decision.isoformat().replace('+00:00','Z')})
    url=ENDPOINT+'?'+params
    blob=fetch(url); payload=json.loads(blob.decode('utf-8'))
    if not isinstance(payload,list): raise RuntimeError(f'Unexpected Coinbase payload {payload}')
    rows=[]
    for item in payload:
        if not isinstance(item,list) or len(item)<6: continue
        rows.append({'time':pd.to_datetime(int(item[0]),unit='s',utc=True),'low':float(item[1]),'high':float(item[2]),'open':float(item[3]),'close':float(item[4]),'volume':float(item[5])})
    df=pd.DataFrame(rows).sort_values('time').drop_duplicates('time')
    df=df[(df['time']>=start)&(df['time']<decision)].copy()
    expected=pd.date_range(start,decision-pd.Timedelta(minutes=5),freq='5min',tz='UTC')
    missing=expected.difference(pd.DatetimeIndex(df['time']))
    meta={'timestamp':str(ts),'url':url,'response_sha256':hashlib.sha256(blob).hexdigest(),'raw_candles':len(payload),'causal_candles':len(df),'missing_candles':[str(x) for x in missing]}
    return df,meta


def path_features(df:pd.DataFrame,event_sign:int,row:pd.Series)->dict:
    if len(df)!=96: raise RuntimeError(f'Expected 96 causal candles, got {len(df)}')
    pre=df.iloc[:48].copy(); ev=df.iloc[48:].copy()
    if pre.iloc[0]['time'].hour!=12 or ev.iloc[0]['time'].hour!=16: raise RuntimeError('Coinbase window alignment failed')
    out={}
    def logret(frame:pd.DataFrame)->float: return float(np.log(float(frame.iloc[-1]['close'])/float(frame.iloc[0]['open'])))
    out['cb_pre_ret_240']=logret(pre); out['cb_event_ret_240']=logret(ev); out['cb_aligned_event_ret_240']=event_sign*out['cb_event_ret_240']
    for minutes in [120,60,30,15,5]:
        n=minutes//5; frame=ev.iloc[-n:]
        ret=logret(frame); aligned=event_sign*ret
        out[f'cb_ret_{minutes}']=ret; out[f'cb_aligned_ret_{minutes}']=aligned
        bcol=f'spot_aligned_ret_{minutes}'
        if bcol in row and pd.notna(row[bcol]):
            out[f'cb_minus_binance_aligned_{minutes}']=aligned-float(row[bcol])
            out[f'cb_binance_sign_agree_{minutes}']=int(np.sign(ret)==np.sign(float(row[bcol])*event_sign))
    total_range=float(ev['high'].max()-ev['low'].min())
    close=float(ev.iloc[-1]['close']); high=float(ev['high'].max()); low=float(ev['low'].min())
    out['cb_event_range_frac']=total_range/float(ev.iloc[0]['open']) if ev.iloc[0]['open'] else np.nan
    out['cb_close_location']=((2*close-high-low)/total_range) if total_range>0 else 0.0
    if event_sign>0:
        extreme_idx=int(np.argmax(ev['high'].to_numpy())); extreme=high; out['cb_retreat_from_extreme']=(extreme-close)/extreme if extreme else 0.0
    else:
        extreme_idx=int(np.argmin(ev['low'].to_numpy())); extreme=low; out['cb_retreat_from_extreme']=(close-extreme)/extreme if extreme else 0.0
    out['cb_extreme_time_frac']=extreme_idx/47.0; out['cb_time_since_extreme_min']=(47-extreme_idx)*5
    volume=float(ev['volume'].sum()); out['cb_volume_last60_share']=float(ev.iloc[-12:]['volume'].sum()/volume) if volume>0 else np.nan
    out['cb_volume_last30_share']=float(ev.iloc[-6:]['volume'].sum()/volume) if volume>0 else np.nan
    first=float(ev.iloc[:24]['volume'].sum()); second=float(ev.iloc[24:]['volume'].sum()); out['cb_volume_half_logratio']=float(np.log((second+1e-12)/(first+1e-12)))
    out['cb_price_volume_efficiency']=abs(out['cb_event_ret_240'])/(volume+1e-12)
    out['cb_late_acceleration']=out['cb_aligned_ret_30']-out['cb_aligned_ret_60']
    out['cb_leadership_score']=out['cb_aligned_event_ret_240']-abs(float(row.get('aligned_taker_gap',0.0)))
    return out


def load_data()->tuple[pd.DataFrame,dict]:
    data=pd.read_csv(ROOT/'01_EVENT_TARGETS.csv'); data['timestamp']=pd.to_datetime(data['timestamp'],utc=True); data=data.sort_values('timestamp').reset_index(drop=True)
    metas=[]; rows=[]; errors=[]
    for _,r in data.iterrows():
        ts=r['timestamp']
        try:
            df,meta=coinbase_window(ts); feats=path_features(df,int(np.sign(float(r['event_sign']))),r); rec=r.to_dict(); rec.update(feats); rows.append(rec); metas.append(meta)
        except Exception as exc:
            errors.append({'timestamp':str(ts),'error':f'{type(exc).__name__}: {exc}'})
        time.sleep(.12)
    out=pd.DataFrame(rows)
    ledger={'events_input':len(data),'events_complete':len(out),'responses':metas,'errors':errors}
    return out,ledger


def fill(train,test,cols):
    tr=train[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float); te=test[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    tr[~np.isfinite(tr)]=np.nan; te[~np.isfinite(te)]=np.nan
    for j in range(tr.shape[1]):
        v=tr[np.isfinite(tr[:,j]),j]; f=float(np.median(v)) if len(v) else 0.0
        tr[~np.isfinite(tr[:,j]),j]=f; te[~np.isfinite(te[:,j]),j]=f
    return tr,te


def model(): return Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=.5,class_weight='balanced',max_iter=4000,random_state=20260731))])

def abs_dir(event_sign,opp): return int((-event_sign if opp else event_sign)>0)

def action(route,event_sign):
    if route=='IMMEDIATE_SNAPBACK': return 1,abs_dir(event_sign,1)
    if route=='DELAYED_REVERSAL': return 6,abs_dir(event_sign,1)
    if route=='CONTINUATION': return 3,abs_dir(event_sign,0)
    return None,None


def horizon_fold(train,test,cols,period,name):
    Xtr,Xte=fill(train,test,cols); probs={}
    for h in HORIZONS:
        y=train[f'opp_h{h}'].astype(int).to_numpy()
        if len(np.unique(y))<2: probs[h]=np.full(len(test),float(y[0]))
        else:
            m=model(); m.fit(Xtr,y); probs[h]=m.predict_proba(Xte)[:,1]
    rows=[]
    for j,(_,r) in enumerate(test.iterrows()):
        h,conf,p=max([(h,abs(float(probs[h][j])-.5),float(probs[h][j])) for h in HORIZONS],key=lambda x:(x[1],-x[0]))
        opp=int(p>=.5); es=int(np.sign(float(r['event_sign']))); pred=abs_dir(es,opp); target=int(r[f'target_dir_h{h}']); base=abs_dir(es,1)
        rows.append({'period':period,'model':name,'policy':'adaptive_horizon','timestamp':r['timestamp'],'year':int(r['year']),'horizon':h,'predicted_dir':pred,'target_dir':target,'correct':int(pred==target),'confidence':.5+conf,'baseline_correct':int(base==target),'route_actual':r['route'],'route_pred':None})
    return rows


def route_fold(train,test,cols,period,name):
    Xtr,Xte=fill(train,test,cols); y=train['route'].astype(str).to_numpy(); m=model(); m.fit(Xtr,y); pr=m.predict_proba(Xte); idx=np.argmax(pr,axis=1); rp=m.classes_[idx]; conf=pr[np.arange(len(test)),idx]
    rows=[]
    for j,(_,r) in enumerate(test.iterrows()):
        es=int(np.sign(float(r['event_sign']))); h,pred=action(str(rp[j]),es)
        if h is None: target=correct=base_correct=None
        else:
            target=int(r[f'target_dir_h{h}']); correct=int(pred==target); base_correct=int(abs_dir(es,1)==target)
        rows.append({'period':period,'model':name,'policy':'route','timestamp':r['timestamp'],'year':int(r['year']),'horizon':h,'predicted_dir':pred,'target_dir':target,'correct':correct,'confidence':float(conf[j]),'baseline_correct':base_correct,'route_actual':r['route'],'route_pred':str(rp[j])})
    return rows


def evaluate(g):
    x=g[g['correct'].notna()].copy()
    if not len(x): return {'n':0,'successes':0,'accuracy':None,'balanced_accuracy':None}
    y=x['target_dir'].astype(int); p=x['predicted_dir'].astype(int); ba=float(balanced_accuracy_score(y,p)) if y.nunique()>1 else None
    return {'n':len(x),'successes':int(x['correct'].sum()),'accuracy':float(x['correct'].mean()),'balanced_accuracy':ba}


def paired(g):
    x=g[g['correct'].notna()&g['baseline_correct'].notna()]; b=int(((x.correct==1)&(x.baseline_correct==0)).sum()); c=int(((x.correct==0)&(x.baseline_correct==1)).sum()); p=float(binomtest(min(b,c),n=b+c,p=.5).pvalue) if b+c else 1.0
    return {'n':len(x),'model_accuracy':float(x.correct.mean()) if len(x) else None,'baseline_accuracy':float(x.baseline_correct.mean()) if len(x) else None,'paired_difference':float(x.correct.mean()-x.baseline_correct.mean()) if len(x) else None,'mcnemar_b':b,'mcnemar_c':c,'mcnemar_p':p}


def safe(x:Any)->Any:
    if isinstance(x,float): return x if math.isfinite(x) else None
    if isinstance(x,dict): return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [safe(v) for v in x]
    return x


def main():
    data,ledger=load_data()
    cb=[c for c in data.columns if c.startswith('cb_')]
    market=[c for c in MARKET if c in data.columns]
    sets={'base':BASE,'coinbase':BASE+cb,'coinbase_market':BASE+market+cb,'market':BASE+market}
    preds=[]; years=sorted(data.year.astype(int).unique())
    for year in years:
        tr=data[data.year.astype(int)!=year]; te=data[data.year.astype(int)==year]
        for name,cols in sets.items(): preds+=horizon_fold(tr,te,cols,'loyo',f'horizon_{name}')
        for name in ['base','coinbase','coinbase_market']: preds+=route_fold(tr,te,sets[name],'loyo',f'route_{name}')
    for i in range(60,len(data)):
        tr=data.iloc[:i]; te=data.iloc[i:i+1]
        for name in ['base','coinbase','coinbase_market']:
            preds+=horizon_fold(tr,te,sets[name],'prequential',f'horizon_{name}')
            preds+=route_fold(tr,te,sets[name],'prequential',f'route_{name}')
    pred=pd.DataFrame(preds); pred['timestamp']=pd.to_datetime(pred.timestamp,utc=True)
    ms=[]; sel=[]; yrs=[]; pts=[]
    for (period,name),g in pred.groupby(['period','model']):
        e=evaluate(g); e.update({'period':period,'model':name,'universe':len(g)}); ms.append(e)
        for year,gy in g.groupby('year'):
            q=evaluate(gy); q.update({'period':period,'model':name,'year':int(year),'universe':len(gy)}); yrs.append(q)
        for t in THRESHOLDS:
            s=g[(g.confidence>=t)&g.correct.notna()]; q=evaluate(s); bas=[]
            for _,sy in s.groupby('year'): bas.append(evaluate(sy).get('balanced_accuracy'))
            bas=[v for v in bas if v is not None and np.isfinite(v)]
            q.update({'period':period,'model':name,'threshold':t,'coverage':len(s)/len(g) if len(g) else 0.0,'years':s.year.nunique(),'min_year_ba':min(bas) if bas else None}); sel.append(q)
            pt=paired(s); pt.update({'period':period,'model':name,'threshold':t}); pts.append(pt)
    ms=pd.DataFrame(ms); sel=pd.DataFrame(sel); yrs=pd.DataFrame(yrs); pts=pd.DataFrame(pts)
    route=pred[pred.policy=='route']; conf=[]
    for (period,name),g in route.groupby(['period','model']):
        labels=sorted(set(g.route_actual.dropna())|set(g.route_pred.dropna())); cm=confusion_matrix(g.route_actual,g.route_pred,labels=labels)
        for i,a in enumerate(labels):
            for j,p in enumerate(labels): conf.append({'period':period,'model':name,'actual':a,'predicted':p,'n':int(cm[i,j])})
    conf=pd.DataFrame(conf)
    passing=[]
    for _,r in sel.iterrows():
        vals=[r.get('accuracy'),r.get('balanced_accuracy'),r.get('n'),r.get('coverage'),r.get('years'),r.get('min_year_ba')]
        if all(v is not None and pd.notna(v) for v in vals) and r.accuracy>=.8 and r.balanced_accuracy>=.8 and r.n>=60 and r.coverage>=.15 and r.years>=4 and r.min_year_ba>=.65: passing.append(r.to_dict())
    gate={'candidate_family':'R21_COINBASE_BINANCE_LEADERSHIP_v0_1','status':'CANDIDATE_FOUND_DEVELOPMENT_ONLY' if passing else 'NO_CANDIDATE_GATE_FAILED','passing_rows':passing,'gate':{'accuracy_min':.8,'balanced_accuracy_min':.8,'minimum_predictions':60,'minimum_coverage':.15,'minimum_years':4,'minimum_year_balanced_accuracy':.65},'governance':{'data_through_2026_07_29_exposed':True,'trading_authorized':False}}
    best=sel.sort_values(['balanced_accuracy','accuracy','n'],ascending=False,na_position='last').head(30)
    result={'id':'R21_COINBASE_BINANCE_LEADERSHIP_v0_1','status':gate['status'],'coverage':{'input_events':ledger['events_input'],'complete_events':ledger['events_complete'],'errors':len(ledger['errors'])},'coinbase_feature_count':len(cb),'model_summary':ms.to_dict('records'),'best_selective':best.to_dict('records'),'gate':gate}
    data.to_csv(OUT/'01_R21_FEATURE_TABLE.csv',index=False); pred.to_csv(OUT/'02_OUT_OF_SAMPLE_PREDICTIONS.csv',index=False); ms.to_csv(OUT/'03_MODEL_SUMMARY.csv',index=False); sel.to_csv(OUT/'04_SELECTIVE_CURVES.csv',index=False); yrs.to_csv(OUT/'05_YEARLY_RESULTS.csv',index=False); pts.to_csv(OUT/'06_PAIRED_TESTS.csv',index=False); conf.to_csv(OUT/'07_ROUTE_CONFUSION.csv',index=False)
    (OUT/'08_SOURCE_LEDGER.json').write_text(json.dumps(safe(ledger),indent=2,ensure_ascii=False),encoding='utf-8'); (OUT/'09_CANDIDATE_GATE_DECISION.json').write_text(json.dumps(safe(gate),indent=2,ensure_ascii=False),encoding='utf-8'); (OUT/'00_RESULT.json').write_text(json.dumps(safe(result),indent=2,ensure_ascii=False),encoding='utf-8')
    report=['# SBC×GANN — R21 Coinbase–Binance Leadership','',f"Statut : **{gate['status']}**",'',f"- événements complets : {len(data)} / {ledger['events_input']} ;",f"- variables Coinbase/cross-venue : {len(cb)} ;",f"- erreurs source : {len(ledger['errors'])}.",'','## Modèles hors entraînement','',ms.to_markdown(index=False),'','## Meilleurs résultats sélectifs','',best.head(20).to_markdown(index=False),'','## Décision',"Le gate 80/80 reste obligatoire. Les réponses Coinbase sont archivées par SHA-256 mais ne disposent pas d'un checksum éditeur. Toutes les observations sont exposées et aucune sortie n'autorise le trading."]
    (OUT/'10_MASTER_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    checks=[]
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name!='SHA256SUMS.txt': checks.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (OUT/'SHA256SUMS.txt').write_text('\n'.join(checks)+'\n',encoding='utf-8')
    print(json.dumps(safe({'status':gate['status'],'coverage':result['coverage'],'best':best.head(5).to_dict('records')}),indent=2,ensure_ascii=False))

if __name__=='__main__': main()
