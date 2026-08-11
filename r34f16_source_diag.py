import json
import pandas as pd
from r34f16_fetch_plain import get_daily

out={}
for date in ['2022-01-05','2022-02-16','2022-06-21']:
    d=get_daily('metrics',[pd.Timestamp(date,tz='UTC')])
    out['metrics_'+date]={
      'rows':len(d),'columns':list(d.columns),
      'non_null':{c:int(d[c].notna().sum()) for c in d.columns},
      'head':d.head(3).astype(str).to_dict('records'),
      'tail':d.tail(3).astype(str).to_dict('records')}
for date in ['2024-12-11','2025-07-02','2026-04-01']:
    sources={s:get_daily(s,[pd.Timestamp(date,tz='UTC')]) for s in ['spot','mark','index','premium']}
    item={}
    for s,d in sources.items():
      item[s]={
        'rows':len(d),
        'first_open_time':None if d.empty else str(d.open_time.iloc[0]),
        'first_close_time':None if d.empty else str(d.close_time.iloc[0]),
        'last_open_time':None if d.empty else str(d.open_time.iloc[-1]),
        'last_close_time':None if d.empty else str(d.close_time.iloc[-1])}
    if all(not d.empty for d in sources.values()):
      c=set(sources['spot'].close_time)
      o=set(sources['spot'].open_time)
      item['close_time_common_all']=len(c & set(sources['mark'].close_time) & set(sources['index'].close_time) & set(sources['premium'].close_time))
      item['open_time_common_all']=len(o & set(sources['mark'].open_time) & set(sources['index'].open_time) & set(sources['premium'].open_time))
      # normalized bucket key independent of close-time precision
      keys=[]
      for s,d in sources.items():
        keys.append(set(d.open_time.dt.floor('5min')))
      item['open_bucket_common_all']=len(set.intersection(*keys))
    out['kline_'+date]=item
open('r34f16_source_diag.json','w').write(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
