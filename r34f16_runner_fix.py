import re,base64,gzip
s=open('r34f16_dynamic_winner_runner.py',encoding='utf-8').read()
m=re.search(r"b64decode\('([^']+)'\)",s)
if not m: raise RuntimeError('embedded R34F16 payload not found')
exec(gzip.decompress(base64.b64decode(m.group(1))).decode('utf-8'))
