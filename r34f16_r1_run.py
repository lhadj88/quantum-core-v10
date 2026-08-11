from pathlib import Path
import pandas as pd
import r34f16_fetch_plain as fetch

# Data-format repair frozen before winner scoring:
# Data Vision spot timestamps switch precision from 2025 onward. Bar identity is open_time;
# canonical 5m close is therefore open_time + 5 minutes. No interpolation is performed.
_original_parse_kline = fetch.parse_kline

def parse_kline_bar_identity(raw):
    d = _original_parse_kline(raw)
    if not d.empty:
        d = d.copy()
        d['close_time'] = d['open_time'] + pd.Timedelta(minutes=5)
    return d

fetch.parse_kline = parse_kline_bar_identity
fetch.FRAME.clear()

import r34f16_run_plain as runner
runner.OUT = Path('out_r34f16r1')
runner.OUT.mkdir(exist_ok=True)

if __name__ == '__main__':
    runner.main()
