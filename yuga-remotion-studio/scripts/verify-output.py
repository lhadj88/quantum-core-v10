#!/usr/bin/env python3
import hashlib
import json
import subprocess
from pathlib import Path

OUT = Path('out')
FILES = [OUT / 'Yuga_Studio_FR.mp4', OUT / 'Yuga_Studio_EN.mp4']

def probe(path: Path) -> dict:
    raw = subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format',
        '-of', 'json', str(path)
    ], text=True)
    return json.loads(raw)

results = {}
for path in FILES:
    if not path.exists() or path.stat().st_size < 1_000_000:
        raise SystemExit(f'Missing or unexpectedly small master: {path}')
    data = probe(path)
    videos = [s for s in data['streams'] if s.get('codec_type') == 'video']
    audios = [s for s in data['streams'] if s.get('codec_type') == 'audio']
    if len(videos) != 1 or len(audios) != 1:
        raise SystemExit(f'Expected one video and one audio stream in {path}')
    video, audio = videos[0], audios[0]
    duration = float(data['format']['duration'])
    fps_num, fps_den = map(int, video['r_frame_rate'].split('/'))
    fps = fps_num / fps_den
    checks = {
        'width': int(video['width']),
        'height': int(video['height']),
        'fps': fps,
        'duration': duration,
        'video_codec': video['codec_name'],
        'audio_codec': audio['codec_name'],
        'sample_rate': int(audio['sample_rate']),
        'channels': int(audio['channels']),
        'size_bytes': path.stat().st_size,
    }
    assert checks['width'] == 1080, checks
    assert checks['height'] == 1920, checks
    assert abs(checks['fps'] - 30.0) < 0.01, checks
    assert 35.90 <= checks['duration'] <= 36.10, checks
    assert checks['video_codec'] == 'h264', checks
    assert checks['audio_codec'] == 'aac', checks
    assert checks['sample_rate'] == 48000, checks
    assert checks['channels'] == 2, checks
    results[path.name] = checks
    (OUT / f'ffprobe-{path.stem}.json').write_text(json.dumps(data, indent=2), encoding='utf-8')

with (OUT / 'SHA256SUMS.txt').open('w', encoding='utf-8') as handle:
    for path in FILES:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        handle.write(f'{digest}  {path.name}\n')

(OUT / 'TECHNICAL_REPORT.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
print('Technical validation passed.')
