#!/usr/bin/env python3
import math, random, struct, wave
from pathlib import Path

SR=48000
DURATION=36.0
N=int(SR*DURATION)
OUT=Path('public/audio/yuga-studio-master.wav')
OUT.parent.mkdir(parents=True, exist_ok=True)
rng=random.Random(20442082)

lp_l=0.0
lp_r=0.0
prev_l=0.0
prev_r=0.0
transitions=[0.0,4.3,9.0,14.6,19.4,24.7,30.8,35.0]
notes=[(32.2,220.0),(33.25,277.18),(34.35,329.63)]

def clamp(v):
    return max(-0.98,min(0.98,v))

def impact(t, ts, idx):
    dt=t-ts
    if dt<0 or dt>1.7:
        return 0.0
    f0=62.0 if idx in (2,5,6) else 78.0
    sweep=math.sin(2*math.pi*(f0*dt + 80*(dt-(1-math.exp(-3*dt))/3))) * math.exp(-4.5*dt)
    click=math.sin(2*math.pi*900*dt)*math.exp(-42*dt)
    metallic=(math.sin(2*math.pi*410*dt)+.5*math.sin(2*math.pi*813*dt))*math.exp(-3.8*dt)
    return .24*sweep+.045*click+.025*metallic

with wave.open(str(OUT),'wb') as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    frames=bytearray()
    for i in range(N):
        t=i/SR
        fade=min(1.0,t/2.0,max(0.0,(DURATION-t)/2.0))
        l=r=0.0
        base=43.65
        for k,amp in ((1,.11),(2,.055),(3,.028),(5,.015)):
            mod=.15*math.sin(2*math.pi*.017*t)
            l += amp*math.sin(2*math.pi*base*k*t+.35*k+mod)*fade
            r += amp*math.sin(2*math.pi*base*k*t+.35*k+.08+.15*math.sin(2*math.pi*.019*t))*fade
        w1=rng.uniform(-1,1)
        w2=rng.uniform(-1,1)
        lp_l=.9982*lp_l+.0018*w1
        lp_r=.9982*lp_r+.0018*w2
        air_l=lp_l-prev_l
        air_r=lp_r-prev_r
        prev_l=lp_l
        prev_r=lp_r
        env=(.035+.02*math.sin(2*math.pi*.05*t))*fade
        l += env*(10.5*lp_l+18*air_l)
        r += env*(10.5*lp_r+18*air_r)
        for idx,ts in enumerate(transitions):
            hit=impact(t,ts,idx)
            if hit:
                pan=-.3 if idx%2==0 else .3
                l += hit*(1-pan)*.7
                r += hit*(1+pan)*.7
        for ts in (23.5,29.6):
            dt=t-ts
            if 0<=dt<=2.4:
                e=math.sin(math.pi*dt/2.4)**1.5
                tone=(math.sin(2*math.pi*(240+55*dt)*dt)+.6*math.sin(2*math.pi*(480+110*dt)*dt))*e*.035
                l += tone
                r += tone*.96
        for ts,f in notes:
            dt=t-ts
            if 0<=dt<=1.6:
                e=(1-math.exp(-20*dt))*math.exp(-1.7*dt)
                note=(math.sin(2*math.pi*f*dt)+.35*math.sin(2*math.pi*2*f*dt))*e*.08
                l += note
                r += note*.98
        l += .018*math.sin(2*math.pi*.11*t)*fade
        r += .018*math.sin(2*math.pi*.13*t+1.2)*fade
        l=math.tanh(l*1.5)/math.tanh(1.5)*.92
        r=math.tanh(r*1.5)/math.tanh(1.5)*.92
        frames += struct.pack('<hh',int(clamp(l)*32767),int(clamp(r)*32767))
        if len(frames)>=SR*4*2:
            wf.writeframes(frames)
            frames.clear()
    if frames:
        wf.writeframes(frames)
print(f'Generated {OUT} ({DURATION:.1f}s, {SR}Hz stereo)')
