from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.io import wavfile

SR = 48000
DURATION = 36.0
N = int(SR * DURATION)
t = np.arange(N, dtype=np.float64) / SR
rng = np.random.default_rng(20442082)


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def env_segment(start, end, attack=0.12, release=0.2):
    e = np.zeros_like(t)
    active = (t >= start) & (t <= end)
    if not np.any(active):
        return e
    local = t[active]
    a = smoothstep((local-start)/max(attack, 1e-6))
    r = smoothstep((end-local)/max(release, 1e-6))
    e[active] = np.minimum(a, r)
    return e


def pan(sig, pan_value):
    angle = (pan_value + 1) * math.pi / 4
    return np.stack([sig * math.cos(angle), sig * math.sin(angle)], axis=1)

mix = np.zeros((N,2), dtype=np.float64)

white = rng.standard_normal(N)
sea = signal.sosfilt(signal.butter(4, [80, 2400], btype='bandpass', fs=SR, output='sos'), white)
sea /= np.max(np.abs(sea)) + 1e-9
sea_env = env_segment(0, 7.0, 1.4, 2.2) * (0.45 + 0.12*np.sin(2*np.pi*0.17*t))
mix += pan(sea * sea_env * 0.085, -0.18)

fund = 43.2 + 0.22*np.sin(2*np.pi*0.031*t)
drone_phase = np.cumsum(2*np.pi*fund/SR)
drone = np.sin(drone_phase)
drone += 0.42*np.sin(1.5*drone_phase + 0.4)
drone += 0.22*np.sin(2.02*drone_phase + 1.1)
drone = np.tanh(drone*0.8)
drone_env = env_segment(0, 36, 2.2, 2.7) * (0.68 + 0.15*np.sin(2*np.pi*0.045*t))
mix += pan(drone * drone_env * 0.19, 0.0)


def sub_hit(at, amp=0.3, freq=48, decay=1.2, panpos=0.0):
    local = t-at
    gate = local >= 0
    env = np.where(gate, np.exp(-np.maximum(local,0)/decay), 0)
    sweep = freq*(1 + 0.5*np.exp(-np.maximum(local,0)/0.14))
    ph = np.cumsum(2*np.pi*sweep/SR)
    sig = np.sin(ph) * env
    click = signal.sosfilt(signal.butter(2, [1100, 7000], btype='bandpass', fs=SR, output='sos'), rng.standard_normal(N))
    click *= np.where(gate, np.exp(-np.maximum(local,0)/0.035), 0)*0.04
    return pan((sig + click)*amp, panpos)

for at in [3.8, 7.1, 10.4, 13.7, 17.0, 20.3, 23.6, 27.0, 31.0, 34.0]:
    mix += sub_hit(at, amp=0.22 if at < 27 else 0.34, freq=46, decay=1.15, panpos=-0.08)
for at in np.arange(7.1, 24.0, 1.65):
    mix += sub_hit(float(at), amp=0.095, freq=72, decay=0.45, panpos=0.10)
for at in np.arange(10.4, 19.0, 0.55):
    mix += sub_hit(float(at), amp=0.036, freq=112, decay=0.18, panpos=0.35 if int(at*10)%2 else -0.35)

for idx, freq in enumerate([181.0, 263.5, 391.0, 587.0]):
    phase = 2*np.pi*freq*t + 0.7*idx + 0.9*np.sin(2*np.pi*(0.07+0.013*idx)*t)
    tone = np.sin(phase) + 0.25*np.sin(phase*1.997)
    e = env_segment(10.5+idx*0.25, 18.5, 1.8, 2.0)
    e *= (0.55 + 0.45*np.sin(2*np.pi*(0.19+idx*0.03)*t + idx)**2)
    mix += pan(tone*e*(0.027/(idx+1)**0.35), -0.55+idx*0.36)

paper = signal.sosfilt(signal.butter(3, [500, 9000], btype='bandpass', fs=SR, output='sos'), rng.standard_normal(N))
paper /= np.max(np.abs(paper))+1e-9
paper_env = env_segment(17.0, 25.3, 0.8, 1.3)
scratch = np.zeros(N)
for at in [17.7, 18.9, 20.2, 21.8, 23.0, 24.2]:
    loc = t-at
    scratch += np.where(loc>=0, np.exp(-np.maximum(loc,0)/0.12)*np.sin(2*np.pi*(1300+800*np.exp(-np.maximum(loc,0)/0.08))*np.maximum(loc,0)), 0)
mix += pan((paper*0.035 + scratch*0.025)*paper_env, -0.15)

silence_dip = 1 - 0.58*env_segment(24.7, 27.4, 0.35, 0.65)
mix *= silence_dip[:,None]

start, end = 27.0, 31.8
loc = np.clip((t-start)/(end-start),0,1)
riser_env = smoothstep(loc) * (1-smoothstep(np.clip((t-end)/0.5,0,1)))
noise = signal.sosfilt(signal.butter(4, [300, 12000], btype='bandpass', fs=SR, output='sos'), rng.standard_normal(N))
noise /= np.max(np.abs(noise))+1e-9
mix += pan(noise*riser_env*0.10, 0.0)
chirp = signal.chirp(t, f0=110, t1=31.8, f1=1500, method='logarithmic')
mix += pan(chirp*riser_env*0.035, 0.2)

mix += sub_hit(31.55, amp=0.55, freq=42, decay=2.3, panpos=0.0)
impact_noise = signal.sosfilt(signal.butter(3, [70, 6000], btype='bandpass', fs=SR, output='sos'), rng.standard_normal(N))
loc2 = t-31.55
impact_env = np.where(loc2>=0, np.exp(-np.maximum(loc2,0)/0.6), 0)
mix += pan(impact_noise*impact_env*0.12, 0.0)

for idx, freq in enumerate([216.0, 323.9, 431.8, 646.7, 970.0]):
    e = env_segment(30.8+0.12*idx, 36.0, 1.1, 2.1)
    tone = np.sin(2*np.pi*freq*t + idx*0.8) * (0.65+0.35*np.sin(2*np.pi*0.11*t+idx))
    mix += pan(tone*e*(0.045/(idx+1)**0.42), -0.6+idx*0.3)

for at, panpos, f in [(5.9,-.6,2600),(9.8,.45,3100),(14.2,-.25,3700),(21.2,.55,2900),(28.9,-.35,4300),(33.2,.2,5100)]:
    loc=t-at
    e=np.where(loc>=0,np.exp(-np.maximum(loc,0)/0.55),0)
    sig=np.sin(2*np.pi*f*np.maximum(loc,0)+2*np.pi*220*np.maximum(loc,0)**2)*e
    mix += pan(sig*0.018,panpos)

mix = np.tanh(mix*1.12)
mid = (mix[:,0]+mix[:,1])*0.5
side = (mix[:,0]-mix[:,1])*0.5*1.16
mix[:,0]=mid+side
mix[:,1]=mid-side
fade_out = 1-smoothstep(np.clip((t-34.7)/1.3,0,1))
mix *= fade_out[:,None]

mix = signal.detrend(mix, axis=0, type='constant')
peak = np.max(np.abs(mix))+1e-12
mix *= (10**(-1.2/20))/peak
out = Path(__file__).resolve().parents[1]/'public'/'audio'/'yuga-final.wav'
out.parent.mkdir(parents=True, exist_ok=True)
wavfile.write(out, SR, (mix*32767).astype(np.int16))
print(out)
