"""Profile reference reel audio: detect TTS-voice segments vs music-only segments.

Strategy:
- RMS energy: when speech dominates, energy has burst pattern aligned with words
- Zero-crossing rate (ZCR): speech has higher ZCR than music
- Spectral centroid + flatness: speech is more peaked, music is more "flat" / harmonic
- Detect voice activity by combining these into a per-second score
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa

AUDIO = Path(__file__).resolve().parent.parent / "reference" / "ref_audio.wav"
print(f"Loading {AUDIO}...")

y, sr = librosa.load(str(AUDIO), sr=22050, mono=True)
duration = len(y) / sr
print(f"Duration: {duration:.2f}s @ {sr}Hz, {len(y)} samples")

# 1-second windows
hop = sr  # 1s hop
n_frames = int(duration)

rms = []
zcr = []
centroid = []
flatness = []

for i in range(n_frames):
    chunk = y[i*hop:(i+1)*hop]
    if len(chunk) < 100:
        continue
    rms.append(float(np.sqrt(np.mean(chunk**2))))
    zcr.append(float(np.mean(librosa.feature.zero_crossing_rate(chunk)[0])))
    centroid.append(float(np.mean(librosa.feature.spectral_centroid(y=chunk, sr=sr)[0])))
    flatness.append(float(np.mean(librosa.feature.spectral_flatness(y=chunk)[0])))

rms = np.array(rms)
zcr = np.array(zcr)
centroid = np.array(centroid)
flatness = np.array(flatness)

# Normalize and combine: voice has high ZCR + LOW flatness + variable centroid
zcr_norm = (zcr - zcr.min()) / (zcr.max() - zcr.min() + 1e-9)
flat_norm = (flatness - flatness.min()) / (flatness.max() - flatness.min() + 1e-9)

# Voice score: high ZCR, LOW flatness, energy present
voice_score = zcr_norm * (1 - flat_norm) * (rms > 0.01).astype(float)

print()
print(f"{'t (s)':>5} {'RMS':>7} {'ZCR':>7} {'centroid':>9} {'flatness':>9} {'voice?':>7}")
print("-" * 50)
for i in range(n_frames):
    tag = "VOICE" if voice_score[i] > 0.15 else ("music" if rms[i] > 0.005 else "silence")
    print(f"{i:>5} {rms[i]:>7.4f} {zcr[i]:>7.4f} {centroid[i]:>9.1f} {flatness[i]:>9.4f} {tag:>7}")

# Segment detection
threshold = 0.15
segments = []
state = "silence"
seg_start = 0
for i, score in enumerate(voice_score):
    if rms[i] < 0.005:
        new_state = "silence"
    elif score > threshold:
        new_state = "voice"
    else:
        new_state = "music"
    if new_state != state:
        if state != "silence" or i > 0:
            segments.append((state, seg_start, i))
        state = new_state
        seg_start = i
segments.append((state, seg_start, n_frames))

print()
print("Detected segments:")
for seg_type, start, end in segments:
    print(f"  t={start:>4.1f}s -> t={end:>4.1f}s  [{end-start:>4.1f}s]  {seg_type}")
