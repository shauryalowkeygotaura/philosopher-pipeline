"""Render the same quote at several pitch depths so the user can A/B the voice.
Outputs output/voice-depth/voice_<n>st.mp3 and profiles median F0 of each."""
import logging, sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import numpy as np, librosa

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
logging.basicConfig(level=logging.WARNING)

from tts import synthesize_quote

QUOTE = "In the depth of winter, I finally learned that within me there lay an invincible summer."
SLOGAN = "A man of solitude must find his own fire."
OUT = BASE / "output" / "voice-depth"
OUT.mkdir(parents=True, exist_ok=True)

def med_f0(path):
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    f0, *_ = librosa.pyin(y, fmin=65, fmax=400, sr=sr, frame_length=2048)
    v = f0[~np.isnan(f0)]
    if v.size == 0: return 0.0, 0.0
    return float(np.median(v)), len(y)/sr

print(f"{'pitch':>8} {'medF0':>8} {'note':>6} {'dur':>7}")
print("-"*36)
for st in (-1.0, -3.0, -4.0):
    res = synthesize_quote(QUOTE, out_dir=OUT, slogan=SLOGAN, pitch_semitones=st)
    dest = OUT / f"voice_{int(abs(st))}st.mp3"
    Path(res.audio_path).replace(dest)
    f0, dur = med_f0(dest)
    note = librosa.midi_to_note(librosa.hz_to_midi(f0)) if f0 else "-"
    print(f"{st:>6.1f}st {f0:>7.1f}Hz {note:>6} {dur:>6.2f}s  -> {dest.name}")
print("\nReference target: ~80-90 Hz")
