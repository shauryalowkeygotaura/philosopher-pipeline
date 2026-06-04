"""Compare fundamental frequency (pitch) + loudness of the @wisdomofhidgon
reference voice against our edge-tts output, so the cinematic pitch shift can
be set to a measured target instead of guessed.

Usage:
  python scripts/compare_voice_profile.py <audio.wav/.mp3> [label] ...
With no args it profiles the reference voice segments + the latest cache/tts files.
"""
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa

ROOT = Path(__file__).resolve().parent.parent


def profile(path: Path, label: str, voice_only: bool = False):
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    dur = len(y) / sr

    # pyin F0 over voiced frames; C2..C5 covers male narration range
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=65, fmax=400, sr=sr, frame_length=2048
    )
    voiced = f0[~np.isnan(f0)]
    if voiced.size == 0:
        print(f"{label:>28}: no voiced frames detected")
        return

    med_f0 = float(np.median(voiced))
    p25, p75 = np.percentile(voiced, [25, 75])
    # convert to a musical note for intuition
    midi = librosa.hz_to_midi(med_f0)
    note = librosa.midi_to_note(midi)

    rms = librosa.feature.rms(y=y)[0]
    rms_db = 20 * np.log10(np.median(rms[rms > 0]) + 1e-9)

    print(f"{label:>28}: dur={dur:5.2f}s  medF0={med_f0:6.1f}Hz ({note})  "
          f"IQR={p25:5.1f}-{p75:5.1f}Hz  medRMS={rms_db:6.1f}dB")
    return med_f0


def main():
    print("=" * 92)
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            p = Path(args[i])
            label = args[i + 1] if i + 1 < len(args) and not Path(args[i + 1]).exists() else p.name
            profile(p, label)
            i += 2 if (i + 1 < len(args) and not Path(args[i + 1]).exists()) else 1
        return

    ref = ROOT / "reference" / "ref_audio.wav"
    if ref.exists():
        ref_f0 = profile(ref, "REFERENCE (whole, w/ music)")

    tts = ROOT / "cache" / "tts"
    if tts.exists():
        # raw quote (pre-cinematic) and final (post-cinematic)
        for pat, lbl in [("*_quote.mp3", "edge-tts raw quote"),
                         ("*_raw.mp3", "edge-tts raw concat"),
                         ("tts-*.mp3", "edge-tts FINAL (cinematic)")]:
            files = sorted(tts.glob(pat), key=lambda f: f.stat().st_mtime)
            # exclude the partials/derived for the FINAL match
            if pat == "tts-*.mp3":
                files = [f for f in files if "_" not in f.stem.split("tts-")[1]]
            if files:
                profile(files[-1], lbl)
    print("=" * 92)


if __name__ == "__main__":
    main()
