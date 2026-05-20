"""Smoke test for compose_kinetic_letterbox: render one reel from cached assets."""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from composer import compose_kinetic_letterbox

AUDIO = BASE / "cache" / "audio" / "_Xm1hmB0KCg.m4a"
PHOTOS = sorted((BASE / "cache" / "photos").glob("albert-camus-*.jpg"))[:4]
FONT = BASE / "fonts" / "PlayfairDisplay-Regular.ttf"
OUT = BASE / "output" / "smoketest-kinetic.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)

QUOTE = "in the depth of winter, I finally learned that within me there lay an invincible summer"
PHILOSOPHER = "Albert Camus"

print(f"Audio: {AUDIO} (exists={AUDIO.exists()})")
print(f"Images: {len(PHOTOS)} photos")
print(f"Font: {FONT} (exists={FONT.exists()})")
print(f"Output: {OUT}")

compose_kinetic_letterbox(
    [str(p) for p in PHOTOS],
    QUOTE,
    PHILOSOPHER,
    str(AUDIO),
    str(OUT),
    str(FONT),
    reel_duration=28.0,
)
print(f"OK: {OUT.stat().st_size} bytes")
