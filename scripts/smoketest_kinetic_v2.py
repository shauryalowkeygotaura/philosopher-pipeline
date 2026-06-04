"""Smoke test for compose_kinetic_v2: 5-beat TTS-driven reel using cached assets."""
import logging
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from composer import compose_kinetic_v2

PHOTOS = sorted((BASE / "cache" / "photos").glob("albert-camus-*.jpg"))[:4]
FONT = BASE / "fonts" / "PlayfairDisplay-Regular.ttf"
OUT = BASE / "output" / "smoketest-kinetic-v2.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)

# Cold cinematic orchestral, transformative, atmospheric, epic. Replaces the
# sparse "stoic grief" piano that read as too thin under the narration.
MUSIC = BASE / "cache" / "audio" / "NS9z2QHcZdY.m4a"

QUOTE = "In the depth of winter, I finally learned that within me there lay an invincible summer."
PHILOSOPHER = "Albert Camus"
SLOGAN = "A man of solitude must find his own fire."

print(f"Images: {len(PHOTOS)} photos")
print(f"Font:   {FONT} (exists={FONT.exists()})")
print(f"Music:  {MUSIC} (exists={MUSIC.exists()})")
print(f"Output: {OUT}")

compose_kinetic_v2(
    [str(p) for p in PHOTOS],
    QUOTE,
    PHILOSOPHER,
    str(OUT),
    str(FONT),
    slogan=SLOGAN,
    voice="daniel",
    music_path=str(MUSIC) if MUSIC.exists() else None,
)
print(f"OK: {OUT.stat().st_size} bytes")
