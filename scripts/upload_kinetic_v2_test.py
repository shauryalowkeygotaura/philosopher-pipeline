"""Render kinetic-v2 test reel WITHOUT the Wisdom-of-hidgon subtitle, then
upload to the @deepahhthinking IG account using pipeline.py's standard
caption builder.

Requires INSTAGRAM_USERNAME, INSTAGRAM_PASSWORD, ELEVENLABS_API_KEY in env
(inject via `doppler run -- python scripts/upload_kinetic_v2_test.py`).
"""
import logging
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("upload_v2")

from composer import compose_kinetic_v2
from pipeline import _build_caption, HOOKS
from uploader import upload_reel

PHOTOS = sorted((BASE / "cache" / "photos").glob("albert-camus-*.jpg"))[:4]
FONT = BASE / "fonts" / "PlayfairDisplay-Regular.ttf"
MUSIC = BASE / "cache" / "audio" / "NS9z2QHcZdY.m4a"
OUT = BASE / "output" / "upload-kinetic-v2.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)

QUOTE = "In the depth of winter, I finally learned that within me there lay an invincible summer."
PHILOSOPHER = "Albert Camus"
SLOGAN = "A man of solitude must find his own fire."
SLUG = "albert-camus"

log.info("Rendering kinetic v2 (brand_subtitle blank)...")
compose_kinetic_v2(
    [str(p) for p in PHOTOS],
    QUOTE,
    PHILOSOPHER,
    str(OUT),
    str(FONT),
    slogan=SLOGAN,
    voice="daniel",
    music_path=str(MUSIC) if MUSIC.exists() else None,
    music_volume=0.30,
    brand_subtitle="",
)
log.info("Render OK: %s (%d bytes)", OUT, OUT.stat().st_size)

import time as _t
# Rotate hook so consecutive uploads don't trigger IG's duplicate-caption damper.
hook_idx = int(_t.time()) % len(HOOKS)
log.info("Hook rotation: HOOKS[%d] -> %r", hook_idx, HOOKS[hook_idx])
caption = _build_caption(
    quote=QUOTE,
    philosopher=PHILOSOPHER,
    hook=HOOKS[hook_idx],
    bio="French-Algerian existentialist; survived TB, occupied Paris, wrote The Stranger at 29.",
    slug_tag=SLUG.replace("-", ""),
)
log.info("Caption:\n%s\n", caption)

log.info("Uploading to Instagram...")
upload_reel(str(OUT), caption)
log.info("Upload complete.")
