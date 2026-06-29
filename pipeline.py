#!/usr/bin/env python3
"""
Philosopher Instagram Pipeline
Usage: python pipeline.py

Generates a 7-second beat-synced Reel per philosopher using a mix of
Renaissance paintings and portraits of the writer, with a translucent
@deepahhthinking watermark. Cuts land on the actual song beats (librosa)
with xfade transitions and Ken Burns zoom — no CapCut step required.

Env vars:
  STYLE           'capcut' (default, 7s fast-cut beat-synced) or 'kinetic'
                  (28s @wisdomofhidgon-style letterbox + red word-by-word reveal).
  USE_BEAT_SYNC   '1' (default) for beat-synced reels under STYLE=capcut,
                  '0' for the old uniform-cut path. Ignored under STYLE=kinetic.
  BEAT_TRANSITION ffmpeg xfade name (default 'fadeblack'). 'auto' rotates through punchy ones.
  KEN_BURNS       '1' (default) enables zoom-pan per slide, '0' to disable.
"""
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

# Load .env file from the pipeline directory if present
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from state import StateManager
from input_parser import parse_philosophers, parse_songs
from fetcher import (
    fetch_quote, fetch_slogan, match_song, fetch_photo,
    fetch_paintings, fetch_portraits, get_bio,
)
from composer import (
    compose_image, compose_reel, compose_frame,
    compose_slideshow, compose_slideshow_beat_synced,
    compose_kinetic_letterbox, compose_kinetic_v2,
)
from scheduler import schedule_uploads
from uploader import upload_reel
import bandit
import run_metrics

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
VAULT_DIR = BASE_DIR.parent

_local_philosophers = BASE_DIR / "philosophers.md"
PHILOSOPHERS_FILE = _local_philosophers if _local_philosophers.exists() else VAULT_DIR / "philosophers.md"

_local_songs = BASE_DIR / "songs.md"
SONGS_FILE = _local_songs if _local_songs.exists() else VAULT_DIR / "songs.md"
STATE_FILE = BASE_DIR / "state.json"
OUTPUT_DIR = BASE_DIR / "output"
CACHE_PHOTOS = BASE_DIR / "cache" / "photos"
CACHE_PAINTINGS = BASE_DIR / "cache" / "paintings"
CACHE_AUDIO = BASE_DIR / "cache" / "audio"
FONT_PATH = BASE_DIR / "fonts" / "PlayfairDisplay-Regular.ttf"
GOTHIC_FONT_PATH = BASE_DIR / "fonts" / "UnifrakturMaguntia-Book.ttf"

# Slideshow mix: 24 paintings + 16 portraits = 40 unique images.
# Composer floors cuts at 0.10s with MIN_CUTS_PER_REEL=36, so a 7s reel
# produces 35-50 cuts. 40 unique images means every cut is a new image
# inside a single loop even at the high end. Bumped from 16+12.
NUM_PAINTINGS = 24
NUM_PORTRAITS = 16


def _env_bool(name, default=True):
    return os.getenv(name, "1" if default else "0").strip().lower() not in ("0", "false", "no", "")


USE_BEAT_SYNC = _env_bool("USE_BEAT_SYNC", default=True)
MIN_CUTS = int(os.getenv("MIN_CUTS", "16"))
# Color grade applied uniformly across every clip so disparate paintings
# read as one cohesive reel: vintage | sepia | noir | cool | warm | off.
COLOR_GRADE = os.getenv("COLOR_GRADE", "vintage").strip()
# STYLE = "capcut" (fast-cut 7s, default) or "kinetic" (letterbox + word-by-word
# red typography, 28s, modeled on @wisdomofhidgon DIA_e3dI9tq).
STYLE = os.getenv("STYLE", "capcut").strip().lower()
REEL_DURATION = 28.0 if STYLE == "kinetic" else 7.0
# Kinetic v2 narration voice. Daniel = deep BBC-style British, matches the
# @wisdomofhidgon reference. Override via Doppler for A/B tests.
KINETIC_VOICE = os.getenv("KINETIC_VOICE", "daniel").strip().lower()
# 0.30 ≈ -10 dB. Loud enough to be present under the cinematic voice without
# competing with it. The 2026-05-28 default of 0.18 was inaudible to a
# casual viewer. Set KINETIC_MUSIC_VOLUME=0 to mute, or bump to 0.50 for a
# more music-forward mix.
KINETIC_MUSIC_VOLUME = float(os.getenv("KINETIC_MUSIC_VOLUME", "0.30"))
# Music-only lead-in seconds before the narration starts (black screen, music
# ~2x its under-voice level, ducking down as the voice enters). 0 disables.
KINETIC_INTRO = float(os.getenv("KINETIC_INTRO", "1.5"))
# How many alternate songs to try when a YouTube audio download fails (e.g. a
# track went private/region-locked, or yt-dlp's extractor hiccups). Before this,
# a single dead video in --single mode produced "No reels generated" and an
# empty run (observed 2026-05-30). The failed URL is blacklisted by
# _download_audio, so retries pick a different vibe-matched track.
SONG_RETRY_LIMIT = int(os.getenv("SONG_RETRY_LIMIT", "4"))

# Hooks rotate by post_count so the same opening line never repeats per
# philosopher, which avoids the IG "duplicate caption" downranking.
HOOKS = [
    "the kind of words that hit at 3am.",
    "this hits different at 25 vs 35.",
    "save this. read it again next week.",
    "screenshot this one.",
    "philosophy that actually changes you.",
    "the words that shaped western thought.",
    "you needed to hear this today.",
    "they wrote this 200+ years ago. still hits.",
]

# Niche targeting tags (who the post is FOR) followed by broad reach tags
# (#fyp / #viral etc.) to push the reel onto Explore + Reels feeds. Niche tags
# go first so IG categorises the content correctly before the reach tags widen it.
NICHE_TAGS = "#philosophy #stoicism #renaissance #wisdom #deepthoughts #mindset #motivation"
REACH_TAGS = "#fyp #fypシ #viral #reels #reelsinstagram #explore #explorepage #trending #foryou #viralreels"
HASHTAGS = NICHE_TAGS + " " + REACH_TAGS

# Soft CTA priorities: saves > follows > likes for IG retention scoring.
CTA_LINE = (
    "save this for the day you need it.\n"
    "follow @deepahhthinking for daily wisdom that actually rewires how you think."
)

RUN_ID = time.strftime("%Y-%m-%dT%H%M%S")


def _interleave(a, b):
    """Interleave two lists alternately, appending the longer tail at the end."""
    out = []
    i = 0
    while i < max(len(a), len(b)):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
        i += 1
    return out


def _build_caption(quote, philosopher, hook, bio, slug_tag):
    parts = [hook, '"' + quote + '"', "- " + philosopher]
    if bio:
        parts.append(bio)
    parts.append(CTA_LINE)
    parts.append(HASHTAGS + " #" + slug_tag)
    return "\n\n".join(parts)


def main(upload_now=True, single=False, generate_only=False):
    for d in [OUTPUT_DIR, CACHE_PHOTOS, CACHE_PAINTINGS, CACHE_AUDIO]:
        d.mkdir(parents=True, exist_ok=True)

    if not FONT_PATH.exists():
        sys.exit(
            "[error] Font not found: " + str(FONT_PATH) + "\n"
            "Run: curl -L <playfair-url> -o " + str(FONT_PATH)
        )

    philosophers = parse_philosophers(PHILOSOPHERS_FILE)
    songs = parse_songs(SONGS_FILE)

    state = StateManager(STATE_FILE)
    state.load()

    blacklisted = set(state.get_blacklisted_songs())
    available_songs = [s for s in songs if s["url"] not in blacklisted]

    if not available_songs:
        sys.exit(
            "[error] No songs available after excluding " + str(len(blacklisted)) +
            " blacklisted URLs.\nAdd entries to " + str(SONGS_FILE)
        )
    if len(available_songs) < len(philosophers):
        log.warning(
            "Only %d songs for %d philosophers, songs will be reused across philosophers.",
            len(available_songs), len(philosophers),
        )

    log.info("Run ID: %s", RUN_ID)
    log.info(
        "Style: %s | duration=%.1fs | grade=%s | min_cuts=%d | gothic_font=%s",
        STYLE,
        REEL_DURATION,
        COLOR_GRADE if USE_BEAT_SYNC and STYLE != "kinetic" else "n/a",
        MIN_CUTS if USE_BEAT_SYNC and STYLE != "kinetic" else 0,
        GOTHIC_FONT_PATH.exists(),
    )
    log.info("Processing %d philosophers...", len(philosophers))

    generated = []
    used_songs_this_run = []

    if single:
        philosophers = sorted(philosophers, key=lambda p: state.get_philosopher(p)["post_count"])[:1]
        log.info("Running in --single mode. Selected %s", philosophers[0])

    for philosopher in philosophers:
        log.info("== %s ==", philosopher)
        phil_state = state.get_philosopher(philosopher)

        log.info("  Fetching quote...")
        try:
            quote_result = fetch_quote(philosopher, phil_state["used_quotes"])
        except Exception as e:
            log.warning("  Quote fetch failed for %s: %s, skipping.", philosopher, e)
            continue
        quote = quote_result["quote"]
        reframed = quote_result["reframed"]
        log.info("  Quote: %s...", quote[:60])

        log.info("  Matching song...")
        try:
            song_url = match_song(
                philosopher, quote,
                songs=available_songs,
                used_in_run=used_songs_this_run,
                used_for_philosopher=phil_state["used_songs"],
            )
        except Exception as e:
            log.warning("  Song match failed for %s: %s, skipping.", philosopher, e)
            continue
        used_songs_this_run.append(song_url)
        log.info("  Song: %s", song_url)

        log.info("  Fetching %d paintings + %d portraits of %s...",
                 NUM_PAINTINGS, NUM_PORTRAITS, philosopher)
        try:
            paintings = fetch_paintings(NUM_PAINTINGS, phil_state["used_photos"], CACHE_PAINTINGS)
        except Exception as e:
            log.warning("  Painting fetch error: %s", e)
            paintings = []
        try:
            portraits = fetch_portraits(philosopher, NUM_PORTRAITS, phil_state["used_photos"], CACHE_PHOTOS)
        except Exception as e:
            log.warning("  Portrait fetch error: %s", e)
            portraits = []

        if not paintings and not portraits:
            log.warning("  No images for %s, skipping.", philosopher)
            continue

        frames = _interleave(paintings, portraits)
        log.info("  Got %d frames (%d paintings + %d portraits)",
                 len(frames), len(paintings), len(portraits))

        log.info("  Downloading audio...")
        audio_path = _download_audio(song_url, CACHE_AUDIO, state)
        # A single unavailable video used to abandon the whole philosopher (and,
        # in --single mode, the entire run). Retry with other vibe-matched songs;
        # the just-failed URL is already in used_songs_this_run so match_song
        # won't hand it back.
        retries = 0
        while not audio_path and retries < SONG_RETRY_LIMIT:
            retries += 1
            log.warning("  Audio download failed (%s); trying another song (%d/%d)...",
                        song_url, retries, SONG_RETRY_LIMIT)
            try:
                song_url = match_song(
                    philosopher, quote,
                    songs=available_songs,
                    used_in_run=used_songs_this_run,
                    used_for_philosopher=phil_state["used_songs"],
                )
            except Exception as e:
                log.warning("  Song re-match failed for %s: %s", philosopher, e)
                break
            if not song_url or song_url in used_songs_this_run:
                log.warning("  No fresh song left for %s.", philosopher)
                break
            used_songs_this_run.append(song_url)
            log.info("  Song (retry %d): %s", retries, song_url)
            audio_path = _download_audio(song_url, CACHE_AUDIO, state)
        if not audio_path:
            log.warning("  Audio download failed for %s after %d retr%s, skipping.",
                        philosopher, retries, "y" if retries == 1 else "ies")
            continue
        log.info("  Audio: %s", audio_path)

        slug = _philosopher_slug(philosopher)
        mp4_path = str(OUTPUT_DIR / (slug + "-" + RUN_ID + ".mp4"))
        cover_jpg = str(OUTPUT_DIR / (slug + "-" + RUN_ID + ".jpg"))

        log.info(
            "  Composing %.0fs %s reel...",
            REEL_DURATION,
            STYLE if STYLE == "kinetic" else ("beat-synced" if USE_BEAT_SYNC else "fast-cut"),
        )
        slogan = None
        try:
            if STYLE == "kinetic":
                slogan = fetch_slogan(quote, philosopher)
                log.info("  Slogan: %s", slogan)
                compose_kinetic_v2(
                    frames, quote, philosopher,
                    mp4_path, str(FONT_PATH),
                    slogan=slogan,
                    voice=KINETIC_VOICE,
                    music_path=audio_path,
                    music_volume=KINETIC_MUSIC_VOLUME,
                    music_intro_sec=KINETIC_INTRO,
                )
            elif USE_BEAT_SYNC:
                compose_slideshow_beat_synced(
                    frames, quote, philosopher,
                    audio_path, mp4_path, str(FONT_PATH),
                    reel_duration=REEL_DURATION,
                    min_cuts=MIN_CUTS,
                    seamless_loop=False,
                    color_grade=COLOR_GRADE,
                    overlay_font_path=str(GOTHIC_FONT_PATH) if GOTHIC_FONT_PATH.exists() else None,
                )
            else:
                compose_slideshow(
                    frames, quote, philosopher,
                    audio_path, mp4_path, str(FONT_PATH),
                )
        except Exception as e:
            log.warning("  Slideshow composition failed for %s: %s, skipping.", philosopher, e)
            continue

        try:
            compose_frame(frames[0], quote, philosopher, cover_jpg, str(FONT_PATH))
        except Exception as e:
            log.warning("  Cover thumbnail failed for %s: %s (using mp4 only)", philosopher, e)
            cover_jpg = None

        if not Path(mp4_path).exists() or Path(mp4_path).stat().st_size == 0:
            log.warning("  Reel file missing or empty for %s, skipping.", philosopher)
            continue

        all_filenames = [Path(p).name for p in (paintings + portraits)]
        state.update_philosopher(philosopher, quote, song_url, all_filenames, reframed)
        log.info("  State updated.")

        bio = get_bio(philosopher)
        slug_tag = slug.replace("-", "")[:20]
        # Self-improving loop: identical to HOOKS[post_count % len(HOOKS)] until
        # the ledger has engagement insights for this hook set, then the bandit
        # biases toward the best-performing hook (bandit.pick_hook).
        hook = bandit.pick_hook(philosopher, phil_state["post_count"], HOOKS)
        caption = _build_caption(quote, philosopher, hook, bio, slug_tag)

        generated.append({
            "philosopher": philosopher,
            "mp4_path": mp4_path,
            "jpg_path": cover_jpg,
            "caption": caption,
            "hook": hook,
            "slogan": slogan,
            "slug": slug,
        })
        log.info("  Reel ready: %s", mp4_path)

    if not generated:
        log.warning("No reels generated. Exiting.")
        run_metrics.write(mode=STYLE, status="degraded",
                          summary="No reels generated (no eligible philosopher / asset fetch failed)")
        return

    uploaded = 0
    if generate_only:
        log.info("--generate-only: %d reel(s) saved to output/, skipping upload.", len(generated))
        for reel in generated:
            log.info("  Ready: %s", reel["mp4_path"])
    elif upload_now:
        log.info("Uploading %d reels immediately...", len(generated))
        for reel in generated:
            log.info("  Uploading %s...", reel["philosopher"])
            try:
                upload_reel(
                    reel["mp4_path"], reel["caption"], reel.get("jpg_path"),
                    meta={
                        "philosopher": reel["philosopher"],
                        "hook": reel.get("hook"),
                        "slogan": reel.get("slogan"),
                        "slug": reel.get("slug"),
                        "style": STYLE,
                    },
                )
                uploaded += 1
                log.info("  Uploaded %s", reel["philosopher"])
            except Exception as e:
                log.error("  Upload failed for %s: %s", reel["philosopher"], e)
    else:
        log.info("Scheduling %d reels...", len(generated))
        schedule_uploads(generated, upload_reel)

    who = ", ".join(r["philosopher"] for r in generated)
    run_metrics.write(
        mode=STYLE, status="ok",
        summary=(f"Generated {len(generated)} {STYLE} reel(s) ({who})"
                 + (f", uploaded {uploaded}" if upload_now else " (not uploaded)")),
        metrics={"generated": len(generated), "uploaded": uploaded,
                 "style": STYLE, "philosophers": [r["philosopher"] for r in generated]},
        budgets={"edge_tts": {"note": "free, no key"},
                 "groq": {"note": "free tier (slogan generation)"}},
    )


def _philosopher_slug(name):
    slug = name.lower()
    replacements = {
        "ø": "o", "ü": "u", "ä": "a", "ö": "o",
        "é": "e", "è": "e", "ê": "e", "ë": "e",
        "ï": "i", "î": "i", "ô": "o", "û": "u",
        "ç": "c", "ñ": "n", " ": "-",
    }
    for char, rep in replacements.items():
        slug = slug.replace(char, rep)
    return "".join(c for c in slug if c.isalnum() or c == "-")


def _yt_dlp_cmd():
    """Resolve how to invoke yt-dlp.

    Prefer a `yt-dlp` launcher on PATH (how GitHub Actions installs it), but
    fall back to `python -m yt_dlp` using the SAME interpreter running the
    pipeline. The module form is what's actually guaranteed by
    requirements.txt (`yt-dlp>=...`); a bare `yt-dlp.exe` only exists if the
    install env's Scripts/ dir happens to be on PATH, which it isn't under
    the local run_pipeline.bat path. Without this, `subprocess.run(["yt-dlp",
    ...])` raises FileNotFoundError locally even though yt_dlp is installed.
    """
    import shutil
    found = shutil.which("yt-dlp")
    return [found] if found else [sys.executable, "-m", "yt_dlp"]


def _download_audio(song_url, cache_dir, state):
    import re
    import subprocess

    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", song_url)
    video_id = match.group(1) if match else hashlib.md5(song_url.encode()).hexdigest()[:11]
    cached = cache_dir / (video_id + ".m4a")

    if cached.exists() and cached.stat().st_size > 0:
        # Trust but verify: an interrupted download can leave a non-empty file
        # with no moov atom (found 2026-06-06: AswXiy8JCsI.m4a), which passes
        # the size check here and then kills the render. A quick ffprobe
        # catches it; on failure fall through to a fresh download.
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(cached)],
                capture_output=True, text=True)
            if probe.returncode == 0:
                return str(cached)
        except OSError:
            # ffprobe missing/unrunnable: can't validate, trust the size check
            # (the composer needs ffmpeg anyway, so this is unlikely locally).
            return str(cached)
        log.warning("Cached audio %s is corrupt, re-downloading.", cached.name)
        try:
            cached.unlink()
        except OSError as e:
            # OneDrive sync can hold a lock; yt-dlp overwrites the path anyway.
            log.warning("Could not delete corrupt %s: %s", cached.name, e)

    cmd = [
        *_yt_dlp_cmd(),
        # Prefer YouTube's native AAC stream so no transcode happens at all
        # (plain `bestaudio` usually grabs opus, and re-encoding opus -> m4a at
        # yt-dlp's default quality was a lossy-to-lossy hit on the music bed).
        # When only opus exists, --audio-quality 0 re-encodes at best VBR.
        "--format", "bestaudio[ext=m4a]/bestaudio",
        "--extract-audio",
        "--audio-format", "m4a",
        "--audio-quality", "0",
        "--output", str(cache_dir / (video_id + ".%(ext)s")),
    ]
    cookies_file = os.environ.get("YOUTUBE_COOKIES_FILE")
    if cookies_file and Path(cookies_file).exists():
        cmd += ["--cookies", cookies_file]
    cmd.append(song_url)

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log.error("yt-dlp error for %s: %s", song_url, result.stderr[-300:])
        state.blacklist_song(song_url)
        return None

    if cached.exists() and cached.stat().st_size > 0:
        return str(cached)

    log.error("Audio file not found after yt-dlp for %s", song_url)
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Philosopher Instagram Pipeline")
    parser.add_argument("--schedule", action="store_true", help="Schedule uploads at optimal times instead of uploading immediately")
    parser.add_argument("--single", action="store_true", help="Process only the philosopher with the fewest posts")
    parser.add_argument("--generate-only", action="store_true", help="Generate reel but do not upload (saves to output/)")
    args = parser.parse_args()
    try:
        main(upload_now=not args.schedule, single=args.single, generate_only=args.generate_only)
    except Exception as e:
        # Record the failure for the dashboard, then re-raise so CI goes red.
        run_metrics.write(mode=STYLE, status="error", summary=f"{type(e).__name__}: {e}")
        raise
