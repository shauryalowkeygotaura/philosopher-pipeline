"""Image and slideshow composition for philosopher reels.

- compose_frame(): one image + quote + translucent watermark -> 1080x1920 JPG.
- compose_slideshow(): N composed frames + audio -> fast-cut MP4 reel (uniform timing).
- compose_slideshow_beat_synced(): cuts land on song beats, xfade transitions,
  optional Ken Burns zoom. The "edited reel without CapCut" path.
- compose_image()/compose_reel(): backward-compat shims for legacy callers.

Tuned for IG Reels algorithm:
- 0.30 s/frame -> energetic but readable
- 8 s reel -> short enough that viewers loop 3-4 times before scrolling
- Seamless loop: last frame matches first frame so the loop boundary is invisible
- Quote uses serif (Playfair); attribution uses lighter sans-serif (Inter) when available
"""
import logging
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import ffmpeg
from PIL import Image, ImageDraw, ImageFilter, ImageFont

log = logging.getLogger(__name__)

REEL_WIDTH = 1080
REEL_HEIGHT = 1920
TEXT_MAX_WIDTH_RATIO = 0.76

# Quote (serif) sizing — smaller, more readable
QUOTE_FONT_START = 54
QUOTE_FONT_MIN = 26
QUOTE_FONT_STEP = 3

# Attribution (philosopher name) sizing — about 60% of fitted quote size
NAME_FONT_RATIO = 0.62
NAME_FONT_MIN = 22

WATERMARK_TEXT = "@deepahhthinking"
WATERMARK_OPACITY = 130
WATERMARK_FONT_SIZE = 30

# Reel pacing
DEFAULT_FRAME_DURATION = 0.30
DEFAULT_REEL_DURATION = 8
SEAMLESS_LOOP = True


def _load_font(font_path, size):
    """Load TTF; fall back to PIL default on failure."""
    try:
        return ImageFont.truetype(str(font_path), size)
    except (IOError, OSError):
        return ImageFont.load_default()


def _resolve_name_font(font_path):
    """Prefer a sans-serif (Inter) for the attribution; fall back to the quote font."""
    p = Path(font_path)
    candidates = [
        p.parent / "Inter-Medium.ttf",
        p.parent / "Inter-Regular.ttf",
        p.parent / "Inter.ttf",
        p,
    ]
    for c in candidates:
        if c.exists():
            return c
    return p


def compose_frame(
    image_path,
    quote,
    philosopher,
    output_path,
    font_path,
    watermark_text=WATERMARK_TEXT,
):
    """Render one slideshow frame: full-color image + quote + watermark."""
    p = Path(image_path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError("Image missing or empty: " + str(image_path))

    src = Image.open(BytesIO(p.read_bytes())).convert("RGB")
    base = _fit_to_reel_color(src)

    name_font_path = _resolve_name_font(font_path)

    quote_text = '“' + quote + '”'
    name_text = "— " + philosopher.upper()

    max_px_width = int(REEL_WIDTH * TEXT_MAX_WIDTH_RATIO)
    max_px_height = int(REEL_HEIGHT * 0.50)

    measure_draw = ImageDraw.Draw(base)

    # Auto-fit the quote
    quote_size = QUOTE_FONT_START
    quote_font = _load_font(font_path, quote_size)
    wrapped = _wrap_text(quote_text, quote_font, measure_draw, max_px_width)

    while quote_size > QUOTE_FONT_MIN:
        quote_font = _load_font(font_path, quote_size)
        wrapped = _wrap_text(quote_text, quote_font, measure_draw, max_px_width)
        bbox = measure_draw.multiline_textbbox((0, 0), wrapped, font=quote_font, spacing=10)
        if (bbox[2] - bbox[0]) <= max_px_width and (bbox[3] - bbox[1]) <= max_px_height:
            break
        quote_size -= QUOTE_FONT_STEP
    else:
        quote_font = _load_font(font_path, QUOTE_FONT_MIN)
        wrapped = _wrap_text(quote_text, quote_font, measure_draw, max_px_width)
        wrapped = _truncate_text(wrapped, quote_font, measure_draw, max_px_width)

    quote_bbox = measure_draw.multiline_textbbox((0, 0), wrapped, font=quote_font, spacing=10)
    quote_w = quote_bbox[2] - quote_bbox[0]
    quote_h = quote_bbox[3] - quote_bbox[1]

    name_size = max(NAME_FONT_MIN, int(quote_size * NAME_FONT_RATIO))
    name_font = _load_font(name_font_path, name_size)
    name_bbox = measure_draw.textbbox((0, 0), name_text, font=name_font)
    name_w = name_bbox[2] - name_bbox[0]
    name_h = name_bbox[3] - name_bbox[1]

    gap = max(18, int(quote_size * 0.45))
    total_h = quote_h + gap + name_h

    cx = REEL_WIDTH // 2
    cy = REEL_HEIGHT // 2
    quote_top = cy - total_h // 2
    name_top = quote_top + quote_h + gap

    # Translucent band behind text
    band_pad_x = 60
    band_pad_y = 48
    band_left = max(0, cx - max(quote_w, name_w) // 2 - band_pad_x)
    band_right = min(REEL_WIDTH, cx + max(quote_w, name_w) // 2 + band_pad_x)
    band_top = max(0, quote_top - band_pad_y)
    band_bot = min(REEL_HEIGHT, name_top + name_h + band_pad_y)

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle((band_left, band_top, band_right, band_bot), fill=(0, 0, 0, 140))

    # Quote (serif) — drop shadow + white
    odraw.multiline_text(
        (cx + 2, quote_top + 2), wrapped, font=quote_font, fill=(0, 0, 0, 220),
        align="center", anchor="ma", spacing=10,
    )
    odraw.multiline_text(
        (cx, quote_top), wrapped, font=quote_font, fill=(255, 255, 255, 255),
        align="center", anchor="ma", spacing=10,
    )

    # Attribution (sans-serif, lighter)
    odraw.text(
        (cx + 1, name_top + 1), name_text, font=name_font, fill=(0, 0, 0, 200),
        anchor="ma",
    )
    odraw.text(
        (cx, name_top), name_text, font=name_font, fill=(220, 220, 220, 245),
        anchor="ma",
    )

    # Watermark
    wfont = _load_font(font_path, WATERMARK_FONT_SIZE)
    wbbox = odraw.textbbox((0, 0), watermark_text, font=wfont)
    wtw = wbbox[2] - wbbox[0]
    wth = wbbox[3] - wbbox[1]
    wx = (REEL_WIDTH - wtw) // 2
    wy = REEL_HEIGHT - wth - 90
    odraw.text((wx + 1, wy + 1), watermark_text, font=wfont, fill=(0, 0, 0, WATERMARK_OPACITY))
    odraw.text((wx, wy), watermark_text, font=wfont, fill=(255, 255, 255, WATERMARK_OPACITY))

    final = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    final.save(output_path, "JPEG", quality=92)


def compose_slideshow(
    image_paths,
    quote,
    philosopher,
    audio_path,
    output_path,
    font_path,
    frame_duration=DEFAULT_FRAME_DURATION,
    reel_duration=DEFAULT_REEL_DURATION,
    seamless_loop=SEAMLESS_LOOP,
):
    """Render N image frames and concat them into a fast-cut reel with audio."""
    if not image_paths:
        raise ValueError("compose_slideshow requires at least one image")

    needed = max(1, int(round(reel_duration / frame_duration)))
    chosen = []
    while len(chosen) < needed:
        chosen.extend(image_paths)
    chosen = chosen[:needed]

    workdir = Path(tempfile.mkdtemp(prefix="reel-frames-"))
    try:
        frame_files = []
        for i, image in enumerate(chosen):
            frame_out = workdir / ("frame-" + str(i).zfill(4) + ".jpg")
            try:
                compose_frame(image, quote, philosopher, str(frame_out), font_path)
            except Exception as e:
                log.warning("frame %d failed (%s) - skipping that image", i, e)
                continue
            frame_files.append(frame_out)

        if not frame_files:
            raise RuntimeError("No frames composed - all images failed")

        log.info("Composed %d/%d frames from %d unique images",
                 len(frame_files), needed, len(image_paths))

        concat_path = workdir / "concat.txt"
        lines = []
        APOS = chr(39)
        for f in frame_files:
            lines.append("file " + APOS + f.as_posix() + APOS)
            lines.append("duration " + str(frame_duration))
        loop_target = frame_files[0] if seamless_loop and len(frame_files) > 1 else frame_files[-1]
        lines.append("file " + APOS + loop_target.as_posix() + APOS)
        concat_path.write_text("\n".join(lines), encoding="utf-8")

        try:
            video = ffmpeg.input(str(concat_path), format="concat", safe=0)
            audio = ffmpeg.input(audio_path, t=reel_duration)
            (
                ffmpeg
                .output(
                    video, audio, output_path,
                    vcodec="libx264", crf=23, pix_fmt="yuv420p",
                    acodec="aac", audio_bitrate="128k",
                    movflags="+faststart",
                    r=30,
                    t=reel_duration,
                )
                .overwrite_output()
                .run(quiet=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError("ffmpeg slideshow failed: " + e.stderr.decode()[-300:])
    finally:
        try:
            for f in workdir.iterdir():
                f.unlink()
            workdir.rmdir()
        except Exception:
            pass


def _fit_to_reel_color(img):
    """Fit image into 1080x1920 with a blurred enlarged copy as background."""
    target_ratio = REEL_WIDTH / REEL_HEIGHT
    w, h = img.size
    img_ratio = w / h

    if img_ratio > target_ratio:
        new_w_fg = REEL_WIDTH
        new_h_fg = max(1, int(REEL_WIDTH / img_ratio))
    else:
        new_h_fg = REEL_HEIGHT
        new_w_fg = max(1, int(REEL_HEIGHT * img_ratio))
    fg = img.resize((new_w_fg, new_h_fg), Image.LANCZOS)

    if img_ratio > target_ratio:
        bg_h = REEL_HEIGHT
        bg_w = max(REEL_WIDTH, int(bg_h * img_ratio))
    else:
        bg_w = REEL_WIDTH
        bg_h = max(REEL_WIDTH, int(bg_w / img_ratio))
    bg = img.resize((bg_w, bg_h), Image.LANCZOS)
    bg_x = (bg_w - REEL_WIDTH) // 2
    bg_y = (bg_h - REEL_HEIGHT) // 2
    bg = bg.crop((bg_x, bg_y, bg_x + REEL_WIDTH, bg_y + REEL_HEIGHT))
    bg = bg.filter(ImageFilter.GaussianBlur(radius=28))
    dim = Image.new("RGB", bg.size, (0, 0, 0))
    bg = Image.blend(bg, dim, 0.30)

    fg_x = (REEL_WIDTH - fg.width) // 2
    fg_y = (REEL_HEIGHT - fg.height) // 2
    bg.paste(fg, (fg_x, fg_y))
    return bg


def _wrap_text(text, font, draw, max_width):
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        current = []
        for word in words:
            test_line = " ".join(current + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] > max_width and current:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
    return "\n".join(lines)


def _truncate_text(text, font, draw, max_width):
    for i in range(len(text), 0, -1):
        candidate = text[:i] + "..."
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return candidate
    return "..."


def compose_image(photo_path, quote, philosopher, output_path, font_path):
    """Legacy entry point, color frame with watermark."""
    compose_frame(photo_path, quote, philosopher, output_path, font_path)


def compose_reel(image_path, audio_path, output_path, duration=30):
    """Legacy single-image static reel kept for backward compatibility."""
    video = ffmpeg.input(image_path, loop=1, framerate=30, t=duration)
    audio = ffmpeg.input(audio_path, t=duration)
    try:
        (
            ffmpeg
            .output(
                video, audio, output_path,
                vcodec="libx264", crf=23, pix_fmt="yuv420p",
                acodec="aac", audio_bitrate="128k",
                movflags="+faststart",
            )
            .overwrite_output()
            .run(quiet=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError("ffmpeg failed: " + e.stderr.decode()[-300:])


# ─── CapCut-style hard-cut reel ──────────────────────────────────────────────
# Hard cuts (no slide/wipe — those read as PowerPoint), in-clip 1.0 -> 1.30
# zoom punch alternating directions, unified color grade across every clip,
# and ONE persistent text overlay PNG riding on top of the whole 7s timeline
# (text is no longer rebaked into every JPG). Images shuffled, no back-to-back
# repeats. Gothic font for the quote when available, fallback otherwise.

# Fast-cut montage pacing. 0.10s is the practical floor: at 30fps that's
# 3 frames, the minimum at which the eye still registers an individual
# image vs just a strobe. Going below turns the reel into a blur.
MIN_SEGMENT_SECONDS = 0.10
MAX_SEGMENT_SECONDS = 0.28
MIN_CUTS_PER_REEL = 36

DEFAULT_COLOR_GRADE = "vintage"
COLOR_GRADES = {
    # Subtle desaturation + warm shadows + slight contrast bump
    "vintage": "eq=saturation=0.62:contrast=1.10:gamma=0.95,curves=preset=vintage",
    # Old-paper sepia tone, strong unification across paintings
    "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
    # Desaturated B/W high contrast
    "noir": "eq=saturation=0:contrast=1.20:gamma=0.95",
    # Cool-blue hazy film
    "cool": "eq=saturation=0.65:gamma_b=1.08:gamma_r=0.94",
    # Warm sun-faded film
    "warm": "eq=saturation=0.72:gamma_r=1.06:gamma_b=0.92",
    "off": "",
}


def detect_hits(audio_path, max_duration=8.0):
    """Return (hit_times, tempo_bpm) combining beats + onset transients.

    Onset detection fires on every drum hit, snare, vocal entry — far more
    cut points than beat tracking alone, so slow songs still cut frequently.
    """
    try:
        import librosa
    except ImportError:
        log.warning("librosa not installed — falling back to fixed 120 BPM grid")
        return _fixed_grid_beats(max_duration, 120.0), 120.0

    try:
        import numpy as np
        y, sr = librosa.load(str(audio_path), sr=None, duration=max_duration + 1.0)

        tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
        beats = np.atleast_1d(beat_times).ravel().tolist()

        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=False)
        onsets = np.atleast_1d(onset_times).ravel().tolist()

        # Union, dedupe within 0.10s, clip to window
        all_hits = sorted(set(round(float(t), 3) for t in (list(beats) + list(onsets))))
        hits = []
        for t in all_hits:
            if 0.0 < t < max_duration and (not hits or t - hits[-1] >= 0.10):
                hits.append(t)

        tempo_arr = np.atleast_1d(tempo).ravel()
        tempo_val = float(tempo_arr[0]) if tempo_arr.size > 0 else 120.0
    except Exception as e:
        log.warning("hit detection failed (%s) — falling back to fixed grid", e)
        return _fixed_grid_beats(max_duration, 120.0), 120.0

    if not hits or hits[0] > 0.05:
        hits.insert(0, 0.0)
    if not hits or hits[-1] < max_duration - 0.05:
        hits.append(float(max_duration))

    return hits, tempo_val


def _fixed_grid_beats(duration, bpm):
    """Generate evenly-spaced 'beat' times when librosa isn't available."""
    period = 60.0 / bpm
    t = 0.0
    out = [0.0]
    while t + period < duration:
        t += period
        out.append(round(t, 4))
    if out[-1] < duration - 0.05:
        out.append(float(duration))
    return out


def _segments_from_hits(hit_times, target_duration, min_cuts=MIN_CUTS_PER_REEL):
    """Convert hit times into segment durations and force minimum cut density.

    If fewer than min_cuts segments emerge from the audio analysis, the
    longest segments get split in half repeatedly until min_cuts is met.
    Net effect: even ballads get 16+ cuts in a 7s reel.
    """
    raw = [hit_times[i + 1] - hit_times[i] for i in range(len(hit_times) - 1)]
    cleaned = []
    for s in raw:
        if s < MIN_SEGMENT_SECONDS:
            if cleaned:
                cleaned[-1] += s
            else:
                cleaned.append(s)
        elif s > MAX_SEGMENT_SECONDS:
            n_splits = int(s / MAX_SEGMENT_SECONDS) + 1
            piece = s / n_splits
            cleaned.extend([piece] * n_splits)
        else:
            cleaned.append(s)

    if not cleaned:
        per = target_duration / max(min_cuts, 1)
        return [per] * min_cuts

    # Force minimum cut count by halving the largest segment until we hit min_cuts
    safety = 0
    while len(cleaned) < min_cuts and safety < 200:
        max_idx = cleaned.index(max(cleaned))
        biggest = cleaned.pop(max_idx)
        cleaned.insert(max_idx, biggest / 2.0)
        cleaned.insert(max_idx, biggest / 2.0)
        safety += 1

    total = sum(cleaned)
    if total <= 0:
        per = target_duration / max(min_cuts, 1)
        return [per] * min_cuts
    scale = target_duration / total
    return [s * scale for s in cleaned]


def _select_images_for_cuts(image_paths, n_cuts, seamless_loop=True):
    """Pick n_cuts images, preserving caller's ordering (FIFO, no shuffle).

    Callers (pipeline.py) hand us a list already interleaved
    painting-portrait-painting-portrait. Shuffling kills that, so two
    near-identical B&W portraits of the same philosopher can land
    back-to-back and read as "one image held for several cuts". FIFO popping
    preserves the alternation: cut 0 is a painting, cut 1 a portrait,
    cut 2 a painting, etc., regardless of how many cuts we need.

    If seamless_loop=True, the last image equals the first so the IG
    auto-replay boundary is invisible.
    """
    if not image_paths:
        return []
    if len(image_paths) == 1:
        return [image_paths[0]] * n_cuts

    pool_template = list(image_paths)
    pool = list(pool_template)
    out = []
    last = None
    while len(out) < n_cuts:
        if not pool:
            pool = list(pool_template)
        pick = pool.pop(0)  # FIFO so the interleave order is honored
        if pick == last and pool:
            alt = pool.pop(0)
            pool.append(pick)  # try `pick` again after the pool cycles
            pick = alt
        out.append(pick)
        last = pick

    if seamless_loop and n_cuts > 1:
        out[-1] = out[0]
    return out


def _resolve_overlay_font(overlay_font_path, fallback_font_path):
    """Pick the gothic font if it exists, else the fallback (Playfair)."""
    if overlay_font_path:
        try:
            ImageFont.truetype(str(overlay_font_path), 12)
            return overlay_font_path
        except (IOError, OSError):
            pass
    return fallback_font_path


def _render_quote_overlay(quote, philosopher, output_path, gothic_font_path, watermark_font_path):
    """Render the persistent transparent 1080x1920 text overlay.

    One PNG drawn once and overlaid onto the whole video timeline — text never
    rebakes per frame, never jitters across cuts. Smaller fonts than the old
    per-frame version, gothic typeface for the quote+name, plain font for the
    watermark.
    """
    img = Image.new("RGBA", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Smaller text than baked-frame version — gothic reads heavier per pixel.
    quote_size_start = 38
    quote_size_min = 18
    quote_size_step = 2
    name_size_ratio = 0.55
    name_size_min = 16
    line_spacing = 8

    quote_text = '"' + quote + '"'
    name_text = "- " + philosopher.upper()

    max_px_width = int(REEL_WIDTH * 0.72)
    max_px_height = int(REEL_HEIGHT * 0.40)

    quote_size = quote_size_start
    quote_font = _load_font(gothic_font_path, quote_size)
    wrapped = _wrap_text(quote_text, quote_font, draw, max_px_width)
    while quote_size > quote_size_min:
        quote_font = _load_font(gothic_font_path, quote_size)
        wrapped = _wrap_text(quote_text, quote_font, draw, max_px_width)
        bbox = draw.multiline_textbbox((0, 0), wrapped, font=quote_font, spacing=line_spacing)
        if (bbox[2] - bbox[0]) <= max_px_width and (bbox[3] - bbox[1]) <= max_px_height:
            break
        quote_size -= quote_size_step
    else:
        quote_font = _load_font(gothic_font_path, quote_size_min)
        wrapped = _wrap_text(quote_text, quote_font, draw, max_px_width)
        wrapped = _truncate_text(wrapped, quote_font, draw, max_px_width)

    quote_bbox = draw.multiline_textbbox((0, 0), wrapped, font=quote_font, spacing=line_spacing)
    quote_w = quote_bbox[2] - quote_bbox[0]
    quote_h = quote_bbox[3] - quote_bbox[1]

    name_size = max(name_size_min, int(quote_size * name_size_ratio))
    name_font = _load_font(gothic_font_path, name_size)
    name_bbox = draw.textbbox((0, 0), name_text, font=name_font)
    name_w = name_bbox[2] - name_bbox[0]
    name_h = name_bbox[3] - name_bbox[1]

    gap = max(12, int(quote_size * 0.40))
    total_h = quote_h + gap + name_h
    cx = REEL_WIDTH // 2
    cy = REEL_HEIGHT // 2
    quote_top = cy - total_h // 2
    name_top = quote_top + quote_h + gap

    # Slim band behind text for legibility against any image
    band_pad_x = 36
    band_pad_y = 24
    band_left = max(0, cx - max(quote_w, name_w) // 2 - band_pad_x)
    band_right = min(REEL_WIDTH, cx + max(quote_w, name_w) // 2 + band_pad_x)
    band_top = max(0, quote_top - band_pad_y)
    band_bot = min(REEL_HEIGHT, name_top + name_h + band_pad_y)
    draw.rectangle((band_left, band_top, band_right, band_bot), fill=(0, 0, 0, 110))

    # Quote (gothic) — soft drop shadow + white
    draw.multiline_text(
        (cx + 1, quote_top + 1), wrapped, font=quote_font, fill=(0, 0, 0, 200),
        align="center", anchor="ma", spacing=line_spacing,
    )
    draw.multiline_text(
        (cx, quote_top), wrapped, font=quote_font, fill=(255, 255, 255, 255),
        align="center", anchor="ma", spacing=line_spacing,
    )

    # Attribution (same gothic, smaller, dimmer)
    draw.text((cx + 1, name_top + 1), name_text, font=name_font, fill=(0, 0, 0, 200), anchor="ma")
    draw.text((cx, name_top), name_text, font=name_font, fill=(220, 220, 220, 240), anchor="ma")

    # Watermark — plain font (gothic blackletter is unreadable at small sizes)
    wm = "@deepahhthinking"
    wm_size = 22
    wm_font = _load_font(watermark_font_path, wm_size)
    wbbox = draw.textbbox((0, 0), wm, font=wm_font)
    wx = (REEL_WIDTH - (wbbox[2] - wbbox[0])) // 2
    wy = REEL_HEIGHT - (wbbox[3] - wbbox[1]) - 80
    draw.text((wx + 1, wy + 1), wm, font=wm_font, fill=(0, 0, 0, 130))
    draw.text((wx, wy), wm, font=wm_font, fill=(255, 255, 255, 130))

    img.save(output_path, "PNG")


# zoompan's x/y expressions do NOT have `d` as a variable (only `on`, `iw`,
# `ih`, `zoom`, `pzoom`, `x`, `y`, `px`, `py`). To make the pan progress run
# from 0 -> 1 across the clip, the frame count is substituted into the
# expression at filter-build time (DUR placeholder -> str(d_frames)).
_PAN_TEMPLATES = [
    ("iw/2-(iw/zoom/2)",                                  "ih/2-(ih/zoom/2)"),                                  # 0 static center
    ("iw/2-(iw/zoom/2)+(iw-iw/zoom)/2*on/DUR",            "ih/2-(ih/zoom/2)"),                                  # 1 pan right
    ("iw/2-(iw/zoom/2)-(iw-iw/zoom)/2*on/DUR",            "ih/2-(ih/zoom/2)"),                                  # 2 pan left
    ("iw/2-(iw/zoom/2)",                                  "ih/2-(ih/zoom/2)+(ih-ih/zoom)/2*on/DUR"),            # 3 pan down
    ("iw/2-(iw/zoom/2)",                                  "ih/2-(ih/zoom/2)-(ih-ih/zoom)/2*on/DUR"),            # 4 pan up
    ("iw/2-(iw/zoom/2)+(iw-iw/zoom)/3*on/DUR",            "ih/2-(ih/zoom/2)-(ih-ih/zoom)/3*on/DUR"),            # 5 diag NE
    ("iw/2-(iw/zoom/2)-(iw-iw/zoom)/3*on/DUR",            "ih/2-(ih/zoom/2)+(ih-ih/zoom)/3*on/DUR"),            # 6 diag SW
]


def _capcut_clip_filter(idx, length_sec, color_grade_filter, flash=False):
    """Per-clip ffmpeg filter chain, CapCut-style.

    Outputs every clip as 1080x1920 30fps yuv420p so the concat filter
    accepts them. Each clip gets:
      - 1.00 -> 1.45 punch zoom alternating in/out by clip index
      - Random parallax pan direction (7-way), pan templates parameterised by
        the clip's frame count so on/DUR maps to a 0 -> 1 progress sweep
      - Per-clip subtle exposure + saturation variance, sells the "edited" feel
      - Unified color grade applied AFTER all motion so cuts read as one reel

    `flash` is accepted for caller compatibility but currently a no-op:
    a per-clip flash would need ffmpeg's `eq=...:eval=frame` and is fragile,
    so the flash punch is instead applied as a post-concat curve later.
    """
    d_frames = max(1, int(length_sec * 30))
    # Bigger zoom range than v3 (1.30 -> 1.45) so even a 0.20s clip has visible motion
    if idx % 2 == 0:
        z_expr = "min(zoom+0.0055,1.45)"
    else:
        z_expr = "if(eq(on,0),1.45,max(zoom-0.0055,1.00))"

    # Deterministic pan-mode pick so retries are stable, but every clip is different
    pan_x_t, pan_y_t = _PAN_TEMPLATES[(idx * 3 + 1) % len(_PAN_TEMPLATES)]
    dur_str = str(max(2, d_frames))  # avoid divide-by-1 making pan instant
    pan_x = pan_x_t.replace("DUR", dur_str)
    pan_y = pan_y_t.replace("DUR", dur_str)

    chain = [
        "scale=2160:3840:force_original_aspect_ratio=increase",
        "crop=2160:3840",
        ("zoompan=z='" + z_expr + "'"
         + ":x='" + pan_x + "':y='" + pan_y + "'"
         + ":d=" + str(d_frames) + ":s=1080x1920:fps=30"),
    ]
    # Per-clip exposure jitter for that hand-cut feel: +/- 4% brightness, +/- 5% saturation
    bright_jitter = ((idx * 37) % 9 - 4) / 100.0
    sat_jitter = 1.0 + ((idx * 53) % 11 - 5) / 100.0
    chain.append("eq=brightness=%.3f:saturation=%.3f" % (bright_jitter, sat_jitter))
    if color_grade_filter:
        chain.append(color_grade_filter)
    chain.append("setsar=1")
    chain.append("format=yuv420p")
    return "[" + str(idx) + ":v]" + ",".join(chain) + "[v" + str(idx) + "]"


def compose_slideshow_beat_synced(
    image_paths,
    quote,
    philosopher,
    audio_path,
    output_path,
    font_path,
    reel_duration=7.0,
    min_cuts=MIN_CUTS_PER_REEL,
    seamless_loop=True,
    color_grade=DEFAULT_COLOR_GRADE,
    overlay_font_path=None,
):
    """CapCut-style hard-cut reel.

    - Cuts on detected beats+onsets, force min_cuts segments in the window
    - Each clip: 1.0 -> 1.30 alternating zoom punch + unified color grade
    - HARD CUTS via concat filter (no slide/wipe — those read as PowerPoint)
    - ONE persistent text overlay PNG sits on top of the entire timeline
    - Gothic overlay font (UnifrakturMaguntia) when available, else Playfair
    """
    if not image_paths:
        raise ValueError("compose_slideshow_beat_synced requires at least one image")

    hits, tempo = detect_hits(audio_path, max_duration=reel_duration)
    segments = _segments_from_hits(hits, reel_duration, min_cuts=min_cuts)
    n_segments = len(segments)

    chosen_images = _select_images_for_cuts(image_paths, n_segments, seamless_loop=seamless_loop)
    unique_count = len(set(chosen_images))

    grade_filter = COLOR_GRADES.get(color_grade, COLOR_GRADES[DEFAULT_COLOR_GRADE])
    overlay_font = _resolve_overlay_font(overlay_font_path, font_path)

    log.info(
        "capcut reel v3: tempo=%.0f BPM, %d hard cuts, %d unique images, grade=%s, gothic=%s, durations=%s",
        tempo, n_segments, unique_count, color_grade,
        bool(overlay_font_path) and str(overlay_font) == str(overlay_font_path),
        ["%.2f" % s for s in segments],
    )

    workdir = Path(tempfile.mkdtemp(prefix="reel-cc3-"))
    try:
        # 1. Pre-render the persistent text overlay (gothic font, smaller text, watermark)
        overlay_png = workdir / "quote_overlay.png"
        _render_quote_overlay(quote, philosopher, str(overlay_png), overlay_font, font_path)

        # 2. ffmpeg: per-clip scale+zoom+grade, concat them, overlay text, mux audio.
        # `-framerate 30` is critical: ffmpeg defaults stills to 25fps, but our
        # zoompan filter uses d=int(seg*30) frames per cycle. A 25-vs-30 mismatch
        # caused zoompan to restart its zoom inside a single clip, reading on
        # screen as 3-4 cuts of the same image before moving on. Locking input
        # to 30fps means d == input-frame-count, exactly one zoom cycle per clip.
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats"]
        for i, src in enumerate(chosen_images):
            cmd += ["-loop", "1", "-framerate", "30", "-t", "%.4f" % segments[i], "-i", str(src)]
        cmd += ["-i", str(overlay_png)]
        overlay_idx = len(chosen_images)
        cmd += ["-i", str(audio_path)]
        audio_idx = len(chosen_images) + 1

        # Flash punches: every 4th cut + the first and last segment. These
        # land where the eye expects a "drop" so the reel reads as edited.
        flash_indices = set([0, len(chosen_images) - 1])
        flash_indices.update(range(3, len(chosen_images), 4))

        parts = []
        for i in range(len(chosen_images)):
            parts.append(_capcut_clip_filter(
                i, segments[i], grade_filter, flash=(i in flash_indices)
            ))
        chain_inputs = "".join("[v%d]" % i for i in range(len(chosen_images)))
        parts.append(chain_inputs + "concat=n=" + str(len(chosen_images)) + ":v=1:a=0[vcat]")
        # Cinematic finishing chain on the full timeline:
        #   vignette  -> darken edges, sells filmic feel
        #   noise     -> subtle film grain (4 alpha, temporal so it moves)
        #   unsharp   -> micro-sharpening, makes paintings feel HD
        parts.append(
            "[vcat]"
            "vignette=PI/5,"
            "noise=alls=6:allf=t,"
            "unsharp=3:3:0.4:3:3:0.0"
            "[vfin]"
        )
        parts.append("[vfin][" + str(overlay_idx) + ":v]overlay=0:0:format=auto[vout]")

        cmd += [
            "-filter_complex", ";".join(parts),
            "-map", "[vout]",
            "-map", "%d:a" % audio_idx,
            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-r", "30",
            "-t", "%.4f" % reel_duration,
            "-shortest",
            str(output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or "")[-700:]
            raise RuntimeError("ffmpeg capcut compose failed: " + tail)
    finally:
        try:
            for f in workdir.iterdir():
                f.unlink()
            workdir.rmdir()
        except Exception:
            pass


# Back-compat aliases so older callers (and earlier function names) still work.
detect_beats = detect_hits
_segments_from_beats = _segments_from_hits


# ─── Kinetic letterbox reel (@wisdomofhidgon-style) ──────────────────────────
# Slow cinematic band of image (~28% of frame height), pure black above/below,
# word-by-word red serif text reveal timed to beat/onset hits. One mid-reel
# splash card with a red banner + philosopher name. 28s long-form bait, ~3-4
# image cuts vs the 36-cut CapCut mode.

KINETIC_REEL_DURATION = 28.0
KINETIC_BAND_HEIGHT = 540   # 28% of 1920, centered vertically (y=690..1230)
KINETIC_RED = (200, 0, 24)  # #C80018, the @wisdomofhidgon crimson
KINETIC_SPLASH_RED = (193, 12, 28)
KINETIC_CREAM = (235, 222, 200)


def _split_quote_to_phrases(quote, n_target=14):
    """Split quote into ~n_target word chunks (1-3 words each), preferring
    natural breaks at commas / dashes / semicolons.
    """
    raw = quote.replace("—", ",").replace("–", ",").replace(";", ",")
    pieces = [p.strip() for p in raw.split(",") if p.strip()]

    chunks = []
    for piece in pieces:
        words = piece.split()
        if not words:
            continue
        if len(words) <= 3:
            chunks.append(" ".join(words))
            continue
        # pack into groups of 2 (with occasional 3) to hit n_target
        i = 0
        while i < len(words):
            take = 2 if (i + 2 <= len(words)) else (len(words) - i)
            chunks.append(" ".join(words[i:i + take]))
            i += take

    # Coerce toward n_target by merging shortest neighbors if too many
    while len(chunks) > n_target and len(chunks) > 1:
        idx = min(range(len(chunks) - 1), key=lambda k: len(chunks[k]) + len(chunks[k + 1]))
        chunks[idx] = chunks[idx] + " " + chunks[idx + 1]
        del chunks[idx + 1]

    return chunks


def _phrase_style(idx, total):
    """Deterministic style permutation per phrase index. Returns one of:
    'plain' | 'bracket' | 'underline' | 'caps'. Last phrase always 'caps' for
    the climax beat (mirrors the reference reel's UPPERCASE finale).
    """
    if idx == total - 1:
        return "caps"
    if idx == 0:
        return "plain"
    return ("plain", "bracket", "underline", "caps")[idx % 4]


def _render_kinetic_phrase_png(text, style, output_path, font_path, color=KINETIC_RED):
    """Render one phrase as a 1080x1920 transparent PNG with red serif text.
    Position: vertically centered in the TOP black bar above the letterbox band.
    """
    img = Image.new("RGBA", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    display = text
    if style == "bracket":
        display = "[" + text + "]"
    elif style == "caps":
        display = text.upper()

    # Auto-fit a single line into the top black bar (y=0..690).
    max_px_width = int(REEL_WIDTH * 0.84)
    max_px_height = int((REEL_HEIGHT - KINETIC_BAND_HEIGHT) / 2 * 0.55)
    size = 110 if style == "caps" else 96
    while size > 36:
        font = _load_font(font_path, size)
        bbox = draw.textbbox((0, 0), display, font=font)
        if (bbox[2] - bbox[0]) <= max_px_width and (bbox[3] - bbox[1]) <= max_px_height:
            break
        size -= 4
    else:
        font = _load_font(font_path, 36)

    bbox = draw.textbbox((0, 0), display, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    # Anchor in the upper-third of the frame, above the letterbox band
    band_top = (REEL_HEIGHT - KINETIC_BAND_HEIGHT) // 2
    cy = band_top // 2
    cx = REEL_WIDTH // 2

    draw.text((cx, cy), display, font=font, fill=color + (255,), anchor="mm")

    if style == "underline":
        underline_y = cy + th // 2 + 14
        thickness = max(3, int(size * 0.06))
        draw.line(
            (cx - tw // 2, underline_y, cx + tw // 2, underline_y),
            fill=color + (255,), width=thickness,
        )

    img.save(output_path, "PNG")


def _render_kinetic_splash_card(philosopher, output_path, font_path, handle="@deepahhthinking"):
    """Mid-reel splash card: full red band with bordered serif name + tiny handle.
    Mirrors the @wisdomofhidgon t=8s frame.
    """
    img = Image.new("RGBA", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)

    band_top = (REEL_HEIGHT - KINETIC_BAND_HEIGHT) // 2
    band_bot = band_top + KINETIC_BAND_HEIGHT
    draw.rectangle((0, band_top, REEL_WIDTH, band_bot), fill=KINETIC_SPLASH_RED + (255,))

    # Fit philosopher name into ~80% of band width
    name_text = philosopher.upper()
    max_w = int(REEL_WIDTH * 0.80)
    max_h = int(KINETIC_BAND_HEIGHT * 0.50)
    size = 180
    while size > 48:
        f = _load_font(font_path, size)
        b = draw.textbbox((0, 0), name_text, font=f)
        if (b[2] - b[0]) <= max_w and (b[3] - b[1]) <= max_h:
            break
        size -= 6
    else:
        f = _load_font(font_path, 48)

    cx = REEL_WIDTH // 2
    cy = band_top + KINETIC_BAND_HEIGHT // 2 - 30
    # Hollow border effect: draw text 4 times slightly offset in cream, then white center
    for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3)):
        draw.text((cx + dx, cy + dy), name_text, font=f, fill=KINETIC_CREAM + (255,), anchor="mm")
    draw.text((cx, cy), name_text, font=f, fill=(255, 255, 255, 255), anchor="mm")

    # Handle, small italic-feel beneath
    handle_size = max(28, size // 6)
    hf = _load_font(font_path, handle_size)
    hcy = cy + (max_h // 2) + handle_size
    draw.text((cx, hcy), handle, font=hf, fill=KINETIC_CREAM + (200,), anchor="mm")

    img.save(output_path, "PNG")


def _kinetic_image_filter(idx, length_sec):
    """Per-image filter chain for the letterbox base.

    Scales/crops the source to a 1080-wide x 540-tall horizontal band, applies
    a slow Ken Burns zoom using crop's continuous `t` variable, then pads to
    full 1080x1920 with black above and below the band.

    Why not zoompan: zoompan's `d` is frames per INPUT image. With our
    `-loop 1 -t SEG -framerate 30` inputs producing SEG*30 input frames AND
    `d=SEG*30`, zoompan emits SEG*30 frames per input frame -> (SEG*30)^2
    total output, blowing past the concat budget and starving every segment
    after the first. The animated-crop pattern below sidesteps that entirely.
    """
    # Slow continuous zoom 1.00 -> ~1.08 across the segment, driven by
    # crop's per-frame `t` variable (seconds since input start).
    zoom = "(1+0.04*min(t/%.4f,1))" % max(0.1, length_sec)
    crop_w = "2160/%s" % zoom
    crop_h = "1080/%s" % zoom
    crop_x = "(2160-2160/%s)/2" % zoom
    crop_y = "(1080-1080/%s)/2" % zoom
    chain = [
        "scale=2160:1080:force_original_aspect_ratio=increase",
        "crop=2160:1080",
        "crop=w='%s':h='%s':x='%s':y='%s'" % (crop_w, crop_h, crop_x, crop_y),
        "scale=1080:540",
        "eq=saturation=0.92:contrast=1.05",
        "pad=1080:1920:0:690:color=black",
        "setsar=1",
        "format=yuv420p",
    ]
    return "[" + str(idx) + ":v]" + ",".join(chain) + "[v" + str(idx) + "]"


def compose_kinetic_letterbox(
    image_paths,
    quote,
    philosopher,
    audio_path,
    output_path,
    font_path,
    reel_duration=KINETIC_REEL_DURATION,
    splash_at_fraction=0.27,
    splash_duration=2.6,
):
    """@wisdomofhidgon-style reel: letterbox band image + word-by-word red
    kinetic typography + mid-reel splash card.

    Pacing: text reveals on beat/onset hits (reuses detect_hits). 4 image
    clips around the splash card, each lingering ~5-7s with a slow 1.00 -> 1.08
    zoom. Pure black 9:16 canvas, image squeezed into a 540-tall horizontal
    band centered vertically.
    """
    if not image_paths:
        raise ValueError("compose_kinetic_letterbox requires at least one image")

    hits, tempo = detect_hits(audio_path, max_duration=reel_duration)

    phrases = _split_quote_to_phrases(quote, n_target=14)

    # Image segment plan: 4 image clips around a centered splash card.
    splash_t = reel_duration * splash_at_fraction
    pre_pool = max(1, min(2, len(image_paths) // 2))
    post_pool = 3
    pre_dur = splash_t / pre_pool
    post_total = reel_duration - splash_t - splash_duration
    post_dur = post_total / post_pool

    # Pick images, no back-to-back repeats across the small set
    chosen = []
    pool = list(image_paths)
    last = None
    for _ in range(pre_pool + post_pool):
        if not pool:
            pool = list(image_paths)
        pick = pool.pop(0)
        if pick == last and pool:
            alt = pool.pop(0)
            pool.append(pick)
            pick = alt
        chosen.append(pick)
        last = pick

    log.info(
        "kinetic letterbox: tempo=%.0f BPM, %d hits, %d phrases, %d image segs (pre=%.1fs each, splash=%.1fs, post=%.1fs each)",
        tempo, len(hits), len(phrases), len(chosen), pre_dur, splash_duration, post_dur,
    )

    workdir = Path(tempfile.mkdtemp(prefix="kinetic-"))
    try:
        # 1. Render splash card PNG (used as one of the image inputs)
        splash_png = workdir / "splash.png"
        _render_kinetic_splash_card(philosopher, str(splash_png), font_path)

        # 2. Render one PNG per phrase
        phrase_pngs = []
        for i, phrase in enumerate(phrases):
            png = workdir / ("phrase_%02d.png" % i)
            style = _phrase_style(i, len(phrases))
            _render_kinetic_phrase_png(phrase, style, str(png), font_path)
            phrase_pngs.append(png)

        # 3. Assemble ffmpeg invocation.
        # Inputs: chosen[0..pre_pool-1] as image segments at pre_dur each,
        #         splash.png as one segment at splash_duration,
        #         chosen[pre_pool..] as post-splash segments at post_dur each,
        #         then all phrase PNGs (still images, not concatted),
        #         then audio.
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-stats"]
        segments = []
        for i in range(pre_pool):
            cmd += ["-loop", "1", "-framerate", "30", "-t", "%.4f" % pre_dur, "-i", str(chosen[i])]
            segments.append(("img", pre_dur))
        cmd += ["-loop", "1", "-framerate", "30", "-t", "%.4f" % splash_duration, "-i", str(splash_png)]
        segments.append(("splash", splash_duration))
        for i in range(pre_pool, pre_pool + post_pool):
            cmd += ["-loop", "1", "-framerate", "30", "-t", "%.4f" % post_dur, "-i", str(chosen[i])]
            segments.append(("img", post_dur))

        n_video_inputs = len(segments)
        for png in phrase_pngs:
            cmd += ["-loop", "1", "-i", str(png)]
        phrase_input_offset = n_video_inputs
        cmd += ["-i", str(audio_path)]
        audio_idx = n_video_inputs + len(phrase_pngs)

        parts = []
        # Per-segment filter: letterbox-zoom for image segments, passthrough for splash
        for i, (kind, dur) in enumerate(segments):
            if kind == "splash":
                parts.append(
                    "[" + str(i) + ":v]scale=1080:1920,setsar=1,format=yuv420p[v" + str(i) + "]"
                )
            else:
                parts.append(_kinetic_image_filter(i, dur))

        chain_inputs = "".join("[v%d]" % i for i in range(n_video_inputs))
        parts.append(chain_inputs + "concat=n=" + str(n_video_inputs) + ":v=1:a=0[vcat]")

        # Phrase overlay timing: spread phrases EVENLY across the timeline,
        # snap each anchor to the nearest detected hit within +/- 0.4s, then
        # hold each phrase from its anchor until the next phrase's anchor.
        # The naive hits[0..N-1] approach clumped every phrase in the first
        # 3s because dense onset detection front-loaded the hit list.
        phrase_times = []
        n_phrases = len(phrase_pngs)
        # Skip the splash window when laying down anchors
        usable_span = reel_duration - splash_duration
        anchor_step = usable_span / max(1, n_phrases)
        anchors = []
        for i in range(n_phrases):
            t = i * anchor_step + anchor_step / 2
            if t >= splash_t:
                t += splash_duration
            anchors.append(t)

        # Snap each anchor to nearest hit within tolerance
        snap_tol = 0.4
        snapped = []
        for a in anchors:
            best = a
            if hits:
                nearest = min(hits, key=lambda h: abs(h - a))
                if abs(nearest - a) <= snap_tol:
                    best = nearest
            # Don't let snapping land inside the splash window
            if splash_t - 0.05 < best < splash_t + splash_duration + 0.05:
                best = a
            snapped.append(best)
        snapped.sort()

        for i, start in enumerate(snapped):
            if i + 1 < len(snapped):
                end = snapped[i + 1]
            else:
                end = reel_duration
            # Clamp around the splash window
            if start < splash_t + splash_duration and end > splash_t:
                if start < splash_t:
                    end = min(end, splash_t)
                elif end > splash_t + splash_duration:
                    start = max(start, splash_t + splash_duration)
                else:
                    continue
            if end - start < 0.20:
                continue
            phrase_times.append((i, start, end))

        # Chain text overlays on top of the concatted letterbox base
        current = "[vcat]"
        for n, (i, start, end) in enumerate(phrase_times):
            out_label = "[vt%d]" % n if n < len(phrase_times) - 1 else "[vout]"
            inp = "[" + str(phrase_input_offset + i) + ":v]"
            parts.append(
                current + inp +
                "overlay=0:0:enable='between(t," + ("%.3f" % start) + "," + ("%.3f" % end) + ")'" +
                out_label
            )
            current = out_label

        if not phrase_times:
            parts.append("[vcat]null[vout]")

        cmd += [
            "-filter_complex", ";".join(parts),
            "-map", "[vout]",
            "-map", "%d:a" % audio_idx,
            "-c:v", "libx264", "-crf", "22", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            "-r", "30",
            "-t", "%.4f" % reel_duration,
            "-shortest",
            str(output_path),
        ]

        log.info("kinetic filter_complex (%d parts): %s", len(parts), " || ".join(parts))
        log.info("kinetic phrase_times: %s", phrase_times)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            tail = (result.stderr or "")[-700:]
            raise RuntimeError("ffmpeg kinetic letterbox failed: " + tail)
    finally:
        try:
            for f in workdir.iterdir():
                f.unlink()
            workdir.rmdir()
        except Exception:
            pass


# ─── Kinetic v2: 5-beat reel matching @wisdomofhidgon's actual format ────────
# Reference: reference/wisdomofhidgon-DIA_e3dI9tq.mp4
# Spec: Projects/philosopher-pipeline/kinetic-format-spec.md

V2_BRAND_BAND_TOP = 400
V2_BRAND_BAND_HEIGHT = 320
V2_BODY_BAND_TOP = 690
V2_BODY_BAND_HEIGHT = 540


def _render_v2_hook_word(word, output_path, font_path, is_last=False):
    """Beat 1: small red italic serif word on pure black + thin red underline.
    `is_last` wraps in [brackets] to mark the key word."""
    img = Image.new("RGBA", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0, 255))
    draw = ImageDraw.Draw(img)
    display = "[" + word + "]" if is_last else word
    font = _load_font(font_path, 96)
    bbox = draw.textbbox((0, 0), display, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    cx = REEL_WIDTH // 2
    cy = int(REEL_HEIGHT * 0.56)
    draw.text((cx, cy), display, font=font, fill=KINETIC_RED + (255,), anchor="mm")
    if not is_last:
        underline_y = cy + th // 2 + 14
        draw.line(
            (cx - tw // 2 - 4, underline_y, cx + tw // 2 + 4, underline_y),
            fill=KINETIC_RED + (255,), width=4,
        )
    img.save(output_path, "PNG")


def _render_v2_black(output_path):
    Image.new("RGB", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0)).save(output_path, "PNG")


def _render_v2_brand_card(philosopher, portrait_path, output_path, font_path,
                          subtitle="Wisdom of hidgon"):
    """Beat 3: red band with philosopher name + portrait cut INSIDE band on right."""
    img = Image.new("RGB", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    band_top = V2_BRAND_BAND_TOP
    band_h = V2_BRAND_BAND_HEIGHT
    band_bot = band_top + band_h
    draw.rectangle((0, band_top, REEL_WIDTH, band_bot), fill=KINETIC_SPLASH_RED)

    portrait_w = 0
    if portrait_path and Path(portrait_path).exists():
        try:
            p = Image.open(portrait_path).convert("RGB")
            target_h = band_h + 80
            target_w = target_h
            pw, ph = p.size
            scale = max(target_w / pw, target_h / ph)
            p_resized = p.resize((int(pw * scale), int(ph * scale)), Image.LANCZOS)
            left = (p_resized.width - target_w) // 2
            # Upper-third bias for portrait sources (faces sit in upper third).
            # Pure center crop on a tall standing-figure photo cuts the face off.
            excess_v = p_resized.height - target_h
            top = int(excess_v * 0.22) if ph > pw else excess_v // 2
            p_cropped = p_resized.crop((left, top, left + target_w, top + target_h))
            paste_x = REEL_WIDTH - target_w + 30
            paste_y = band_top - 40
            img.paste(p_cropped, (paste_x, paste_y))
            portrait_w = target_w
        except Exception as e:
            log.warning("Brand card portrait paste failed: %s", e)

    parts = philosopher.upper().strip().split()
    line1 = parts[0] if parts else ""
    line2 = " ".join(parts[1:]) if len(parts) >= 2 else ""

    avail_w = REEL_WIDTH - portrait_w - 80
    title_size = 130
    while title_size > 40:
        title_f = _load_font(font_path, title_size)
        b1 = draw.textbbox((0, 0), line1, font=title_f)
        b2 = draw.textbbox((0, 0), line2, font=title_f) if line2 else (0, 0, 0, 0)
        if (b1[2] - b1[0]) <= avail_w and (b2[2] - b2[0]) <= avail_w:
            break
        title_size -= 6
    else:
        title_f = _load_font(font_path, 40)

    lx = 50
    ly1 = band_top + 30
    draw.text((lx, ly1), line1, font=title_f, fill=(245, 235, 215), anchor="lt")
    if line2:
        ly2 = ly1 + title_size + 4
        draw.text((lx, ly2), line2, font=title_f, fill=(0, 0, 0), anchor="lt")

    sub_f = _load_font(font_path, 30)
    sub_y = band_bot - 28
    draw.text((lx, sub_y), subtitle, font=sub_f, fill=(245, 235, 215), anchor="lb")

    img.save(output_path, "PNG")


def _render_v2_body_frame(image_path, text, output_path, font_path, is_bracket=False):
    """Beat 4: letterbox image band + red italic phrase.

    Short phrases sit in the black strip BELOW the band. Long phrases (that would
    overflow one line at full size) are auto-shrunk, wrapped, and laid OVER the
    top of the image with a dark scrim so they stay readable (user: 'sometimes i
    cant read it ... put the new text on top of the picture instead of bottom')."""
    img = Image.new("RGB", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0))
    band_top = V2_BODY_BAND_TOP
    band_h = V2_BODY_BAND_HEIGHT

    if image_path and Path(image_path).exists():
        try:
            p = Image.open(image_path).convert("RGB")
            pw, ph = p.size
            scale = max(REEL_WIDTH / pw, band_h / ph)
            p_resized = p.resize((int(pw * scale), int(ph * scale)), Image.LANCZOS)
            left = (p_resized.width - REEL_WIDTH) // 2
            # Portraits: bias crop toward upper portion so faces stay in frame.
            # Landscape paintings: center crop keeps the composition.
            excess_v = p_resized.height - band_h
            top = int(excess_v * 0.22) if ph > pw else excess_v // 2
            p_cropped = p_resized.crop((left, top, left + REEL_WIDTH, top + band_h))
            img.paste(p_cropped, (0, band_top))
        except Exception as e:
            log.warning("Body band image failed: %s", e)

    overlay = Image.new("RGBA", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    display = "[" + text + "]" if is_bracket else text
    cx = REEL_WIDTH // 2
    margin = 56
    max_w = REEL_WIDTH - 2 * margin

    # Size to fit: 76 -> 60 -> 48 until the phrase wraps to at most two lines, so
    # even long reveals stay on a couple of readable lines instead of one
    # clipped line running off both edges of the frame.
    font = _load_font(font_path, 76)
    wrapped = _wrap_text(display, font, draw, max_w)
    for size in (76, 60, 48):
        font = _load_font(font_path, size)
        wrapped = _wrap_text(display, font, draw, max_w)
        if wrapped.count("\n") + 1 <= 2:
            break
    n_lines = wrapped.count("\n") + 1

    if n_lines == 1:
        # Short: red phrase in the black strip below the band (the original look).
        cy = band_top + band_h + 90
        draw.text((cx + 2, cy + 2), wrapped, font=font, fill=(0, 0, 0, 200), anchor="mm")
        draw.text((cx, cy), wrapped, font=font, fill=KINETIC_RED + (255,), anchor="mm")
    else:
        # Long: lay it OVER the top of the picture, on a dark scrim for contrast.
        spacing = 14
        top_y = band_top + 40
        tb = draw.multiline_textbbox((cx, top_y), wrapped, font=font,
                                     anchor="ma", align="center", spacing=spacing)
        pad = 26
        scrim_top = max(0, tb[1] - pad)
        scrim_bot = min(REEL_HEIGHT, tb[3] + pad)
        scrim = Image.new("RGBA", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0, 0))
        ImageDraw.Draw(scrim).rectangle(
            [0, scrim_top, REEL_WIDTH, scrim_bot], fill=(0, 0, 0, 165))
        overlay = Image.alpha_composite(overlay, scrim)
        draw = ImageDraw.Draw(overlay)
        draw.multiline_text((cx + 2, top_y + 2), wrapped, font=font,
                            fill=(0, 0, 0, 210), anchor="ma", align="center", spacing=spacing)
        draw.multiline_text((cx, top_y), wrapped, font=font,
                            fill=KINETIC_RED + (255,), anchor="ma", align="center", spacing=spacing)

    final = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    final.save(output_path, "PNG")


def _render_v2_slogan_card(slogan_text, image_path, output_path, font_path):
    """Beat 5: BIG mixed-size red overlay on full-frame darkened image."""
    img = Image.new("RGB", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0))
    if image_path and Path(image_path).exists():
        try:
            from PIL import ImageEnhance
            bg = Image.open(image_path).convert("RGB")
            bw, bh = bg.size
            scale = max(REEL_WIDTH / bw, REEL_HEIGHT / bh)
            bg = bg.resize((int(bw * scale), int(bh * scale)), Image.LANCZOS)
            left = (bg.width - REEL_WIDTH) // 2
            top = (bg.height - REEL_HEIGHT) // 2
            bg = bg.crop((left, top, left + REEL_WIDTH, top + REEL_HEIGHT))
            bg = ImageEnhance.Brightness(bg).enhance(0.45)
            img = bg
        except Exception as e:
            log.warning("Slogan card bg failed: %s", e)

    overlay = Image.new("RGBA", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    lines = _v2_split_slogan(slogan_text)
    base_sizes = [120, 160, 140, 130]
    max_w = int(REEL_WIDTH * 0.90)

    # Auto-fit each line: shrink until it fits 90% of frame width.
    # Without this, longer lines like "SOLITUDE MUST" clip off the right edge.
    fitted = []
    for i, line in enumerate(lines):
        size = base_sizes[i % len(base_sizes)]
        upper = line.upper()
        while size > 56:
            f = _load_font(font_path, size)
            b = draw.textbbox((0, 0), upper, font=f)
            if (b[2] - b[0]) <= max_w:
                break
            size -= 6
        else:
            f = _load_font(font_path, 56)
        fitted.append((upper, f, size))

    total_h = sum(s + 12 for _, _, s in fitted)
    y = (REEL_HEIGHT - total_h) // 2
    cx = REEL_WIDTH // 2
    for upper, f, size in fitted:
        draw.text((cx, y), upper, font=f, fill=KINETIC_RED + (255,), anchor="mt")
        y += size + 12

    final = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    final.save(output_path, "PNG")


_V2_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "so", "yet", "as", "if", "than", "that", "this",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "his", "she",
    "her", "it", "its", "they", "them", "their",
}


def _v2_pick_content_word(words, fallback="wisdom"):
    """Pick the first content word from a list of (word, start, end) tuples,
    skipping stopwords. Falls back to the longest word if all are stopwords,
    or the provided fallback if the list is empty."""
    if not words:
        return fallback
    for w, _, _ in words:
        if w.lower() not in _V2_STOPWORDS and len(w) >= 3:
            return w
    longest = max(words, key=lambda t: len(t[0]))
    return longest[0]


def _v2_split_slogan(slogan):
    words = slogan.replace(".", "").replace(",", "").strip().split()
    if not words:
        return []
    n = len(words)
    if n <= 3:
        return [" ".join(words)]
    target = 4 if n >= 9 else 3
    per = n // target
    extra = n % target
    out = []
    i = 0
    for L in range(target):
        take = per + (1 if L < extra else 0)
        out.append(" ".join(words[i:i + take]))
        i += take
    return out


def _v2_kenburns_chain(idx, length_sec):
    """Per-beat motion for compose_kinetic_v2: a slow pull-back (Ken Burns
    zoom-OUT) so each slide 'extends' / settles instead of sitting dead-still
    (user, 2026-06-05). The beat starts ~7% enlarged and eases to the full frame
    across its duration; since zoom only ever decreases toward 1.0 (full frame),
    no black edge is revealed.

    Implementation note: the zoom is driven by zoompan's `on` (output-frame
    index), not crop's `t`. crop evaluates w/h once at config time, so a t-based
    crop SIZE never animates. zoompan re-evaluates z per frame; d=1 keeps a
    `-loop 1 -t SEG` input's SEG*30 frames 1:1 and avoids the d=SEG*30 ->
    (SEG*30)^2 frame blowup the older _kinetic_image_filter warns about.
    """
    # Drive the zoom with zoompan's `on` (output-frame index), NOT crop's `t`:
    # the crop filter evaluates w/h ONCE at config time, so a t-based crop size
    # never actually animates. zoompan re-evaluates z every frame. Use d=1 (one
    # output frame per input frame) so a `-loop 1 -t SEG` input of SEG*30 frames
    # yields SEG*30 output frames, sidestepping the d=SEG*30 -> (SEG*30)^2 blowup.
    frames = max(1, int(round(length_sec * 30)))
    denom = max(1, frames - 1)
    z = "1.07-0.07*min(on/%d,1)" % denom  # zoom 1.07 -> 1.00 (zoompan z is >= 1)
    chain = (
        "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
        "zoompan=z='%s':d=1:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        ":s=1080x1920:fps=30,setsar=1,format=yuv420p"
    ) % z
    return "[%d:v]%s[v%d]" % (idx, chain, idx)


def compose_kinetic_v2(
    image_paths,
    quote,
    philosopher,
    output_path,
    font_path,
    slogan=None,
    voice="daniel",
    music_path=None,
    music_volume=0.30,
    slogan_hold=3.5,
    brand_subtitle="",
):
    """V2 kinetic reel: TTS-driven 5-beat structure matching @wisdomofhidgon.

    Parameters
    ----------
    music_path: optional path to background music file. Mixed UNDER the voice
        at `music_volume` (default 0.18 ≈ -15 dB) so the narration stays
        forward and the music is atmospheric.
    slogan_hold: seconds to hold the closing slogan card on screen AFTER the
        voice ends. Without this the final card flashes by in <2s and viewers
        miss the punchline.
    """
    from tts import synthesize_quote

    if not image_paths:
        raise ValueError("compose_kinetic_v2 requires at least one image")
    if slogan is None:
        slogan = "A seeker of truth must find their own light"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Synthesizing narration (voice=%s)...", voice)
    tts = synthesize_quote(
        quote, out_dir=Path("cache/tts"), voice=voice,
        slogan=slogan, inject_breaks=True,
    )
    log.info("TTS: %.2fs duration, %d word timings", tts.duration_sec, len(tts.word_timings))

    quote_n = len([w for w in quote.split() if w.strip(".,;:!?")])
    quote_words = tts.word_timings[:quote_n]
    slogan_words = tts.word_timings[quote_n:]

    # Hold the closing slogan for slogan_hold seconds after voice ends so the
    # punchline registers. Without this, slogan flashes in <2s.
    total = tts.duration_sec + slogan_hold

    n_hook = min(4, len(quote_words))
    hook_times = [(qw[0], qw[1], qw[2]) for qw in quote_words[:n_hook]]
    hook_end = hook_times[-1][2] + 0.3 if hook_times else 1.5
    # Choose which hook word gets the [brackets] marker. We want it on a
    # content word, not a stopword like "of" or "the". Prefer the last content
    # word in the hook range; fall back to the actual last word if all are
    # stopwords.
    bracket_idx = n_hook - 1
    for i in range(n_hook - 1, -1, -1):
        w = hook_times[i][0].lower()
        if w not in _V2_STOPWORDS and len(w) >= 4:
            bracket_idx = i
            break

    if n_hook < len(quote_words):
        breath_end = quote_words[n_hook][1]
    else:
        breath_end = hook_end + 0.5
    breath_end = max(breath_end, hook_end + 0.3)

    # Brand card holds for a tight 1.6s — long enough to register the name,
    # short enough that the voice/visual gap doesn't feel like a stall. The
    # user feedback "image popped up for a long time" tracked to a 2.5s brand
    # card; pulling it down to 1.6s frees runway for the body phrase reveals.
    brand_end = breath_end + 1.6

    # Body ends when the SLOGAN voice section starts so the slogan card is
    # on screen the entire time voice reads the slogan (was: slogan_words[-3],
    # which left the visual climax mistimed and felt out-of-sync).
    if slogan_words:
        body_end = slogan_words[0][1] - 0.1
    else:
        body_end = total - 2.0
    body_end = max(body_end, brand_end + 1.5)
    body_end = min(body_end, total - 1.5)

    log.info(
        "Beats: hook 0-%.2fs | breath %.2f-%.2fs | brand %.2f-%.2fs | body %.2f-%.2fs | slogan %.2f-%.2fs",
        hook_end, hook_end, breath_end, breath_end, brand_end, brand_end, body_end, body_end, total,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmpd = Path(tmp)
        beat_inputs = []

        # Hook: group into 2-word phrases (matches reference reel's
        # "I would" / "life of" pattern). 4 single-word cuts at ~0.7s each
        # read as choppy; 2 phrase cuts at ~1.5s breathe.
        hook_phrases = []
        for i in range(0, n_hook, 2):
            group = hook_times[i:i + 2]
            text = " ".join(w for w, _, _ in group)
            hook_phrases.append((text, group[0][1], group[-1][2], group))

        # Bracket the last hook phrase since it leads into the brand card.
        # If any phrase contains the picked content word (bracket_idx), prefer
        # that one so the bracket lands on the meaning, not just the timing.
        bracket_phrase_idx = len(hook_phrases) - 1
        for pi, (_, _, _, group) in enumerate(hook_phrases):
            start_word_idx = pi * 2
            if start_word_idx <= bracket_idx < start_word_idx + len(group):
                bracket_phrase_idx = pi
                break

        for pi, (text, ws, we, _) in enumerate(hook_phrases):
            png = tmpd / ("hook_p_%d.png" % pi)
            is_last = (pi == bracket_phrase_idx)
            _render_v2_hook_word(text, str(png), font_path, is_last=is_last)
            if pi + 1 < len(hook_phrases):
                dur = hook_phrases[pi + 1][1] - ws
            else:
                dur = hook_end - ws
            beat_inputs.append((png, max(0.4, dur)))

        breath_png = tmpd / "breath.png"
        _render_v2_black(str(breath_png))
        beat_inputs.append((breath_png, max(0.2, breath_end - hook_end)))

        brand_png = tmpd / "brand.png"
        _render_v2_brand_card(philosopher, image_paths[0], str(brand_png), font_path, subtitle=brand_subtitle)
        beat_inputs.append((brand_png, max(0.5, brand_end - breath_end)))

        # Beat 4: PHRASE reveals (not per-word). User feedback: "word by word
        # cuts feels cut off" — 7-15 PNG cuts at 0.5-0.7s each reads as
        # choppy. Group consecutive TTS words into 2-3 word phrases and hold
        # each for ~1.2-1.8s. This matches the reference reel's "here to" /
        # "warrior" pattern of 2-word labels at painting-cut cadence.
        body_pool = image_paths[1:] if len(image_paths) > 1 else image_paths
        body_pool = body_pool[:3] if len(body_pool) >= 3 else body_pool

        body_window_words = [
            (w, s, e) for (w, s, e) in tts.word_timings
            if brand_end <= s < body_end
        ]

        # Group into phrases. Target ~1.5s per phrase, BUT enforce minimum 2
        # words per phrase. Without the floor, `len // target_phrases` rounds
        # to 1 for short body windows (e.g. 5 words / 4 phrases = 1) which
        # reproduces the choppy word-by-word feel we were trying to fix.
        body_duration = max(0.5, body_end - brand_end)
        target_phrases = max(2, int(round(body_duration / 1.5)))
        phrases = []  # (text, start, end)
        if body_window_words:
            per = max(2, (len(body_window_words) + target_phrases - 1) // target_phrases)
            i = 0
            while i < len(body_window_words):
                group = body_window_words[i:i + per]
                # Show the phrase as spoken (no stopword stripping inside).
                # Reference reel uses "here to" / "warrior" — natural prose,
                # not filtered content words. Previous aggressive filter
                # stripped "me" from "within me" and produced single-word
                # labels that read as choppy cuts.
                text = " ".join(w for w, _, _ in group)
                phrases.append((text, group[0][1], group[-1][2]))
                i += per

        if phrases:
            for bi, (text, ws, we) in enumerate(phrases):
                png = tmpd / ("body_p_%d.png" % bi)
                img_idx = (bi * len(body_pool)) // max(1, len(phrases))
                body_image = body_pool[min(img_idx, len(body_pool) - 1)]
                is_last = (bi == len(phrases) - 1)
                _render_v2_body_frame(body_image, text, str(png), font_path, is_bracket=is_last)
                if bi + 1 < len(phrases):
                    dur = phrases[bi + 1][1] - ws
                else:
                    # Cap the trailing phrase at 1.8s so the closing word
                    # doesn't camp on screen for 3s+ when there's silence
                    # between body end and slogan start.
                    dur = min(1.8, body_end - ws)
                beat_inputs.append((png, max(0.6, dur)))
        else:
            body_label = _v2_pick_content_word(quote_words[-5:], fallback="wisdom")
            png = tmpd / "body_fallback.png"
            _render_v2_body_frame(body_pool[0], body_label, str(png), font_path, is_bracket=True)
            beat_inputs.append((png, body_duration))

        slogan_png = tmpd / "slogan.png"
        slogan_image = image_paths[-1]
        _render_v2_slogan_card(slogan, slogan_image, str(slogan_png), font_path)
        beat_inputs.append((slogan_png, max(0.5, total - body_end)))

        cmd = ["ffmpeg", "-y"]
        for png, dur in beat_inputs:
            cmd += ["-loop", "1", "-t", "%.3f" % dur, "-framerate", "30", "-i", str(png)]
        cmd += ["-i", str(tts.audio_path)]
        tts_idx = len(beat_inputs)
        music_idx = None
        if music_path and Path(music_path).exists():
            cmd += ["-stream_loop", "-1", "-i", str(music_path)]
            music_idx = tts_idx + 1

        n = len(beat_inputs)
        # Slow pull-back on every beat so the slides move (user, 2026-06-05).
        scale_chains = [
            _v2_kenburns_chain(i, beat_inputs[i][1])
            for i in range(n)
        ]
        concat_chain = "".join("[v%d]" % i for i in range(n)) + "concat=n=" + str(n) + ":v=1:a=0[vout]"

        if music_idx is not None:
            # Mix TTS at full volume + music at low volume. Music looped via
            # -stream_loop -1 so it never runs out mid-reel. amix duration=longest
            # would extend past TTS; use first to anchor to TTS length.
            audio_mix = (
                "[%d:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,volume=1.0[voice];"
                "[%d:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,volume=%.3f,"
                "afade=t=out:st=%.2f:d=1.2[bg];"
                "[voice][bg]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
            ) % (tts_idx, music_idx, music_volume, max(0.5, total - 1.2))
            filter_complex = ";".join(scale_chains) + ";" + concat_chain + ";" + audio_mix
            cmd += [
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-map", "[aout]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                "-c:a", "aac", "-b:a", "192k",
                "-t", "%.3f" % total,
                str(output_path),
            ]
        else:
            filter_complex = ";".join(scale_chains) + ";" + concat_chain
            cmd += [
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-map", "%d:a" % tts_idx,
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                "-c:a", "aac", "-b:a", "192k",
                "-t", "%.3f" % total,
                str(output_path),
            ]

        log.info("Rendering v2 reel (%d beats)...", n)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("ffmpeg v2 failed: %s", (result.stderr or "")[-1500:])
            raise RuntimeError("compose_kinetic_v2 ffmpeg failed")

    log.info("OK: %s", output_path)
    return str(output_path)
