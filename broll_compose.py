"""B-roll reel composer: quote typography over looping motion footage.

A new, opt-in reel STYLE ('broll') that lays the philosopher quote over real
motion B-roll fetched by ``broll_fetcher.py``. It is deliberately SELF-CONTAINED
and does NOT modify the locked ``capcut`` / ``kinetic`` composers in
composer.py -- it only imports their shared constants and font helpers so the
typography matches the rest of the brand.

Pipeline:
  1. Normalize each B-roll clip -> 1080x1920, 30fps, h264, silent.
  2. Concat the normalized clips, then loop to fill the reel duration.
  3. Overlay a PIL-rendered quote card (with a dark legibility scrim) so the
     text reads over arbitrary footage.
  4. Mux the matched song under it, trimmed to duration.

Public API:
    compose_broll_reel(clips, quote, philosopher, audio_path, output_path,
                       font_path, reel_duration=20.0, music_volume=0.7) -> str
"""

from __future__ import annotations

import logging
import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from composer import (  # shared brand constants + font helpers (read-only reuse)
    REEL_WIDTH,
    REEL_HEIGHT,
    WATERMARK_TEXT,
    WATERMARK_OPACITY,
    WATERMARK_FONT_SIZE,
    _load_font,
    _resolve_name_font,
)

log = logging.getLogger("broll_compose")

FPS = 30
QUOTE_FONT_START = 64
QUOTE_FONT_MIN = 30
QUOTE_FONT_STEP = 3
NAME_FONT_SIZE = 40
SCRIM_ALPHA = 120           # full-frame dark scrim so text reads over footage
TEXT_MAX_WIDTH_RATIO = 0.84
TEXT_MAX_HEIGHT_RATIO = 0.52


def _wrap(draw, text, font, max_w):
    """Greedy word-wrap to a pixel width."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def _build_text_overlay(quote, philosopher, font_path, out_png):
    """Render a transparent 1080x1920 PNG: scrim + centered quote + attribution
    + watermark. Auto-shrinks the quote font until it fits the text box."""
    overlay = Image.new("RGBA", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Full-frame dark scrim for legibility over unpredictable footage.
    scrim = Image.new("RGBA", (REEL_WIDTH, REEL_HEIGHT), (0, 0, 0, SCRIM_ALPHA))
    overlay = Image.alpha_composite(overlay, scrim)
    draw = ImageDraw.Draw(overlay)

    max_w = int(REEL_WIDTH * TEXT_MAX_WIDTH_RATIO)
    max_h = int(REEL_HEIGHT * TEXT_MAX_HEIGHT_RATIO)
    quote_text = f"“{quote.strip()}”"

    # Auto-fit: shrink until wrapped block fits the height budget.
    size = QUOTE_FONT_START
    while size >= QUOTE_FONT_MIN:
        qfont = _load_font(font_path, size)
        lines = _wrap(draw, quote_text, qfont, max_w)
        line_h = int(size * 1.32)
        block_h = line_h * len(lines)
        if block_h <= max_h:
            break
        size -= QUOTE_FONT_STEP
    else:
        qfont = _load_font(font_path, QUOTE_FONT_MIN)
        lines = _wrap(draw, quote_text, qfont, max_w)
        line_h = int(QUOTE_FONT_MIN * 1.32)
        block_h = line_h * len(lines)

    # Center the quote block vertically (slightly above true center).
    y = (REEL_HEIGHT - block_h) // 2 - 60
    for line in lines:
        w = draw.textlength(line, font=qfont)
        x = (REEL_WIDTH - w) // 2
        # soft shadow then fill for contrast
        draw.text((x + 3, y + 3), line, font=qfont, fill=(0, 0, 0, 200))
        draw.text((x, y), line, font=qfont, fill=(245, 245, 240, 255))
        y += line_h

    # Attribution under the quote.
    name_font = _load_font(_resolve_name_font(font_path), NAME_FONT_SIZE)
    name_text = f"— {philosopher}"
    nw = draw.textlength(name_text, font=name_font)
    ny = y + 28
    draw.text(((REEL_WIDTH - nw) // 2 + 2, ny + 2), name_text, font=name_font, fill=(0, 0, 0, 180))
    draw.text(((REEL_WIDTH - nw) // 2, ny), name_text, font=name_font, fill=(210, 180, 140, 255))

    # Watermark, bottom-center.
    wm_font = _load_font(_resolve_name_font(font_path), WATERMARK_FONT_SIZE)
    ww = draw.textlength(WATERMARK_TEXT, font=wm_font)
    draw.text(((REEL_WIDTH - ww) // 2, REEL_HEIGHT - 110), WATERMARK_TEXT,
              font=wm_font, fill=(255, 255, 255, WATERMARK_OPACITY))

    overlay.save(out_png)
    return out_png


def _run(cmd):
    """Run ffmpeg, raising with a trimmed stderr tail on failure."""
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-400:]
        raise RuntimeError("ffmpeg failed: " + tail)


def _normalize_clip(src, dst):
    """Scale-cover to 1080x1920, crop, 30fps, h264, drop audio."""
    vf = (
        f"scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={REEL_WIDTH}:{REEL_HEIGHT},fps={FPS},setsar=1"
    )
    _run([
        "ffmpeg", "-y", "-i", str(src),
        "-an", "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        str(dst),
    ])


def compose_broll_reel(
    clips,
    quote,
    philosopher,
    audio_path,
    output_path,
    font_path,
    reel_duration: float = 20.0,
    music_volume: float = 0.7,
) -> str:
    """Compose a 1080x1920 reel: looping B-roll background + quote overlay + song.

    Raises ValueError if no usable clips were supplied (caller decides whether
    to fall back to an image style).
    """
    # Coerce/clamp the numbers that get formatted into the ffmpeg filter graph,
    # so a bad env value can't produce a malformed (or absurd) filter string.
    try:
        reel_duration = float(reel_duration)
    except (TypeError, ValueError):
        reel_duration = 20.0
    if not math.isfinite(reel_duration):
        reel_duration = 20.0
    reel_duration = min(max(reel_duration, 1.0), 120.0)
    try:
        music_volume = float(music_volume)
    except (TypeError, ValueError):
        music_volume = 0.7
    if not math.isfinite(music_volume):
        music_volume = 0.7
    music_volume = min(max(music_volume, 0.0), 2.0)

    clips = [Path(c) for c in clips if Path(c).exists() and Path(c).stat().st_size > 0]
    if not clips:
        raise ValueError("compose_broll_reel requires at least one B-roll clip")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="broll_") as td:
        td = Path(td)
        # 1. normalize each clip
        normalized = []
        for i, c in enumerate(clips):
            dst = td / f"norm_{i}.mp4"
            try:
                _normalize_clip(c, dst)
                normalized.append(dst)
            except RuntimeError as e:
                log.warning("  B-roll clip %s failed to normalize: %s", c.name, e)
        if not normalized:
            raise ValueError("all B-roll clips failed to normalize")

        # 2. concat normalized clips (concat demuxer)
        concat_list = td / "list.txt"
        concat_list.write_text(
            "".join(f"file '{p.as_posix()}'\n" for p in normalized), encoding="utf-8"
        )
        bg = td / "bg.mp4"
        # -safe 0 is required because the concat entries are absolute Windows
        # paths (drive letter ':') which the demuxer's "safe" mode rejects. It
        # is NOT a user-input surface: every path in concat_list is a temp file
        # this function just created (norm_*.mp4 under TemporaryDirectory), never
        # caller- or network-supplied.
        _run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", str(bg),
        ])

        # 3. text overlay
        overlay_png = td / "overlay.png"
        _build_text_overlay(quote, philosopher, font_path, overlay_png)

        # 4. loop bg to fill duration, overlay text, mux song trimmed to duration
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-t", f"{reel_duration:.2f}", "-i", str(bg),
            "-i", str(overlay_png),
        ]
        has_audio = audio_path and Path(audio_path).exists()
        if has_audio:
            cmd += ["-i", str(audio_path)]
        cmd += [
            "-filter_complex",
            "[0:v][1:v]overlay=0:0:format=auto[v]"
            + (f";[2:a]volume={music_volume:.2f},atrim=0:{reel_duration:.2f}[a]" if has_audio else ""),
            "-map", "[v]",
        ]
        if has_audio:
            cmd += ["-map", "[a]", "-c:a", "aac", "-b:a", "160k"]
        cmd += [
            "-c:v", "libx264", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-t", f"{reel_duration:.2f}", str(output_path),
        ]
        _run(cmd)

    log.info("  B-roll reel written: %s", output_path)
    return str(output_path)
