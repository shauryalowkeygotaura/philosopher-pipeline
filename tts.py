"""TTS client for the kinetic reel style.

Uses Microsoft Edge's neural TTS endpoint via `edge-tts` (free, no key) and
applies a cinematic-narrator post-processing chain so the result matches the
deep BBC-baritone vibe of the @wisdomofhidgon reference.

Quote is rendered in TWO halves with a configurable silence gap between
them so the composer's brand-card beat (which sits between hook and body
in the reference reel) has actual silence to land in — the previous
continuous-narration build mismatched voice and visuals end-to-end.

Returns:
  - path to the rendered audio (mp3, ~24 kHz mono)
  - per-word timestamps in seconds: [(word, start, end), ...]

The duration_sec includes the trailing silence on the slogan clip so the
visual composer can pin its `slogan_hold` extension on top of it.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import edge_tts

log = logging.getLogger(__name__)


# Legacy ElevenLabs slugs mapped to Edge neural voices. `daniel` defaults to
# Andrew Multilingual because the user A/B'd it against Brian/Roger and
# picked it as the closest to the @wisdomofhidgon BBC-narrator vibe (with
# the cinematic post-processing chain below).
VOICES = {
    "daniel":  "en-US-AndrewMultilingualNeural",
    "andrew":  "en-US-AndrewMultilingualNeural",
    "brian":   "en-US-BrianMultilingualNeural",
    "roger":   "en-US-RogerNeural",
    "george":  "en-GB-ThomasNeural",
    "ryan":    "en-GB-RyanNeural",
    "thomas":  "en-GB-ThomasNeural",
    "adam":    "en-US-GuyNeural",
    "antoni":  "en-US-ChristopherNeural",
}


# Cinematic post-processing chain. Pitch is a measured target, not a guess:
# the @wisdomofhidgon reference voice profiles at ~80-90 Hz median F0, while raw
# edge-tts (Andrew) lands at ~104 Hz.
#
# Voice brief = "old, deep, full of wisdom" (user, 2026-06-04). We want an aged
# sage, not just a deep announcer, so we push past the earlier -3 (~87 Hz) to
# -4.5 semitones: 104 Hz * 2^(-4.5/12) = ~80 Hz, sitting at the FLOOR of the
# reference band (the gravest, oldest end). Slower default rate (-22%) makes
# the delivery measured/deliberate, and a slightly longer reverb tail adds the
# resonant, contemplative space of an elder speaking in a large room.
#   asetrate=44100*ratio + atempo=1/ratio -> pitch shift at unchanged speed
#   aecho                                  -> longer-tail body (60/110 ms taps)
#   acompressor 2.5:1                      -> gentle narrator compression
#   loudnorm -14 LUFS / -1.5 TP            -> IG-friendly loudness target
PITCH_SEMITONES = -4.5


def _build_cinematic_filter(semitones: float = PITCH_SEMITONES) -> str:
    ratio = 2 ** (semitones / 12.0)  # <1 lowers pitch
    return (
        f"asetrate=44100*{ratio:.4f},aresample=44100,atempo={1/ratio:.4f},"
        "aecho=0.7:0.4:60|110:0.12|0.08,"
        "acompressor=threshold=-18dB:ratio=2.5:attack=10:release=120,"
        "loudnorm=I=-14:TP=-1.5:LRA=8"
    )


_CINEMATIC_FILTER = _build_cinematic_filter()


@dataclass
class TTSResult:
    audio_path: Path
    duration_sec: float
    word_timings: List[Tuple[str, float, float]]


def synthesize_quote(
    quote: str,
    out_dir: Path | str,
    voice: str = "daniel",
    slogan: str | None = None,
    rate: str = "-22%",
    volume: str = "+0%",
    hook_word_count: int = 4,
    brand_gap_sec: float = 1.2,
    slogan_gap_sec: float = 1.2,
    cinematic: bool = True,
    pitch_semitones: float = PITCH_SEMITONES,
    # Legacy ElevenLabs kwargs kept so existing call sites don't break.
    stability: float = 0.55,
    similarity_boost: float = 0.85,
    style: float = 0.40,
    use_speaker_boost: bool = True,
    model_id: str = "",
    inject_breaks: bool = True,
    hook_pause_count: int = 0,
    hook_pause_sec: float = 0.45,
    speed: float = 1.0,
    post_speed: float = 1.0,
) -> TTSResult:
    """Render narration in 2-3 segments with silence gaps so the kinetic v2
    beats (hook → breath → brand → body → slogan) align with the voice.

    Layout:
        [hook (N words)] + [brand_gap silence] + [rest of quote] +
        [slogan_gap silence] + [slogan]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_voice = VOICES.get(voice.lower(), voice)
    slug = "tts-" + str(abs(hash((quote, slogan or "", edge_voice, rate))) % 10_000_000)

    words = quote.strip().split()
    n_hook = min(hook_word_count, len(words))

    log.info("TTS: %d words, voice=%s (edge=%s), rate=%s, cinematic=%s",
             len(words), voice, edge_voice, rate, cinematic)

    # 1. Render the FULL quote in ONE TTS call so prosody flows naturally —
    # the previous hook/rest split sounded like two different takes because
    # each TTS call starts with sentence-initial prosody. Single call =
    # one continuous voice character throughout the quote.
    quote_mp3 = out_dir / f"{slug}_quote.mp3"
    quote_word_timings = asyncio.run(_render(quote, edge_voice, rate, volume, quote_mp3))
    quote_dur = _probe_duration(quote_mp3)

    # 2. Surgically slice the quote audio at the natural word boundary
    # between word N and word N+1, then re-concat with `brand_gap_sec` of
    # silence inserted. Cut point lands midway in the inter-word silence so
    # we never clip a phoneme.
    segments: List[dict] = []
    all_timings: List[Tuple[str, float, float]] = []
    cursor = 0.0

    if len(quote_word_timings) > n_hook and n_hook > 0:
        hook_end = quote_word_timings[n_hook - 1][2]
        rest_start = quote_word_timings[n_hook][1]
        t_split = (hook_end + rest_start) / 2

        hook_part = out_dir / f"{slug}_hook_part.mp3"
        rest_part = out_dir / f"{slug}_rest_part.mp3"
        _cut_at(quote_mp3, 0.0, t_split, hook_part)
        _cut_at(quote_mp3, t_split, quote_dur, rest_part)

        segments.append({"path": hook_part})
        segments.append({"silence": brand_gap_sec})
        segments.append({"path": rest_part})

        # Hook words keep their original timings; rest words shift by +gap.
        for i, (w, s, e) in enumerate(quote_word_timings):
            if i < n_hook:
                all_timings.append((w, s, e))
            else:
                all_timings.append((w, s + brand_gap_sec, e + brand_gap_sec))
        cursor = quote_dur + brand_gap_sec
    else:
        segments.append({"path": quote_mp3})
        all_timings.extend(quote_word_timings)
        cursor = quote_dur

    # 3. Slogan is a separate utterance by design — fresh sentence-initial
    # prosody is correct here (it's the punchline, not a continuation).
    slogan_dur = 0.0
    if slogan and slogan.strip():
        slogan_mp3 = out_dir / f"{slug}_slogan.mp3"
        slogan_words_raw = asyncio.run(_render(slogan, edge_voice, rate, volume, slogan_mp3))
        slogan_dur = _probe_duration(slogan_mp3)
        slogan_offset = cursor + slogan_gap_sec
        all_timings += [(w, s + slogan_offset, e + slogan_offset) for (w, s, e) in slogan_words_raw]
        segments.append({"silence": slogan_gap_sec})
        segments.append({"path": slogan_mp3})
        cursor = slogan_offset + slogan_dur

    raw_final = out_dir / f"{slug}_raw.mp3"
    _concat_segments(segments, raw_final)
    total_duration = cursor

    if cinematic:
        final_mp3 = out_dir / f"{slug}.mp3"
        _apply_cinematic(raw_final, final_mp3, pitch_semitones)
        audio_path = final_mp3
    else:
        audio_path = raw_final

    log.info("TTS: rendered %.2fs (quote=%.2f + brand_gap=%.2f + slogan_gap=%.2f + slogan=%.2f), %d word timings",
             total_duration, quote_dur,
             brand_gap_sec if len(quote_word_timings) > n_hook else 0,
             slogan_gap_sec if slogan else 0, slogan_dur,
             len(all_timings))
    return TTSResult(audio_path=audio_path, duration_sec=total_duration,
                     word_timings=all_timings)


async def _render(text: str, voice: str, rate: str, volume: str,
                  output_path: Path) -> List[Tuple[str, float, float]]:
    """Stream synth from Edge TTS, capture audio bytes + WordBoundary events.

    `boundary="WordBoundary"` is REQUIRED — the default `SentenceBoundary`
    emits one event per sentence and silently breaks word-by-word reveals.
    `offset` and `duration` are 100-nanosecond ticks (Windows FILETIME);
    divide by 1e7 for seconds.
    """
    communicate = edge_tts.Communicate(
        text, voice, rate=rate, volume=volume, boundary="WordBoundary",
    )
    audio_bytes = bytearray()
    words: List[Tuple[str, float, float]] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_bytes.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            start = chunk["offset"] / 1e7
            end = (chunk["offset"] + chunk["duration"]) / 1e7
            words.append((chunk["text"], start, end))
    if not audio_bytes:
        raise RuntimeError(f"edge-tts produced no audio for voice {voice!r}")
    output_path.write_bytes(bytes(audio_bytes))
    return words


def _probe_duration(mp3_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(mp3_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(out.stdout.strip())
    except Exception as e:
        log.warning("ffprobe failed (%s); falling back to 0.0", e)
        return 0.0


def _cut_at(src: Path, t_start: float, t_end: float, out: Path) -> None:
    """Sample-accurate cut of `src` from t_start to t_end seconds. Output
    is 44.1kHz stereo mp3 to match the rest of the chain. `-ss` placed
    AFTER `-i` so we get sample-accurate seeking (slower but precise);
    cut points come from word-boundary timings where precision matters."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-ss", "%.3f" % t_start,
        "-to", "%.3f" % t_end,
        "-af", "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo",
        "-ar", "44100",
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _concat_segments(segments: List[dict], out: Path) -> None:
    """Concat a mixed list of {'path': Path} (audio file) and {'silence': float}
    (silence seconds) into a single 44.1kHz stereo mp3.

    The 44.1kHz target is non-optional: edge-tts emits 24kHz mono, but the
    cinematic chain in `_apply_cinematic` hardcodes `asetrate=44100*0.91` for
    its pitch-shift math. If you concat at 24kHz and then feed the result to
    a 44.1kHz-tuned asetrate, ffmpeg reinterprets the audio at 40131Hz
    (44100*0.91), playback becomes 1.67x faster, and pitch jumps ~9
    semitones up. Result: squeaky voice ~45% short, total visual/audio
    desync. Lock the rate here so the downstream formula stays sound.
    """
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    n = len(segments)
    for seg in segments:
        if "path" in seg:
            cmd += ["-i", str(seg["path"])]
        else:
            cmd += ["-f", "lavfi", "-t", "%.3f" % seg["silence"],
                    "-i", "anullsrc=r=44100:cl=stereo"]
    # Normalize every input to 44.1kHz stereo BEFORE concat so the filter
    # doesn't fail on heterogeneous rates / layouts.
    normalize = ";".join(
        f"[{i}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[a{i}]"
        for i in range(n)
    )
    concat = "".join(f"[a{i}]" for i in range(n)) + f"concat=n={n}:v=0:a=1[aout]"
    cmd += [
        "-filter_complex", normalize + ";" + concat,
        "-map", "[aout]",
        "-ar", "44100",
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _apply_cinematic(src: Path, dst: Path, pitch_semitones: float = PITCH_SEMITONES) -> None:
    """Pitch shift (default -3 semitones), hall reverb, narrator compression, -14 LUFS.

    Pitch shift uses the asetrate+atempo dance because rubberband isn't on
    every ffmpeg build. The asetrate value MUST match the source rate; the
    `_concat_segments` step locks input to 44.1kHz so this filter is safe.
    The atempo factor 1/ratio restores playback speed so word_timings stay
    valid through the chain.
    """
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-af", _build_cinematic_filter(pitch_semitones),
        "-ar", "44100",
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def quote_to_phrase_groups(word_timings, n_target=10):
    """Group consecutive word_timings into ~n_target phrases. Used by callers
    that want phrase-level reveals instead of per-word. compose_kinetic_v2 does
    its own grouping; this helper stays for external scripts that import it."""
    if not word_timings:
        return []
    n = len(word_timings)
    per_phrase = max(1, (n + n_target - 1) // n_target)
    out = []
    i = 0
    while i < n:
        group = word_timings[i : i + per_phrase]
        text = " ".join(w for w, _, _ in group)
        out.append((text, group[0][1], group[-1][2]))
        i += per_phrase
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    test_quote = "I would rather live a short life of glory than a long one of obscurity."
    result = synthesize_quote(
        test_quote,
        out_dir=Path("cache/tts"),
        slogan="A man of ambition must find his way to the light",
    )
    print(f"Audio: {result.audio_path}")
    print(f"Duration: {result.duration_sec:.2f}s")
    print("Word timings:")
    for w, s, e in result.word_timings:
        print(f"  {s:6.2f}s -> {e:6.2f}s  '{w}'")
