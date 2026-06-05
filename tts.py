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
# Voice tuning history: -3 (~87 Hz) -> -4.5 "old/deep" (~82 Hz) -> -2.5 "lighter"
# (user, 2026-06-05) -> -1.0 "lighter still" (user, 2026-06-05). Each "lighter"
# step raises pitch and eases the heaviness levers. "Lighter still" = ~98 Hz
# (104 Hz * 2^(-1.0/12)), the bright upper edge of the @wisdomofhidgon band, plus
# rate eased to -12% (less ponderous, see the rate default below), a drier/shorter
# reverb tail (40/70 ms taps, lower decay) so the voice sits forward instead of in
# a cavernous room, and gentler 2:1 compression. Still cinematic, just airier.
#   asetrate=44100*ratio + atempo=1/ratio -> pitch shift at unchanged speed
#   aecho                                  -> short-tail body (40/70 ms taps)
#   acompressor 2:1                        -> gentle narrator compression
#   loudnorm -14 LUFS / -1.5 TP            -> IG-friendly loudness target
PITCH_SEMITONES = -1.0


def _build_cinematic_filter(semitones: float = PITCH_SEMITONES) -> str:
    ratio = 2 ** (semitones / 12.0)  # <1 lowers pitch
    return (
        f"asetrate=44100*{ratio:.4f},aresample=44100,atempo={1/ratio:.4f},"
        "aecho=0.6:0.3:40|70:0.07|0.04,"
        "acompressor=threshold=-18dB:ratio=2.0:attack=10:release=120,"
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
    # Hook->body silence. 0 = continuous (user, 2026-06-05: the 1.2s gap still
    # read as an awkward "I would rather <pause>"). The composer's brand card is a
    # fixed 1.6s hold (composer.py: brand_end = breath_end + 1.6), so it survives
    # a 0 gap by overlaying the body's opening instead of sitting in dead air.
    brand_gap_sec: float = 0.0,
    slogan_gap_sec: float = 1.2,
    cinematic: bool = True,
    pitch_semitones: float = PITCH_SEMITONES,
    # Cadence controls for a slower, "wisdomful" delivery (added 2026-06-05).
    # Mid-line pauses felt awkward (user, 2026-06-05) so they default OFF; the
    # drawn-out opening + slow rate + the structural brand beat carry the weight.
    phrase_pause_sec: float = 0.0,    # beat of silence at punctuation cuts (0 = off)
    pause_every_words: int = 0,       # also insert a beat every N words (0 = off)
    drawl_words: int = 3,             # elongate the first N words ("I  wouuuld  ratheer")
    drawl_factor: float = 1.38,       # how much to stretch them (1.0 = off)
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
    """Render narration as word-aligned chunks with silence beats between them
    so the delivery breathes (hook -> brand beat -> body, paused a thought at a
    time) and the kinetic v2 reveals stay in sync with the voice.

    The opening `drawl_words` words are time-stretched for a slow, deliberate
    "I  wouuuld  ratheer" entrance; pauses land at the hook boundary, at
    punctuation, and every `pause_every_words` words.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_voice = VOICES.get(voice.lower(), voice)
    slug = "tts-" + str(abs(hash((quote, slogan or "", edge_voice, rate))) % 10_000_000)

    words = quote.strip().split()
    n_hook = min(hook_word_count, len(words))

    log.info("TTS: %d words, voice=%s (edge=%s), rate=%s, drawl=%dx%.2f, pause=%.2fs/%d, cinematic=%s",
             len(words), voice, edge_voice, rate, drawl_words, drawl_factor,
             phrase_pause_sec, pause_every_words, cinematic)

    # 1. Render the FULL quote in ONE TTS call so prosody flows naturally — the
    # previous hook/rest split sounded like two takes because each TTS call
    # starts with sentence-initial prosody. One call = one continuous character.
    quote_mp3 = out_dir / f"{slug}_quote.mp3"
    quote_word_timings = asyncio.run(_render(quote, edge_voice, rate, volume, quote_mp3))
    quote_dur = _probe_duration(quote_mp3)

    # 2. A "wisdomful" delivery breathes and lands one thought at a time, so we
    # cut the single render into chunks and insert a beat of silence after the
    # hook (the brand-card beat), at punctuation, and on a regular cadence. The
    # first `drawl_words` words are isolated and time-stretched — the drawn-out
    # "I  wouuuld  ratheer" opening. Cuts land midway in inter-word silence so no
    # phoneme is clipped, and every word timing is recomputed against the new
    # layout so the kinetic on-screen reveals stay in sync.
    # edge-tts may emit a different number of word boundaries than `quote.split()`
    # (punctuation/merging), so index everything off quote_word_timings — the
    # thing we actually slice — to avoid an out-of-range cut index.
    n_words = len(quote_word_timings)
    n_hook = min(hook_word_count, n_words)
    pause_after: dict = {}
    if 0 < n_hook < n_words:
        pause_after[n_hook - 1] = brand_gap_sec
    if phrase_pause_sec > 0:
        for i in range(n_words - 1):
            if quote_word_timings[i][0].strip()[-1:] in ",.;:!?":
                pause_after[i] = max(pause_after.get(i, 0.0), phrase_pause_sec)
    if pause_every_words > 0:
        for i in range(pause_every_words - 1, n_words - 1, pause_every_words):
            pause_after.setdefault(i, phrase_pause_sec)

    n_drawl = min(drawl_words, n_words - 1) if drawl_factor > 1.0 else 0
    if n_drawl > 0:
        pause_after.setdefault(n_drawl - 1, 0.0)  # 0s: just isolate the chunk to stretch it

    cut_indices = sorted(pause_after)
    chunk_times, chunk_word_ranges = _midpoint_splits(quote_word_timings, cut_indices, quote_dur)

    segments: List[dict] = []
    all_timings: List[Tuple[str, float, float]] = []
    cursor = 0.0
    for j, ((ta, tb), (ws, we)) in enumerate(zip(chunk_times, chunk_word_ranges)):
        part = out_dir / f"{slug}_chunk{j}.mp3"
        _cut_at(quote_mp3, ta, tb, part)
        # Drawl only the opening chunk (the one that starts at word 0).
        stretch = drawl_factor if (n_drawl > 0 and ws == 0) else 1.0
        if stretch != 1.0:
            drawled = out_dir / f"{slug}_chunk{j}_drawl.mp3"
            _atempo_stretch(part, drawled, stretch)
            part = drawled
        for k in range(ws, we):
            w, s, e = quote_word_timings[k]
            all_timings.append((w, cursor + (s - ta) * stretch, cursor + (e - ta) * stretch))
        segments.append({"path": part})
        cursor += (tb - ta) * stretch
        gap = pause_after.get(we - 1, 0.0)
        if gap > 0:
            segments.append({"silence": gap})
            cursor += gap

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


def _midpoint_splits(word_timings, cut_after, total_dur):
    """Split a quote render into chunks, cutting AFTER each word index in
    `cut_after`. Each cut time lands midway in the inter-word silence between the
    cut word and the next, so no phoneme is clipped. Returns two per-chunk lists:
    (time_range (t_start, t_end), word_index_range (start, end_exclusive)).
    Empty `cut_after` yields a single chunk spanning the whole render."""
    cuts = sorted(i for i in set(cut_after) if 0 <= i < len(word_timings) - 1)
    bounds = []
    for idx in cuts:
        e = word_timings[idx][2]
        s = word_timings[idx + 1][1] if idx + 1 < len(word_timings) else total_dur
        bounds.append((e + s) / 2.0)
    times = [0.0] + bounds + [total_dur]
    starts = [0] + [c + 1 for c in cuts]
    ends = [c + 1 for c in cuts] + [len(word_timings)]
    return list(zip(times[:-1], times[1:])), list(zip(starts, ends))


def _atempo_stretch(src: Path, dst: Path, factor: float) -> None:
    """Lengthen `src` by `factor` (1.4 = 40% longer / slower) via ffmpeg atempo,
    which changes tempo without touching pitch. atempo slows when its argument is
    < 1, so we pass 1/factor. atempo's valid range is 0.5-2.0, i.e. factor in
    [0.5, 2.0]; we clamp to stay safe."""
    factor = max(0.5, min(2.0, factor))
    tempo = 1.0 / factor
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-filter:a", f"atempo={tempo:.4f}",
        "-ar", "44100",
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(dst),
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
