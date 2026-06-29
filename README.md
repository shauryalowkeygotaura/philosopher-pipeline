# Philosopher Instagram Pipeline

## Reel styles (`STYLE` env var)

- `capcut` (default) — 7s fast-cut, beat-synced slideshow of paintings/portraits.
- `kinetic` — 28s letterbox + word-by-word typography (locked 5-beat format).
- `broll` — 15s quote over looping, keyless motion footage.

### B-roll style

Lays the quote over real, license-clean motion footage fetched per reel from
keyless sources (Wikimedia Commons video, Archive.org). No API keys, no paid
services, no torch. The quote is mapped to an atmospheric visual theme
(`derive_broll_theme`), e.g. "suffering" → storm clouds. If no footage matches,
it falls back to the beat-synced slideshow so a reel is always produced.

```bash
STYLE=broll python pipeline.py --single --generate-only
```

Env knobs: `BROLL_CLIPS` (clips fetched per reel, default 4),
`BROLL_MUSIC_VOLUME` (default 0.7). Modules: `broll_fetcher.py` (fetch +
theme), `broll_compose.py` (compose). These are self-contained and do NOT touch
the locked `capcut` / `kinetic` composers.
