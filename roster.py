"""
roster.py -- the candidate bench the pipeline promotes from on its own.

WHY THIS EXISTS
---------------
Groq can only recall a bounded number of genuinely documented quotes per
thinker (roughly 15-30 survive the attribution gate). Twelve philosophers is
therefore a hard ceiling of a few hundred reels, after which even a perfectly
self-refilling pool degrades to spaced repeats. The only unbounded axis is
MORE THINKERS, so the roster has to be able to grow without a human.

Every candidate here is:
  - long dead and firmly in the public domain (no living-person attribution risk)
  - heavily quoted in the training distribution, so recall actually works
  - visually servable: fetch_portraits() can find period portraits or busts

Promotion is append-only and one-at-a-time. A promoted name is written into
philosophers.md, which is the pipeline's real source of truth; this file only
supplies the bio and song-matching vibes it would otherwise be missing.

ORDERING IS DELIBERATE
----------------------
Candidates are promoted top-down. The list opens with Stoics and moralists
whose registers sit closest to the existing 12, so the account's voice drifts
slowly rather than lurching. Reorder this list to change what gets promoted
next; nothing else needs touching.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_DIR = Path(__file__).parent.resolve()
PHILOSOPHERS_FILE = _DIR / "philosophers.md"

# name -> (bio, vibes). Vibes feed match_song(); bios feed the caption builder.
CANDIDATES: dict[str, tuple[str, list[str]]] = {
    "Seneca": (
        "Roman Stoic philosopher and statesman (4 BC-65 AD). Adviser to Nero, "
        "author of the Letters to Lucilius on time, death, and self-command.",
        ["stoic", "resolute", "dominant", "cold", "contemplative", "powerful"],
    ),
    "Epictetus": (
        "Greek Stoic philosopher (50-135 AD), born enslaved. His Discourses and "
        "Enchiridion taught that only our own judgements are ours to control.",
        ["stoic", "resolute", "mechanical", "dominant", "cold", "relentless"],
    ),
    "Michel de Montaigne": (
        "French Renaissance thinker (1533-1592) who invented the personal essay "
        "and made honest self-examination a philosophical method.",
        ["contemplative", "nostalgic", "warm", "cinematic", "drift", "ironic"],
    ),
    "Baruch Spinoza": (
        "Dutch rationalist (1632-1677), excommunicated for identifying God with "
        "nature. His Ethics argues freedom is understanding necessity.",
        ["cold", "precise", "mechanical", "ethereal", "atmospheric", "spiritual"],
    ),
    "Lao Tzu": (
        "Chinese sage traditionally dated to the 6th century BC, credited with "
        "the Tao Te Ching and its teaching of yielding, emptiness, and flow.",
        ["ethereal", "drift", "contemplative", "spiritual", "atmospheric", "hypnotic"],
    ),
    "Ralph Waldo Emerson": (
        "American transcendentalist (1803-1882) whose essays on self-reliance "
        "and nature shaped a whole national temperament.",
        ["warm", "cinematic", "resolute", "nostalgic", "contemplative", "powerful"],
    ),
    "Henry David Thoreau": (
        "American naturalist and dissenter (1817-1862). Walden and Civil "
        "Disobedience argue for deliberate living and principled refusal.",
        ["contemplative", "nostalgic", "drift", "atmospheric", "warm", "ethereal"],
    ),
    "Simone Weil": (
        "French philosopher and mystic (1909-1943) who wrote on attention, "
        "affliction, and the moral weight of paying real attention to others.",
        ["ethereal", "sorrowful", "spiritual", "melancholic", "haunting", "contemplative"],
    ),
    "Hannah Arendt": (
        "German-American political theorist (1906-1975) who analysed "
        "totalitarianism and the banality of evil.",
        ["cold", "precise", "cinematic", "ominous", "atmospheric", "resolute"],
    ),
    "Ludwig Wittgenstein": (
        "Austrian-British philosopher (1889-1951) who twice remade the "
        "philosophy of language and insisted on the limits of what can be said.",
        ["cold", "eerie", "precise", "mechanical", "hypnotic", "surreal"],
    ),
    "Heraclitus": (
        "Pre-Socratic Greek philosopher (c. 535-475 BC), the obscure one, who "
        "held that everything flows and strife is the father of all.",
        ["dark", "ominous", "atmospheric", "relentless", "eerie", "powerful"],
    ),
    "Epicurus": (
        "Greek philosopher (341-270 BC) who taught that pleasure is the absence "
        "of pain and fear, and that death is nothing to us.",
        ["warm", "contemplative", "drift", "nostalgic", "atmospheric"],
    ),
    "Boethius": (
        "Roman statesman and philosopher (477-524 AD) who wrote The Consolation "
        "of Philosophy in prison awaiting execution.",
        ["sorrowful", "spiritual", "melancholic", "ethereal", "haunting", "cinematic"],
    ),
    "Niccolo Machiavelli": (
        "Florentine political thinker (1469-1527) whose The Prince separated "
        "how power is actually held from how moralists say it should be.",
        ["dominant", "cold", "ironic", "aggressive", "punchy", "relentless"],
    ),
    "Thomas Hobbes": (
        "English philosopher (1588-1679) whose Leviathan argued that without a "
        "sovereign, life is solitary, poor, nasty, brutish and short.",
        ["dark", "heavy", "ominous", "dominant", "crushing", "cold"],
    ),
    "David Hume": (
        "Scottish empiricist (1711-1776) who showed reason is the slave of the "
        "passions and that causation is a habit of mind.",
        ["ironic", "contemplative", "precise", "cinematic", "wit", "atmospheric"],
    ),
    "Georg Wilhelm Friedrich Hegel": (
        "German idealist (1770-1831) whose dialectic made history itself the "
        "unfolding of reason.",
        ["heavy", "cinematic", "powerful", "mechanical", "intense", "atmospheric"],
    ),
    "Marcus Tullius Cicero": (
        "Roman orator and philosopher (106-43 BC) who carried Greek thought "
        "into Latin and died defending the republic.",
        ["resolute", "dominant", "cinematic", "powerful", "contemplative"],
    ),
}


def bio(name: str) -> str:
    entry = CANDIDATES.get(name) or CANDIDATES.get(name.title())
    return entry[0] if entry else ""


def vibes(name: str) -> list[str]:
    entry = CANDIDATES.get(name) or CANDIDATES.get(name.title())
    return list(entry[1]) if entry else []


def _lookup_ci(name: str) -> str | None:
    """Case-insensitive candidate lookup returning the canonical spelling."""
    lowered = name.lower()
    for candidate in CANDIDATES:
        if candidate.lower() == lowered:
            return candidate
    return None


def available(active: list[str]) -> list[str]:
    """Candidates not already on the active roster, in promotion order."""
    on_roster = {n.lower() for n in active}
    return [c for c in CANDIDATES if c.lower() not in on_roster]


def promote(name: str, path: Path | None = None) -> bool:
    """Append `name` to philosophers.md. Returns False if already present.

    Append-only and idempotent: the file is the pipeline's source of truth and
    a promoted philosopher must never be silently dropped, because state.json
    keeps their history keyed by name.
    """
    target = Path(path) if path is not None else PHILOSOPHERS_FILE
    canonical = _lookup_ci(name)
    if canonical is None:
        log.error("promote: %r is not on the candidate bench; refusing.", name)
        return False

    try:
        text = target.read_text(encoding="utf-8")
    except OSError as e:
        log.error("promote: cannot read %s: %s", target, e)
        return False

    existing = {
        line.strip()[2:].strip().lower()
        for line in text.splitlines()
        if line.strip().startswith("- ")
    }
    if canonical.lower() in existing:
        return False

    if not text.endswith("\n"):
        text += "\n"
    try:
        target.write_text(text + f"- {canonical}\n", encoding="utf-8")
    except OSError as e:
        log.error("promote: cannot write %s: %s", target, e)
        return False
    log.info("promote: added %s to the roster.", canonical)
    return True
