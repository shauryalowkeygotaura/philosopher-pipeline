import ctypes
import hashlib
import logging
import random
import sys
import time
import requests
from pathlib import Path

_FILE_ATTRIBUTE_OFFLINE = 0x1000
_FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000
_IS_WINDOWS = sys.platform == "win32"


def _is_cloud_only(path: Path) -> bool:
    if not _IS_WINDOWS:
        return False
    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    if attrs == -1:
        return True
    return bool(attrs & (_FILE_ATTRIBUTE_OFFLINE | _FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS))


log = logging.getLogger(__name__)

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks for a contact URL in the User-Agent for high-volume scripted access.
HEADERS = {
    "User-Agent": (
        "PhilosopherPipeline/1.1 "
        "(+https://github.com/shauryalowkeygotaura/philosopher-pipeline; "
        "instagram-content-bot)"
    ),
    "Accept-Encoding": "gzip",
}

# The Met Open Access: free, no auth, ~470k artworks. primaryImage is a
# direct images.metmuseum.org CDN URL with no Cloudflare bot challenge.
# Different host from Wikimedia so it gives the pipeline a second independent
# source when commons.wikimedia.org rate-limits us.
MET_SEARCH_API = "https://collectionapi.metmuseum.org/public/collection/v1/search"
MET_OBJECT_API = "https://collectionapi.metmuseum.org/public/collection/v1/objects"
MET_QUERIES = [
    "renaissance",
    "baroque",
    "italian renaissance",
    "dutch golden age",
    "rembrandt",
    "caravaggio",
    "raphael",
    "titian",
    "religious painting",
    "portrait",
]

# Throttle: Wikimedia documents ~10 req/s as a hard ceiling for unauthenticated
# scripts; we stay well under it. 429 retries honor Retry-After.
WIKIMEDIA_THUMB_WIDTH = 1800  # was 1200; bigger so upscale to 2160 stays crisp
WIKIMEDIA_SEARCH_SLEEP = 1.0
WIKIMEDIA_INFO_SLEEP = 0.5
DOWNLOAD_SLEEP = 0.25
MAX_RETRIES_429 = 3

PHILOSOPHER_QUOTES = {
    "albert camus": [
        "In the midst of winter, I found there was, within me, an invincible summer.",
        "You will never be happy if you continue to search for what happiness consists of.",
        "Don't walk in front of me, I may not follow. Don't walk behind me, I may not lead. Walk beside me, just be my friend.",
        "The only way to deal with an unfree world is to become so absolutely free that your very existence is an act of rebellion.",
        "I rebel; therefore I exist.",
        "Man is the only creature who refuses to be what he is.",
        "In the depth of winter, I finally learned that within me there lay an invincible summer.",
    ],
    "fyodor dostoevsky": [
        "Pain and suffering are always inevitable for a large intelligence and a deep heart.",
        "The mystery of human existence lies not in just staying alive, but in finding something to live for.",
        "To love someone means to see them as God intended them.",
        "Beauty will save the world.",
        "If you want to be respected by others, the great thing is to respect yourself.",
        "Above all, don't lie to yourself. The man who lies to himself and listens to his own lie comes to a point where he cannot distinguish the truth.",
    ],
    "franz kafka": [
        "There is infinite hope, but not for us.",
        "A book must be the axe for the frozen sea within us.",
        "I am a cage, in search of a bird.",
        "In the struggle between yourself and the world, second the world.",
        "Start with what is right rather than what is acceptable.",
        "Don't bend; don't water it down; don't try to make it logical; don't edit your own soul according to the fashion.",
    ],
    "voltaire": [
        "Common sense is not so common.",
        "Judge a man by his questions rather than his answers.",
        "The art of medicine consists in amusing the patient while nature cures the disease.",
        "Think for yourself and let others enjoy the privilege of doing so too.",
        "It is dangerous to be right in matters on which the established authorities are wrong.",
        "God is a comedian playing to an audience that is too afraid to laugh.",
    ],
    "friedrich nietzsche": [
        "He who has a why to live can bear almost any how.",
        "Without music, life would be a mistake.",
        "That which does not kill us, makes us stronger.",
        "In individuals, insanity is rare; but in groups, parties, nations and epochs, it is the rule.",
        "To live is to suffer; to survive is to find some meaning in the suffering.",
        "One must still have chaos in oneself to be able to give birth to a dancing star.",
    ],
    "simone de beauvoir": [
        "One is not born, but rather becomes, a woman.",
        "Change your life today. Don't gamble on the future, act now, without delay.",
        "I am awfully greedy; I want everything from life.",
        "One's life has value so long as one attributes value to the life of others.",
        "The most mediocre of males feels himself a demigod as compared with women.",
        "Representation of the world, like the world itself, is the work of men; they describe it from their own point of view.",
    ],
    "soren kierkegaard": [
        "Life can only be understood backwards; but it must be lived forwards.",
        "The most common form of despair is not being who you are.",
        "Once you label me you negate me.",
        "Anxiety is the dizziness of freedom.",
        "People demand freedom of speech as a compensation for the freedom of thought which they seldom use.",
        "Face the facts of being what you are, for that is what changes what you are.",
    ],
    "arthur schopenhauer": [
        "Talent hits a target no one else can hit; genius hits a target no one else can see.",
        "A man can do what he wants, but not want what he wants.",
        "Compassion is the basis of morality.",
        "The person who writes for fools is always sure of a large audience.",
        "It is difficult to find happiness within oneself, but it is impossible to find it anywhere else.",
        "We forfeit three-fourths of ourselves in order to be like other people.",
    ],
    "marcus aurelius": [
        "You have power over your mind, not outside events. Realize this, and you will find strength.",
        "The impediment to action advances action. What stands in the way becomes the way.",
        "Very little is needed to make a happy life; it is all within yourself, in your way of thinking.",
        "Waste no more time arguing about what a good man should be. Be one.",
        "The best revenge is to be unlike him who performed the injustice.",
        "Accept the things to which fate binds you, and love the people with whom fate brings you together.",
    ],
    "immanuel kant": [
        "Act only according to that maxim whereby you can at the same time will that it should become a universal law.",
        "Two things fill the mind with ever new and increasing admiration and awe: the starry heavens above me and the moral law within me.",
        "Happiness is not an ideal of reason, but of imagination.",
        "Science is organized knowledge. Wisdom is organized life.",
        "Out of the crooked timber of humanity, no straight thing was ever made.",
        "Seek not the favor of the multitude; it is seldom got by honest and lawful means.",
    ],
    "jean-paul sartre": [
        "Existence precedes essence.",
        "Hell is other people.",
        "Man is condemned to be free; because once thrown into the world, he is responsible for everything he does.",
        "Words are loaded pistols.",
        "Life has no meaning the moment you lose the illusion of being eternal.",
        "Everything has been figured out, except how to live.",
    ],
    "blaise pascal": [
        "The heart has its reasons which reason knows nothing of.",
        "Man's greatness lies in his power of thought.",
        "All of humanity's problems stem from man's inability to sit quietly in a room alone.",
        "We are generally the better persuaded by the reasons we discover ourselves than by those given to us by others.",
        "There is a God-shaped vacuum in the heart of each man which cannot be satisfied by any created thing.",
        "Kind words do not cost much. Yet they accomplish much.",
    ],
}
PHILOSOPHER_QUOTES["søren kierkegaard"] = PHILOSOPHER_QUOTES["soren kierkegaard"]


PHILOSOPHER_VIBES = {
    "albert camus":         ["existential", "melancholic", "absurd", "rebellion", "dark", "atmospheric"],
    "fyodor dostoevsky":    ["dark", "heavy", "suffering", "crushing", "slowed", "sorrowful", "aggressive"],
    "franz kafka":          ["anxious", "eerie", "surreal", "ominous", "haunting", "cold"],
    "voltaire":             ["ironic", "cynical", "dominant", "punchy", "absurd", "wit"],
    "friedrich nietzsche":  ["dominant", "aggressive", "powerful", "intense", "dark", "relentless"],
    "simone de beauvoir":   ["atmospheric", "ethereal", "cinematic", "melancholic", "drift"],
    "soren kierkegaard":    ["melancholic", "contemplative", "sorrowful", "nostalgic", "ethereal"],
    "søren kierkegaard":    ["melancholic", "contemplative", "sorrowful", "nostalgic", "ethereal"],
    "arthur schopenhauer":  ["crushing", "heavy", "sorrowful", "dark", "slowed", "pessimistic"],
    "marcus aurelius":      ["dominant", "stoic", "resolute", "mechanical", "cold", "powerful"],
    "immanuel kant":        ["cold", "mechanical", "precise", "cinematic", "atmospheric"],
    "jean-paul sartre":     ["dark", "eerie", "existential", "nocturnal", "hypnotic"],
    "blaise pascal":        ["ethereal", "nostalgic", "contemplative", "spiritual", "cinematic"],
}


PHILOSOPHER_BIOS = {
    "albert camus": "French-Algerian writer and absurdist philosopher (1913-1960). Nobel laureate, author of The Stranger and The Myth of Sisyphus.",
    "fyodor dostoevsky": "Russian novelist (1821-1881) who plumbed faith, suffering, and the human soul in Crime and Punishment and The Brothers Karamazov.",
    "franz kafka": "Czech-German writer (1883-1924) whose surreal, anxiety-laced stories like The Metamorphosis gave us the word Kafkaesque.",
    "voltaire": "French Enlightenment writer and wit (1694-1778), tireless critic of dogma and intolerance, author of Candide.",
    "friedrich nietzsche": "German philosopher (1844-1900) who challenged morality, religion, and meaning itself, coining the Ubermensch and 'God is dead.'",
    "simone de beauvoir": "French existentialist (1908-1986) whose The Second Sex laid the foundation for modern feminist thought.",
    "soren kierkegaard": "Danish theologian (1813-1855), widely regarded as the father of existentialism. Wrote on anxiety, despair, and the leap of faith.",
    "søren kierkegaard": "Danish theologian (1813-1855), widely regarded as the father of existentialism. Wrote on anxiety, despair, and the leap of faith.",
    "arthur schopenhauer": "German philosopher (1788-1860) of pessimistic metaphysics who argued life is driven by a blind, restless Will.",
    "marcus aurelius": "Roman emperor and Stoic philosopher (121-180 AD). His private journal, Meditations, remains a classic manual of self-discipline.",
    "immanuel kant": "Prussian philosopher (1724-1804) who synthesized rationalism and empiricism in the Critique of Pure Reason.",
    "jean-paul sartre": "French existentialist (1905-1980) who argued that 'existence precedes essence', we are radically free and condemned to choose.",
    "blaise pascal": "French mathematician and theologian (1623-1662). Pioneer of probability theory, author of the Pensees.",
}


# Expanded Renaissance + Baroque categories for higher hit rate
RENAISSANCE_CATEGORIES = [
    "Italian_Renaissance_paintings",
    "Northern_Renaissance_paintings",
    "High_Renaissance_paintings",
    "Early_Renaissance_paintings",
    "Paintings_of_the_Italian_Renaissance",
    "Paintings_by_Caravaggio",
    "Paintings_by_Rembrandt",
    "Baroque_paintings",
    "16th-century_oil_paintings",
    "17th-century_oil_paintings",
    "Paintings_by_Raphael",
    "Paintings_by_Titian",
    "Paintings_by_Leonardo_da_Vinci",
    "Paintings_by_Michelangelo",
]

# Multiple portrait search queries to maximize image yield per philosopher
PORTRAIT_QUERIES = [
    "{name} portrait painting",
    "{name} portrait",
    "{name} oil painting",
    "{name} bust",
    "{name} engraving",
    "{name} photograph",
]


def get_bio(philosopher: str) -> str:
    return PHILOSOPHER_BIOS.get(philosopher.lower(), "")


def fetch_quote(philosopher: str, used_quotes: list) -> dict:
    all_quotes = PHILOSOPHER_QUOTES.get(philosopher.lower(), [])
    used_set = set(used_quotes)
    fresh = [q for q in all_quotes if q not in used_set]

    if fresh:
        return {"quote": fresh[0], "reframed": False}
    if all_quotes:
        return {"quote": all_quotes[0], "reframed": True}
    return {"quote": "The unexamined life is not worth living.", "reframed": True}


# Static prefix kept verbatim at the top of every fetch_slogan() call so the
# Groq prompt cache hit-rate stays high across philosophers (saves ~half the
# input tokens per call). Interpolated variables only appear in the user msg.
_SLOGAN_SYSTEM_PROMPT = """You write the closing-climax line of a philosophy Instagram Reel modeled on @wisdomofhidgon.

The line appears full-frame in BIG mixed-size red serif over a portrait of the philosopher, broken into 3-4 lines by the renderer. It is the punchline of the reel.

Rules:
- Output ONE line only. No quotes, no trailing period, no preamble, no explanation.
- 4 to 7 words. Never longer.
- Declarative. Present tense. Image-rich. No abstractions, no academic words.
- Echo the underlying theme of the quote without restating its words.
- Avoid cliches: "find your why", "follow your heart", "live your truth", "be yourself", "embrace the journey", "trust the process".
- Avoid em dashes. Use commas or nothing at all.
- Match the philosopher's voice (Stoic, absurdist, existentialist, etc.) but stay accessible.

Good examples:
  Quote: "I would rather live a short life of glory than a long one of obscurity."
  Slogan: A man of ambition must find his way to the light

  Quote: "He who has a why to live can bear almost any how."
  Slogan: Carry the why, the how follows

  Quote: "The unexamined life is not worth living."
  Slogan: Look inward or you will never arrive"""


def fetch_slogan(quote: str, philosopher: str) -> str:
    """Return a 4-7 word climax slogan for the kinetic reel.

    On any failure (missing key, network, malformed output) returns a sane
    fallback so the pipeline keeps rendering instead of aborting the reel.
    """
    import os
    fallback = "Truth is found alone in the dark"
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        log.warning("fetch_slogan: GROQ_API_KEY missing; using fallback")
        return fallback
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _SLOGAN_SYSTEM_PROMPT},
                {"role": "user", "content": f"Philosopher: {philosopher}\nQuote: {quote}\n\nSlogan:"},
            ],
            max_tokens=24,
            temperature=0.85,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = raw.strip('"').strip("'").rstrip(".").strip()
        # Reject pathological outputs (LLM ignored the brevity instruction).
        if not raw or len(raw.split()) > 9 or len(raw) > 80:
            log.warning("fetch_slogan: bad output %r; using fallback", raw)
            return fallback
        log.info("fetch_slogan: %r", raw)
        return raw
    except Exception as e:
        log.warning("fetch_slogan: %s; using fallback", e)
        return fallback


def match_song(philosopher, quote, songs, used_in_run, used_for_philosopher):
    last_3_used = set(used_for_philosopher[-3:])
    run_used = set(used_in_run)

    available = [s for s in songs
                 if s["url"] not in run_used and s["url"] not in last_3_used]
    if not available:
        available = [s for s in songs if s["url"] not in run_used]
    if not available:
        available = songs

    vibes = PHILOSOPHER_VIBES.get(philosopher.lower(), [])

    def score(song):
        label_words = song["label"].lower().split()
        return sum(1 for v in vibes if any(v in w for w in label_words))

    ranked = sorted(available, key=score, reverse=True)
    return ranked[0]["url"]


def fetch_photo(philosopher, used_photos, cache_dir):
    """Single-portrait fetch (legacy; backed by fetch_portraits)."""
    portraits = fetch_portraits(philosopher, 1, used_photos, cache_dir)
    return portraits[0] if portraits else None


def _polite_get(url, params=None, timeout=30, pre_sleep=0.0):
    """GET with Retry-After-aware backoff for HTTP 429.

    Returns a Response on success or raises the last exception. Honors
    the server-advertised Retry-After header up to a 30s ceiling so we
    do not stall the whole pipeline on a hostile cache.
    """
    if pre_sleep:
        time.sleep(pre_sleep)
    last_exc = None
    for attempt in range(MAX_RETRIES_429):
        resp = requests.get(url, params=params, timeout=timeout, headers=HEADERS)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        retry_after = resp.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after else 2.0 * (attempt + 1)
        except ValueError:
            wait = 2.0 * (attempt + 1)
        wait = min(wait, 30.0)
        log.info("HTTP 429 on %s; sleeping %.1fs (attempt %d/%d)",
                 url, wait, attempt + 1, MAX_RETRIES_429)
        time.sleep(wait)
        last_exc = requests.HTTPError("429 after retries", response=resp)
    if last_exc:
        raise last_exc
    return resp


def _wikimedia_search(query, srlimit=40):
    params = {
        "action": "query", "list": "search", "format": "json",
        "srnamespace": "6", "srsearch": query, "srlimit": str(srlimit),
    }
    resp = _polite_get(WIKIMEDIA_API, params=params, pre_sleep=WIKIMEDIA_SEARCH_SLEEP)
    return resp.json().get("query", {}).get("search", [])


def _wikimedia_imageinfo(title):
    """Imageinfo with iiurlwidth so we get a thumbnail CDN URL.

    The thumbnail CDN (upload.wikimedia.org/.../thumb/...) is served by a
    different cache tier from the originals and is far more tolerant of
    scripted access. Wikimedia's own 429 message recommends this path.
    """
    params = {
        "action": "query", "titles": title, "prop": "imageinfo",
        "iiprop": "url|size", "iiurlwidth": str(WIKIMEDIA_THUMB_WIDTH),
        "format": "json",
    }
    resp = _polite_get(WIKIMEDIA_API, params=params, pre_sleep=WIKIMEDIA_INFO_SLEEP)
    return resp.json().get("query", {}).get("pages", {})


def _pick_image_url(info):
    """Prefer thumburl when available, fall back to original url."""
    return info.get("thumburl") or info.get("url")


def _download_to_cache(url, cached_path):
    if cached_path.exists() and cached_path.stat().st_size > 0:
        return True
    try:
        img_resp = _polite_get(url, timeout=60, pre_sleep=DOWNLOAD_SLEEP)
        cached_path.write_bytes(img_resp.content)
        return cached_path.stat().st_size > 0
    except Exception as e:
        log.warning("Image download failed (%s): %s", url, e)
        return False


def fetch_portraits(philosopher, count, used_portraits, cache_dir):
    """Fetch up to `count` distinct images of the philosopher.

    Strategy: try every PORTRAIT_QUERIES variant, lowered min size 400x400.
    Falls back hard to local cache (any era) when API yields nothing,
    so the slideshow always has frames even if Wikimedia rate-limits.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = philosopher.lower().replace(" ", "-")
    used_set = set(used_portraits)
    collected = []

    for query_template in PORTRAIT_QUERIES:
        if len(collected) >= count:
            break
        query = query_template.format(name=philosopher)
        try:
            results = _wikimedia_search(query, srlimit=30)
        except Exception as e:
            log.warning("Portrait search failed (%s): %s", query, e)
            continue

        for item in results:
            if len(collected) >= count:
                break
            title = item.get("title", "")
            if not any(title.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                continue
            if any(skip in title.lower() for skip in [
                "svg", "flag", "coat", "seal", "map",
                "collage", "comparison", "composite", "group",
                "versus", "_vs_", "caricature", "cartoon",
            ]):
                continue
            try:
                pages = _wikimedia_imageinfo(title)
            except Exception as e:
                log.warning("Portrait info failed (%s): %s", title, e)
                continue

            for page in pages.values():
                infos = page.get("imageinfo", [])
                if not infos:
                    continue
                info = infos[0]
                w, h = info.get("width", 0), info.get("height", 0)
                if w < 400 or h < 400:
                    continue
                dl_url = _pick_image_url(info)
                if not dl_url:
                    continue

                # Hash the canonical original URL so the same image keeps the
                # same filename whether we downloaded thumb or full-res.
                key_url = info.get("url") or dl_url
                url_hash = hashlib.md5(key_url.encode()).hexdigest()[:10]
                filename = "portrait-" + slug + "-" + url_hash + ".jpg"
                if filename in used_set:
                    continue

                cached = cache_dir / filename
                if not _download_to_cache(dl_url, cached):
                    continue
                if _is_cloud_only(cached):
                    continue
                collected.append(str(cached))

    # Fallback: any non-used local file matching the philosopher
    if len(collected) < count:
        for pattern in ("portrait-" + slug + "-*.jpg", slug + "-*.jpg"):
            for p in sorted(cache_dir.glob(pattern)):
                sp = str(p)
                if sp in collected:
                    continue
                if p.stat().st_size > 0 and not _is_cloud_only(p):
                    collected.append(sp)
                    if len(collected) >= count:
                        break
            if len(collected) >= count:
                break

    log.info("Portraits for %s: %d/%d", philosopher, len(collected), count)
    return collected[:count]


def _fetch_paintings_met(count, used_set, seen_urls, cache_dir):
    """Pull paintings from The Met Open Access (free, no auth).

    Met serves images from images.metmuseum.org with no Cloudflare bot
    challenge, so headless scripts can download directly. Used as the
    primary source so the pipeline does not depend on Wikimedia.
    """
    collected = []
    queries = list(MET_QUERIES)
    random.shuffle(queries)
    for query in queries:
        if len(collected) >= count:
            break
        try:
            params = {"q": query, "hasImages": "true", "medium": "Paintings"}
            resp = _polite_get(MET_SEARCH_API, params=params, pre_sleep=0.4)
            object_ids = resp.json().get("objectIDs") or []
        except Exception as e:
            log.warning("Met search %r failed: %s", query, e)
            continue

        random.shuffle(object_ids)
        for object_id in object_ids[:60]:
            if len(collected) >= count:
                break
            try:
                obj_resp = _polite_get(
                    "%s/%s" % (MET_OBJECT_API, object_id), pre_sleep=0.2
                )
                obj = obj_resp.json()
            except Exception as e:
                log.warning("Met object %s failed: %s", object_id, e)
                continue
            if not obj.get("isPublicDomain"):
                continue
            url = obj.get("primaryImage") or ""
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            url_hash = hashlib.md5(url.encode()).hexdigest()[:10]
            filename = "met-" + url_hash + ".jpg"
            if filename in used_set:
                continue

            cached = cache_dir / filename
            if not _download_to_cache(url, cached):
                continue
            if _is_cloud_only(cached):
                continue
            collected.append(str(cached))
    log.info("Met paintings: %d/%d", len(collected), count)
    return collected


def _fetch_paintings_wikimedia(count, used_set, seen_urls, cache_dir):
    """Pull Renaissance/Baroque paintings from Wikimedia Commons categories.

    Used as a top-up when AIC has not produced enough images for the reel.
    Always uses the thumbnail CDN via _wikimedia_imageinfo.
    """
    collected = []
    categories = list(RENAISSANCE_CATEGORIES)
    random.shuffle(categories)

    for category in categories:
        if len(collected) >= count:
            break
        try:
            params = {
                "action": "query", "list": "categorymembers", "format": "json",
                "cmtype": "file", "cmtitle": "Category:" + category,
                "cmlimit": "200",
            }
            resp = _polite_get(WIKIMEDIA_API, params=params, pre_sleep=WIKIMEDIA_SEARCH_SLEEP)
            members = resp.json().get("query", {}).get("categorymembers", [])
            random.shuffle(members)
        except Exception as e:
            log.warning("Wikimedia category %s failed: %s", category, e)
            continue

        for item in members:
            if len(collected) >= count:
                break
            title = item.get("title", "")
            if not any(title.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                continue
            try:
                pages = _wikimedia_imageinfo(title)
            except Exception as e:
                log.warning("Wikimedia info failed (%s): %s", title, e)
                continue

            for page in pages.values():
                infos = page.get("imageinfo", [])
                if not infos:
                    continue
                info = infos[0]
                w, h = info.get("width", 0), info.get("height", 0)
                if w < 600 or h < 600:
                    continue
                dl_url = _pick_image_url(info)
                key_url = info.get("url") or dl_url
                if not dl_url or key_url in seen_urls:
                    continue
                seen_urls.add(key_url)

                url_hash = hashlib.md5(key_url.encode()).hexdigest()[:10]
                filename = "renaissance-" + url_hash + ".jpg"
                if filename in used_set:
                    continue

                cached = cache_dir / filename
                if not _download_to_cache(dl_url, cached):
                    continue
                if _is_cloud_only(cached):
                    continue
                collected.append(str(cached))
    return collected


def fetch_paintings(count, used_paintings, cache_dir):
    """Fetch up to `count` Renaissance/Baroque paintings.

    Source order: Art Institute of Chicago (no rate limit) -> Wikimedia
    Commons (thumb CDN) -> local cache. The two-source strategy keeps the
    reel populated when one host is throttling.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    used_set = set(used_paintings)
    seen_urls = set()
    collected = []

    met = _fetch_paintings_met(count, used_set, seen_urls, cache_dir)
    collected.extend(met)

    if len(collected) < count:
        remaining = count - len(collected)
        wm = _fetch_paintings_wikimedia(remaining, used_set, seen_urls, cache_dir)
        collected.extend(wm)

    # Fallback: any cached painting still on disk
    if len(collected) < count:
        existing = (
            list(cache_dir.glob("renaissance-*.jpg"))
            + list(cache_dir.glob("met-*.jpg"))
            + list(cache_dir.glob("aic-*.jpg"))  # legacy filename, preserved for old caches
        )
        random.shuffle(existing)
        for p in existing:
            sp = str(p)
            if sp in collected:
                continue
            if p.stat().st_size > 0 and not _is_cloud_only(p):
                collected.append(sp)
                if len(collected) >= count:
                    break

    log.info("Paintings: %d/%d (met=%d wikimedia+cache=%d)",
             len(collected), count, len(met), len(collected) - len(met))
    return collected[:count]
