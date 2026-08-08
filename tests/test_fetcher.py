import hashlib
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fetcher import (
    fetch_quote, match_song, fetch_photo,
    fetch_paintings, fetch_portraits, get_bio,
    PHILOSOPHER_BIOS,
)


def make_mock_client(response_text: str):
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=response_text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    return mock_client


SONGS = [
    {"url": "https://youtube.com/watch?v=aaa", "label": "Dark ambient"},
    {"url": "https://youtube.com/watch?v=bbb", "label": "Melancholic piano"},
    {"url": "https://youtube.com/watch?v=ccc", "label": "Stoic orchestral"},
]

MOCK_WIKIMEDIA_RESPONSE = {
    "query": {
        "search": [
            {"title": "File:Voltaire portrait.jpg"},
            {"title": "File:Voltaire young.jpg"},
        ]
    }
}

MOCK_IMAGE_INFO = {
    "query": {
        "pages": {
            "-1": {
                "imageinfo": [{
                    "url": "https://upload.wikimedia.org/test.jpg",
                    "width": 600,
                    "height": 800,
                }]
            }
        }
    }
}

MOCK_CATEGORY_RESPONSE = {
    "query": {
        "categorymembers": [
            {"title": "File:Mona_Lisa.jpg"},
            {"title": "File:Birth_of_Venus.jpg"},
        ]
    }
}

MOCK_PAINTING_INFO = {
    "query": {
        "pages": {
            "-1": {
                "imageinfo": [{
                    "url": "https://upload.wikimedia.org/painting1.jpg",
                    "thumburl": "https://upload.wikimedia.org/thumb/painting1.jpg/1200px.jpg",
                    "width": 1200,
                    "height": 1500,
                }]
            }
        }
    }
}

# The Met Open Access: search returns {objectIDs: [...]}, each object
# fetched separately returns {primaryImage, isPublicDomain}.
# The pipeline fetches Met first, so any test that exercises Wikimedia
# painting paths must seed enough empty Met responses to drain the Met
# query loop (one response per query in fetcher.MET_QUERIES).
MOCK_MET_EMPTY = {"total": 0, "objectIDs": None}
MOCK_MET_SEARCH_HIT = {"total": 1, "objectIDs": [12345]}
MOCK_MET_OBJECT = {
    "objectID": 12345,
    "isPublicDomain": True,
    "primaryImage": "https://images.metmuseum.org/test.jpg",
    "title": "Test Painting",
    "artistDisplayName": "Test Artist",
}


def _met_query_count():
    from fetcher import MET_QUERIES
    return len(MET_QUERIES)


def _empty_met_responses():
    """One MagicMock per Met query so Met fetch drains harmlessly."""
    out = []
    for _ in range(_met_query_count()):
        r = MagicMock()
        r.json.return_value = MOCK_MET_EMPTY
        r.raise_for_status = MagicMock()
        r.status_code = 200
        out.append(r)
    return out


# Quote tests

def test_fetch_quote_returns_string():
    result = fetch_quote("Voltaire", used_quotes=[])
    assert isinstance(result["quote"], str)
    assert len(result["quote"]) > 0


def test_fetch_quote_passes_used_quotes_in_prompt():
    first = fetch_quote("Voltaire", used_quotes=[])["quote"]
    result = fetch_quote("Voltaire", used_quotes=[first])
    assert result["quote"] != first


def test_fetch_quote_caps_length_at_280_chars():
    result = fetch_quote("Voltaire", used_quotes=[])
    assert len(result["quote"]) <= 280
    assert len(result["quote"]) > 0


def test_fetch_quote_marks_reframed_when_flagged(tmp_path):
    """Exhausting the pool marks the result as a replay.

    Uses an isolated pool_path: since 2026-08-01 the live pool also contains
    Groq-generated quotes, so spending only the static PHILOSOPHER_QUOTES no
    longer exhausts a philosopher. allow_generation=False keeps it offline.
    """
    from fetcher import PHILOSOPHER_QUOTES
    all_voltaire = PHILOSOPHER_QUOTES["voltaire"]
    result = fetch_quote(
        "Voltaire",
        used_quotes=list(all_voltaire),
        allow_generation=False,
        pool_path=tmp_path / "pool.json",
    )
    assert result["reframed"] is True


def test_fetch_quote_not_reframed_by_default():
    result = fetch_quote("Voltaire", used_quotes=[])
    assert result["reframed"] is False


# Song match tests

def test_match_song_returns_url_from_list():
    result = match_song("Kafka", "Quote text", songs=SONGS,
                        used_in_run=[], used_for_philosopher=[])
    assert result == "https://youtube.com/watch?v=aaa"


def test_match_song_excludes_used_in_run():
    result = match_song("Camus", "Quote", songs=SONGS,
                        used_in_run=["https://youtube.com/watch?v=aaa"],
                        used_for_philosopher=[])
    assert result != "https://youtube.com/watch?v=aaa"


def test_match_song_excludes_last_3_used_for_philosopher():
    result = match_song(
        "Friedrich Nietzsche", "Quote", songs=SONGS,
        used_in_run=[],
        used_for_philosopher=[
            "https://youtube.com/watch?v=aaa",
            "https://youtube.com/watch?v=bbb",
        ]
    )
    assert result not in ["https://youtube.com/watch?v=aaa", "https://youtube.com/watch?v=bbb"]


def test_match_song_fallback_when_all_excluded():
    result = match_song(
        "Voltaire", "Quote",
        songs=[{"url": "https://youtube.com/watch?v=aaa", "label": "Ambient"}],
        used_in_run=[],
        used_for_philosopher=["https://youtube.com/watch?v=aaa",
                               "https://youtube.com/watch?v=aaa",
                               "https://youtube.com/watch?v=aaa"]
    )
    assert result is not None


def test_match_song_only_excludes_last_3_not_full_history():
    result = match_song(
        "Voltaire", "Quote",
        songs=SONGS,
        used_in_run=[],
        used_for_philosopher=[
            "https://youtube.com/watch?v=aaa",
            "https://youtube.com/watch?v=bbb",
            "https://youtube.com/watch?v=ccc",
            "https://youtube.com/watch?v=bbb",
        ]
    )
    assert result == "https://youtube.com/watch?v=aaa"


def test_match_song_invalid_response_falls_back_to_first_available():
    result = match_song("Kafka", "Quote", songs=SONGS,
                        used_in_run=[], used_for_philosopher=[])
    assert result in [s["url"] for s in SONGS]


# Photo fetch tests (legacy single portrait)

def test_fetch_photo_returns_path(tmp_path):
    with patch("fetcher.requests.get") as mock_get:
        search_resp = MagicMock()
        search_resp.json.return_value = MOCK_WIKIMEDIA_RESPONSE
        search_resp.raise_for_status = MagicMock()
        info_resp = MagicMock()
        info_resp.json.return_value = MOCK_IMAGE_INFO
        info_resp.raise_for_status = MagicMock()
        img_resp = MagicMock()
        img_resp.content = b"FAKEJPEG"
        img_resp.raise_for_status = MagicMock()
        mock_get.side_effect = [search_resp, info_resp, img_resp]

        result = fetch_photo("Voltaire", used_photos=[], cache_dir=tmp_path)
        assert result is not None
        assert Path(result).exists()


def test_fetch_photo_skips_used_photos(tmp_path):
    """A used filename in the used_photos set must be skipped, returning None."""
    mock_url = "https://upload.wikimedia.org/test.jpg"
    url_hash = hashlib.md5(mock_url.encode()).hexdigest()[:10]
    already_used_filename = "portrait-voltaire-" + url_hash + ".jpg"

    with patch("fetcher.requests.get") as mock_get:
        search_resp = MagicMock()
        search_resp.json.return_value = MOCK_WIKIMEDIA_RESPONSE
        search_resp.raise_for_status = MagicMock()
        search_resp.status_code = 200
        info_resp = MagicMock()
        info_resp.json.return_value = MOCK_IMAGE_INFO
        info_resp.raise_for_status = MagicMock()
        info_resp.status_code = 200
        mock_get.return_value = info_resp
        mock_get.side_effect = None
        # First call returns search; everything after returns the same info_resp
        # (which would normally feed info+download). With used_set hit, the loop
        # never reaches download, so we don't need a download mock.
        def _seq(*args, **kwargs):
            if "list" in (kwargs.get("params") or {}) and (kwargs["params"]).get("list") == "search":
                return search_resp
            return info_resp
        mock_get.side_effect = _seq

        result = fetch_photo("Voltaire", used_photos=[already_used_filename], cache_dir=tmp_path)
        assert result is None


# Bio tests

def test_get_bio_known_philosopher():
    bio = get_bio("Voltaire")
    assert isinstance(bio, str)
    assert len(bio) > 0
    assert "Enlightenment" in bio


def test_get_bio_unknown_philosopher():
    assert get_bio("Some Random Person") == ""


def test_get_bio_case_insensitive():
    assert get_bio("VOLTAIRE") == get_bio("voltaire")


def test_philosopher_bios_covers_all_quote_authors():
    """Every philosopher with quotes should also have a bio."""
    from fetcher import PHILOSOPHER_QUOTES
    for name in PHILOSOPHER_QUOTES.keys():
        assert get_bio(name), "Missing bio for " + name


# Painting fetch tests

def test_fetch_paintings_returns_list_of_paths(tmp_path):
    """Met responds with a public-domain painting; image downloads."""
    with patch("fetcher.requests.get") as mock_get:
        search_resp = MagicMock()
        search_resp.json.return_value = MOCK_MET_SEARCH_HIT
        search_resp.raise_for_status = MagicMock()
        search_resp.status_code = 200
        object_resp = MagicMock()
        object_resp.json.return_value = MOCK_MET_OBJECT
        object_resp.raise_for_status = MagicMock()
        object_resp.status_code = 200
        img_resp = MagicMock()
        img_resp.content = b"FAKEMETPAINTING"
        img_resp.raise_for_status = MagicMock()
        img_resp.status_code = 200
        # Met search -> object lookup -> image download.
        mock_get.side_effect = [search_resp, object_resp, img_resp]

        result = fetch_paintings(1, used_paintings=[], cache_dir=tmp_path)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert Path(result[0]).exists()


def test_fetch_paintings_falls_through_to_wikimedia(tmp_path):
    """When Met yields no hits across every query, Wikimedia is consulted."""
    with patch("fetcher.requests.get") as mock_get:
        cat_resp = MagicMock()
        cat_resp.json.return_value = MOCK_CATEGORY_RESPONSE
        cat_resp.raise_for_status = MagicMock()
        cat_resp.status_code = 200
        info_resp = MagicMock()
        info_resp.json.return_value = MOCK_PAINTING_INFO
        info_resp.raise_for_status = MagicMock()
        info_resp.status_code = 200
        img_resp = MagicMock()
        img_resp.content = b"WIKIMEDIA_PAINTING"
        img_resp.raise_for_status = MagicMock()
        img_resp.status_code = 200

        sequence = _empty_met_responses() + [cat_resp, info_resp, img_resp]
        mock_get.side_effect = sequence

        result = fetch_paintings(1, used_paintings=[], cache_dir=tmp_path)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert Path(result[0]).exists()


def test_fetch_paintings_returns_empty_when_no_data(tmp_path):
    with patch("fetcher.requests.get") as mock_get:
        empty_resp = MagicMock()
        empty_resp.json.return_value = {
            "objectIDs": None,
            "query": {"categorymembers": []},
        }
        empty_resp.raise_for_status = MagicMock()
        empty_resp.status_code = 200
        mock_get.return_value = empty_resp

        result = fetch_paintings(3, used_paintings=[], cache_dir=tmp_path)
        assert result == []


# Portrait fetch tests (multi-portrait)

def test_fetch_portraits_returns_list(tmp_path):
    with patch("fetcher.requests.get") as mock_get:
        search_resp = MagicMock()
        search_resp.json.return_value = MOCK_WIKIMEDIA_RESPONSE
        search_resp.raise_for_status = MagicMock()
        info_resp = MagicMock()
        info_resp.json.return_value = MOCK_IMAGE_INFO
        info_resp.raise_for_status = MagicMock()
        img_resp = MagicMock()
        img_resp.content = b"FAKEPORTRAIT"
        img_resp.raise_for_status = MagicMock()
        mock_get.side_effect = [search_resp, info_resp, img_resp,
                                 info_resp, img_resp]

        result = fetch_portraits("Voltaire", count=1, used_portraits=[], cache_dir=tmp_path)
        assert isinstance(result, list)
        assert len(result) >= 1


def test_fetch_portraits_falls_back_to_cache_on_error(tmp_path):
    """When network fails entirely, cached portraits should still be returned."""
    cached_path = tmp_path / "portrait-voltaire-deadbeef00.jpg"
    cached_path.write_bytes(b"PRECACHED")

    with patch("fetcher.requests.get", side_effect=Exception("network down")):
        result = fetch_portraits("Voltaire", count=2, used_portraits=[], cache_dir=tmp_path)
        assert any("portrait-voltaire-deadbeef00" in p for p in result)
