"""
models/audio/word_lookup.py — Word lookup logic for profanity detection.

Single source of truth: models/audio/words.json
Format: { "word": { "severity": "mild|moderate|severe|clean", "custom": bool } }

  custom: false  built-in word shipped with CleanStream
  custom: true   user added or modified via the Word List tab

"clean" severity means explicitly whitelisted — not flagged even if
it appears in the text.
"""

import json
import re
from pathlib import Path
from loguru import logger

WORDS_PATH = Path("models/audio/words.json")

# In-memory lookup: word → severity
_WORD_LIST: dict[str, str] = {}


# ── JSON helpers ──────────────────────────────────────────────────────────────

def load_words() -> dict:
    """Load words.json. Returns {word: {severity, custom}}."""
    if not WORDS_PATH.exists():
        logger.error(f"Word list not found at {WORDS_PATH}")
        return {}
    try:
        return json.loads(WORDS_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to load words.json: {e}")
        return {}


def save_words(words: dict) -> None:
    """Save words.json."""
    WORDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    WORDS_PATH.write_text(
        json.dumps(words, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


def add_word(word: str, severity: str) -> None:
    """Add or update a word. Always marks as custom=True."""
    assert severity in ("mild", "moderate", "severe", "clean"), \
        f"Invalid severity: {severity}"
    words = load_words()
    words[word.strip().lower()] = {"severity": severity, "custom": True}
    save_words(words)
    rebuild()


def remove_word(word: str) -> None:
    """
    Remove a custom word entirely.
    Built-in words cannot be removed — set severity to 'clean' to whitelist.
    """
    words = load_words()
    key = word.strip().lower()
    entry = words.get(key)
    if entry and not entry.get("custom", False):
        logger.warning(f"Cannot remove built-in word '{word}' — set to 'clean' to whitelist it")
        return
    words.pop(key, None)
    save_words(words)
    rebuild()


def set_severity(word: str, severity: str) -> None:
    """Change severity of any word. Marks as custom=True."""
    assert severity in ("mild", "moderate", "severe", "clean"), \
        f"Invalid severity: {severity}"
    words = load_words()
    key = word.strip().lower()
    if key in words:
        words[key]["severity"] = severity
        words[key]["custom"] = True
    else:
        words[key] = {"severity": severity, "custom": True}
    save_words(words)
    rebuild()


# ── Word list builder ─────────────────────────────────────────────────────────

def rebuild() -> None:
    """Rebuild the in-memory lookup from words.json."""
    global _WORD_LIST
    words = load_words()
    _WORD_LIST = {word: entry["severity"] for word, entry in words.items()}
    logger.debug(f"Word lookup rebuilt: {len(_WORD_LIST)} entries")


def get_word_list() -> dict[str, str]:
    """Return {word: severity} lookup table."""
    if not _WORD_LIST:
        rebuild()
    return _WORD_LIST


# ── Lookup API ────────────────────────────────────────────────────────────────

def lookup(word: str) -> str | None:
    """
    Return severity of a single word, or None if not in list.
    Returns None for unknown words and for whitelisted ('clean') words.
    """
    result = get_word_list().get(word.strip().lower())
    return None if result == "clean" else result


def scan(text: str) -> list[tuple[str, str]]:
    """
    Scan text for all flagged words/phrases.
    Returns [(matched_word, severity), ...].
    Multi-word phrases checked before single words.
    """
    if not text:
        return []

    wl = get_word_list()
    text_lower = text.lower()
    hits = []
    matched_positions: set[int] = set()

    # Multi-word phrases first (longest match wins)
    phrases = [(w, s) for w, s in wl.items() if " " in w and s != "clean"]
    for phrase, severity in sorted(phrases, key=lambda x: -len(x[0])):
        start = 0
        while True:
            idx = text_lower.find(phrase, start)
            if idx == -1:
                break
            positions = set(range(idx, idx + len(phrase)))
            if not positions & matched_positions:
                hits.append((phrase, severity))
                matched_positions |= positions
            start = idx + 1

    # Single words
    for match in re.finditer(r"\b\w+\b", text_lower):
        if match.start() in matched_positions:
            continue
        word = match.group()
        sev = wl.get(word)
        if sev and sev != "clean":
            hits.append((word, sev))
            matched_positions |= set(range(match.start(), match.end()))

    return hits


def highest_severity(hits: list[tuple[str, str]]) -> str:
    """Return the highest severity from a list of hits."""
    rank = {"clean": 0, "mild": 1, "moderate": 2, "severe": 3}
    if not hits:
        return "clean"
    return max((sev for _, sev in hits), key=lambda s: rank.get(s, 0))


# Build on import
rebuild()
