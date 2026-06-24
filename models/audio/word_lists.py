"""
word_lists.py — Compatibility shim. All logic is in word_lookup.py.
All word data is in words.json.
"""
from models.audio.word_lookup import (
    get_word_list,
    load_words,
    save_words,
    add_word,
    remove_word,
    set_severity,
    rebuild,
    lookup,
    scan as lookup_phrase,
    highest_severity,
)

WORD_LIST = get_word_list()
