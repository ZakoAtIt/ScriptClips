"""Lexical evidence scoring used before semantic reranking."""
import re

STOPWORDS = {"a", "an", "the", "who", "that", "with", "from", "for", "of", "in", "at", "to", "and", "or",
             "is", "are", "was", "were", "this", "video", "clip", "footage", "person", "man", "woman", "guy"}


def tokens(text: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9']+", text.lower()) if len(word) > 1 and word not in STOPWORDS}


def stem(word: str) -> str:
    for suffix in ("ation", "ment", "ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def score(query: str, text: str) -> float:
    """Content-word coverage; a one-word name can never become a perfect hit."""
    wanted, observed = {stem(word) for word in tokens(query)}, {stem(word) for word in tokens(text)}
    if not wanted or not observed:
        return 0.0
    return len(wanted & observed) / len(wanted)


def phrase_hit(query: str, text: str) -> float:
    """1.0 if the normalized query appears verbatim inside the transcript text."""
    q = re.sub(r"[^a-z0-9' ]+", " ", query.lower()).strip()
    t = re.sub(r"[^a-z0-9' ]+", " ", text.lower())
    return 1.0 if q and q in t else 0.0
