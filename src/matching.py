"""Dialogue matching: sliding windows, fuzzy pre-filter, then semantic scoring (ADR-4)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

from rapidfuzz import fuzz

SEMANTIC_WEIGHT = 0.6
FUZZY_WEIGHT = 0.4
_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"

_semantic_model = None


@dataclass
class WordTiming:
    word: str
    start_sec: float
    end_sec: float


@dataclass
class Candidate:
    text: str
    start_sec: float
    end_sec: float
    fuzzy_score: float
    semantic_score: float
    combined_score: float


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation so fuzzy scores are case/punct invariant (T3.5)."""
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def score_fuzzy(candidate_text: str, target_dialogue: str) -> float:
    """Lexical similarity in [0, 1] via rapidfuzz token_sort_ratio (ADR-4)."""
    return fuzz.token_sort_ratio(_normalize(candidate_text), _normalize(target_dialogue)) / 100.0


def score_semantic(candidate_text: str, target_dialogue: str) -> float:
    """Cosine similarity of MiniLM embeddings, clipped to [0, 1].

    Loaded lazily so the fast test suite can patch this function without
    downloading or running the sentence-transformer (T3.9).
    """
    global _semantic_model
    if _semantic_model is None:
        from sentence_transformers import SentenceTransformer

        _semantic_model = SentenceTransformer(_DEFAULT_MODEL_NAME)

    embeddings = _semantic_model.encode(
        [candidate_text, target_dialogue],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    cosine = float(embeddings[0] @ embeddings[1])
    return max(0.0, min(1.0, cosine))


def find_candidates(
    words: Sequence[WordTiming],
    target_dialogue: str,
    window_slack: int = 2,
    prefilter_top_n: int = 20,
) -> List[Candidate]:
    """Slide word-count windows, fuzzy-score all, semantically score top-N.

    N = len(target_dialogue.split()) on the target as given. Window sizes are
    the inclusive range [max(1, N − window_slack), N + window_slack].
    Only the (at most) prefilter_top_n fuzzy-ranked windows are returned,
    sorted by combined_score = 0.6 * semantic + 0.4 * fuzzy.
    """
    if not words:
        return []

    target_word_count = len(target_dialogue.split())
    min_size = max(1, target_word_count - window_slack)
    max_size = target_word_count + window_slack
    n_words = len(words)

    windows: list[tuple[float, Sequence[WordTiming], str]] = []
    for size in range(min_size, max_size + 1):
        if size > n_words:
            continue
        for start in range(0, n_words - size + 1):
            span = words[start : start + size]
            text = " ".join(w.word for w in span)
            fuzzy = score_fuzzy(text, target_dialogue)
            windows.append((fuzzy, span, text))

    windows.sort(key=lambda item: item[0], reverse=True)
    prefiltered = windows[:prefilter_top_n]

    candidates: list[Candidate] = []
    for fuzzy, span, text in prefiltered:
        semantic = score_semantic(text, target_dialogue)
        combined = SEMANTIC_WEIGHT * semantic + FUZZY_WEIGHT * fuzzy
        candidates.append(
            Candidate(
                text=text,
                start_sec=span[0].start_sec,
                end_sec=span[-1].end_sec,
                fuzzy_score=fuzzy,
                semantic_score=semantic,
                combined_score=combined,
            )
        )

    candidates.sort(key=lambda c: c.combined_score, reverse=True)
    return candidates
