"""
Text preprocessing for IR-based fault localization.

Implements the 3-stage pipeline used by both BoostN and Blues:
  1. Split  -- CamelCase + underscore splitting, lowercasing
  2. Stop   -- stopword removal (English + Java keywords)
  3. Stem   -- Porter stemming

Reference: CorpusPreprocessor.java in FlexFL / BoostNSift.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from nltk.stem import PorterStemmer  # type: ignore[import-untyped]

# Module-level singleton -- NLTK PorterStemmer is thread-safe for reads.
_stemmer = PorterStemmer()


# maxsize=2 so the Java and Python stopword lists can both stay cached within a
# single process (e.g. a mixed test run), keyed by their distinct paths.
@lru_cache(maxsize=2)
def load_stopwords(stopwords_path: str | Path | None = None) -> frozenset[str]:
    """Load stopwords from file. Defaults to bundled StopwordsPlusJava.txt."""
    if stopwords_path is None:
        from src.common.config import STOPWORDS_FILE

        stopwords_path = STOPWORDS_FILE
    stopwords_path = Path(stopwords_path)
    with open(stopwords_path, encoding="utf-8") as f:
        return frozenset(line.strip().lower() for line in f if line.strip())


def split_tokens(text: str) -> list[str]:
    """Stage 1: CamelCase split + underscore split + lowercase.

    Examples:
        "getMinValue"  -> ["get", "min", "value"]
        "MAX_VALUE"    -> ["max", "value"]
        "XMLParser"    -> ["xml", "parser"]
        "parseHTML5"   -> ["parse", "html", "5"]
    """
    # CamelCase: insert space between lower->Upper and Upper->UpperLower
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    # Underscore -> space
    text = text.replace("_", " ")
    # Split on non-alphanum boundaries
    tokens = re.findall(r"[a-zA-Z0-9]+", text)
    return [t.lower() for t in tokens]


def remove_stopwords(
    tokens: list[str],
    stopwords: frozenset[str] | None = None,
) -> list[str]:
    """Stage 2: Remove stopwords, single-char tokens, pure digits."""
    if stopwords is None:
        stopwords = load_stopwords()
    return [t for t in tokens if t not in stopwords and len(t) > 1 and not t.isdigit()]


def stem_tokens(tokens: list[str]) -> list[str]:
    """Stage 3: Porter stemming."""
    return [_stemmer.stem(t) for t in tokens]


def preprocess(
    text: str,
    stopwords: frozenset[str] | None = None,
) -> list[str]:
    """Full 3-stage pipeline: Split -> Stop -> Stem.

    Returns a list of preprocessed tokens ready for indexing / querying.
    """
    tokens = split_tokens(text)
    tokens = remove_stopwords(tokens, stopwords)
    tokens = stem_tokens(tokens)
    return tokens
