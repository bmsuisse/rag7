"""Adversarial robustness variants built deterministically from base cases.

Strategy: take queries from hit-case lists that the
pipeline currently handles well, apply mechanical perturbations (typos,
compound splits, reorderings, abbreviations), and expect the SAME
retrieval IDs as the original. Measures graceful degradation.

Each variant keeps the original's expected IDs — ideal retrieval is
robust to the perturbation.
"""

from __future__ import annotations

import random
import re
from typing import Callable

HIT_CASE = tuple[str, list[str], str]

_UMLAUT = str.maketrans({"ä": "a", "ö": "o", "ü": "u", "ß": "ss",
                         "Ä": "A", "Ö": "O", "Ü": "U"})


def _swap_adjacent(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 2)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def _drop_char(word: str, rng: random.Random) -> str:
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    return word[:i] + word[i + 1:]


def typo_variants(query: str, n: int = 3, seed: int = 0) -> list[str]:
    """Return `n` typo variants: umlaut stripped, adjacent swap, single drop."""
    rng = random.Random(hash((query, seed)) & 0xFFFFFFFF)
    words = query.split()
    if not words:
        return []
    out: list[str] = []

    stripped = query.translate(_UMLAUT)
    if stripped != query:
        out.append(stripped)

    longest_idx = max(range(len(words)), key=lambda i: len(words[i]))
    longest = words[longest_idx]
    for transform in (_swap_adjacent, _drop_char):
        if len(out) >= n:
            break
        new_word = transform(longest, rng)
        if new_word != longest:
            variant = words[:longest_idx] + [new_word] + words[longest_idx + 1:]
            joined = " ".join(variant)
            if joined not in out and joined != query:
                out.append(joined)
    return out[:n]


_COMPOUND_SPLIT_RE = re.compile(r"(?<=[a-zäöü])(?=[A-ZÄÖÜ])")
_GERMAN_JOIN_POINTS = re.compile(
    r"(?i)(beton|mörtel|schraube|platte|profil|dämmung|schaum|klebe[rn]?|"
    r"öffner|becken|mauer|boden|decke|wand|dach|stein|pflaster|fliese[nr]?|"
    r"fuge|putz|grund|bahn|rohr|kessel|zement|halter|bügel|haken|nagel)"
)


def compound_split_variants(query: str) -> list[str]:
    """Split German compound words into space-separated tokens.

    Two strategies: camelCase split (rare but informative) and suffix split
    at common compound roots.
    """
    out: list[str] = []

    camel = _COMPOUND_SPLIT_RE.sub(" ", query)
    if camel != query:
        out.append(camel)

    def suffix_split(match: re.Match[str]) -> str:
        return " " + match.group(1)

    suffixed = _GERMAN_JOIN_POINTS.sub(suffix_split, query)
    suffixed = re.sub(r"\s+", " ", suffixed).strip()
    if suffixed != query and suffixed not in out:
        out.append(suffixed)

    return out


def reorder_variant(query: str) -> str | None:
    """Reverse word order for multi-word queries (3+ words)."""
    words = query.split()
    if len(words) < 3:
        return None
    reversed_q = " ".join(reversed(words))
    return reversed_q if reversed_q != query else None


def lowercase_variant(query: str) -> str | None:
    low = query.lower()
    return low if low != query else None


def expand_cases(
    base: list[HIT_CASE],
    transforms: list[Callable[[str], list[str] | str | None]],
) -> list[HIT_CASE]:
    """Apply each transform to each base case, keeping original expected IDs."""
    out: list[HIT_CASE] = []
    seen: set[tuple[str, str]] = set()
    for query, ids, field in base:
        for transform in transforms:
            result = transform(query)
            variants = result if isinstance(result, list) else [result] if result else []
            for variant in variants:
                key = (variant, field)
                if variant and variant != query and key not in seen:
                    seen.add(key)
                    out.append((variant, ids, field))
    return out


def build_adversarial_cases(base: list[HIT_CASE]) -> list[HIT_CASE]:
    """Compose all mechanical variants from a base case list."""
    return expand_cases(
        base,
        [
            lambda q: typo_variants(q, n=2),
            compound_split_variants,
            lambda q: [v] if (v := reorder_variant(q)) else [],
            lambda q: [v] if (v := lowercase_variant(q)) else [],
        ],
    )
