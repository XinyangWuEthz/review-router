"""Non-model signals consumed by policy rules.

identity_term_present is a policy input, not a measurement: it is a fixed word
list, matched on word boundaries, that marks a comment as one where a false
positive would land on an identity mention. The list is deliberately small and
lives here so a reviewer can read it in one screen.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

__all__ = ["IDENTITY_TERMS", "identity_term_present"]

IDENTITY_TERMS: tuple[str, ...] = (
    "gay",
    "gays",
    "lesbian",
    "lesbians",
    "bisexual",
    "transgender",
    "trans",
    "queer",
    "lgbt",
    "homosexual",
    "homosexuals",
    "heterosexual",
    "muslim",
    "muslims",
    "islam",
    "islamic",
    "jew",
    "jews",
    "jewish",
    "christian",
    "christians",
    "catholic",
    "catholics",
    "hindu",
    "hindus",
    "buddhist",
    "buddhists",
    "atheist",
    "atheists",
    "black",
    "blacks",
    "white",
    "whites",
    "asian",
    "asians",
    "latino",
    "latina",
    "hispanic",
    "african",
    "africans",
    "arab",
    "arabs",
    "mexican",
    "mexicans",
    "chinese",
    "indian",
    "indians",
    "immigrant",
    "immigrants",
    "refugee",
    "refugees",
    "woman",
    "women",
    "female",
    "females",
    "male",
    "males",
    "feminist",
    "feminists",
    "disabled",
    "disability",
    "autistic",
    "elderly",
)

_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in IDENTITY_TERMS) + r")\b", re.IGNORECASE
)


def identity_term_present(texts: Iterable[str]) -> np.ndarray:
    """1.0 where any identity term occurs as a whole word, else 0.0."""
    return np.array([1.0 if _PATTERN.search(str(text)) else 0.0 for text in texts], dtype=float)
