from __future__ import annotations

from review_router.signals import identity_term_present


def test_matches_whole_words_only() -> None:
    out = identity_term_present(["the Muslim editor", "blackboard", "a gay man", ""])
    assert out.tolist() == [1.0, 0.0, 1.0, 0.0]
