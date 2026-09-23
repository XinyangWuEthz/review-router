from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from review_router.policy import (
    DEFAULT_HIGH_RISK_EDGES,
    DEFAULT_POLICY_PATH,
    AgreementSegments,
    load_policy,
)


def test_shipped_policy_parses() -> None:
    policy = load_policy()
    assert policy.version == 3
    assert policy.decision_mode == "human_confirmation"
    assert policy.rules, "shipped policy declares no rules"


def test_shipped_policy_actions_are_known_tiers() -> None:
    policy = load_policy()
    assert {r.action for r in policy.rules} <= set(policy.tiers)


def test_every_rule_states_its_rationale() -> None:
    """A tier choice without a written reason is not reviewable policy."""
    policy = load_policy()
    for rule in policy.rules:
        assert len(rule.description.strip()) > 40, f"{rule.id} lacks a rationale"


def test_priority_review_floor_is_stricter_than_human_review() -> None:
    policy = load_policy()
    assert policy.tier_precision_floors == {"priority_review": 0.95, "human_review": 0.90}


def test_highest_matching_tier_wins() -> None:
    policy = load_policy()
    signals = {
        "p_threat": 0.9,
        "p_severe_toxic": 0.0,
        "p_toxic": 1.0,
        "identity_term_present": 0.0,
        "p_max": 1.0,
    }
    assert policy.route(signals) == "priority_review"


def test_no_matching_rule_falls_through_to_allow() -> None:
    policy = load_policy()
    signals = {
        "p_threat": 0.0,
        "p_severe_toxic": 0.0,
        "p_toxic": 0.0,
        "identity_term_present": 0.0,
        "p_max": 0.1,
    }
    assert policy.route(signals) == "allow"


def test_hierarchy_violation_is_routed_to_a_human() -> None:
    """Independent heads can violate the observed label hierarchy."""
    policy = load_policy()
    signals = {
        "p_threat": 0.0,
        "p_severe_toxic": 0.8,
        "p_toxic": 0.1,
        "identity_term_present": 0.0,
        "p_max": 0.8,
    }
    assert policy.route(signals) == "priority_review"


def test_unknown_action_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 1\n"
        "tiers: {allow: 1}\n"
        "rules:\n"
        "  - {id: X, description: x, action: nuke, conditions: []}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown action"):
        load_policy(bad)


def test_unknown_op_is_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 1\n"
        "tiers: {allow: 1}\n"
        "rules:\n"
        "  - id: X\n"
        "    description: x\n"
        "    action: allow\n"
        "    conditions: [{signal: p_toxic, op: '~=', threshold: 0.5}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown op"):
        load_policy(bad)


def test_policy_file_ships_with_the_package() -> None:
    assert DEFAULT_POLICY_PATH.is_file()


def test_retired_rules_are_not_active() -> None:
    """A withdrawn rule stays in the file as evidence, never as policy."""
    raw = yaml.safe_load(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    retired = {entry["id"] for entry in raw.get("retired_rules") or []}
    assert "R103_identity_term_low_confidence" in retired
    active = {rule.id for rule in load_policy().rules}
    assert not retired & active
    for entry in raw.get("retired_rules") or []:
        assert entry.get("retired_on") and len(str(entry.get("why", ""))) > 80, entry["id"]


def test_subgroup_threshold_declaration_needs_a_floored_tier(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "version: 1\ntiers: {allow: 1, human_review: 2}\n"
        "tier_precision_floors: {human_review: 0.9}\n"
        "subgroup_thresholds: {allow: [{signal: x, rationale: y}]}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a tier with a precision floor"):
        load_policy(bad)


def test_shipped_subgroup_threshold_states_its_rationale() -> None:
    policy = load_policy()
    assert policy.subgroup_thresholds
    for declared in policy.subgroup_thresholds:
        assert len(declared.rationale.strip()) > 40, declared


@pytest.mark.parametrize("tier", ["allow", "custom_tier"])
def test_subgroup_threshold_rejects_unsupported_floored_tier(tmp_path: Path, tier: str) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "tiers": {tier: 1},
                "tier_precision_floors": {tier: 0.9},
                "subgroup_thresholds": {tier: [{"signal": "identity_term_present"}]},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not a model-driven tier"):
        load_policy(path)


def test_legacy_snapshot_preserves_automatic_action_semantics(tmp_path: Path) -> None:
    path = tmp_path / "legacy-policy.yaml"
    path.write_text(
        "version: 1\ntiers: {allow: 1, human_review: 2, auto_action: 3}\n"
        "tier_precision_floors: {human_review: 0.9, auto_action: 0.99}\n"
        "subgroup_thresholds: {auto_action: [{signal: identity_term_present}]}\n",
        encoding="utf-8",
    )
    policy = load_policy(path)
    assert policy.version == 1
    assert policy.decision_mode == "legacy"
    assert policy.tiers["auto_action"] == 3
    assert policy.tier_precision_floors["auto_action"] == 0.99
    assert policy.subgroup_thresholds[0].tier == "auto_action"
    assert "priority_review" not in policy.tiers


@pytest.mark.parametrize("mode", [None, "legacy"])
def test_v2_requires_human_confirmation(tmp_path: Path, mode: str | None) -> None:
    raw = yaml.safe_load(DEFAULT_POLICY_PATH.read_text())
    if mode is None:
        del raw["decision_mode"]
    else:
        raw["decision_mode"] = mode
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="v2 requires decision_mode: human_confirmation"):
        load_policy(path)


def test_human_confirmation_rejects_automatic_action_tier(tmp_path: Path) -> None:
    raw = yaml.safe_load(DEFAULT_POLICY_PATH.read_text())
    raw["tiers"]["auto_action"] = 4
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="automatic action tiers are forbidden"):
        load_policy(path)


def test_human_confirmation_cannot_fall_through_to_an_automatic_action() -> None:
    with pytest.raises(ValueError, match="forbids fallback tier 'auto_action'"):
        load_policy().route({}, default="auto_action")


def _shipped_raw() -> dict[str, object]:
    raw: dict[str, object] = yaml.safe_load(DEFAULT_POLICY_PATH.read_text(encoding="utf-8"))
    return raw


def _write(tmp_path: Path, raw: dict[str, object]) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_shipped_policy_declares_the_segment_rule_for_priority_review() -> None:
    policy = load_policy()
    assert policy.version == 3
    assert policy.tier_selection_rules == {
        "priority_review": "segment_agreement",
        "human_review": "cumulative_precision",
    }
    segments = policy.agreement_segments
    assert segments is not None
    assert segments.edges == (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0)
    assert segments.high_risk_edges == (0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0)
    assert segments.min_rows == 30 and segments.use_wilson_lower_bound is False
    assert segments.edges_for(10.0, 5.0) == segments.high_risk_edges
    assert segments.edges_for(1.0, 5.0) == segments.edges


def test_selection_defaults_when_keys_are_absent(tmp_path: Path) -> None:
    raw = _shipped_raw()
    raw.pop("tier_selection_rules")
    raw.pop("agreement_segments")
    policy = load_policy(_write(tmp_path, raw))
    assert policy.tier_selection_rules == {
        "human_review": "cumulative_precision",
        "priority_review": "cumulative_precision",
    }
    assert policy.agreement_segments == AgreementSegments(
        edges=(0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0),
        high_risk_edges=DEFAULT_HIGH_RISK_EDGES,
        min_rows=30,
        use_wilson_lower_bound=False,
    )


def test_unknown_selection_rule_is_rejected(tmp_path: Path) -> None:
    raw = _shipped_raw()
    raw["tier_selection_rules"] = {"priority_review": "vibes"}
    with pytest.raises(ValueError, match="unknown rule 'vibes'"):
        load_policy(_write(tmp_path, raw))


def test_selection_rule_needs_a_floored_tier(tmp_path: Path) -> None:
    raw = _shipped_raw()
    raw["tier_selection_rules"] = {"allow": "cumulative_precision"}
    with pytest.raises(ValueError, match="not a tier with a precision floor"):
        load_policy(_write(tmp_path, raw))


@pytest.mark.parametrize(
    ("edges", "message"),
    [
        ([0.5, 0.5, 1.0], "strictly increasing"),
        ([0.9, 0.8, 1.0], "strictly increasing"),
        ([0.5, 0.9], "must end at 1.0"),
        ([1.0], "at least two edges"),
        ([-0.1, 1.0], "within"),
    ],
)
def test_bad_segment_edges_are_rejected(tmp_path: Path, edges: list[float], message: str) -> None:
    raw = _shipped_raw()
    raw["agreement_segments"] = {"edges": edges, "min_rows": 30}
    with pytest.raises(ValueError, match=message):
        load_policy(_write(tmp_path, raw))
    raw["agreement_segments"] = {"edges": [0.5, 1.0], "high_risk_edges": edges, "min_rows": 30}
    with pytest.raises(ValueError, match=message):
        load_policy(_write(tmp_path, raw))


@pytest.mark.parametrize("min_rows", [0, -3, 2.5, "30"])
def test_bad_min_rows_is_rejected(tmp_path: Path, min_rows: object) -> None:
    raw = _shipped_raw()
    raw["agreement_segments"] = {"edges": [0.5, 1.0], "min_rows": min_rows}
    with pytest.raises(ValueError, match="min_rows"):
        load_policy(_write(tmp_path, raw))
