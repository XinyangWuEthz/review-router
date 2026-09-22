from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from review_router.policy import DEFAULT_POLICY_PATH, load_policy


def test_shipped_policy_parses() -> None:
    policy = load_policy()
    assert policy.version == 1
    assert policy.rules, "shipped policy declares no rules"


def test_shipped_policy_actions_are_known_tiers() -> None:
    policy = load_policy()
    assert {r.action for r in policy.rules} <= set(policy.tiers)


def test_every_rule_states_its_rationale() -> None:
    """A tier choice without a written reason is not reviewable policy."""
    policy = load_policy()
    for rule in policy.rules:
        assert len(rule.description.strip()) > 40, f"{rule.id} lacks a rationale"


def test_auto_action_floor_is_stricter_than_human_review() -> None:
    policy = load_policy()
    assert (
        policy.tier_precision_floors["auto_action"] > policy.tier_precision_floors["human_review"]
    )


def test_highest_matching_tier_wins() -> None:
    policy = load_policy()
    signals = {
        "p_threat": 0.9,
        "p_severe_toxic": 0.0,
        "p_toxic": 1.0,
        "identity_term_present": 0.0,
        "p_max": 1.0,
    }
    assert policy.route(signals) == "human_review"


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
    """severe_toxic asserted without toxic is logically impossible."""
    policy = load_policy()
    signals = {
        "p_threat": 0.0,
        "p_severe_toxic": 0.8,
        "p_toxic": 0.1,
        "identity_term_present": 0.0,
        "p_max": 0.8,
    }
    assert policy.route(signals) == "human_review"


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
