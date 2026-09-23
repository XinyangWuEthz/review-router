"""Policy loader: review priorities, routing rules and CI gate floors.

Policy lives in YAML rather than code so the enforcement-cost reasoning is
reviewable by someone who does not read Python, and so the CI floors are
declared exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Condition",
    "Rule",
    "SubgroupThreshold",
    "AgreementSegments",
    "SELECTION_RULES",
    "DEFAULT_SEGMENT_EDGES",
    "DEFAULT_HIGH_RISK_EDGES",
    "Policy",
    "load_policy",
    "DEFAULT_POLICY_PATH",
]

# Threshold-selection rules a tier may declare. cumulative_precision is the
# original rule (lowest t with precision(p >= t) >= floor); segment_agreement
# requires every declared score segment at or above the threshold to meet the
# floor on its own.
SELECTION_RULES: tuple[str, ...] = ("cumulative_precision", "segment_agreement")
DEFAULT_SEGMENT_EDGES: tuple[float, ...] = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0)
DEFAULT_HIGH_RISK_EDGES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.8, 0.9, 1.0)

DEFAULT_POLICY_PATH = Path(__file__).parent / "policy.yaml"

_OPS: dict[str, Any] = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


@dataclass(frozen=True)
class Condition:
    signal: str
    op: str
    threshold: float

    def holds(self, signals: dict[str, float]) -> bool:
        if self.signal not in signals:
            raise KeyError(f"signal {self.signal!r} not present in {sorted(signals)}")
        return bool(_OPS[self.op](signals[self.signal], self.threshold))


@dataclass(frozen=True)
class Rule:
    id: str
    description: str
    action: str
    conditions: tuple[Condition, ...]

    def matches(self, signals: dict[str, float]) -> bool:
        """Conditions within a rule are ANDed."""
        return all(c.holds(signals) for c in self.conditions)


@dataclass(frozen=True)
class SubgroupThreshold:
    """A per-label tier threshold target fitted to rows where `signal` is 1."""

    tier: str
    signal: str
    rationale: str


def _validate_edges(name: str, edges: tuple[float, ...]) -> None:
    if len(edges) < 2:
        raise ValueError(f"agreement_segments.{name} needs at least two edges")
    if any(not isfinite(e) or not 0.0 <= e <= 1.0 for e in edges):
        raise ValueError(f"agreement_segments.{name} must lie within [0, 1]")
    if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"agreement_segments.{name} must be strictly increasing")
    if edges[-1] != 1.0:
        raise ValueError(f"agreement_segments.{name} must end at 1.0")


@dataclass(frozen=True)
class AgreementSegments:
    """Score segments for the segment_agreement rule and the agreement tables.

    Segments are [edges[i], edges[i+1]) with the last one closed at 1.0.
    high_risk_edges, when given, replace edges for labels whose severity weight
    reaches the high-risk weight. min_rows is the smallest segment count that
    counts as evidence; use_wilson_lower_bound compares the Wilson 95% lower
    bound instead of the point estimate against the floor.
    """

    edges: tuple[float, ...] = DEFAULT_SEGMENT_EDGES
    high_risk_edges: tuple[float, ...] | None = DEFAULT_HIGH_RISK_EDGES
    min_rows: int = 30
    use_wilson_lower_bound: bool = False

    def __post_init__(self) -> None:
        _validate_edges("edges", self.edges)
        if self.high_risk_edges is not None:
            _validate_edges("high_risk_edges", self.high_risk_edges)
        if isinstance(self.min_rows, bool) or not isinstance(self.min_rows, int):
            raise ValueError("agreement_segments.min_rows must be an integer")
        if self.min_rows <= 0:
            raise ValueError("agreement_segments.min_rows must be positive")

    def edges_for(self, weight: float, high_risk_min_weight: float) -> tuple[float, ...]:
        """The edge set for a label of the given severity weight."""
        if self.high_risk_edges is not None and weight >= high_risk_min_weight:
            return self.high_risk_edges
        return self.edges


@dataclass(frozen=True)
class Policy:
    version: int
    tiers: dict[str, int]
    severity_weights: dict[str, float]
    tier_precision_floors: dict[str, float]
    rules: tuple[Rule, ...]
    gates: dict[str, Any]
    subgroup_thresholds: tuple[SubgroupThreshold, ...] = ()
    decision_mode: str = "legacy"
    # Per-tier selection rule; tiers absent from the mapping use cumulative_precision.
    tier_selection_rules: dict[str, str] = field(default_factory=dict)
    agreement_segments: AgreementSegments | None = None

    def __post_init__(self) -> None:
        for tier, rule in self.tier_selection_rules.items():
            if tier not in self.tier_precision_floors:
                raise ValueError(
                    f"tier_selection_rules: {tier!r} is not a tier with a precision floor"
                )
            if rule not in SELECTION_RULES:
                raise ValueError(
                    f"tier_selection_rules: unknown rule {rule!r} for {tier!r}; "
                    f"choose from {SELECTION_RULES}"
                )
        # Fill the defaults so every consumer sees a complete declaration.
        rules_filled = {
            tier: self.tier_selection_rules.get(tier, "cumulative_precision")
            for tier in self.tier_precision_floors
        }
        object.__setattr__(self, "tier_selection_rules", rules_filled)
        if self.agreement_segments is None:
            min_rows = int(self.gates.get("min_predicted_positives_for_precision", 30))
            object.__setattr__(self, "agreement_segments", AgreementSegments(min_rows=min_rows))
        if self.decision_mode not in {"legacy", "human_confirmation"}:
            raise ValueError(f"unknown decision_mode {self.decision_mode!r}")
        if self.version >= 2 and self.decision_mode != "human_confirmation":
            raise ValueError("policy v2 requires decision_mode: human_confirmation")
        if self.decision_mode == "human_confirmation":
            expected = {"allow": 1, "human_review": 2, "priority_review": 3}
            if self.tiers != expected:
                raise ValueError(
                    "human_confirmation requires allow/human_review/priority_review tiers "
                    "with ranks 1/2/3; automatic action tiers are forbidden"
                )
            if set(self.tier_precision_floors) != {"human_review", "priority_review"}:
                raise ValueError(
                    "human_confirmation requires empirical targets for both review tiers"
                )
            if any(rule.action not in expected for rule in self.rules):
                raise ValueError("human_confirmation rules must use review tiers or allow")
        if any(not isfinite(v) or not 0 <= v <= 1 for v in self.tier_precision_floors.values()):
            raise ValueError("tier precision targets must be finite values between 0 and 1")

    def route(self, signals: dict[str, float], default: str = "allow") -> str:
        """Highest matching tier wins; nothing matching means the default tier."""
        if self.decision_mode == "human_confirmation" and default not in self.tiers:
            raise ValueError(f"human_confirmation forbids fallback tier {default!r}")
        matched = [r.action for r in self.rules if r.matches(signals)]
        if not matched:
            return default
        return max(matched, key=lambda action: self.tiers[action])


def load_policy(path: Path | str = DEFAULT_POLICY_PATH) -> Policy:
    """Parse and validate a policy file. Raises ValueError on malformed policy."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("policy file must contain a mapping")

    tiers = raw.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        raise ValueError("policy must define a non-empty 'tiers' mapping")

    rules: list[Rule] = []
    for entry in raw.get("rules") or []:
        rule_id = entry.get("id", "<missing id>")
        action = entry.get("action")
        if action not in tiers:
            raise ValueError(f"rule {rule_id}: unknown action {action!r}")

        conditions: list[Condition] = []
        for cond in entry.get("conditions") or []:
            op = cond.get("op")
            if op not in _OPS:
                raise ValueError(f"rule {rule_id}: unknown op {op!r}")
            conditions.append(
                Condition(
                    signal=str(cond["signal"]),
                    op=str(op),
                    threshold=float(cond["threshold"]),
                )
            )

        rules.append(
            Rule(
                id=str(rule_id),
                description=str(entry.get("description", "")),
                action=str(action),
                conditions=tuple(conditions),
            )
        )

    floors = {str(k): float(v) for k, v in (raw.get("tier_precision_floors") or {}).items()}
    subgroups: list[SubgroupThreshold] = []
    for tier, entries in (raw.get("subgroup_thresholds") or {}).items():
        if tier not in tiers or tier not in floors:
            raise ValueError(f"subgroup_thresholds: {tier!r} is not a tier with a precision floor")
        if tier not in {"human_review", "priority_review", "auto_action"}:
            raise ValueError(f"subgroup_thresholds: {tier!r} is not a model-driven tier")
        for entry in entries or []:
            signal = entry.get("signal")
            if not signal:
                raise ValueError(f"subgroup_thresholds: {tier}: entry without a signal")
            subgroups.append(
                SubgroupThreshold(
                    tier=str(tier), signal=str(signal), rationale=str(entry.get("rationale", ""))
                )
            )

    gates = dict(raw.get("gates") or {})
    selection_rules = raw.get("tier_selection_rules") or {}
    if not isinstance(selection_rules, dict):
        raise ValueError("tier_selection_rules must be a mapping of tier -> rule name")
    segments = _parse_agreement_segments(raw.get("agreement_segments"), gates)

    return Policy(
        version=int(raw.get("version", 1)),
        tiers={str(k): int(v) for k, v in tiers.items()},
        severity_weights={str(k): float(v) for k, v in (raw.get("severity_weights") or {}).items()},
        tier_precision_floors=floors,
        rules=tuple(rules),
        gates=gates,
        subgroup_thresholds=tuple(subgroups),
        decision_mode=str(raw.get("decision_mode", "legacy")),
        tier_selection_rules={str(k): str(v) for k, v in selection_rules.items()},
        agreement_segments=segments,
    )


def _parse_agreement_segments(raw: Any, gates: dict[str, Any]) -> AgreementSegments | None:
    """Absent -> None (the Policy fills the defaults); present -> validated declaration."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("agreement_segments must be a mapping")
    min_rows_raw = raw.get("min_rows", gates.get("min_predicted_positives_for_precision", 30))
    if isinstance(min_rows_raw, bool) or not isinstance(min_rows_raw, int):
        raise ValueError("agreement_segments.min_rows must be an integer")
    high_risk = raw.get("high_risk_edges", DEFAULT_HIGH_RISK_EDGES)
    return AgreementSegments(
        edges=tuple(float(e) for e in raw.get("edges", DEFAULT_SEGMENT_EDGES)),
        high_risk_edges=None if high_risk is None else tuple(float(e) for e in high_risk),
        min_rows=min_rows_raw,
        use_wilson_lower_bound=bool(raw.get("use_wilson_lower_bound", False)),
    )
