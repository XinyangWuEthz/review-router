"""Policy loader: enforcement tiers, routing rules and CI gate floors.

Policy lives in YAML rather than code so the enforcement-cost reasoning is
reviewable by someone who does not read Python, and so the CI floors are
declared exactly once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["Condition", "Rule", "Policy", "load_policy", "DEFAULT_POLICY_PATH"]

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
class Policy:
    version: int
    tiers: dict[str, int]
    severity_weights: dict[str, float]
    tier_precision_floors: dict[str, float]
    rules: tuple[Rule, ...]
    gates: dict[str, Any]

    def route(self, signals: dict[str, float], default: str = "allow") -> str:
        """Highest matching tier wins; nothing matching means the default tier."""
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

    return Policy(
        version=int(raw.get("version", 1)),
        tiers={str(k): int(v) for k, v in tiers.items()},
        severity_weights={
            str(k): float(v) for k, v in (raw.get("severity_weights") or {}).items()
        },
        tier_precision_floors={
            str(k): float(v)
            for k, v in (raw.get("tier_precision_floors") or {}).items()
        },
        rules=tuple(rules),
        gates=dict(raw.get("gates") or {}),
    )
