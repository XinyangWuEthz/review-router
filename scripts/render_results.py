#!/usr/bin/env python
"""Render a README results block from a finished run.

The run's policy snapshot supplies precision targets and comparison gates.
An explicit --policy replaces comparison gates, never the run's targets.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from review_router.policy import load_policy  # noqa: E402

START, END = "<!-- results:start -->", "<!-- results:end -->"
AUTO, PRIORITY, HUMAN = "auto_action", "priority_review", "human_review"


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, dict) and "mean" in value:
        return f"{fmt(value['mean'], digits)} ± {fmt(value.get('std'), digits)}"
    if isinstance(value, float):
        return f"{value:.{digits}f}" if finite(value) else "n/a"
    return str(value)


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or not denominator:
        return None
    if not finite(numerator) or not finite(denominator):
        return None
    return numerator / denominator


def ci(value: list[float] | None, digits: int = 2) -> str:
    if not value or len(value) != 2 or not all(finite(v) for v in value):
        return "n/a"
    return f"{value[0]:.{digits}f} to {value[1]:.{digits}f}"


def gate_status(value: Any, limit: Any, *, upper: bool = False) -> str:
    if limit is None:
        return "not set"
    if not finite(value) or not finite(limit):
        return "unavailable"
    passed = value <= limit if upper else value >= limit
    return "green" if passed else "**red**"


def fairness_gate_status(section: dict[str, Any], ceiling: Any, minimum: int) -> str:
    """Match the evidence states of the subgroup regression gate."""
    if ceiling is None:
        return "not set"
    if not finite(ceiling):
        return "unavailable"
    counts = [section.get(group, {}).get("n_predicted_positive") for group in
              ("with_identity_term", "without_identity_term")]
    if not all(finite(n) for n in counts):
        return "unavailable"
    if any(n < minimum for n in counts):
        return f"skipped (subgroup n < {minimum})"
    interval = section.get("false_discovery_rate_ratio_ci95")
    if (not isinstance(interval, (list, tuple)) or len(interval) != 2
            or not all(finite(v) for v in interval) or not 0 <= interval[0] <= interval[1]):
        return "unavailable"
    if interval[0] > ceiling:
        return "**red**"
    if interval[1] <= ceiling:
        return "green"
    return "inconclusive"


def _group_cell(group: dict[str, Any], metric: str, denominator: str) -> str:
    return (f"{fmt(group.get(metric))} "
            f"({fmt(group.get('n_false_positive'))}/{fmt(group.get(denominator))})")


def _identity_lines(
    report: dict[str, Any], floors: dict[str, float], high_tier: str = AUTO
) -> list[str]:
    identity = report.get("identity_false_positives", {}).get("test", {})
    terms = identity.get("identity_terms")
    vocabulary = (f"The run recorded {len(terms)} whole-word terms in its `report.json`."
                  if isinstance(terms, list) else "This run did not record its identity-term vocabulary.")
    correctness = (
        "For both priority and standard human review, a flagged comment is a false discovery when no label is true. "
        if high_tier == PRIORITY else
        "For auto-action, a decision is wrong when none of its triggering labels is true; "
        "for human review, it is wrong when no label is true. "
    )
    high_name = "priority-review candidates" if high_tier == PRIORITY else "auto-actions"
    lines = [
        "**Identity mentions.** " + vocabulary + " This word-list diagnostic is a proxy, not an identity annotation.",
        "",
        "False discovery rate (FDR) divides wrong positive decisions by all positive decisions in each slice. "
        + correctness + "Ratios compare comments with an identity term to those without one.",
        "",
        "| decision basis | tier | FDR with term (wrong / decisions) | FDR without term | ratio | ratio 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    sources = (
        ("population_tier", high_tier, "population thresholds"),
        ("model_tier", high_tier, "after subgroup threshold, before rules"),
        ("final_tier", "predicted_positive", "final routing"),
        ("final_tier", high_tier, "final routing"),
        ("final_tier", HUMAN, "final routing"),
    )
    for basis, tier, name in sources:
        e = identity.get(basis, {}).get(tier, {})
        with_term = _group_cell(e.get("with_identity_term", {}), "false_discovery_rate", "n_predicted_positive")
        without = _group_cell(e.get("without_identity_term", {}), "false_discovery_rate", "n_predicted_positive")
        lines.append(f"| {name} | {tier} | {with_term} | {without} | "
                     f"{fmt(e.get('false_discovery_rate_ratio'), 2)} | {ci(e.get('false_discovery_rate_ratio_ci95'))} |")
    lines += [
        "",
        "Conventional false-positive rate (FPR) uses all actually negative comments in the slice as its denominator, "
        "including allowed comments. Here 'negative' means no positive label. It measures the fraction of clean comments "
        "sent to each tier. " + (
            "Both review tiers use the same any-positive-label definition of correctness."
            if high_tier == PRIORITY else
            "It does not count wrong-trigger auto-actions on comments that have another positive label."
        ),
        "",
        "| final tier | FPR with term (clean routed / clean total) | FPR without term | ratio | ratio 95% CI |",
        "|---|---:|---:|---:|---|",
    ]
    final = identity.get("final_tier", {})
    for tier in ("predicted_positive", high_tier, HUMAN):
        e = final.get("clean_negative_false_positives", {}).get(tier, {})
        with_term = _group_cell(e.get("with_identity_term", {}), "false_positive_rate", "n_actual_negative")
        without = _group_cell(e.get("without_identity_term", {}), "false_positive_rate", "n_actual_negative")
        lines.append(f"| {tier} | {with_term} | {without} | "
                     f"{fmt(e.get('false_positive_rate_ratio'), 2)} | {ci(e.get('false_positive_rate_ratio_ci95'))} |")
    sub = report.get("subgroup_thresholds", {})
    if high_tier in sub.get("thresholds", {}):
        thresholds = sub["thresholds"][high_tier].get("identity_term_present", {})
        descriptions = [f"{label}: {fmt(value, 4) if value is not None else 'no qualifying slice threshold'}"
                        for label, value in thresholds.items()]
        selection = sub.get("threshold_selection", {}).get(high_tier, {})
        test = sub.get("test", {}).get(high_tier, {})
        lines += [
            "",
            f"The subgroup threshold uses the run's {high_tier} precision target of {fmt(floors.get(high_tier), 2)} "
            "on identity-term comments in the threshold-selection split, at or above each population threshold. "
            f"A label without a qualifying slice threshold cannot trigger {high_tier} on that slice. "
            "These are empirical selection targets, not statistical guarantees on test data.",
            "",
            "Slice thresholds: " + ("; ".join(descriptions) or "unavailable") + ". "
            f"They removed {fmt(selection.get('n_removed_by_subgroup_threshold'))} of "
            f"{fmt(selection.get('n_population_tier'))} population {high_name} on selection data and "
            f"{fmt(test.get('n_removed_by_subgroup_threshold'))} of {fmt(test.get('n_population_tier'))} on test data. "
            f"Of the removed test decisions, {fmt(test.get('n_removed_truly_positive_any_label'))} had a positive label.",
        ]
    omission = final.get("allow_false_omission", {})
    base = final.get("base_rate", {})
    if omission or base:
        with_term, without = (omission.get(group, {}) for group in ("with_identity_term", "without_identity_term"))
        lines += [
            "",
            "False omission rate uses allowed comments as its denominator. "
            f"It is {fmt(with_term.get('false_omission_rate'))} with a term "
            f"({fmt(with_term.get('n_truly_positive'))}/{fmt(with_term.get('n_allowed'))} allowed comments have a positive label) "
            f"and {fmt(without.get('false_omission_rate'))} without "
            f"({fmt(without.get('n_truly_positive'))}/{fmt(without.get('n_allowed'))}). "
            f"The overall positive-label base rates are {fmt(base.get('with_identity_term', {}).get('rate'))} "
            f"and {fmt(base.get('without_identity_term', {}).get('rate'))}, respectively.",
        ]
    terms_rows = final.get("by_term", [])[:5]
    if terms_rows:
        lines += ["", "The five terms with the most positive decisions are shown below; comments can match multiple terms.", "",
                  "| term | wrong decisions | positive decisions | FDR |", "|---|---:|---:|---:|"]
        for row in terms_rows:
            lines.append(f"| {row['term']} | {row['n_false_positive']} | {row['n_predicted_positive']} | "
                         f"{fmt(row.get('false_discovery_rate'))} |")
    return lines


def _simulation_lines(sim: dict[str, Any], high_tier: str = AUTO) -> list[str]:
    summary = sim.get("summary", {})
    if not summary:
        return ["**Queue simulation.** " + sim.get("note", "No queue simulation results are available.")]
    a = sim["assumptions"]
    review_pool = (
        "Arrival rates are post-admission review jobs per hour, sampled from both priority_review and human_review. "
        "They are not incoming platform-comment rates. Every admitted item uses reviewer capacity. "
        if high_tier == PRIORITY else ""
    )
    if high_tier == PRIORITY and a.get("primary_order"):
        review_pool += f"Primary ordering ({sim.get('primary', {}).get('strategy', 'unavailable')}): {a['primary_order']}. "
    if high_tier == PRIORITY and a.get("priority_order"):
        review_pool += f"Priority-band ordering: {a['priority_order']}. "
    wait_population = (
        "all jobs that started, including reviews still in progress at the horizon"
        if "time_metrics" in sim else "completed items"
    )
    lines = [
        f"**Queue simulation.** {a['reviewers']} reviewers, {a['handle_minutes']:g} min per item, {a['horizon_hours']:g} h, "
        f"capacity {a['capacity_per_hour']:g}/h. The sampled pool has {a['queued_jobs']} queued comments "
        f"and {a['queued_high_risk']} high-risk comments; high-risk means {a.get('high_risk', 'definition unavailable')}. "
        f"The run uses {len(a['seeds'])} seeds with identical arrivals and handle times for every ordering within each seed.",
        "",
        review_pool + f"Arrival assumption: {a.get('arrivals', 'unavailable')}. Handle times: {a.get('handle_time', 'unavailable')}. "
        f"Harm proxy: {a.get('harm_proxy', 'unavailable')}. Waits are minutes until review starts, measured over {wait_population}. "
        "High-risk left includes jobs still in service; backlog counts jobs not yet started. "
        "The table reports seed means and standard deviations. Completion counts at a finite horizon do not establish queue stability.",
        "",
        "| load/h | ordering | handled | high-risk handled | high-risk left | harm / reviewer-h | high-risk wait p50 | high-risk wait p90 | wait p90, all items | backlog at end |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in summary.values():
        lines.append(
            f"| {e['load_per_hour']:g} | {e['strategy']} | {fmt(e.get('n_handled'), 0)} | {fmt(e.get('high_risk_handled'), 1)} "
            f"| {fmt(e.get('high_risk_unhandled'), 1)} | {fmt(e.get('harm_per_reviewer_hour'), 1)} | {fmt(e.get('high_risk_wait_p50'), 1)} "
            f"| {fmt(e.get('high_risk_wait_p90'), 1)} | {fmt(e.get('wait_p90'), 1)} | {fmt(e.get('backlog_end'), 0)} |"
        )
    paired = sim.get("paired", {})
    if paired:
        lines += [
            "", "Paired comparisons use shared arrivals within each seed. Differences are first ordering minus second. "
            "Each cell gives the mean difference and the number of seeds in which the first ordering was better on that metric.", "",
            "| comparison | harm / reviewer-h difference | high-risk handled difference | high-risk wait p90 difference |",
            "|---|---:|---:|---:|",
        ]
        for name, fields in paired.items():
            cells = []
            for field in ("harm_per_reviewer_hour", "high_risk_handled", "high_risk_wait_p90"):
                stats = fields.get(field)
                cells.append(f"{fmt(stats['mean_diff'], 2)}; better {stats['n_seeds_first_better']}/{stats['n_seeds']}"
                             if stats else "n/a")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += ["", "Harm and high-risk status use the same policy weights as severity ordering. "
                  "These comparisons are conditional on that weight vector and the stated arrival model."]
    if "time_metrics" in sim:
        lines += _time_lines(sim)
    return lines


def _time_lines(sim: dict[str, Any]) -> list[str]:
    head = sim["headline"]
    selected = head["selected"]
    reduction = (
        f"Reduction: {100 * selected['reduction']:.1f}%."
        if finite(selected.get("reduction")) else
        f"Absolute reduction: {fmt(selected.get('absolute_difference_min'))} min; percentage unavailable."
    )
    lines = [
        "", f"**Headline time metric.** {head['metric']} p{head['percentile']} at {head['load_per_hour']:g}/h, "
        f"pooled over seeds: {head['router_strategy']} {fmt(selected['router'])} vs FIFO {fmt(selected['fifo'])} min "
        f"(n={selected['n_router']} / {selected['n_fifo']}). {reduction}", "",
        "Waiting time uses all started reviews. Completion latency uses reviews completed within the horizon. "
        "Never-started jobs have no observed wait; their counts and age at the horizon remain in the records. "
        "Completion means simulated review service finished; actual moderation outcomes are not observed. "
        "The two strategies use the same arrivals, but their started and completed subsets may differ.", "",
        "| load/h | ordering | completed / in progress / not started | high-risk completed / in progress / not started | wait p50 / p90 / p99 (n) | completion latency p50 / p90 / p99 (n) | p99 reliable, wait / completion |",
        "|---:|---|---|---|---|---|---|",
    ]
    for key, block in sim["time_metrics"].items():
        strategy, load = key.split("@")
        tm = block["pooled_over_seeds"]
        status, hr = tm["status"], tm["high_risk_status"]
        wait, latency = tm["wait"], tm["completion_latency"]
        cells = [load, strategy]
        cells += [" / ".join(str(s[k]) for k in ("completed", "in_progress", "not_started")) for s in (status, hr)]
        cells += [" / ".join(fmt(s[k]) for k in ("p50", "p90", "p99")) + f" ({s['n']})" for s in (wait, latency)]
        cells.append(" / ".join("yes" if s["p99_reliable"] else "no" for s in (wait, latency)))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "All times are minutes. A p99 with fewer than 100 observations is flagged unreliable. "
              "Counts pool seeds, whereas the preceding simulation table reports seed means."]
    return lines


def _gate_lines(
    report: dict[str, Any], gates: dict[str, Any], source: Path, high_tier: str = AUTO
) -> list[str]:
    sim = report.get("simulation", {})
    routing, fairness = gates.get("routing", {}), gates.get("fairness", {})
    capacity = gates.get("capacity", routing)
    classification = gates.get("classification", gates)
    minimum = gates.get("min_predicted_positives_for_precision", 30)
    rows = []
    if high_tier == PRIORITY:
        contract = report.get("decision_contract", {})
        complete = all(key in contract for key in
                       ("mode", "requires_human_confirmation", "automatic_actions"))
        valid = (contract.get("mode") == "human_confirmation"
                 and contract.get("requires_human_confirmation") is True
                 and contract.get("automatic_actions") == 0)
        status = "unavailable" if not complete else "green" if valid else "**red**"
        rows.append(("human confirmation before every moderation action",
                     f"mode={fmt(contract.get('mode'))}; automatic actions={fmt(contract.get('automatic_actions'))}",
                     "human_confirmation; required=true; actions=0", status))
    specs = {label: spec for label, spec in classification.get("per_label_average_precision", {}).items()
             if spec.get("floor") is not None}
    statuses = [gate_status(report.get("per_label", {}).get(label, {}).get("average_precision"), spec["floor"])
                for label, spec in specs.items()]
    ap_status = ("not set" if not statuses else "**red**" if "**red**" in statuses
                 else "unavailable" if "unavailable" in statuses else "green")
    rows.append((f"per-label AP ({len(specs)} configured labels)", "see label table", "snapshot/override floors", ap_status))
    for label, spec in classification.get("per_label_precision_at_human_threshold", {}).items():
        stats = report.get("per_label", {}).get(label, {}).get("at_human_review", {})
        floor, value = spec.get("floor"), stats.get("precision")
        status = gate_status(value, floor)
        if floor is not None:
            count = stats.get("n_predicted_positive")
            if not finite(count):
                status = "unavailable"
            elif count < minimum or stats.get("threshold") is None:
                status = "**red**"
        rows.append((f"{label} precision at human threshold", fmt(value), f"≥ {fmt(floor)}; n ≥ {minimum}", status))
    for label, spec in classification.get("volume_overshoot_max", {}).items():
        value = report.get("prevalence_shift", {}).get(label, {}).get("volume_overshoot_model")
        limit = spec.get("ceiling")
        rows.append((f"{label} predicted volume overshoot", fmt(value), f"≤ {fmt(limit)}", gate_status(value, limit, upper=True)))
    for tier in (high_tier, HUMAN):
        stats = report.get("tiers", {}).get(tier, {})
        value, floor = stats.get("precision"), routing.get(f"{tier}_precision_floor")
        status = gate_status(value, floor)
        if floor is not None:
            count = stats.get("n_predicted_positive")
            if not finite(count):
                status = "unavailable"
            elif count < minimum:
                status = "**red**" if tier == HUMAN else f"skipped (n < {minimum})"
        requirement = "diagnostic; no test floor" if floor is None and high_tier == PRIORITY else f"≥ {fmt(floor)}"
        rows.append((f"{tier} precision", fmt(value), requirement, status))
    primary = sim.get("primary", {})
    thesis = sim.get("thesis", primary)
    for metric, config_name, scenario, upper in (
        ("harm_per_reviewer_hour", "harm_per_reviewer_hour_vs_fifo_min", thesis, False),
        ("high_risk_wait_p90", "high_risk_wait_p90_vs_fifo_max", primary, True),
    ):
        values = sim.get(metric, {})
        value = ratio(values.get("router"), values.get("fifo"))
        limit = capacity.get(config_name)
        name = f"{metric}, {scenario.get('strategy', 'router')} / FIFO at {fmt(scenario.get('load_per_hour'))}/h"
        measured = fmt(value)
        status = gate_status(value, limit, upper=upper)
        if (high_tier == PRIORITY and metric == "high_risk_wait_p90"
                and finite(values.get("router")) and values["router"] > 0
                and values.get("fifo") == 0 and finite(limit)):
            measured = f"{fmt(values['router'])} min / zero FIFO wait"
            status = "**red**"
        rows.append((name, measured, f"{'≤' if upper else '≥'} {fmt(limit)}", status))
    if high_tier == PRIORITY:
        for scenario in ("primary", "thesis"):
            values = sim.get("high_risk_handled", {}).get(scenario, {})
            value = ratio(values.get("router"), values.get("fifo"))
            limit = capacity.get("high_risk_handled_vs_fifo_min")
            strategy = (thesis if scenario == "thesis" else primary).get("strategy", "router")
            name = f"high-risk completed, {strategy} / FIFO at {fmt(values.get('load_per_hour'))}/h ({scenario})"
            rows.append((name, fmt(value), f"≥ {fmt(limit)}", gate_status(value, limit)))
    rows.extend(_ordering_rows(sim, capacity.get("ordering_vs_alternatives")))
    for metric, config_name, upper in (
        ("completion_ratio", "completion_ratio_min", False),
        ("backlog_end", "backlog_end_max", True),
        ("wait_p50", "wait_p50_max_min", True),
    ):
        if config_name not in capacity:
            continue
        value = sim.get(metric)
        if metric == "wait_p50":
            key = f"{primary.get('strategy')}@{primary.get('load_per_hour', 0):g}"
            value = sim.get("time_metrics", {}).get(key, {}).get("pooled_over_seeds", {}).get("wait", {}).get("p50")
        limit = capacity[config_name]
        rows.append((f"{metric} at {fmt(primary.get('load_per_hour'))}/h", fmt(value),
                     f"{'≤' if upper else '≥'} {fmt(limit)}", gate_status(value, limit, upper=upper)))
    for metric, config_name, upper in (
        ("queue_depth_p95", "queue_depth_p95_max", True),
        ("reviewer_utilization", "reviewer_utilization_min", False),
    ):
        value, limit = sim.get(metric), capacity.get(config_name)
        status = gate_status(value, limit, upper=upper)
        if (high_tier == AUTO and metric == "reviewer_utilization"
                and capacity.get("queue_depth_p95_max") is None):
            status = "skipped (queue-depth gate not set)"
        rows.append((f"{metric} at {fmt(primary.get('load_per_hour'))}/h", fmt(value),
                     f"{'≤' if upper else '≥'} {fmt(limit)}", status))
    value = report.get("consistency", {}).get("hierarchy_violation_rate")
    limit = classification.get("hierarchy_violation_rate_max", gates.get("consistency", {}).get("hierarchy_violation_rate_max"))
    rows.append(("hierarchy violation rate", fmt(value, 4), f"≤ {fmt(limit)}", gate_status(value, limit, upper=True)))
    identity = report.get("identity_false_positives", {}).get("test", {})
    ceiling = fairness.get("identity_false_discovery_rate_ratio_max")
    fairness_sources = (
        (("final_tier", "predicted_positive", "pooled, final tier"),
         ("final_tier", PRIORITY, "priority_review, final tier"),
         ("final_tier", HUMAN, "human_review, final tier"))
        if high_tier == PRIORITY else
        (("final_tier", "predicted_positive", "pooled, final tier"),
         ("model_tier", AUTO, "auto_action, after subgroup threshold before rules"))
    )
    for basis, tier, name in fairness_sources:
        section = identity.get(basis, {}).get(tier, {})
        measured = f"{fmt(section.get('false_discovery_rate_ratio'))} [{ci(section.get('false_discovery_rate_ratio_ci95'))}]"
        rows.append((f"identity FDR ratio, {name}", measured, f"≤ {fmt(ceiling)}", fairness_gate_status(section, ceiling, minimum)))
    lines = [f"Gates read from `{source}`:", "",
             "For identity FDR ratios, green requires the entire 95% interval to be at or below the ceiling. "
             "An interval entirely above it is red; a crossing interval is inconclusive. "
             "Subgroups below the minimum count are skipped, and missing estimates are unavailable. "
             "Neither state is a passing fairness result.", "",
             "| gate | measured | requirement | status |", "|---|---:|---|---|"]
    lines.extend(f"| {name} | {measured} | {requirement} | {status} |" for name, measured, requirement, status in rows)
    if "reproducibility" in gates:
        spec = gates["reproducibility"]
        lines += ["", f"Reproducibility checks use `{spec['records_file']}` with absolute tolerance {spec['float_tolerance']:g}. "
                  "The gate suite recomputes record-derived metrics and replays the configured scenario. "
                  "Policy/config hashes and, when required, a clean matching commit are checked separately. "
                  "This table displays measured criteria; rendering it does not execute or certify those checks."]
    return lines


def _ordering_rows(
    sim: dict[str, Any], ordering: dict[str, Any] | None
) -> list[tuple[str, str, str, str]]:
    """Primary ordering versus each declared alternative: mean per-seed difference at the thesis load.

    Rows are skipped when the block is absent, the run has no thesis scenario, the alternative
    is the primary ordering itself, or the run has no paired entry for it. The requirement is a
    declared tolerance, not a ratio.
    """
    primary, thesis = sim.get("primary", {}), sim.get("thesis")
    if not ordering or not thesis or not finite(thesis.get("load_per_hour")):
        return []
    load = thesis["load_per_hour"]
    rows = []
    for alt in ordering.get("alternatives", []):
        entry = sim.get("paired", {}).get(f"{primary.get('strategy')}_vs_{alt}@{load:g}")
        if alt == primary.get("strategy") or not entry:
            continue
        for field, label, config_name in (
            ("high_risk_handled", "high-risk completed", "high_risk_handled_tolerance"),
            ("harm_per_reviewer_hour", "harm per reviewer-hour", "harm_per_reviewer_hour_tolerance"),
        ):
            tolerance = ordering.get(config_name)
            value = (entry.get(field) or {}).get("mean_diff")
            limit = -tolerance if finite(tolerance) else None
            rows.append((f"ordering vs {alt} at {fmt(load)}/h: {label} mean diff", fmt(value),
                         f"≥ -{fmt(tolerance)}", gate_status(value, limit)))
    return rows


def _criteria_lines(
    report: dict[str, Any], gates: dict[str, Any], floors: dict[str, float], source: Path
) -> list[str]:
    routing = gates.get("capacity", gates.get("routing", {}))
    sim = report.get("simulation", {})
    primary, thesis = sim.get("primary", {}), sim.get("thesis", {})
    ordering = routing.get("ordering_vs_alternatives") or {}
    ordering_text = (
        f"At {fmt(thesis.get('load_per_hour'))}/h the primary ordering's mean per-seed difference against each of "
        f"{', '.join(str(alt) for alt in ordering.get('alternatives', []) if alt != primary.get('strategy'))} must be at least "
        f"-{fmt(ordering.get('high_risk_handled_tolerance'))} high-risk completions and "
        f"-{fmt(ordering.get('harm_per_reviewer_hour_tolerance'))} harm per reviewer-hour; wait quantiles are reported and do not decide. "
        if ordering else ""
    )
    return [
        "**Decision contract and evaluation criteria.** Every moderation action requires human confirmation. "
        "The model routes comments to priority_review, human_review or allow; priority_review schedules a human review earlier "
        "and does not authorize an automatic moderation action. Automatic actions must remain zero.", "",
        f"The run's per-label selection targets are {fmt(floors.get(PRIORITY), 2)} for priority_review "
        f"and {fmt(floors.get(HUMAN), 2)} for human_review. These are empirical targets on the selection split, "
        "not guarantees of test precision. Per-tier test precision remains diagnostic when its gate is not set.", "",
        f"Comparison gates come from `{source}`. The router is the configured primary ordering, "
        f"{fmt(primary.get('strategy'))}; the band-first priority ordering is one of the compared alternatives. "
        "At equal reviewer capacity, the router/FIFO harm-per-reviewer-hour ratio "
        f"at {fmt(thesis.get('load_per_hour'))}/h must be ≥ {fmt(routing.get('harm_per_reviewer_hour_vs_fifo_min'))}; "
        f"the high-risk wait p90 ratio at {fmt(primary.get('load_per_hour'))}/h must be ≤ "
        f"{fmt(routing.get('high_risk_wait_p90_vs_fifo_max'))}. The high-risk completed-count ratio must be ≥ "
        f"{fmt(routing.get('high_risk_handled_vs_fifo_min'))} at both named loads. "
        "These comparisons use ratios of seed means. They are descriptive checks that the measured result is no worse than FIFO, "
        "not statistical non-inferiority tests. " + ordering_text
        + "Historical automatic-enforcement results use a different contract.", "",
    ]


def _workload_lines(report: dict[str, Any]) -> list[str]:
    workload = report.get("review_workload", {})
    contract = report.get("decision_contract", {})
    return [
        "**Total human-review workload.** Both flagged tiers enter the same review pool and consume reviewer capacity. "
        f"The report records {fmt(workload.get('n_requires_human_review'))} of {fmt(workload.get('n_total'))} comments "
        f"requiring review, a fraction of {fmt(workload.get('review_fraction'))}: "
        f"{fmt(workload.get('n_priority_review'))} priority reviews and {fmt(workload.get('n_human_review'))} standard reviews. "
        f"Of the flagged comments, {fmt(workload.get('n_truly_positive'))} have at least one positive label. "
        f"Combined review precision is {fmt(workload.get('precision'))}, with 95% interval "
        f"{ci(workload.get('precision_ci95'), 3)}. Recorded automatic actions: {fmt(contract.get('automatic_actions'))}.",
        "",
    ]


def _prevalence_lines(report: dict[str, Any]) -> list[str]:
    shift = report.get("prevalence_shift")
    if not shift:
        return []
    lines = [
        "", "**Prevalence and predicted volume.** Expected positives are the sum of calibrated probabilities on test rows. "
        "Overshoot is expected divided by actual positives, minus one. This measures aggregate probability calibration; "
        "it does not by itself establish why the distributions differ.", "",
        "| label | train-file prevalence | test prevalence | actual positives | model-expected positives | volume overshoot |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in report.get("per_label", {}):
        s = shift[label]
        lines.append(f"| {label} | {fmt(s['prevalence_train_file'], 4)} | {fmt(s['prevalence_test'], 4)} "
                     f"| {s['actual_positives_test']} | {fmt(s['model_expected_positives_test'], 1)} | {fmt(s['volume_overshoot_model'])} |")
    return lines


def render(run_dir: Path, policy_path: Path | None = None) -> str:
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("evaluate_test") is False:
        raise ValueError(f"{run_dir} is a development run (evaluate_test: false): test rows not scored, so there is no results block to render")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    snapshot_path = run_dir / "policy.yaml"
    snapshot = load_policy(snapshot_path)
    high_tier = PRIORITY if PRIORITY in snapshot.tiers else AUTO
    high_label = "priority" if high_tier == PRIORITY else "auto"
    floors = snapshot.tier_precision_floors
    gates = load_policy(policy_path).gates if policy_path is not None else snapshot.gates
    gate_source = policy_path if policy_path is not None else snapshot_path
    counts = manifest["split_label_counts"]
    commit = manifest.get("git_commit")
    provenance = f"commit `{commit[:12]}`" if commit else "an unavailable Git commit"
    if manifest.get("git_dirty"):
        provenance += " with uncommitted changes (manifest `git_dirty: true`)"
    synthetic = report.get("synthetic", False)
    title = "synthetic pipeline check" if synthetic else "Jigsaw scored test rows"
    heading = "Review-routing results" if high_tier == PRIORITY else "Round-1 results"
    lines = [f"## {heading} ({title}), run `{manifest['run_id']}`", ""]
    if synthetic:
        lines += ["**SYNTHETIC corpus: pipeline check only, not Jigsaw results.**", ""]
    lines += [
        f"Generated from this run's `report.json` and `manifest.json`. The run used {provenance}, "
        f"seed {manifest['seed']}, scikit-learn {manifest['dependencies']['scikit-learn']}. "
        "The input files' SHA-256 hashes are recorded in the manifest.", "",
        f"Development split: {counts['train']['rows']:,} / {counts['calib']['rows']:,} / {counts['thresh']['rows']:,} rows "
        f"for train / calibration / threshold selection; {counts['test_scored']['rows']:,} scored test rows evaluated separately. "
        f"Precision targets come from the run snapshot `{snapshot_path}`.", "",
    ]
    if policy_path is not None:
        lines += [f"Comparison gates are explicitly overridden by `{policy_path}`; the run's precision targets remain those in its snapshot.", ""]
    if high_tier == PRIORITY:
        lines += _criteria_lines(report, gates, floors, gate_source)
    lines += [
        f"**Per label.** Thresholds were selected for precision targets {fmt(floors.get(HUMAN), 2)} (human) and "
        f"{fmt(floors.get(high_tier), 2)} ({high_label}). Selection and test AP are both shown. "
        "An AP difference describes a measured performance gap; it does not by itself identify the cause.", "",
        f"| label | positives | AP selection | AP test | ROC-AUC test | human thr | P@human | R@human | {high_label} thr | P@{high_label} | R@{high_label} |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selection = report.get("threshold_selection", {})
    for label, e in report.get("per_label", {}).items():
        h, a = e["at_human_review"], e[f"at_{high_tier}"]
        lines.append(f"| {label} | {e['positives']} | {fmt(selection.get('per_label', {}).get(label, {}).get('average_precision'))} "
                     f"| {fmt(e['average_precision'])} | {fmt(e['roc_auc'])} | {fmt(h['threshold'], 4)} "
                     f"| {fmt(h['precision'])} | {fmt(h['recall'])} | {fmt(a['threshold'], 4)} | {fmt(a['precision'])} | {fmt(a['recall'])} |")
    rules = report.get("rules", {})
    if high_tier == PRIORITY:
        rule_description = (
            "**Routing tiers.** Matching policy rules promoted "
            f"{fmt(rules.get('n_promoted_to_priority_review'))} comments to priority review and added "
            f"{fmt(rules.get('n_added_to_queue_from_allow'))} comments from allow to the review pool. "
            "These aggregate counts can overlap. Both review tiers count a comment as positive when any label is true."
        )
        lines += [""] + _workload_lines(report)
    else:
        rule_description = (
            "**Routing tiers.** Matching policy rules moved "
            f"{fmt(rules.get('n_downgraded_from_auto_action'))} auto-actions and "
            f"{fmt(rules.get('n_added_to_queue_from_allow'))} allows to human review. These counts aggregate all matching rules."
        )
    lines += ["", rule_description, "",
              "| tier | n | coverage | precision (test) | 95% CI | precision (selection split) | target |",
              "|---|---:|---:|---:|---|---:|---:|"]
    for tier, e in report.get("tiers", {}).items():
        lines.append(f"| {tier} | {e['n_predicted_positive']} | {fmt(e['coverage'])} | {fmt(e['precision'])} "
                     f"| {ci(e['precision_ci95'], 3)} | {fmt(selection.get('tiers', {}).get(tier, {}).get('precision'), 4)} "
                     f"| {fmt(floors.get(tier))} |")
    tier_note = ("Per-label precision targets need not hold after pooling labels, partitioning the review pool "
                 if high_tier == PRIORITY else
                 "Per-label precision targets need not hold after pooling labels, removing auto-actions from the human queue, ")
    lines += ["", tier_note + "or applying rules. Test precision is measured separately. The gates below retain the declared limits; "
              "changes to models, calibration or threshold selection need a fresh evaluation.", ""]
    lines += _identity_lines(report, floors, high_tier)
    lines += _prevalence_lines(report)
    lines += [""] + _simulation_lines(report.get("simulation", {}), high_tier)
    lines += [""] + _gate_lines(report, gates, gate_source, high_tier)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--readme", type=Path, default=Path(__file__).resolve().parent.parent / "README.md")
    parser.add_argument("--policy", type=Path, default=None,
                        help="explicit comparison-gate override; defaults to run_dir/policy.yaml; tier targets always use the run snapshot")
    args = parser.parse_args()
    try:
        block = render(args.run_dir, args.policy)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    text = args.readme.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"{args.readme} has no {START} / {END} markers")
    text = re.sub(re.escape(START) + ".*?" + re.escape(END), lambda _m: f"{START}\n{block}\n{END}", text, flags=re.S)
    args.readme.write_text(text, encoding="utf-8")
    print(f"rendered {args.run_dir} into {args.readme}")


if __name__ == "__main__":
    main()
