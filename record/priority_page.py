"""English Step 6 page, rendered only from the collected evaluation artifacts."""

from __future__ import annotations

from html import escape
from typing import Any

from scripts.render_results import fairness_gate_status

PRIORITY, HUMAN = "priority_review", "human_review"


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, dict):
        return f"{_number(value['mean'], digits)} ± {_number(value['std'], digits)}"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def _percent(value: Any) -> str:
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(item)}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(item)}</td>" for item in row) + "</tr>" for row in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def build_body(data: dict[str, Any]) -> str:
    run = data["run"]
    selection = data["threshold_selection"]
    cumulative = selection["alternatives"]["cumulative_precision"]
    test_cumulative = data["test_cumulative"]
    policy = run["policy_snapshot"]
    segments = policy["agreement_segments"]
    workload = run["review_workload"]
    sim = run["simulation"]
    primary = sim["primary"]["strategy"]
    thesis_load = sim["thesis"]["load_per_hour"]
    paired = sim["paired"][f"{primary}_vs_priority@{thesis_load:g}"]
    ci = data["ci"]
    ci_state = str(ci.get("conclusion") or ci.get("status") or "unavailable")
    ci_link = escape(str(ci["url"]), quote=True)
    ci_run_id = str(ci["url"]).rstrip("/").rsplit("/", 1)[-1]
    download_dir = f"reports/ci-{ci_run_id}"

    comparison_rows = []
    for split, arm, tiers in (
        ("Selection, in sample", "Cumulative", cumulative["tiers"]),
        ("Selection, in sample", "Segment", selection["tiers"]),
        ("Selection, cross-fitted", "Segment", selection["cross_fitted"]["tiers"]),
        ("Scored test, reused", "Cumulative", test_cumulative["tiers"]),
        ("Scored test, reused", "Segment", run["tiers"]),
    ):
        band = tiers[PRIORITY]
        comparison_rows.append(
            [
                split,
                arm,
                _number(band["n_predicted_positive"]),
                _percent(band["coverage"]),
                _percent(band["precision"]),
            ]
        )
    comparisons = _table(
        ["Rows", "Priority rule", "Priority count", "Coverage of all rows", "Priority precision"],
        comparison_rows,
    )

    thresholds = _table(
        ["Label", "Cumulative priority", "Segment priority", "Human review"],
        [
            [
                label,
                "disabled" if value is None else _number(value, 4),
                "disabled"
                if run["thresholds"][PRIORITY][label] is None
                else _number(run["thresholds"][PRIORITY][label], 4),
                "disabled"
                if run["thresholds"][HUMAN][label] is None
                else _number(run["thresholds"][HUMAN][label], 4),
            ]
            for label, value in cumulative["thresholds"][PRIORITY].items()
        ],
    )
    composition_rows = []
    for arm, metrics in (("Cumulative", test_cumulative["review_workload"]), ("Segment", workload)):
        for band, values in metrics["high_risk_by_band"].items():
            composition_rows.append(
                [
                    arm,
                    band,
                    _number(values["n"]),
                    _number(values["n_high_risk"]),
                    _percent(values["high_risk_share"]),
                ]
            )
    composition = _table(
        ["Test arm", "Band", "Comments", "High-risk comments", "High-risk share of band"],
        composition_rows,
    )

    queue_rows = []
    loads = list(dict.fromkeys((sim["primary"]["load_per_hour"], thesis_load)))
    for load in loads:
        for strategy in ("severity", "priority", "prob", "fifo"):
            stats = sim["summary"][f"{strategy}@{load:g}"]
            queue_rows.append(
                [
                    f"{load:g}",
                    strategy,
                    _number(stats["high_risk_handled"], 1),
                    _number(stats["harm_per_reviewer_hour"], 2),
                    _number(stats["high_risk_wait_p90"], 2),
                    _number(stats["high_risk_unhandled"], 1),
                ]
            )
    queue = _table(
        [
            "Review arrivals/h",
            "Ordering",
            "High-risk completed",
            "Harm proxy/reviewer-hour",
            "High-risk wait p90, min",
            "High-risk unfinished",
        ],
        queue_rows,
    )

    fairness = run["identity_false_positives"]["test"]["final_tier"][PRIORITY]
    limit = policy["gates"]["fairness"]["identity_false_discovery_rate_ratio_max"]
    fairness_status = fairness_gate_status(
        fairness, limit, policy["gates"]["min_predicted_positives_for_precision"]
    ).replace("**", "")
    interval = fairness.get("false_discovery_rate_ratio_ci95")
    interval_text = (
        f"[{_number(interval[0])}, {_number(interval[1])}]" if interval else "unavailable"
    )
    same_pool = test_cumulative["same_admitted_rows"]
    admission_note = (
        "The two rules admit exactly the same test comments. Since severity ignores the review "
        "band, changing only these bands cannot change its queue results."
        if same_pool
        else "The two rules admit different test comments, so their queue effects must be compared "
        "on a shared incoming comment stream."
    )
    hashes = "".join(
        f"<li><code>{escape(name)}</code>: <code>{escape(digest)}</code></li>"
        for name, digest in data["artifact_sha256"].items()
    )
    priority = run["tiers"][PRIORITY]
    old_priority = test_cumulative["tiers"][PRIORITY]
    total_hr = workload["n_high_risk_total"]
    hr_admitted = sum(
        workload["high_risk_by_band"][band]["n_high_risk"] for band in (PRIORITY, HUMAN)
    )
    return f"""
<span class="tag">Step 6</span><span class="tag">Policy v{escape(str(policy["version"]))}</span>
<h1>Severity ordering and confidence-segment review bands</h1>
<p class="lede">A real-data evaluation of the word model with severity as the primary queue ordering,
development-only diagnostics, and a stricter rule for assigning the priority label. Every moderation action still requires human confirmation.</p>
<div class="box"><b>Recorded conclusion.</b> On the same scored test rows, segment selection changes priority precision
from {_percent(old_priority["precision"])} to {_percent(priority["precision"])}, while the priority band shrinks
from {_number(old_priority["n_predicted_positive"])} to {_number(priority["n_predicted_positive"])} comments.
The current review pool contains {_number(workload["n_requires_human_review"])} comments and captures
{_number(hr_admitted)} of {_number(total_hr)} high-risk comments. {admission_note}</div>
<div class="toc"><a href="#method">What changed</a><a href="#bands">Band precision and coverage</a>
<a href="#composition">Where high-risk comments go</a><a href="#queue">Queue results</a>
<a href="#limits">Evidence and limits</a><a href="#reproduce">Reproduce this step</a></div>

<h2 id="method">What changed</h2>
<p>The primary ordering is <code>{escape(primary)}</code>: for each comment, take the maximum over labels of
calibrated probability × that label's severity weight, then review the highest-scoring comment first.
The <code>priority_review</code> label no longer guarantees earlier service.
Band-first priority, probability and FIFO remain comparisons on the same admitted comments and arrival streams.</p>
<p>For each label, the priority threshold is the lowest declared score edge for which every segment above it
meets the {_percent(policy["tier_precision_floors"][PRIORITY])} empirical agreement target with at least
{_number(segments["min_rows"])} rows. These are score ranges, not statistical confidence intervals.
Human-review thresholds and identity-term subgroup thresholds retain cumulative precision selection.</p>
<p>The rule is a project extension inspired by score-based triage in
<a href="https://arxiv.org/abs/2406.12800">Thomas et al., Supporting Human Raters</a>.
The paper does not prescribe this all-segments rule or these bin edges. The edges were designed after earlier test diagnostics were inspected.</p>
{thresholds}
<p>Disabled means that no label-specific threshold qualified; other labels or policy rules can still admit a comment.
The table shows population thresholds. Subgroup thresholds are stored in <a href="priority_run.json">the source record</a>.</p>

<h2 id="bands">Band precision and coverage</h2>
{comparisons}
<p>Precision here means that at least one original Jigsaw label is positive, after rules and subgroup thresholds.
It is not an independent measure of whether a comment deserves review. Cumulative and segment test arms use
the same saved model scores and selection data; neither fits thresholds on test labels.</p>
<p>The {_number(selection["cross_fitted"]["k"])}-fold development diagnostic refits thresholds on the other folds
before routing each held-out fold. It estimates sensitivity to threshold selection within the same distribution;
it is not a formal bound on optimism or evidence of generalization to a new distribution.</p>

<h2 id="composition">Where high-risk comments go</h2>
{composition}
<p>High risk uses the saved policy's true-label harm proxy and the simulation threshold
<code>{escape(str(sim["assumptions"]["high_risk"]))}</code>. A larger high-risk share in a smaller band can coexist
with fewer high-risk comments in that band. {_number(test_cumulative["n_changed_band"])} test comments change bands
between these two rules. {admission_note}</p>

<h2 id="queue">Queue results</h2>
<p>These rates are admitted review jobs per hour, not all incoming platform comments. Capacity is
{_number(sim["assumptions"]["capacity_per_hour"], 0)} jobs/hour with {_number(sim["assumptions"]["reviewers"])}
reviewers, {_number(sim["assumptions"]["handle_minutes"], 1)} minutes per job and an
{_number(sim["assumptions"]["horizon_hours"], 0)}-hour shift. Cells show mean ± standard deviation across
{_number(len(sim["assumptions"]["seeds"]))} paired simulation seeds.</p>
{queue}
<p>At {_number(thesis_load, 0)} review arrivals/hour, the paired mean difference, {escape(primary)} minus band-first priority,
is {_number(paired["high_risk_handled"]["mean_diff"], 2)} high-risk completions and
{_number(paired["harm_per_reviewer_hour"]["mean_diff"], 3)} harm proxy per reviewer-hour.
These are the current bands; the earlier v2 comparison is a different experiment.</p>
<p>Wait p90 includes jobs that started, including reviews still in progress. Never-started jobs are excluded from
waiting-time quantiles and remain in the unfinished count. These seeds characterize simulated arrivals on this fixed
corpus; they do not measure uncertainty over new corpora. Harm is a label-weighted proxy, not measured real-world harm.</p>

<h2 id="limits">Evidence and limits</h2>
<p><a href="{ci_link}">Recorded GitHub real-evaluation run</a>: <b>{escape(ci_state)}</b>.
The record includes the workflow jobs and artifact hashes. A successful run does not mean every diagnostic met a
precision target or that fairness was established. Per-band test precision gates remain unset.</p>
<p>For priority-review false discoveries, the identity-term subgroup ratio has 95% interval
{interval_text}, against a limit of {_number(limit)}. Its recorded-metric status is <b>{escape(fairness_status)}</b>.
FDR concerns false discoveries among flagged comments; it does not replace the separately recorded FPR on clean comments.</p>
<p>The public test set has been inspected repeatedly. This is a benchmark rerun and a descriptive comparison, not a fresh holdout.
The segment rule concentrates the priority band; it does not establish an improvement in model recall or actual human review value.</p>
<p>Two clean retrains with the same data, split, configuration and recorded dependency versions differed in two admitted comment IDs.
Those membership changes also change which comments the queue's random indices select. Compare orderings within this saved run;
differences between retrains cannot be attributed to the ordering alone.
The <a href="retraining_check.json">retraining check</a> found different vocabulary selections at the feature cap;
the exact platform cause remains unresolved.</p>
<p>Keep the word model and human confirmation. Jev evaluation remains unattempted following the reported registration limit.
An independent human evaluation of review worthiness remains <b>deferred and not started</b> because of annotation workload.
Original Jigsaw labels are retained.</p>

<h2 id="reproduce">Reproduce this step</h2>
<pre><code># Download and collect the recorded CI run
gh run download {escape(ci_run_id)} --name evaluation-{escape(run["git_commit"])} --dir {escape(download_dir)}
gh run view {escape(ci_run_id)} --json headSha,status,conclusion,url,jobs &gt;{escape(download_dir)}-ci.json
PYTHONPATH=. python record/collect_priority.py {escape(download_dir)} --ci-metadata {escape(download_dir)}-ci.json
python record/render.py
# Development diagnostics without scoring test rows
python scripts/run_pipeline.py --config configs/dev.yaml
# Rerun the benchmark protocol from the recorded code revision
python scripts/run_pipeline.py --config configs/baseline.yaml
REVIEW_ROUTER_EVAL_REPORT=reports/&lt;run&gt;/report.json pytest -q tests/test_gate.py
python scripts/render_results.py reports/&lt;run&gt;</code></pre>
<p class="prov">Run <code>{escape(run["run_id"])}</code>, commit <code>{escape(run["git_commit"])}</code>,
{"working tree had uncommitted changes" if run["git_dirty"] else "clean working tree"}.
The <a href="priority_run.json">Step 6 source record</a> contains policy, thresholds, development diagnostics,
the matched-score test comparison and CI provenance. Steps 1 to 5 retain their saved results.</p>
<details><summary>Collected artifact SHA-256 hashes</summary><ul>{hashes}</ul></details>
"""
