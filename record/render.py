#!/usr/bin/env python
"""Build the project record: index.html plus one page per project step.

    python record/render.py

Inputs: runs.json with saved policy snapshots (collect_runs.py), results.json
and figures/ (diagnose.py), and class-count arithmetic from the package.
Measured results come from these artifacts; prose explains their limits.
"""

from __future__ import annotations

import json
import re
import sys
from html import escape, unescape
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from review_router.ceilings import SCORED_TEST_COUNTS, ceiling_table  # noqa: E402
from scripts.render_results import fairness_gate_status, gate_status  # noqa: E402

RUNS = json.loads((HERE / "runs.json").read_text())
R = json.loads((HERE / "results.json").read_text())
HUMAN_RUN_PATH = HERE / "human_review_run.json"
HUMAN_RUN = json.loads(HUMAN_RUN_PATH.read_text()) if HUMAN_RUN_PATH.is_file() else None
POLICY = RUNS["final"]["policy_snapshot"]
LABELS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
AUTO_LABELS = ("toxic", "obscene", "insult")
AUTO, HUMAN, ALLOW = "auto_action", "human_review", "allow"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8983"

CSS = """
:root { --ink: #0b0b0b; --ink2: #52514e; --line: #e6e5e1; --surface: #fcfcfb; --accent: #185ea8; --warn: #a73816; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface); color: var(--ink); font: 16px/1.65 -apple-system, "PingFang SC", "Noto Sans CJK SC", "Helvetica Neue", Arial, sans-serif; }
main { max-width: 1000px; margin: 0 auto; padding: 32px 20px 80px; }
nav.top { display: flex; gap: 16px; font-size: 14px; color: var(--ink2); margin-bottom: 24px; flex-wrap: wrap; }
nav.top a, a { color: var(--accent); text-underline-offset: 3px; }
a:hover { text-decoration-thickness: 2px; }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 4px; }
nav.top a[aria-current="page"] { font-weight: 700; color: var(--ink); }
h1 { font-size: 26px; line-height: 1.3; margin: 0 0 8px; }
h2 { font-size: 20px; margin: 40px 0 10px; border-top: 1px solid var(--line); padding-top: 20px; }
h3 { font-size: 17px; margin: 26px 0 8px; }
h4 { font-size: 15px; margin: 18px 0 6px; color: var(--ink2); }
p { margin: 12px 0; max-width: 78ch; }
li { margin: 8px 0; }
h2, h3 { scroll-margin-top: 16px; }
.lede { color: var(--ink2); font-size: 15px; }
.tag { display: inline-block; font-size: 12px; padding: 2px 8px; border: 1px solid var(--line); border-radius: 999px; color: var(--ink2); margin-right: 6px; }
.tag.red { border-color: var(--warn); color: var(--warn); }
.table-scroll { overflow-x: auto; margin: 12px 0 18px; }
.table-scroll:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--ink2); font-weight: 600; }
td.n, th.n { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
figure { margin: 18px 0; }
figure img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
figcaption { font-size: 14px; color: var(--ink2); margin-top: 8px; max-width: 78ch; }
.box { border: 1px solid var(--line); border-left: 4px solid var(--accent); padding: 10px 14px; margin: 14px 0; border-radius: 4px; background: #fff; }
.box.warn { border-left-color: var(--warn); }
.steps { list-style: none; padding: 0; margin: 0; }
.steps li { border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; margin: 8px 0; background: #fff; }
.steps li a { font-weight: 600; }
.steps li .one { color: var(--ink2); font-size: 14px; margin-top: 2px; }
.toc { font-size: 15px; color: var(--ink2); columns: 2; margin: 8px 0 16px; }
.toc a { display: block; padding: 3px 0; break-inside: avoid; }
.flow { display: flex; gap: 8px; align-items: stretch; flex-wrap: wrap; margin: 14px 0; }
.flow .node { flex: 1 1 120px; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; background: #fff; font-size: 13px; }
.flow .node b { display: block; font-size: 14px; margin-bottom: 2px; }
.flow .arrow { align-self: center; color: var(--ink2); }
.examples li { margin: 6px 0; font-size: 14px; }
.examples blockquote { margin: 8px 0 18px; padding-left: 12px; border-left: 2px solid var(--line); overflow-wrap: anywhere; }
details { margin: 18px 0; border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; }
summary { cursor: pointer; font-weight: 600; }
.examples code, code { background: #f3f2ef; padding: 1px 4px; border-radius: 3px; font-size: 13px; }
pre { background: #f3f2ef; padding: 10px 12px; border-radius: 6px; font-size: 13px; overflow-x: auto; }
pre code { background: none; padding: 0; }
.prov { font-size: 13px; color: var(--ink2); }
code { overflow-wrap: anywhere; }
pre code { overflow-wrap: normal; }
nav.bottom { display: flex; justify-content: space-between; margin-top: 40px; font-size: 14px; }
@media (max-width: 640px) {
  main { padding: 24px 16px 48px; }
  h1 { font-size: 24px; }
  .toc { columns: 1; }
  .flow { flex-direction: column; }
  .flow .node { flex: auto; }
  .flow .arrow { display: none; }
  nav.bottom { gap: 16px; flex-wrap: wrap; }
}
"""

STEPS = [
    (
        "step1-baseline.html",
        "第 1 步：建立 baseline",
        "把实验封装成一条可重跑的命令：数据划分、模型、阈值、规则、队列仿真、报告和回归门槛",
    ),
    (
        "step2-round1.html",
        "第 2 步：首轮结果与诊断",
        "在 Jigsaw 数据上评估 baseline，移除 R103，并检查自动处置精确率未达标的原因",
    ),
]
PENDING = (
    "第 3 步：验证改进方案",
    "待做。比较更保守的阈值选择方法和字符特征，核查标注分歧，并用新增留出数据评估改进",
)
PENDING_STATUS = "待做"
if HUMAN_RUN is not None:
    STEPS.append((
        "step3-human-review.html", "第 3 步：优先人工审核",
        "取消自动处置权限，按全部人工工作量重新比较排序效果",
    ))
    PENDING = ("第 4 步：独立数据验证", "冻结候选方案后，用新增数据验证质量与工作量")
FEATURES_PATH = HERE / "features_run.json"
FEATURES = json.loads(FEATURES_PATH.read_text()) if FEATURES_PATH.is_file() else None
if FEATURES is not None and HUMAN_RUN is not None:
    STEPS.append((
        "step4-features.html", "第 4 步：误报复核与字符 n-gram",
        "人工复核自动处置误报，并在首轮和 v2 两种设计下比较词特征与词+字符特征",
    ))
    PENDING = (
        "第 5 步：独立数据验证",
        "仅保留为可选方向，尚未启动；待资源允许时再决定是否开展，不影响当前保留 word 的决定",
    )
    PENDING_STATUS = "暂缓"

JEV_PATH = HERE / "jev_run.json"
JEV = json.loads(JEV_PATH.read_text()) if JEV_PATH.is_file() else None
if JEV is not None and FEATURES is not None and HUMAN_RUN is not None:
    STEPS.append((
        "step5-jev.html", "第 5 步：Jev 与词模型对照",
        "注册容量已满，Jev 实测暂缓；已完成固定样本、接入代码与配对 word 基线"
        if JEV["status"] == "blocked_registration" else
        "固定样本、问题和政策，比较六标签概率及有限审核容量下的表现",
    ))
    PENDING = (
        "第 6 步：独立人工评估",
        "继续暂缓；本轮沿用原始标签，尚不能证明实际送审价值改善",
    )


DEFERRED_REVIEW_EVALUATION = """
<h2 id="future-review-evaluation">后续考虑：独立人工评估集，暂缓</h2>
<p>2026-09-23 记录。当前用“任一真实标签为阳性”近似衡量审核价值。对有歧义、需要结合语境判断的评论，即使人工最后确认没有违规，送审也可能合理；但缺少语境时，人工也可能无法判断。</p>
<p>后续可考虑建立独立评估集，覆盖优先审核、普通审核和放行，分别记录审核前是否值得送审、紧急程度与所需语境，以及审核后的结论、仍无法判断的情况和耗时。若开展，评估只提供实际可获得的语境，保留原始 Jigsaw 标签；已用于分析的审计样本作为开发案例，最终评估另用未参与调参的新样本。</p>
<p><b>状态：暂缓，尚未启动。</b>新增抽样与独立人工标注的工作量较大，本次仅记录为后续可选方向，待资源允许时再决定是否开展。</p>
"""


def f(v: Any, d: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, dict) and "mean" in v:
        return f"{v['mean']:.{d}f} ± {v['std']:.{d}f}"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def ci(v: list[float] | None, d: int = 3) -> str:
    return f"[{v[0]:.{d}f}, {v[1]:.{d}f}]" if v else ""


def page(
    name: str, title: str, body: str, prev_: tuple[str, str] | None, next_: tuple[str, str] | None
) -> None:
    def accessible_table(match: re.Match[str]) -> str:
        header = re.sub(r"<th(?=[ >])", '<th scope="col"', match[1])
        return '<div class="table-scroll" role="region" aria-label="数据表，可横向滚动" tabindex="0"><table><thead><tr>' + header + '</tr></thead><tbody>'

    body = re.sub(r"<table><tr>(.*?)</tr>", accessible_table, body, flags=re.S)
    body = body.replace("</table>", "</tbody></table></div>")
    nav_prev = (
        f'<a href="{prev_[0]}">← {prev_[1]}</a>' if prev_ else '<a href="index.html">← 目录</a>'
    )
    nav_next = f'<a href="{next_[0]}">{next_[1]} →</a>' if next_ else ""
    navigation = ''.join(
        f'<a href="{path}"' + (' aria-current="page"' if path == name else '')
        + f'>{label}</a>'
        for path, label in [("index.html", "目录"), *((s[0], s[1]) for s in STEPS)]
    )
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{CSS}</style></head><body><main>
<nav class="top" aria-label="项目步骤">{navigation}<span>{PENDING[0]}，{PENDING_STATUS}</span></nav>
{body}
<nav class="bottom" aria-label="前后步骤"><span>{nav_prev}</span><span>{nav_next}</span></nav>
</main></body></html>"""
    (HERE / name).write_text(doc, encoding="utf-8")


# ============================================================================ figures for step 1
def ceilings_figure() -> None:
    table = ceiling_table()
    labels = list(table)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    x = list(range(len(labels)))
    w = 0.36
    b1 = ax.bar(
        [i - w / 2 - 0.01 for i in x],
        [table[l][0.01] for l in labels],
        w,
        color=BLUE,
        label="FPR 1%",
    )
    b2 = ax.bar(
        [i + w / 2 + 0.01 for i in x],
        [table[l][0.001] for l in labels],
        w,
        color=AQUA,
        label="FPR 0.1%",
    )
    for bars in (b1, b2):
        for b in bars:
            ax.annotate(
                f"{b.get_height():.2f}",
                (b.get_x() + b.get_width() / 2, b.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#52514e",
                xytext=(0, 2),
                textcoords="offset points",
            )
    ax.axhline(0.99, color=GRAY, lw=1)
    ax.text(
        len(labels) - 0.5, 0.995, "auto_action floor 0.99", ha="right", fontsize=8, color="#52514e"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{l}\n({SCORED_TEST_COUNTS[l].positives:,} pos)" for l in labels], fontsize=8
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("precision at assumed 50% recall")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    ax.set_title("Precision under fixed recall and false-positive rate assumptions")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "baseline_ceilings.png")
    plt.close(fig)


# ============================================================================ index
def build_index() -> None:
    if HUMAN_RUN is not None:
        build_current_index()
        return
    fin = RUNS["final"]
    body = f"""
<span class="tag">项目记录</span><span class="tag">2026-09-22 起</span>
<h1>review-router 项目记录</h1>
<p class="lede">按步骤记录实验设计、结果和后续判断。每一步一页；结果表从保存的运行产物生成，分析中的假设单独说明。</p>

<h2>项目在做什么</h2>
<p>模型先给评论打分，再按政策分到放行、转人工、自动处置三层。对于需要人工审核的评论，实验比较先到先审、按预测概率排序、按预测严重度排序。数据使用 Jigsaw 英文维基讨论页评论，共六个标签。</p>
<p>要检验的问题是：审核人力相同时，按预测风险排序，能否更早处理按真实标签定义的高危评论？</p>
<div class="box warn"><b>当前结果：</b>在本次仿真条件下，严重度排序优于 FIFO；自动处置精确率仍为 {100 * fin["tiers"][AUTO]["precision"]:.1f}%，未达到 99% 的政策目标。性能差距的原因尚未确定。</div>

<h2>步骤</h2>
<ol class="steps">
<li><a href="{STEPS[0][0]}">{STEPS[0][1]}</a><div class="one">{STEPS[0][2]}。</div><div class="one"><b>结果：</b>一条命令 <code>python scripts/run_pipeline.py --config configs/baseline.yaml</code>，跑出 manifest、逐条预测和报告；在合成数据上验证跑通。</div></li>
<li><a href="{STEPS[1][0]}">{STEPS[1][1]}</a><div class="one">{STEPS[1][2]}。</div><div class="one">每小时到达 180 条时，严重度排序每审核人时处理的危害代理值为 FIFO 的 {fin["simulation"]["harm_per_reviewer_hour"]["router"] / fin["simulation"]["harm_per_reviewer_hour"]["fifo"]:.2f} 倍。每小时 108 条时，已完成高危评论的等待时间 p90 为 {fin["simulation"]["high_risk_wait_p90"]["router"]:.1f} 分钟，FIFO 为 {fin["simulation"]["high_risk_wait_p90"]["fifo"]:.1f} 分钟。这些结论限于本次权重、负载和到达模型。</div><div class="one">R103 已移除，改用子组阈值。诊断显示选择集与测试集表现不同，但尚不能量化模型误判、文本差异和标注差异各自的贡献。</div></li>
<li><span style="font-weight:600;color:#52514e">{PENDING[0]}</span><div class="one">{PENDING[1]}。</div></li>
</ol>

<h2>怎么重新生成这份记录</h2>
<p>已有 <code>runs.json</code> 和 <code>results.json</code> 时，安装分析依赖即可重新生成 HTML 和图表，无需重跑模型：</p>
<pre><code>python -m pip install -e ".[ml,analysis]"
python record/diagnose.py --render-only
python record/render.py</code></pre>
<details><summary>更新实验结果时使用的命令</summary>
<p>下面的三个路径分别指向 R103 生效、移除 R103 但尚未加子组阈值、最终策略的已有运行目录。单次 baseline 只生成一种配置的结果，不能代替三次运行。</p>
<pre><code>python scripts/run_pipeline.py --config configs/baseline.yaml
python record/collect_runs.py --with-r103 reports/&lt;run a&gt; --without-r103 reports/&lt;run b&gt; --final reports/&lt;run c&gt;
python record/diagnose.py reports/&lt;run c&gt;
python record/render.py</code></pre>
</details>
<p class="prov">当前引用的最终运行：<code>{fin["run_id"]}</code>，commit <code>{fin["git_commit"][:12]}</code>{"（工作树有未提交改动）" if fin["git_dirty"] else ""}。数据 sha256：train {fin["data_sha256"]["train"][:12]}…，test {fin["data_sha256"]["test"][:12]}…，test_labels {fin["data_sha256"]["test_labels"][:12]}…。</p>
"""
    page("index.html", "review-router 项目记录", body, None, STEPS[0])


# ============================================================================ step 1: baseline
ORIGINAL_R103 = """- id: R103_identity_term_low_confidence
  description: >
    The documented failure mode on this corpus is false positives on benign
    identity mentions. Suppress auto-action in the borderline band so
    over-enforcement does not concentrate on one group; a human decides.
  action: human_review
  conditions:
    - {signal: identity_term_present, op: "==", threshold: 1}
    - {signal: p_max, op: "<", threshold: 0.95}"""


def build_step1() -> None:
    ceilings_figure()
    table = ceiling_table()
    fin = RUNS["final"]
    counts = fin["split_label_counts"]
    rows = "".join(
        f"<tr><td>{l}</td><td class='n'>{SCORED_TEST_COUNTS[l].positives:,}</td><td class='n'>{SCORED_TEST_COUNTS[l].negatives:,}</td><td class='n'>{table[l][0.01]:.3f}</td><td class='n'>{table[l][0.001]:.3f}</td></tr>"
        for l in table
    )
    weights = "，".join(f"{k} {v:g}" for k, v in RUNS["with_r103"]["policy_snapshot"]["severity_weights"].items())
    body = f"""
<span class="tag">第 1 步</span><span class="tag">commit 93e9093 → 8bb758e</span>
<h1>Baseline：把实验封装成一条可重跑的命令</h1>
<p class="lede">先把数据划分、模型训练、路由和仿真连成可重跑的流程。复现时需要相同的数据、代码、配置和依赖环境；这些信息随运行保存。</p>
<p class="prov">本页记录第一步的实现。R103 当时仍启用，后来已移除；真实数据结果和后续改动见<a href="step2-round1.html">第二步</a>。下文也修正了早期对精确率算例的过度解释。</p>

<h2>起点：项目脚手架</h2>
<p>起点是政策文件加载器、评分行过滤和一组类别比例算例。Jigsaw 的 test.csv 有 153,164 行，其中 63,978 行有评分标签，其余标签为 -1，不参与评估。</p>
<p>下表固定召回率为 50%，再分别假设 FPR 为 1% 和 0.1%。精确率按 <code>TP / (TP + FP)</code> 计算，其中 <code>TP = 阳性数 × 召回率</code>，<code>FP = 阴性数 × FPR</code>。</p>
<table><tr><th>标签</th><th class="n">阳性</th><th class="n">阴性</th><th class="n">精确率，FPR 1%</th><th class="n">精确率，FPR 0.1%</th></tr>{rows}</table>
<figure><img src="figures/baseline_ceilings.png" alt="固定召回率 50% 时，各标签在两种 FPR 假设下的精确率"><figcaption>固定工作点的算例，不是模型实测结果，也不是所有模型的精确率上限。实际能否达到 99%，取决于模型能达到的召回率与 FPR 组合。</figcaption></figure>
<p>这些算例说明，稀有标签的精确率对假阳性尤其敏感。因此，评估需要检查实际阈值下的精确率和覆盖率，不能仅凭 ROC-AUC 判断自动处置是否可靠。</p>

<h2>实验协议</h2>
<div class="flow">
<div class="node"><b>数据划分</b>train.csv 按种子随机切 60 / 20 / 20；官方评分测试行单独保留</div><div class="arrow">→</div>
<div class="node"><b>模型</b>TF-IDF（1-2 gram）+ 六个独立逻辑回归；在校准集上做 Platt 校准</div><div class="arrow">→</div>
<div class="node"><b>阈值</b>在选择集上按精确率门槛逐标签挑：人工 0.90，自动 0.99，至少 30 条；挑不到就关闭</div><div class="arrow">→</div>
<div class="node"><b>规则</b>政策规则优先于模型层级；规则加进队列的条数计入所有指标</div><div class="arrow">→</div>
<div class="node"><b>队列仿真</b>4 名审核员，每条 2 分钟，8 小时；60 / 108 / 180 条每小时；FIFO、概率、严重度三种顺序共用同一批到达</div><div class="arrow">→</div>
<div class="node"><b>报告与门槛</b>manifest、逐条预测、报告；回归门槛读报告</div>
</div>
<table><tr><th>部分</th><th>行数</th><th>职责</th><th>为什么这样分</th></tr>
<tr><td>train</td><td class="n">{counts["train"]["rows"]:,}</td><td>学词表，训练六个分类器</td><td>基础模型只在这部分训练</td></tr>
<tr><td>calib</td><td class="n">{counts["calib"]["rows"]:,}</td><td>Platt 校准</td><td>用独立数据的预测分数和标签拟合概率映射</td></tr>
<tr><td>thresh，选择集</td><td class="n">{counts["thresh"]["rows"]:,}</td><td>选择路由阈值</td><td>避免在模型训练数据上评价候选阈值</td></tr>
<tr><td>scored test</td><td class="n">{counts["test_scored"]["rows"]:,}</td><td>评估冻结后的流程</td><td>不参与本次模型、校准器和路由阈值拟合</td></tr></table>
<p>先声明政策目标，再选择满足目标、覆盖最多评论的阈值。自动处置直接产生执行结果，因此目标设为 0.99，人工审核设为 0.90。这是政策选择，并非由实测错误成本推导出的最优值。至少 30 条只是小样本过滤条件，不能证明真实精确率达到 99%。</p>
<p>这几部分数据互不重叠，但官方测试行已用于多轮结果分析。它现在是迭代诊断基准；后续改进需要新的独立评估数据。</p>
<p><b>严重度权重是政策输入，不是测量：</b>{weights}。仿真里的"危害"是命中标签的最高权重，它是实验里的危害代理，不能称为真实业务中减少的伤害。</p>

<h2>政策文件里的三条规则（baseline 时的写法）</h2>
<ul><li>R101：当 <code>p_threat ≥ 0.30</code> 时转人工，可覆盖模型原先的任一层级。这是针对高危预测的政策选择，不能用上面的算例证明它是唯一可行方案。</li>
<li>R102：当模型同时给出 <code>p_severe_toxic ≥ 0.50</code> 和 <code>p_toxic &lt; 0.50</code> 时转人工。它标记违反训练标签包含关系的预测，不能单独证明输入来自分布外。</li>
<li>R103：含身份词且 <code>p_max &lt; 0.95</code> 时转人工。原意是避免自动误处置，实际作用在第二步中重新检查。</li></ul>
<details><summary>R103 的历史配置，现已退役</summary><pre><code>{escape(ORIGINAL_R103)}</code></pre></details>

<h2>门槛（回归检查）</h2>
<p>门槛写在政策文件里。未设置门槛或未指定评估报告时，相关测试跳过；显式指定的报告不存在时，测试失败。代码 CI 通过不等于真实数据上的评估要求全部通过。</p>
<p>第二步根据实测值设置回归检查范围，用来发现退步。自动处置 0.99 等事先声明的政策要求单独保留。</p>

<h2>产物</h2>
<p>一次运行写出：<code>manifest.json</code>（数据哈希、各组标签计数、commit、依赖版本、种子）、<code>splits.csv</code>、<code>thresholds.json</code>、<code>model.pkl</code>、<code>predictions.csv</code>（逐条）、<code>report.json</code>（门槛读它）、<code>report.md</code>。</p>

<h2>先在合成数据上跑通</h2>
<p>真实数据当时不在机器上，先用一个和 Jigsaw 文件布局相同的合成语料（<code>scripts/make_synthetic_corpus.py</code>，含 -1 未评分行以检验过滤）把全链路跑通。合成数据的数字只用来检查流水线，不作为结果。这一步在合成数据上就暴露了 R103 的问题：它把大量良性 identity 提及推进了人工队列。</p>

<h2>这一步交付了什么</h2>
<p>第一步交付了可重跑的实验命令、逐条预测记录，以及在相同审核资源下比较三种排序策略的流程。合成语料只验证代码路径，效果判断留给真实数据实验。</p>
"""
    page(STEPS[0][0], STEPS[0][1], body, None, STEPS[1])


# ============================================================================ step 2: round 1 run + analysis
def tier_table(run: dict[str, Any], sel: bool = True) -> str:
    rows = ""
    for tier in (AUTO, HUMAN, ALLOW):
        e = run["tiers"][tier]
        s = run["threshold_selection_tiers"][tier]["precision"] if sel else None
        rows += f"<tr><td>{tier}</td><td class='n'>{e['n_predicted_positive']}</td><td class='n'>{f(e['coverage'])}</td><td class='n'>{f(e['precision'])}</td><td>{ci(e['precision_ci95'])}</td><td class='n'>{f(s, 4)}</td></tr>"
    return f"<table><tr><th>层级</th><th class='n'>n</th><th class='n'>覆盖率</th><th class='n'>测试集精确率</th><th>95% 区间</th><th class='n'>选择集精确率</th></tr>{rows}</table>"


def build_step2() -> None:
    a, b, fin = RUNS["with_r103"], RUNS["without_r103"], RUNS["final"]
    s1, s2, s3, s4, s5, s6 = R["step1"], R["step2"], R["step3"], R["step4"], R["step5"], R["step6"]
    P = R["provenance"]
    floor = P["floor"]
    lex = s4["lexicon"]
    hits = lex["test_false_positives"]["n_with_lexicon_hit"]
    n_auto = s4["n_toxic_auto_test"]
    tp = n_auto - s4["n_false_positive"]
    counterfactual = (tp + hits) / n_auto  # hypothetical relabeling, not causal attribution
    gap = floor - s1["per_label_test"]["toxic"]["precision"]
    gap_fraction = (counterfactual - s1["per_label_test"]["toxic"]["precision"]) / gap
    pth = s4["p_toxic_given_lexicon_hit"]
    oovg = s4["oov_by_group"]
    counts = fin["split_label_counts"]

    # ---- per label
    pl = ""
    for l in LABELS:
        e = fin["per_label"][l]
        sp = fin["threshold_selection_per_label"][l]
        h, au = e["at_human_review"], e["at_auto_action"]
        pl += f"<tr><td>{l}</td><td class='n'>{e['positives']}</td><td class='n'>{f(sp['average_precision'])}</td><td class='n'>{f(e['average_precision'])}</td><td class='n'>{f(e['roc_auc'])}</td><td class='n'>{f(h['threshold'], 4)}</td><td class='n'>{f(h['precision'])}</td><td class='n'>{f(h['recall'])}</td><td class='n'>{f(au['threshold'], 4)}</td><td class='n'>{f(au['precision'])}</td><td class='n'>{f(au['recall'])}</td></tr>"

    # ---- R103 comparison
    def harm_ratio(run: dict[str, Any]) -> float:
        h = run["simulation"]["harm_per_reviewer_hour"]
        return h["router"] / h["fifo"]

    def wait_pair(run: dict[str, Any]) -> str:
        w = run["simulation"]["high_risk_wait_p90"]
        return f"{w['router']:.1f} / {w['fifo']:.1f}"

    r103 = "".join(
        f"<tr><td>{name}</td><td><code>{run['run_id'].split('-')[0]}</code></td><td class='n'>{run['tiers'][HUMAN]['n_predicted_positive']}</td><td class='n'>{f(run['tiers'][HUMAN]['precision'])}</td><td class='n'>{run['tiers'][AUTO]['n_predicted_positive']}</td><td class='n'>{f(run['tiers'][AUTO]['precision'])}</td><td class='n'>{harm_ratio(run):.2f}</td><td class='n'>{wait_pair(run)}</td></tr>"
        for name, run in (("R103 生效", a), ("R103 去掉", b), ("R103 去掉 + 子组阈值（最终）", fin))
    )
    a_rules = a["rules"]
    a_id = a["identity_summary"]
    f_id = fin["identity_summary"]
    sub = fin["subgroup_thresholds"]
    sub_thr = sub["thresholds"].get(AUTO, {}).get("identity_term_present", {})
    slice_parts = "；".join(
        f"{l} {f(sub_thr.get(l), 4)}"
        if sub_thr.get(l) is not None
        else f"{l} 未找到合格阈值，在该子组内不能触发自动处置"
        for l in AUTO_LABELS
    )
    auto_pop = f_id["test_population_tier_auto_action"]
    auto_model = f_id["test_model_tier_auto_action"]
    fair_max = POLICY["gates"]["fairness"]["identity_false_discovery_rate_ratio_max"]
    identity_test = fin["identity_false_positives"]["test"]
    identity_terms = identity_test["identity_terms"]
    human_fdr = identity_test["final_tier"][HUMAN]
    human_fpr = identity_test["final_tier"]["clean_negative_false_positives"][HUMAN]

    # ---- simulation table
    sim = fin["simulation"]
    sim_rows = "".join(
        f"<tr><td class='n'>{e['load_per_hour']:g}</td><td>{e['strategy']}</td><td class='n'>{f(e['n_handled'], 0)}</td><td class='n'>{f(e['high_risk_handled'], 1)}</td><td class='n'>{f(e['high_risk_unhandled'], 1)}</td><td class='n'>{f(e['harm_per_reviewer_hour'], 1)}</td><td class='n'>{f(e['high_risk_wait_p50'], 1)}</td><td class='n'>{f(e['high_risk_wait_p90'], 1)}</td><td class='n'>{f(e['backlog_end'], 0)}</td></tr>"
        for e in sim["summary"].values()
    )
    hi, mid = sim["thesis"]["load_per_hour"], sim["primary"]["load_per_hour"]
    sp_hi = sim["paired"][f"severity_vs_prob@{hi:g}"]
    sp_mid = sim["paired"][f"severity_vs_prob@{mid:g}"]

    # ---- gates
    g = POLICY["gates"]
    gate_rows = ""
    for l, spec in g["per_label_average_precision"].items():
        v = fin["per_label"][l]["average_precision"]
        gate_rows += f"<tr><td>AP {l}</td><td class='n'>{f(v)}</td><td>≥ {spec['floor']}</td><td>{'绿' if v >= spec['floor'] else '<b>红</b>'}</td></tr>"
    rt = g["routing"]
    checks = [
        (
            "auto_action 精确率",
            fin["tiers"][AUTO]["precision"],
            f"≥ {rt['auto_action_precision_floor']}",
            fin["tiers"][AUTO]["precision"] >= rt["auto_action_precision_floor"],
        ),
        (
            "human_review 精确率",
            fin["tiers"][HUMAN]["precision"],
            f"≥ {rt['human_review_precision_floor']}",
            fin["tiers"][HUMAN]["precision"] >= rt["human_review_precision_floor"],
        ),
        (
            f"危害/人时 严重度÷FIFO @ {hi:g}/h",
            harm_ratio(fin),
            f"≥ {rt['harm_per_reviewer_hour_vs_fifo_min']}",
            harm_ratio(fin) >= rt["harm_per_reviewer_hour_vs_fifo_min"],
        ),
        (
            f"高危等待 p90 严重度÷FIFO @ {mid:g}/h",
            sim["high_risk_wait_p90"]["router"] / sim["high_risk_wait_p90"]["fifo"],
            f"≤ {rt['high_risk_wait_p90_vs_fifo_max']}",
            sim["high_risk_wait_p90"]["router"] / sim["high_risk_wait_p90"]["fifo"]
            <= rt["high_risk_wait_p90_vs_fifo_max"],
        ),
        (
            f"队列深度 p95 @ {mid:g}/h",
            sim["queue_depth_p95"],
            f"≤ {rt['queue_depth_p95_max']}",
            sim["queue_depth_p95"] <= rt["queue_depth_p95_max"],
        ),
        (
            f"审核员利用率 @ {mid:g}/h",
            sim["reviewer_utilization"],
            f"≥ {rt['reviewer_utilization_min']}",
            sim["reviewer_utilization"] >= rt["reviewer_utilization_min"],
        ),
        (
            "层级一致性违反率",
            fin["consistency"]["hierarchy_violation_rate"],
            f"≤ {g['consistency']['hierarchy_violation_rate_max']}",
            fin["consistency"]["hierarchy_violation_rate"]
            <= g["consistency"]["hierarchy_violation_rate_max"],
        ),
    ]
    for name, v, req, ok in checks:
        gate_rows += f"<tr><td>{name}</td><td class='n'>{f(v)}</td><td>{req}</td><td>{'绿' if ok else '<b>红</b>'}</td></tr>"
    for basis, tier, name in (
        ("final_tier", "predicted_positive", "身份词 FDR 比，最终路由汇总"),
        ("model_tier", AUTO, "身份词 FDR 比，子组阈值之后、规则之前的自动处置"),
    ):
        comparison = identity_test[basis][tier]
        status = fairness_gate_status(comparison, fair_max, g["min_predicted_positives_for_precision"])
        status = {"green": "绿", "**red**": "<b>红</b>", "inconclusive": "不确定，跳过", "unavailable": "不可用"}.get(status, status)
        value = f(comparison["false_discovery_rate_ratio"])
        interval = ci(comparison["false_discovery_rate_ratio_ci95"], 2)
        gate_rows += f"<tr><td>{name}</td><td class='n'>{value}，95% 区间 {interval}</td><td>区间上界 ≤ {fair_max}</td><td>{status}</td></tr>"

    # ---- analysis tables (from the diagnosis)
    cvt = s2["cv"]
    a1 = ""
    for l in AUTO_LABELS:
        x, y = s1["per_label_in_sample"][l], s1["per_label_test"][l]
        a1 += f"<tr><td>{l}</td><td class='n'>{f(x['threshold'], 4)}</td><td class='n'>{x['n']}</td><td class='n'>{f(x['precision'])}</td><td class='n'>{y['n']}</td><td class='n'>{f(y['precision'])}</td><td>{ci(y['ci95'])}</td></tr>"
    x, y = s1["tier_in_sample"], s1["tier_test"]
    a1 += f"<tr><th>自动处置层</th><td></td><td class='n'>{x['n']}</td><td class='n'>{f(x['precision'])}</td><td class='n'>{y['n']}</td><td class='n'>{f(y['precision'])}</td><td>{ci(y['ci95'])}</td></tr>"
    a2 = ""
    for l in AUTO_LABELS:
        x = s1["per_label_in_sample"][l]
        h = cvt["per_label"][l]
        t = s1["per_label_test"][l]
        vals = [v for v in h["threshold_fold_values"] if v is not None]
        spread = f"{min(vals):.4f} 到 {max(vals):.4f}" if vals else "无"
        a2 += f"<tr><td>{l}</td><td class='n'>{f(x['precision'])}</td><td class='n'>{f(h['precision_mean'])}</td><td class='n'>{h['n_mean']:.0f}</td><td class='n'>{f(t['precision'])}</td><td>{spread}</td></tr>"
    a2 += f"<tr><th>自动处置层</th><td class='n'>{f(s1['tier_in_sample']['precision'])}</td><td class='n'>{f(cvt['tier']['precision_mean'])}</td><td class='n'>{cvt['tier']['n_mean']:.0f}</td><td class='n'>{f(s1['tier_test']['precision'])}</td><td>每个种子：{', '.join(f(v) for v in cvt['tier']['precision_per_seed'])}</td></tr>"
    rel_rows = ""
    for i, bn in enumerate(s3["reliability"]["toxic"]["thresh"]):
        bt = s3["reliability"]["toxic"]["test"][i]
        rel_rows += f"<tr><td>{bn['bin']}</td><td class='n'>{bn['n']}</td><td class='n'>{f(bn['observed'])}</td><td class='n'>{bt['n']}</td><td class='n'>{f(bt['observed'])}</td><td>{ci(bt['ci95'])}</td></tr>"
    tx = s3["text"]
    text_rows = "".join(
        f"<tr><td>{name}</td><td class='n'>{tx[k]['sample']:,}</td><td class='n'>{tx[k]['chars_median']:.0f}</td><td class='n'>{tx[k]['chars_p90']:.0f}</td><td class='n'>{f(tx[k]['oov_token_rate_mean'])}</td><td class='n'>{f(tx[k]['oov_token_rate_p90'])}</td><td class='n'>{f(tx[k]['share_no_vocab_token'])}</td></tr>"
        for k, name in (("train", "训练集"), ("thresh", "选择集"), ("test", "测试集"))
    )
    best = s3["best_toxic_precision_on_test_n30"]
    lex_rows = "".join(
        f"<tr><td>{name}</td><td class='n'>{lex[k]['n']}</td><td class='n'>{lex[k]['n_with_lexicon_hit']}</td><td class='n'>{f(lex[k]['share'], 2)}</td><td>{ci(lex[k]['ci95'], 2)}</td></tr>"
        for k, name in (
            ("test_false_positives", "测试集：假阳性（toxic=0，p ≥ 阈值）"),
            ("test_true_positives", "测试集：真阳性（toxic=1，p ≥ 阈值）"),
            ("test_all_toxic_zero", "测试集：全部 toxic=0 行"),
            ("selection_false_positives", "选择集：假阳性"),
            ("selection_true_positives", "选择集：真阳性"),
            ("selection_all_toxic_zero", "选择集：全部 toxic=0 行"),
        )
    )
    ex = "".join(
        f"<li><code>p={e['p_toxic']:.4f}</code> {'含词表词' if e['lexicon_hit'] else '不含词表词'}；{'其他标签：' + ', '.join(e['other_labels_true']) if e['other_labels_true'] else '六个标签全为 0'}<blockquote>{escape(unescape(e['text']))}</blockquote></li>"
        for e in s4["examples"]
    )
    rule_rows = ""
    rule_labels = {
        "point_min30": "点估计 ≥ 0.99，至少 30 条，当前规则",
        "lb_min30": "Wilson 下界 ≥ 0.99，至少 30 条",
        "point_min100": "点估计 ≥ 0.99，至少 100 条",
        "point_min300": "点估计 ≥ 0.99，至少 300 条",
        "lb_min100": "Wilson 下界 ≥ 0.99，至少 100 条",
    }
    for k, v in s5["rules"].items():
        thr = (
            ", ".join(f"{l} {f(t, 4)}" for l, t in v["thresholds"].items() if t is not None) or "无"
        )
        rule_rows += f"<tr><td>{rule_labels.get(k, escape(v['label']))}</td><td class='n'>{f(v['in_sample']['precision'])}</td><td class='n'>{f(v['cv']['precision_mean'])}</td><td class='n'>{v['cv']['n_mean']:.0f}</td><td class='n'>{f(v['test_readout']['precision'])}</td><td class='n'>{v['test_readout']['n']}</td><td>{thr}</td></tr>"
    lb, pt = s5["rules"]["lb_min30"], s5["rules"]["point_min30"]

    body = f"""
<span class="tag">第 2 步，历史实验</span><span class="tag red">原自动处置要求未达标</span><span class="tag">2026-09-22</span>
<h1>首轮结果与诊断</h1>
{'<p class="box">本页保留原自动处置实验及其 99% 目标。当前方案已改为优先人工审核，见<a href="step3-human-review.html">第三步</a>。历史结果不按新标准重判。</p>' if HUMAN_RUN is not None else ''}
<p class="lede">本轮完成了真实数据评估，移除 R103，并检查自动处置精确率的缺口。先列最终路由结果，再比较策略改动，最后讨论诊断能支持哪些判断。</p>
<div class="box warn"><b>先看结论：</b>自动处置精确率为 {100 * fin["tiers"][AUTO]["precision"]:.1f}%，未达到 99%；严重度排序在本次仿真中优于 FIFO。文本与标签统计能提示问题，但还不足以确认性能差距的原因。</div>
<nav class="toc" aria-label="本页目录">
<a href="#data">1. 数据</a><a href="#results">2. R1 结果</a><a href="#r103">3. R103 的处理</a><a href="#gates">4. 门槛</a>
<a href="#gap">5. 未达标的诊断</a><a href="#gap1">5.1 固定阈值复核</a><a href="#gap2">5.2 选择集内部验证</a><a href="#gap3">5.3 两份数据的差异</a><a href="#gap4">5.4 假阳性样本检查</a><a href="#gap5">5.5 阈值规则对比</a><a href="#gap6">5.6 证据与限制</a><a href="#next">6. 下一步</a>
</nav>

<h2 id="data">1. 数据</h2>
<p>数据来自 Hugging Face 镜像 <code>thesofakillers/jigsaw-toxic-comment-classification-challenge</code>，保存在 <code>data/jigsaw/</code>，不提交语料文件。train 共 {counts["train"]["rows"] + counts["calib"]["rows"] + counts["thresh"]["rows"]:,} 行，评分测试行共 {counts["test_scored"]["rows"]:,} 行；行数和标签计数与项目记录的基准计数一致。manifest 保存三个文件的 SHA-256，便于检查是否使用同一份输入。</p>
<p class="prov">最终运行 <code>{fin["run_id"]}</code>，commit <code>{fin["git_commit"][:12]}</code>，工作区干净。下文中的政策要求来自该运行保存的快照。测试行未参与本次模型拟合，但已用于多轮诊断，不再是后续改进的全新盲测数据。</p>

<h2 id="results">2. R1 结果</h2>
<h3>最终路由结果</h3>
{tier_table(fin)}
<p>自动处置的测试集精确率为 {f(fin["tiers"][AUTO]["precision"])}，选择集为 {f(fin["threshold_selection_tiers"][AUTO]["precision"], 4)}；人工审核层分别为 {f(fin["tiers"][HUMAN]["precision"])} 和 {f(fin["threshold_selection_tiers"][HUMAN]["precision"], 4)}。第 5 节检查这类跨数据集差距。人工审核的 0.90 是逐标签选阈值时的目标，不是合并标签、移除自动处置、应用规则后的队列保证。</p>
<details><summary>逐标签结果与指标说明</summary>
<p>AP 是平均精确率，描述整条精确率与召回率曲线；P 为精确率，R 为召回率。下表 P/R 均在测试集测量，使用总体阈值，尚未加入子组限制和规则。n/a 表示没有合格阈值。</p>
<table><tr><th>标签</th><th class="n">阳性</th><th class="n">AP 选择集</th><th class="n">AP 测试集</th><th class="n">ROC-AUC</th><th class="n">人工阈值</th><th class="n">P@人工</th><th class="n">R@人工</th><th class="n">自动阈值</th><th class="n">P@自动</th><th class="n">R@自动</th></tr>{pl}</table>
<p>severe_toxic、threat、identity_hate 在选择集上没有同时满足精确率目标与至少 30 条条件的阈值，因此不能凭自身阈值触发路由。含这些标签的评论仍可能通过其他重叠标签或政策规则进入人工队列。</p>
</details>
<h3>队列仿真</h3>
<p>{sim["assumptions"]["reviewers"]} 名审核员，每条 {sim["assumptions"]["handle_minutes"]:g} 分钟，持续 {sim["assumptions"]["horizon_hours"]:g} 小时，容量 {sim["assumptions"]["capacity_per_hour"]:g} 条/小时。从 {sim["assumptions"]["queued_jobs"]} 条人工审核候选评论中有放回抽样，按泊松过程到达。候选池含 {sim["assumptions"]["queued_high_risk"]} 条高危评论，高危按真实标签的最大严重度权重 ≥ 5 定义。</p>
<p>表格为 {len(sim["assumptions"]["seeds"])} 个种子的均值 ± 标准差，同一种子下三种排序共用到达时间与处理时长。等待时间只统计仿真结束前已完成的评论，单位为分钟；p90 表示其中 90% 的等待不超过该值。高危剩余包含正在处理的任务，结束积压只计尚未开始的任务。</p>
<table><tr><th class="n">负载/h</th><th>顺序</th><th class="n">处理</th><th class="n">高危处理</th><th class="n">高危剩余</th><th class="n">危害/人时</th><th class="n">高危等待 p50</th><th class="n">高危等待 p90</th><th class="n">结束积压</th></tr>{sim_rows}</table>
<p>在 {mid:g} 条/小时时，严重度排序的高危等待 p90 为 {sim["high_risk_wait_p90"]["router"]:.1f} 分钟，FIFO 为 {sim["high_risk_wait_p90"]["fifo"]:.1f} 分钟。此时队列结束仍有少量未完成任务，不能称为全部清空。</p>
<p>在超容量的 {hi:g} 条/小时时，各策略完成数量相同，但处理对象不同。严重度排序每审核人时处理的危害代理值为 FIFO 的 {harm_ratio(fin):.2f} 倍，剩余高危任务为 {sim["summary"][f"severity@{hi:g}"]["high_risk_unhandled"]["mean"]:.1f} 条，FIFO 为 {sim["summary"][f"fifo@{hi:g}"]["high_risk_unhandled"]["mean"]:.1f} 条。</p>
<p>与概率排序相比，严重度排序在 {hi:g} 条/小时的 {sp_hi["high_risk_handled"]["n_seeds_first_better"]}/{sp_hi["high_risk_handled"]["n_seeds"]} 个种子中处理了更多高危任务，平均多 {sp_hi["high_risk_handled"]["mean_diff"]:.1f} 条。在 {mid:g} 条/小时的 {sp_mid["high_risk_wait_p90"]["n_seeds_first_better"]}/{sp_mid["high_risk_wait_p90"]["n_seeds"]} 个种子中，其高危等待 p90 更短。</p>
<p class="prov">这些结果支持本次仿真条件下的排序收益。排序与评价使用同一套严重度权重，结论依赖这套权重和到达假设，尚不能推断真实业务中的伤害减少。</p>

<h2 id="r103">3. R103 的处理</h2>
<p>R103 会把含身份词、最大预测概率低于 0.95 的评论转人工。下表比较启用规则、移除规则、再加入子组阈值三种配置。前两次运行来自有未提交改动的工作区；最终运行对应干净提交。各次运行信息保存在 <a href="runs.json">runs.json</a>。</p>
<table><tr><th>状态</th><th>运行</th><th class="n">人工 n</th><th class="n">人工精确率</th><th class="n">自动 n</th><th class="n">自动精确率</th><th class="n">危害/人时 严重度÷FIFO @180</th><th class="n">高危等待 p90 严重度 / FIFO @108</th></tr>{r103}</table>
<h3>移除原因</h3>
<p>本次总体自动处置阈值为 toxic {f(a["thresholds"]["auto_action"]["toxic"], 4)}、obscene {f(a["thresholds"]["auto_action"]["obscene"], 4)}、insult {f(a["thresholds"]["auto_action"]["insult"], 4)}，均高于 0.95。因此，R103 在这组阈值下无法拦截自动处置，却会把原来应放行的评论送进人工队列。</p>
<p>选择集上有 {a_id["selection_identity_moved_to_positive_by_rules"]:,} 条含身份词评论因此进入正向路由，其中 {f_id["selection_identity_allowed"] - f_id["selection_identity_allowed_truly_positive"]:,} 条没有任何阳性标签。后一个数字根据最终运行中恢复放行的 {f_id["selection_identity_allowed"]:,} 条评论及其中 {f_id["selection_identity_allowed_truly_positive"]} 条阳性计算。</p>
<p>启用 R103 的测试运行中，R103 匹配 {a_rules["matched_per_rule"].get("R103_identity_term_low_confidence", 0):,} 行。该次运行所有规则合计把 {a_rules["n_added_to_queue_from_allow"]:,} 条放行评论转人工，其中 {a_rules["n_added_to_queue_from_allow_true_positive"]} 条有阳性标签；人工队列精确率为 {f(a["tiers"][HUMAN]["precision"])}。</p>
<p class="prov">不同策略改变了人工候选池，仿真又对各候选池设定相同的队列到达率。因此，表中 {harm_ratio(a):.2f} 与 {harm_ratio(fin):.2f} 的收益比只能分别解释各自池内的排序效果，不能直接表示端到端政策收益。</p>
<h3>替换为子组阈值</h3>
<p>一种选择是把所有含身份词的自动处置一律改为人工。当前采用更细的限制：在含身份词的选择集子组内，逐标签寻找满足相同精确率目标与样本数要求的阈值，且不得低于总体阈值。找不到时，该标签在子组内不能触发自动处置。这是样本上的筛选条件，不是统计认证。</p>
<p>当前阈值为：{slice_parts}。这项限制从选择集的 {sub["threshold_selection"][AUTO]["n_population_tier"]} 条总体自动处置中移除了 {sub["threshold_selection"][AUTO]["n_removed_by_subgroup_threshold"]} 条；从测试集的 {sub["test"][AUTO]["n_population_tier"]} 条中移除了 {sub["test"][AUTO]["n_removed_by_subgroup_threshold"]} 条，其中 {sub["test"][AUTO]["n_removed_truly_positive_any_label"]} 条有至少一个阳性标签。</p>
<h3>FDR 与 FPR 分别说明什么</h3>
<p>身份词来自运行保存的 {len(identity_terms)} 个整词匹配词条。这是文本分组的代理，不是作者身份标注。FDR 的分母是被标记的评论；FPR 的分母是所有实际正常的评论。</p>
<table><tr><th>人工审核指标</th><th>含身份词</th><th>不含身份词</th><th>问题</th></tr>
<tr><td>FDR，错误发现率</td><td class="n">{100 * human_fdr["with_identity_term"]["false_discovery_rate"]:.1f}%</td><td class="n">{100 * human_fdr["without_identity_term"]["false_discovery_rate"]:.1f}%</td><td>进入队列的评论中，多少没有阳性标签？</td></tr>
<tr><td>FPR，假阳性率</td><td class="n">{100 * human_fpr["with_identity_term"]["false_positive_rate"]:.1f}%</td><td class="n">{100 * human_fpr["without_identity_term"]["false_positive_rate"]:.1f}%</td><td>所有没有阳性标签的评论中，多少进入队列？</td></tr></table>
<p>两组人工队列的 FDR 接近，但正常评论进入人工队列的比例仍相差 {human_fpr["false_positive_rate_ratio"]:.2f} 倍。因此，FDR 检查通过不能解释为两组误报风险相同。</p>
<p>自动处置的 FDR 比在总体阈值下为 {f(auto_pop["ratio"], 2)}，95% 区间 {ci(auto_pop["ci95"], 2)}；子组阈值后为 {f(auto_model["ratio"], 2)}，区间 {ci(auto_model["ci95"], 2)}。自动处置按触发标签判断正误，FPR 则只统计所有标签均为 0 的正常评论，两者的错误数量也可能不同。</p>

<h2 id="gates">4. 门槛</h2>
<p>回归门槛以首轮实测值为参照，用来发现后续退步，不能作为独立的效果证明。政策快照保存了各门槛的来源。自动处置 0.99、利用率 0.60、层级违例上限和 FDR 比上限等预设要求不随本次结果降低。</p>
<p>FDR 比的完整 95% 区间都不高于 {fair_max} 才通过；整个区间高于它则失败，跨越它则标为不确定并跳过。样本不足也跳过。通过只表示满足这项诊断要求。</p>
<table><tr><th>门槛</th><th class="n">实测</th><th>要求</th><th>状态</th></tr>{gate_rows}</table>
<p>本次已配置的检查中，自动处置精确率未通过。99% 要求保持不变；以下分析用于提出下一轮需要验证的改动。</p>

<h2 id="gap">5. 未达标的诊断</h2>
<p>这一节先检查阈值选择的稳定性，再比较选择集与测试集的预测、文本和标签统计。分析只重新选择阈值，模型与校准器保持冻结；它不能把差距精确分解为某几个原因。</p>
<div class="box"><b>两种口径：</b>最终路由包含子组阈值和规则，测试精确率为 {f(fin["tiers"][AUTO]["precision"])}。以下诊断使用总体阈值，暂不加入这两层政策处理，测试精确率为 {f(s1["tier_test"]["precision"])}。两者对应不同的评论集合。</div>

<h3 id="gap1">5.1 固定阈值复核</h3>
<p>将选择集上得到的总体阈值原样用于测试集，先确认差距出现在哪里。</p>
<table><tr><th>标签</th><th class="n">阈值</th><th class="n">选择集 n</th><th class="n">选择集精确率</th><th class="n">测试集 n</th><th class="n">测试集精确率</th><th>测试集 95% 区间</th></tr>{a1}</table>
<figure><img src="figures/gap1_precision_by_split.png" alt="固定总体阈值在选择集与测试集上的精确率"><figcaption>总体自动处置的 {y["n"]} 条测试评论中，{s1["share_of_auto_rows_by_label_test"]["toxic"]} 条触发 toxic，因此该层精确率主要由 toxic 决定。按当前测试标签计算的区间上界低于 {floor}。</figcaption></figure>

<h3 id="gap2">5.2 选择集内部验证</h3>
<p>将选择集随机分为 {s2["folds"]} 折，每次在 {s2["folds"] - 1} 折选择阈值，在剩余一折测量精确率，再合并留出预测。共使用 {len(s2["seeds"])} 个划分种子；这些种子重复使用同一批数据，不是三份独立样本。</p>
<table><tr><th>标签</th><th class="n">全选择集拟合并评估</th><th class="n">留出折</th><th class="n">每个种子留出预测数均值</th><th class="n">官方测试集</th><th>各折阈值范围</th></tr>{a2}</table>
<figure><img src="figures/gap2_optimism_vs_shift.png" alt="总体阈值的样本内、交叉验证和测试精确率对照"><figcaption>总体自动处置的样本内与留出折精确率相差 {f(s6["optimism"])}，留出折与官方测试行相差 {f(s6["shift"])}。这是描述性比较，不是原因分解。</figcaption></figure>
<p>留出验证能检查选择集内部的稳定性，但各折使用更少的数据、不同的阈值。样本内与留出折的差不能完全归为阈值选择的乐观偏差，剩余差距也不能直接归为分布漂移。</p>
<p>insult 的留出预测平均只有 {cvt["per_label"]["insult"]["n_mean"]:.0f} 条，每折为 {min(cvt["per_label"]["insult"]["held_out_fold_counts"])} 到 {max(cvt["per_label"]["insult"]["held_out_fold_counts"])} 条。其精确率估计对少量错误很敏感；至少 30 条的条件不能替代不确定性评估。</p>

<h3 id="gap3">5.3 两份数据的差异</h3>
<p>以下检查使用冻结的模型分数。它们能描述差异，还不能确定差异来自文本、标注还是模型。</p>
<h4>a. 排序质量</h4>
<figure><img src="figures/gap3a_ap_shift.png" alt="六个标签在选择集和测试集上的平均精确率 AP"><figcaption>toxic AP 从 {f(s3["ap"]["toxic"]["selection"])} 降到 {f(s3["ap"]["toxic"]["test"])}，obscene 从 {f(s3["ap"]["obscene"]["selection"])} 降到 {f(s3["ap"]["obscene"]["test"])}，insult 从 {f(s3["ap"]["insult"]["selection"])} 降到 {f(s3["ap"]["insult"]["test"])}。toxic 阳性率分别为 {f(s3["prevalence"]["toxic"]["thresh"], 4)} 和 {f(s3["prevalence"]["toxic"]["test"], 4)}，相对接近。</figcaption></figure>
<p>AP 不依赖某个单独阈值。这说明需要同时检查模型排序质量，不能只检查阈值选得是否合适。</p>
<h4>b. 概率校准</h4>
<figure><img src="figures/gap3b_reliability.png" alt="预测概率与实际阳性比例在两份数据上的对应关系"><figcaption>横轴为预测概率，纵轴为该概率档内的实际阳性比例；越接近对角线，校准越好。只画至少 20 行的档位。</figcaption></figure>
<details><summary>toxic 各概率档的样本量与实际阳性比例</summary>
<table><tr><th>预测概率档</th><th class="n">选择集 n</th><th class="n">选择集阳性比例</th><th class="n">测试集 n</th><th class="n">测试集阳性比例</th><th>测试集 95% 区间</th></tr>{rel_rows}</table>
</details>
<p>在最高概率档，选择集实际阳性比例为 {f(s3["reliability"]["toxic"]["thresh"][-1]["observed"])}，测试集为 {f(s3["reliability"]["toxic"]["test"][-1]["observed"])}。同一概率映射在测试行上表现较差，但这个差异本身不能识别原因。</p>
<h4>c. 当前 toxic 分数的阈值范围</h4>
<figure><img src="figures/gap3c_precision_curve_toxic.png" alt="当前 toxic 模型的阈值与精确率关系"><figcaption>曲线展示 0.90 到 0.9999 的阈值网格。另对满足至少 30 条预测阳性的 {best["n_distinct_thresholds_checked"]:,} 个不同分数阈值逐个扫描，测试精确率最高为 {best["precision"]:.4f}，对应 {best["n"]} 条评论。</figcaption></figure>
<p>这次事后扫描说明：当前冻结的 toxic 模型，仅调整单一分数阈值，未在这些测试行上达到 {floor}。它不限制其他模型、特征或路由方法的可达性能。扫描使用了测试标签，因此不能把这个最大值当作新方法的独立验证结果。</p>
<h4>d. 文本与词表覆盖</h4>
<table><tr><th>数据</th><th class="n">抽样行数</th><th class="n">字符数中位</th><th class="n">字符数 p90</th><th class="n">每条评论词表外比例均值</th><th class="n">词表外比例 p90</th><th class="n">无词表内 token 的评论比例</th></tr>{text_rows}</table>
<figure><img src="figures/gap3d_text_shift.png" alt="评论长度和近似词表外比例的分布"><figcaption>选择集每条评论的词表外比例均值为 {f(s3["text"]["thresh"]["oov_token_rate_mean"])}，测试集为 {f(s3["text"]["test"]["oov_token_rate_mean"])}。词表从训练集拟合，因此选择集比训练集更适合作为留出对照。</figcaption></figure>
<p class="prov">这里的词表外比例是近似诊断，分词时未完全复用模型的重音符号归一化。它描述文本覆盖差异，不能直接量化这种差异造成了多少误判。</p>

<h3 id="gap4">5.4 假阳性样本检查</h3>
<p>取 toxic 预测概率 ≥ {f(s4["toxic_threshold"], 4)}、但 toxic 标签为 0 的 {s4["n_false_positive"]} 条测试评论。其中 {s4["n_fp_with_other_label_true"]} 条有其他阳性标签，{s4["n_fp_all_labels_zero"]} 条的六个标签均为 0。以下检查以数据集标签为评估依据，并未人工重标。</p>
<p>使用 {len(lex["lexicon"])} 个词的辱骂词表做整词匹配，比较假阳性、真阳性和全部阴性评论。词表无法区分辱骂、引用或讨论词语本身，部分词也有多种含义，所以匹配不等于评论应被标为 toxic。</p>
<table><tr><th>集合</th><th class="n">n</th><th class="n">匹配词表</th><th class="n">比例</th><th>95% 区间</th></tr>{lex_rows}</table>
<figure><img src="figures/gap4_fp_composition.png" alt="假阳性的标签构成与各评论集合的词表匹配比例"><figcaption>词表匹配描述用词，不判断标签是否正确。高分评论由词级模型筛选，因此其中某些词较常见也可能与筛选过程有关。</figcaption></figure>
<details><summary>查看 {len(s4["examples"])} 条抽样原文，含辱骂内容</summary>
<p>按是否匹配词表分层抽样，种子为 {s4["example_seed"]}。其中 {s4["n_examples_with_lexicon_hit"]} 条匹配、{len(s4["examples"]) - s4["n_examples_with_lexicon_hit"]} 条不匹配；每条只显示前 240 个字符，片段可能缺少上下文。</p>
<ol class="examples">{ex}</ol>
</details>
<p>在所有匹配词表的评论中，选择集有 {pth["selection"]["n_toxic"]} / {pth["selection"]["n_with_hit"]} 被标为 toxic，比例 {f(pth["selection"]["share"], 2)}；测试集为 {pth["test"]["n_toxic"]} / {pth["test"]["n_with_hit"]}，比例 {f(pth["test"]["share"], 2)}。两组匹配了同一词表，但评论内容和语境并不相同，不能仅据此认定测试标注更宽松。</p>
<div class="box"><b>假设情景：</b>如果把 {hits} 条匹配词表的假阳性全部改标为 toxic，固定预测下的测试精确率会从 {f(s1["per_label_test"]["toxic"]["precision"])} 升到 {f(counterfactual)}。数值上填补了到 {floor} 目标之间约 {100 * gap_fraction:.0f}% 的差距。这个计算没有证明这些评论应改标，不能解释为标签因素的贡献或上界。</div>
<p>其余 {s4["n_false_positive"] - hits} 条未匹配词表的假阳性也未逐条判定。它们的近似词表外比例为 {f(oovg["test_fp_no_lexicon_hit"])}，高于匹配词表的假阳性 {f(oovg["test_fp_lexicon_hit"])} 和真阳性 {f(oovg["test_true_positives"])}。这是后续抽样复核的线索，尚不能归因。</p>
<details><summary>诊断词表</summary><p class="prov">{", ".join(lex["lexicon"])}。</p><p>这份额外词表只用于诊断，没有作为单独特征输入模型；其中的词可能与模型学到的 TF-IDF 词表重叠。</p></details>

<h3 id="gap5">5.5 阈值规则对比</h3>
<p>比较五种规则，每种都在选择集内做相同的阈值交叉验证，再用全选择集确定阈值并报告测试读数。候选规则按留出折结果比较；由于同一组验证结果也用于选择候选，最好结果还需要新的独立验证。</p>
<table><tr><th>规则</th><th class="n">样本内精确率</th><th class="n">留出折精确率</th><th class="n">留出预测数均值</th><th class="n">测试精确率</th><th class="n">测试预测数</th><th>全选择集阈值</th></tr>{rule_rows}</table>
<figure><img src="figures/gap5_selection_rules.png" alt="五种阈值规则的精确率与预测数量对比"><figcaption>Wilson 下界规则的留出折精确率为 {f(lb["cv"]["precision_mean"])}，预测数量从 {pt["cv"]["n_mean"]:.0f} 降到 {lb["cv"]["n_mean"]:.0f}，约少 {100 * (1 - lb["cv"]["n_mean"] / pt["cv"]["n_mean"]):.0f}%。测试精确率为 {f(lb["test_readout"]["precision"])}，仍未达到 {floor}。</figcaption></figure>
<p>Wilson 区间用于表达比例估计的不确定性。以区间下界选阈值，比只看精确率点估计更保守；但这里扫描了多个候选阈值，单点区间并不是选择后的 95% 保证。</p>
<p>本次两种下界规则 <code>lb_min30</code> 和 <code>lb_min100</code> 的结果相同，均只保留 toxic 的自动处置阈值。按双侧 95% Wilson 区间计算，即使没有错误，下界达到 0.99 也需要至少约 381 条，所以 30 与 100 的最低样本数在本轮均未成为约束。</p>
<p>将点估计规则的最小样本数提高到 100 或 300，会关闭 insult 的自动处置阈值。但它原先触发的评论同时触发其他标签，因此总体自动处置数量在本轮保持不变。</p>

<h3 id="gap6">5.6 证据与限制</h3>
<table><tr><th>总体阈值的评估方式</th><th class="n">自动处置精确率</th><th>描述性差值</th></tr>
<tr><td>全选择集拟合并评估</td><td class='n'>{f(s6["in_sample"])}</td><td>参照值</td></tr>
<tr><td>选择集留出折均值</td><td class='n'>{f(s6["held_out_cv"])}</td><td>比样本内低 {100 * s6["optimism"]:.2f} 个百分点</td></tr>
<tr><td>官方评分测试行</td><td class='n'>{f(s6["test"])}</td><td>比留出折低 {100 * s6["shift"]:.2f} 个百分点</td></tr></table>
<figure><img src="figures/gap6_decomposition.png" alt="三种评估方式的精确率差距，不代表因果分解"><figcaption>不同评估方式的读数对照。差值受样本、阈值和数据差异共同影响，不能据此给各原因分配贡献。</figcaption></figure>
<ul>
<li>当前流程在测试集上未达到自动处置要求，这个结论同时出现在最终路由和总体阈值诊断中。</li>
<li>更保守的阈值规则在选择集内部验证中表现更好，并减少自动处置数量。它值得进入下一轮比较，尚未解决测试精确率缺口。</li>
<li>词表、概率校准和 AP 的差异提供了调查方向。没有统一标准下的人工复核，就不能确认标注差异，更不能量化它对缺口的贡献。</li>
<li>当前 toxic 分数的阈值扫描不证明 99% 对其他方法不可达。后续仍需按事先确定的标签标准、目标和覆盖要求评估。</li>
</ul>

<h2 id="next">6. 首轮提出的下一步</h2>
{'<p>以下是首轮结束时的计划。其中保留自动处置权限和 99% 要求的决定，已由第三步的人工确认方案替代；其余条目保留为当时提出的验证建议，当前安排见目录和最新一步的决定。</p>' if HUMAN_RUN is not None else ''}
<ol>
<li>把 Wilson 下界规则作为候选，与当前点估计规则在相同协议下比较，同时报告精确率、覆盖数量和关闭的标签。</li>
<li>准备新增留出数据，并预先区分模型训练、校准、阈值选择和最终评估的用途。已经反复查看过的官方测试行继续用于诊断，不能切两半后当作从未使用的数据。</li>
<li>尝试字符 n-gram 或子词特征，检查它们是否减少当前词级模型的错误。重点比较精确率与覆盖率、AP 和高概率段的校准表现；词表外比例目前只提供尝试的动机。</li>
<li>按预先写明的标注标准抽样复核，包括含词和不含词的假阳性，以及必要的对照样本。先量化分歧，再决定是否修订标注或评估口径。</li>
<li>保留 99% 自动处置要求，不因本轮未通过而降低。新的方法应在冻结配置、足够样本和明确覆盖率下接受独立评估。</li>
</ol>
<p class="prov">本节数字来自 <a href="results.json">results.json</a>，运行对照来自 <a href="runs.json">runs.json</a>。测试标签用于事后诊断和阈值范围扫描，但没有用扫描结果替换本轮冻结的路由阈值。以上改进方向都还需要新的验证；原始评估标签在本轮保持不变。</p>
"""
    page(STEPS[1][0], STEPS[1][1], body, STEPS[0], STEPS[2] if HUMAN_RUN is not None else None)


def build_current_index() -> None:
    run = HUMAN_RUN
    workload = run["review_workload"]
    steps = "".join(
        f'<li><a href="{path}">{title}</a><div class="one">{description}。</div></li>'
        for path, title, description in STEPS
    )
    body = f"""
<span class="tag">项目记录</span><span class="tag">当前：人工确认</span>
<h1>review-router 项目记录</h1>
<p class="lede">按步骤记录实验设计、测量结果和决定。前两页保留自动处置实验；第三页记录改成人工确认后的新流程与评估{'；第四页记录误报人工复核和字符特征探索' if len(STEPS) > 3 else ''}{'；第五页记录 Jev 对照的进展与结果' if len(STEPS) > 4 else ''}。</p>
<h2>当前项目在做什么</h2>
<p>系统将评论分为放行、普通人工审核、优先人工审核。高置信预测及规则标记的高风险评论进入优先档；任何处置都需要人工确认，两个人工档都占用审核容量。</p>
<p>本轮共评估 {workload["n_total"]:,} 条评论，{workload["n_requires_human_review"]:,} 条进入人工审核，占 {100 * workload["review_fraction"]:.2f}%。这些数字描述保存的评估运行，实际人工审核质量仍待验证。</p>
{'<p>当前决定是默认保持词特征 <code>word</code>。独立审核价值人工评估集仅记录为后续可选方向，因工作量较大而暂缓，尚未启动；本次不改变数据标签或验收门槛。</p>' if FEATURES is not None else ''}
<div class="box">99% 是首轮自动处置实验的历史要求。当前 95% 与 90% 是选择集上的分档参数；测试精确率如实报告，项目不再以达到 99% 来判定人工辅助方案是否有效。</div>
<h2>步骤</h2><ol class="steps">{steps}<li>{PENDING[0]}，{PENDING_STATUS}。{PENDING[1]}。</li></ol>
{'<p>补充研究依据：<a href="step4-features.html#label-quality-reference">Jigsaw 的非预期偏差指标与标签可靠性前提</a>。记录为何需要复核标签质量，以及论文不能替本项目证明的部分。</p>' if FEATURES is not None else ''}
{DEFERRED_REVIEW_EVALUATION}
<h2>重新生成页面</h2>
<pre><code>python -m pip install -e ".[ml,analysis]"
python record/diagnose.py --render-only
python record/render.py</code></pre>
<p>以上命令读取保存的数据生成页面和图表。更新当前人工审核实验时：</p>
<pre><code>python scripts/run_pipeline.py --config configs/baseline.yaml
python record/collect_runs.py --human-review reports/&lt;new run&gt;
python record/render.py</code></pre>
<p class="prov">当前运行 <code>{run["run_id"]}</code>。源数据与策略快照保存在 <a href="human_review_run.json">human_review_run.json</a>；历史运行保存在 <a href="runs.json">runs.json</a>。</p>
"""
    page("index.html", "review-router 项目记录", body, None, STEPS[0])


def _human_time_section(sim: dict[str, Any]) -> str:
    """New records time started and completed jobs separately; old runs keep their meaning."""
    metrics = sim.get("time_metrics") or {}
    if not metrics:
        return """<h2>时间指标口径</h2>
<p>这份历史人工审核运行只保存了按种子汇总的时间指标，等待时间只统计窗口内已完成的任务。
它没有保存可用于本页的逐任务时间汇总，因此不补算已开始但未完成任务的等待，也不补写新的主指标。</p>"""
    headline = sim.get("headline") or {}
    selected = headline.get("selected") or {}
    metric = headline.get("metric")
    metric_name = {"wait": "开始审核前的等待", "completion_latency": "完成审核的总耗时"}.get(metric, str(metric))
    population = "所有已经开始审核的任务，含窗口结束时仍在审核的任务" if metric == "wait" else "窗口内已经完成审核的任务"
    reduction = selected.get("reduction")
    relative = (f"相对缩短 {100 * reduction:.2f}%（负值表示更慢）" if reduction is not None
                else "不报告百分比；FIFO 为零或读数缺失时，比值没有定义")
    reliable = ""
    if headline.get("percentile") == 99:
        reliable = "该主指标的 p99 样本量检查：" + ("达到报告设定的最小样本数。" if selected.get("p99_reliable") else "未达到，不能据此作稳定的尾部比较。")
    rows = ""
    for key, block in metrics.items():
        data = block["pooled_over_seeds"]
        strategy, load = key.split("@")
        status, high_risk = data["status"], data["high_risk_status"]
        wait, completion = data["wait"], data["completion_latency"]
        def quantiles(values: dict[str, Any]) -> str:
            return " / ".join(f(values.get(name), 2) for name in ("p50", "p90", "p99"))
        reliability = "等待：" + ("足够" if wait.get("p99_reliable") else "不足")
        reliability += "；完成：" + ("足够" if completion.get("p99_reliable") else "不足")
        rows += f"<tr><td>{escape(load)}</td><td>{escape(strategy)}</td><td class='n'>{data['n_arrivals']}</td><td class='n'>{status['completed']}</td><td class='n'>{status['in_progress']}</td><td class='n'>{status['not_started']}</td><td class='n'>{high_risk['completed']} / {high_risk['in_progress']} / {high_risk['not_started']}</td><td class='n'>{quantiles(wait)}<br>n={wait['n']}</td><td class='n'>{quantiles(completion)}<br>n={completion['n']}</td><td>{reliability}</td></tr>"
    return f"""<h2>逐任务时间与主指标</h2>
<p>每次仿真保存 <code>simulation_jobs.csv</code>，记录任务到达、开始和完成时间。
等待时间等于开始时间减到达时间，统计所有已开始任务，包括仍在审核的任务；完成耗时等于完成时间减到达时间，只统计窗口内已完成任务。
尚未开始的任务没有观测到完整等待时间，列入状态数量，不按零等待计入分位数。未完成任务也不按零耗时计入完成分位数。</p>
<p>本次配置选择的主指标为 <b>{escape(metric_name)} p{headline.get('percentile')}</b>，负载为 {f(headline.get('load_per_hour'))} 条/小时。
人群是{population}，先合并各随机种子的逐任务记录再计算分位数。它与下方“每个种子先算分位数、再取均值”的表不同，不应混用。</p>
<table><tr><th>主指标</th><th>{escape(str(headline.get('router_strategy', '路由策略')))}，分钟</th><th>FIFO，分钟</th><th>样本数 路由/FIFO</th><th>绝对缩短 FIFO−路由，分钟</th></tr>
<tr><td>{escape(metric_name)} p{headline.get('percentile')}</td><td class='n'>{f(selected.get('router'))}</td><td class='n'>{f(selected.get('fifo'))}</td><td class='n'>{f(selected.get('n_router'))} / {f(selected.get('n_fifo'))}</td><td class='n'>{f(selected.get('absolute_difference_min'))}</td></tr></table>
<p>{relative}。{reliable}高危子集的等待用于补充诊断，不替换全体待审任务的主指标。不同策略在有限窗口内启动或完成的任务可能不同，比较时需同时看下面的状态数量。</p>
<table><tr><th>入队负载/h</th><th>排序</th><th>到达</th><th>完成</th><th>审核中</th><th>尚未开始</th><th>高危 完成/审核中/未开始</th><th>等待 p50/p90/p99，分钟</th><th>完成耗时 p50/p90/p99，分钟</th><th>p99 样本量</th></tr>{rows}</table>
<p>p99 样本量判据为该时间统计至少 {f((sim.get('assumptions') or {}).get('p99_min_samples'))} 条，达到此门槛仍不代表尾部分位数没有抽样误差。
完成审核不等于已证实采取正确处罚；这里没有另行模拟人工准确率、申诉或执行流程。</p>"""


def _human_gate_categories(run: dict[str, Any]) -> str:
    gates = run["policy_snapshot"]["gates"]
    if "classification" not in gates:
        return ""
    classification, routing = gates["classification"], gates["routing"]
    capacity = gates.get("capacity", {})
    sim = run["simulation"]
    rows = ""

    def row(category: str, name: str, value: Any, limit: Any, upper: bool = False) -> str:
        status = {"green": "通过", "**red**": "未通过", "not set": "未设门槛，仅报告", "unavailable": "读数缺失"}[gate_status(value, limit, upper=upper)]
        requirement = "未设定" if limit is None else ("≤ " if upper else "≥ ") + f(limit)
        return f"<tr><td>{category}</td><td>{name}</td><td class='n'>{f(value)}</td><td>{requirement}</td><td>{status}</td></tr>"

    for label, spec in classification.get("per_label_average_precision", {}).items():
        rows += row("分类质量", f"AP {escape(label)}", run["per_label"][label]["average_precision"], spec.get("floor"))
    for label, spec in classification.get("per_label_precision_at_human_threshold", {}).items():
        value = run["per_label"][label]["at_human_review"]["precision"]
        rows += row("分类质量", f"普通审核阈值处精确率 {escape(label)}", value, spec.get("floor"))
    rows += row("分类质量", "标签层级违例率", run["consistency"].get("hierarchy_violation_rate"), classification.get("hierarchy_violation_rate_max"), True)
    for label, spec in classification.get("volume_overshoot_max", {}).items():
        value = (run.get("prevalence_shift") or {}).get(label, {}).get("volume_overshoot_model")
        rows += row("分类质量", f"预测阳性量相对超出 {escape(label)}", value, spec.get("ceiling"), True)
    for tier in ("priority_review", HUMAN):
        rows += row("路由质量", f"{tier} 精确率", run["tiers"][tier]["precision"], routing.get(f"{tier}_precision_floor"))
    primary = sim.get("primary") or {}
    key = f"{primary.get('strategy')}@{primary.get('load_per_hour'):g}" if primary.get("load_per_hour") is not None else ""
    pooled = (sim.get("time_metrics") or {}).get(key, {}).get("pooled_over_seeds", {})
    for name, value, limit, upper in (
        ("近容量完成率", sim.get("completion_ratio"), capacity.get("completion_ratio_min"), False),
        ("近容量结束时未开始数量，种子均值", sim.get("backlog_end"), capacity.get("backlog_end_max"), True),
        ("近容量等待 p50，合并逐任务记录，分钟", pooled.get("wait", {}).get("p50"), capacity.get("wait_p50_max_min"), True),
        ("近容量队列深度 p95，种子均值", sim.get("queue_depth_p95"), capacity.get("queue_depth_p95_max"), True),
        ("近容量审核员利用率", sim.get("reviewer_utilization"), capacity.get("reviewer_utilization_min"), False),
    ):
        rows += row("容量", name, value, limit, upper)
    reproducibility = gates.get("reproducibility") or {}
    scenario = reproducibility.get("scenario") or {}
    tolerance = reproducibility.get("float_tolerance")
    tolerance_text = f"{tolerance:g}" if isinstance(tolerance, (float, int)) else "未设定"
    return f"""<h2>四类回归检查</h2>
<p>门槛来自这次运行保存的政策快照。分类质量检查标签表现和预测量；路由质量检查送审队列；容量检查有限人力下的完成与积压；可复现性检查相同输入能否重放。
身份词差异在后面单独诊断。未设定的门槛不显示为通过。排序相对 FIFO 的四项检查见上表。</p>
<table><tr><th>类别</th><th>检查</th><th>读数</th><th>要求</th><th>结果</th></tr>{rows}</table>
<p><b>可复现性：</b>配置要求读取 <code>{escape(str(reproducibility.get('records_file', '未设定')))}</code>，按负载 <code>{escape(str(scenario.get('load', '未设定')))}</code>、种子 <code>{escape(str(scenario.get('seed', '未设定')))}</code> 重放场景，浮点容差为 <code>{tolerance_text}</code>。
本页只展示保存的配置和指标；重算逐任务时间及重放顺序的检查由回归测试执行，不根据这张静态报告宣称已通过。</p>"""


def build_step3() -> None:
    run = HUMAN_RUN
    if run is None:
        return
    workload, sim = run["review_workload"], run["simulation"]
    policy = run["policy_snapshot"]
    gates = policy["gates"].get("capacity", policy["gates"]["routing"])
    has_job_times = bool(sim.get("time_metrics"))
    wait_population = "已开始高危任务（含审核中）" if has_job_times else "已完成高危任务（历史口径）"
    tier_rows = ""
    for tier in ("priority_review", HUMAN, ALLOW):
        stats = run["tiers"][tier]
        tier_rows += f"<tr><td>{tier}</td><td class='n'>{stats['n_predicted_positive']:,}</td><td class='n'>{100 * stats['coverage']:.2f}%</td><td class='n'>{f(stats['precision'])}</td><td>{ci(stats['precision_ci95'])}</td></tr>"
    comparisons = [
        ("超容量：危害代理值 / 审核人时", sim["harm_per_reviewer_hour"], gates["harm_per_reviewer_hour_vs_fifo_min"], False),
        (f"近容量：{wait_population}的等待 p90", sim["high_risk_wait_p90"], gates["high_risk_wait_p90_vs_fifo_max"], True),
        ("近容量：高危完成数量", sim["high_risk_handled"]["primary"], gates.get("high_risk_handled_vs_fifo_min"), False),
        ("超容量：高危完成数量", sim["high_risk_handled"]["thesis"], gates.get("high_risk_handled_vs_fifo_min"), False),
    ]
    comparison_rows = ""
    failed = 0
    for label, values, limit, upper in comparisons:
        ratio = values["router"] / values["fifo"] if values["router"] is not None and values["fifo"] else None
        status = "不可比较"
        if limit is None:
            status = "未设门槛，仅报告"
        elif ratio is not None:
            ok = ratio <= limit if upper else ratio >= limit
            status = "通过" if ok else "未通过"
            failed += not ok
        elif upper and values["fifo"] == 0 and values["router"] is not None and values["router"] > 0:
            status = "未通过，FIFO 等待为零而路由等待为正"
            failed += 1
        comparison_rows += f"<tr><td>{label}</td><td class='n'>{f(values['router'])}</td><td class='n'>{f(values['fifo'])}</td><td class='n'>{f(ratio)}</td><td>{'≤' if upper else '≥'} {f(limit)}</td><td>{status}</td></tr>"
    sim_rows = ""
    for stats in sim["summary"].values():
        sim_rows += f"<tr><td>{stats['load_per_hour']:g}</td><td>{stats['strategy']}</td><td class='n'>{f(stats['n_handled'], 1)}</td><td class='n'>{f(stats['high_risk_handled'], 1)}</td><td class='n'>{f(stats['high_risk_unhandled'], 1)}</td><td class='n'>{f(stats['harm_per_reviewer_hour'], 2)}</td><td class='n'>{f(stats['high_risk_wait_p90'], 2)}</td><td class='n'>{f(stats['backlog_end'], 1)}</td></tr>"
    fairness_rows = ""
    for tier in ("predicted_positive", "priority_review", HUMAN):
        stats = run["identity_false_positives"]["test"]["final_tier"][tier]
        fpr = run["identity_false_positives"]["test"]["final_tier"]["clean_negative_false_positives"][tier]
        status = fairness_gate_status(stats, policy["gates"]["fairness"]["identity_false_discovery_rate_ratio_max"], policy["gates"]["min_predicted_positives_for_precision"])
        status = {"green": "通过", "**red**": "未通过", "inconclusive": "不确定，跳过"}.get(status, status)
        fairness_rows += f"<tr><td>{tier}</td><td class='n'>{f(stats['false_discovery_rate_ratio'])}</td><td>{ci(stats['false_discovery_rate_ratio_ci95'])}</td><td>{status}</td><td class='n'>{f(fpr['false_positive_rate_ratio'])}</td><td>{ci(fpr['false_positive_rate_ratio_ci95'])}</td></tr>"
    comparison_key = f"priority_vs_severity@{sim['thesis']['load_per_hour']:g}"
    severity_comparison = sim["paired"][comparison_key]
    time_section = _human_time_section(sim)
    gate_sections = _human_gate_categories(run)
    time_scope = (
        "等待统计所有已开始任务，含窗口结束时仍在审核的任务；未开始任务另外计数。"
        if has_job_times else
        "这份历史运行的等待只统计已完成任务。"
    )
    body = f"""
<span class="tag">第 3 步</span><span class="tag">政策 v2</span>
<h1>改为优先人工审核</h1>
<p class="lede">用户决定将自动处置改为高置信优先审核，所有处置由人工确认。本轮据此改变路由、队列工作量和验收标准。</p>
<h2>改变了什么</h2>
<ul><li><code>priority_review</code> 替代当前流程中的 <code>auto_action</code>，不再绕过人工。</li>
<li>高置信档先采用选择集精确率 0.95 作为分档参数，普通档为 0.90。这是本轮预先固定的初始设置，不是测试精确率保证。</li>
<li>威胁和标签关系不一致规则将评论送入优先档。优先排序先区分档位，再按预测严重度排序。</li>
<li>两个待审档都进入同一容量模型。FIFO、概率、严重度和优先档排序使用相同候选池与到达序列。</li></ul>
<p>高置信不等于高危。先处理高置信评论可能挤占其他高危评论的审核时间，因此保留严重度排序作为对照，并直接检查高危完成数量。</p>
<h2>本轮工作量与质量</h2>
<table><tr><th>层级</th><th>评论数</th><th>占全部评论</th><th>测试精确率</th><th>95% 区间</th></tr>{tier_rows}</table>
<p>总人工需求为 {workload["n_requires_human_review"]:,} / {workload["n_total"]:,} 条，占 {100 * workload["review_fraction"]:.2f}%；合并人工队列精确率为 {f(workload["precision"])}，95% 区间 {ci(workload["precision_ci95"])}。这里用任一真实标签为阳性作为审核价值的代理指标，并未独立测量一条评论是否值得人工审核。</p>
<p>运行声明的自动处置数为 {run["decision_contract"]["automatic_actions"]}，所有待审评论均要求人工确认。这里评估的是路由建议，未模拟人工复核的正确率或实际处罚结果。</p>
{time_section}
<h2>新验收标准与结果</h2>
<p>本轮以 FIFO 为参照：超容量下处理的危害代理值不降低；近容量下高危等待 p90 不增加；两个负载下的高危完成数量都不减少。{time_scope}因此同时检查完成数量，避免只看较早启动的任务。下面的要求在看新结果前设定，是实测比值要求，不是统计非劣性证明。</p>
<table><tr><th>指标</th><th>优先排序</th><th>FIFO</th><th>比值</th><th>比值要求</th><th>结果</th></tr>{comparison_rows}</table>
<p>上述四项排序检查中有 {failed} 项未通过；未设门槛或不可比较的项目另列，不计为通过。没有再要求测试精确率达到 99%，但新的排序策略仍需接受这些容量与完成量检查。</p>
<p>与严重度排序的配对比较也需要保留。超容量时，优先排序的高危完成数量平均差值为 {severity_comparison['high_risk_handled']['mean_diff']:+.1f} 条，危害代理值/人时平均差值为 {severity_comparison['harm_per_reviewer_hour']['mean_diff']:+.3f}，方向均为优先排序减去严重度排序。优于 FIFO 并不说明它优于所有对照。</p>
{gate_sections}
<h2>四种排序的仿真结果</h2>
<p>容量为 {sim["assumptions"]["capacity_per_hour"]:g} 条/小时，{sim["assumptions"]["reviewers"]} 名审核员，每条 {sim["assumptions"]["handle_minutes"]:g} 分钟。表格为 {len(sim["assumptions"]["seeds"])} 个种子的均值 ± 标准差；等待单位为分钟。{time_scope}高危剩余包括仍在服务中的任务，积压只计尚未开始的任务。</p>
<div class="box">60 / 108 / 180 条每小时是进入人工队列后的到达率。新策略改变了候选池及入队比例，不能把这里的排序收益与旧自动处置策略的收益比直接当作端到端提升。</div>
<table><tr><th>入队负载/h</th><th>排序</th><th>完成</th><th>高危完成</th><th>高危剩余</th><th>危害代理值/人时</th><th>高危等待 p90</th><th>结束积压</th></tr>{sim_rows}</table>
<h2>身份词子组诊断</h2>
<p>FDR 比较待审评论中的误报占比，FPR 比较正常评论被送审的比例；比值方向均为含身份词除以不含。FDR 只在完整区间不高于设定上限时通过，跨越则不确定。FPR 单独报告，不能用 FDR 通过替代它。</p>
<table><tr><th>层级</th><th>FDR 比</th><th>95% 区间</th><th>FDR 检查</th><th>FPR 比</th><th>95% 区间</th></tr>{fairness_rows}</table>
<h2>如何复现</h2>
<pre><code>python scripts/run_pipeline.py --config configs/baseline.yaml
REVIEW_ROUTER_EVAL_REPORT=reports/&lt;run&gt;/report.json pytest -q tests/test_gate.py
python scripts/render_results.py reports/&lt;run&gt;
python record/collect_runs.py --human-review reports/&lt;run&gt;
python record/render.py</code></pre>
<p class="prov">运行 <code>{run["run_id"]}</code>，commit <code>{run["git_commit"][:12]}</code>{'，工作区含未提交改动' if run['git_dirty'] else '，工作区干净'}。本页读取保存的 <a href="human_review_run.json">运行数据与政策快照</a>。首轮 99% 目标及未达标结果保留在<a href="step2-round1.html">历史记录</a>，没有按新标准重算。</p>
"""
    page(STEPS[2][0], STEPS[2][1], body, STEPS[1], STEPS[3] if len(STEPS) > 3 else None)


# ============================================================================ step 4
def capture_figure(v2: dict[str, Any]) -> str:
    """Positives and high-risk rows captured against how many rows are flagged."""
    name = "step4_capture_curve.png"
    curves = v2["curves"]
    labels = list(curves)
    colors = {labels[0]: BLUE, labels[1]: ORANGE}
    display = {labels[0]: "word", labels[1]: "word+char"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, key, title in (
        (axes[0], "positives", "Truly positive comments captured"),
        (axes[1], "high_risk", "High-risk comments captured"),
    ):
        for label in labels:
            c = curves[label]
            ax.plot(c["k"], c[key], color=colors[label], lw=2, label=display[label])
            op = c["operating_point"]
            ax.plot(
                op["flagged"], op[key], "o", ms=8, color=colors[label],
                markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=5,
            )
        total = curves[labels[0]]["total_" + key]
        xmax = max(curves[labels[0]]["k"])
        for label, (fx, fy) in zip(labels, ((0.50, 0.42), (0.70, 0.60))):
            op = curves[label]["operating_point"]
            ax.annotate(
                f"{display[label]} flags {op['flagged']:,}\ncaptures {op[key]:,}",
                (op["flagged"], op[key]),
                xytext=(fx * xmax, fy * total),
                fontsize=8, color="#52514e",
                arrowprops={"arrowstyle": "-", "color": "#8a8983", "lw": 0.8},
            )
        ax.axhline(total, color=GRAY, lw=1, ls=":")
        ax.text(200, total, f"all {total:,} in the test set", va="bottom", fontsize=8, color="#52514e")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("comments flagged (top k by max label probability)")
        ax.grid(color="#e6e5e1", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlim(0, max(curves[labels[0]]["k"]))
        ax.set_ylim(0, total * 1.08)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / name, dpi=150)
    plt.close(fig)
    return name


def _pct(v: float, d: int = 1) -> str:
    return f"{100 * v:.{d}f}%"


def build_step4() -> None:
    data = FEATURES
    if data is None or len(STEPS) < 4:
        return
    r1, audit, v2 = data["round1"], data["audit"], data["v2"]
    human = audit["human"]
    word, wc = v2["runs"][0], v2["runs"][1]
    lw, lwc = word["label"], wc["label"]

    # ---- audit
    implied = human["implied_precision"]
    change_rows = "".join(
        f"<tr><td>{escape(c['text'])}</td><td>{c['preread']}</td><td>{c['human']}</td><td>{escape(c['note'])}</td></tr>"
        for c in audit["changed"]
    )
    strata_rows = "".join(
        f"<tr><td>{escape(k)}</td><td class='n'>{v}</td><td class='n'>{audit['sample_strata'].get(k, 0)}</td></tr>"
        for k, v in audit["population_strata"].items()
    )

    # ---- round 1
    r1_rows = ""
    for run in r1["runs"]:
        a, h = run["tiers"]["auto_action"], run["tiers"]["human_review"]
        r1_rows += (
            f"<tr><td>{escape(run['label'])}</td><td>{escape(run['model'].get('analyzer', 'word'))}</td>"
            f"<td class='n'>{f(a['test_precision'])}</td><td>{ci(a['test_precision_ci95'])}</td>"
            f"<td class='n'>{a['test_n_predicted_positive']:,}</td><td class='n'>{f(a['selection_precision'])}</td>"
            f"<td class='n'>{f(h['test_precision'])}</td><td class='n'>{h['test_n_predicted_positive']:,}</td></tr>"
        )
    ap_head = "".join(f"<th class='n'>{escape(r['label'])}</th>" for r in r1["runs"])
    ap_rows = "".join(
        f"<tr><td>{lb}</td>" + "".join(f"<td class='n'>{f(r['per_label'][lb]['average_precision'])}</td>" for r in r1["runs"]) + "</tr>"
        for lb in LABELS
    )
    pf = r1["paired_false_positives"]["word-char"]
    overlap = (r1.get("audit_overlap") or {}).get("word-char", {})

    # ---- v2
    tier_rows = ""
    for run in (word, wc):
        for tier in ("priority_review", HUMAN):
            t = run["tiers"][tier]
            tier_rows += (
                f"<tr><td>{escape(run['model'].get('analyzer', 'word'))}</td><td>{tier}</td>"
                f"<td class='n'>{t['test_n_predicted_positive']:,}</td><td class='n'>{f(t['test_precision'])}</td>"
                f"<td>{ci(t['test_precision_ci95'])}</td><td class='n'>{f(t['selection_precision'])}</td></tr>"
            )
    ww, wwc = word["review_workload"], wc["review_workload"]
    rw, rwc = word["recall"], wc["recall"]

    def cell(e: dict[str, Any]) -> str:
        return f"{e['k']:,} / {e['n']:,}（{_pct(e['share'])}）"

    recall_rows = "".join(
        f"<tr><td>{escape(run['model'].get('analyzer', 'word'))}</td><td class='n'>{run['recall']['flagged']['k']:,}</td>"
        f"<td class='n'>{cell(run['recall']['positives_flagged'])}</td><td class='n'>{cell(run['recall']['high_risk_flagged'])}</td>"
        f"<td class='n'>{cell(run['recall']['high_risk_in_top'])}</td><td class='n'>{run['recall']['high_risk_in_allow']['k']:,}</td></tr>"
        for run in (word, wc)
    )
    mv = v2["matched_volume"]
    mv_rows = ""
    for i, k in enumerate(mv["k"]):
        a, b = mv["runs"][lw][i], mv["runs"][lwc][i]
        mv_rows += (
            f"<tr><td class='n'>{k:,}</td><td class='n'>{a['positives']:,}</td><td class='n'>{b['positives']:,}</td>"
            f"<td class='n'>{b['positives'] - a['positives']:+,}</td><td class='n'>{a['high_risk']:,}</td>"
            f"<td class='n'>{b['high_risk']:,}</td><td class='n'>{b['high_risk'] - a['high_risk']:+,}</td></tr>"
        )
    k_word, k_wc = rw["flagged"]["k"], rwc["flagged"]["k"]
    word_at_wc = mv["runs"][lw][mv["k"].index(k_wc)]
    gain_total = rwc["positives_flagged"]["k"] - rw["positives_flagged"]["k"]
    gain_volume = word_at_wc["positives"] - rw["positives_flagged"]["k"]

    eqr_rows = ""
    for run in (word, wc):
        for key in ("priority@108", "priority@180"):
            st = run["queue_equal_review_load"]["by_load"][key]
            eqr_rows += (
                f"<tr><td>{escape(run['model'].get('analyzer', 'word'))}</td><td>{key.split('@')[1]}</td>"
                f"<td class='n'>{f(st['harm_per_reviewer_hour'], 2)}</td><td class='n'>{f(st['high_risk_handled'], 1)}</td>"
                f"<td class='n'>{f(st['high_risk_unhandled'], 1)}</td><td class='n'>{f(st['high_risk_wait_p90'], 2)}</td></tr>"
            )
    eq = v2["equal_input"]
    eqi_rows = ""
    diff = eq["differences_vs_reference"][lwc]
    for rate in eq["runs"][lw]["by_input"]:
        for run in (word, wc):
            pr = eq["runs"][run["label"]]["by_input"][rate]["priority"]
            eqi_rows += (
                f"<tr><td class='n'>{float(rate):,.0f}</td><td>{escape(run['model'].get('analyzer', 'word'))}</td>"
                f"<td class='n'>{f(pr['review_arrivals'], 0)}</td><td class='n'>{f(pr['completion_ratio'], 2)}</td>"
                f"<td class='n'>{f(pr['backlog_end'], 1)}</td><td class='n'>{f(pr['harm_handled_per_hour'], 1)}</td>"
                f"<td class='n'>{f(pr['high_risk_left_in_allow'], 1)}</td><td class='n'>{f(pr['high_risk_unhandled_in_queue'], 1)}</td>"
                f"<td class='n'>{f(pr['high_risk_not_reviewed'], 1)}</td></tr>"
            )
    diff_rows = ""
    for rate, per in diff.items():
        for strategy in ("priority", "fifo"):
            d = per[strategy]
            def pm(name: str, digits: int = 1, d: dict[str, Any] = d) -> str:
                return f"{d[name]['mean']:+.{digits}f} ± {d[name]['se']:.{digits}f}"
            diff_rows += (
                f"<tr><td class='n'>{float(rate):,.0f}</td><td>{strategy}</td><td class='n'>{pm('review_arrivals', 0)}</td>"
                f"<td class='n'>{pm('backlog_end')}</td><td class='n'>{pm('harm_handled_per_hour')}</td>"
                f"<td class='n'>{pm('high_risk_not_reviewed')}</td></tr>"
            )
    rates = list(diff)
    mid, top_rate = diff[rates[1]]["priority"], diff[rates[-1]]
    ecp = v2["equal_count_precision"]
    ecp_rows = ""
    for label, (a, b) in ecp.items():
        ka, kb = str(a["own_count"]), str(b["own_count"])
        ecp_rows += (
            f"<tr><td>{label}</td><td class='n'>{a['own_count']:,}</td><td class='n'>{b['own_count']:,}</td>"
            f"<td class='n'>{f(a['precision_at_count'][ka])}</td><td class='n'>{f(b['precision_at_count'][ka])}</td>"
            f"<td class='n'>{f(a['precision_at_count'][kb])}</td><td class='n'>{f(b['precision_at_count'][kb])}</td></tr>"
        )
    hr_miss_w = rw["high_risk_flagged"]["n"] - mv["runs"][lw][mv["k"].index(k_word)]["high_risk"]
    hr_miss_wc = rw["high_risk_flagged"]["n"] - mv["runs"][lwc][mv["k"].index(k_word)]["high_risk"]
    hr_volume = word_at_wc["high_risk"] - rw["high_risk_flagged"]["k"]
    hr_total = rwc["high_risk_flagged"]["k"] - rw["high_risk_flagged"]["k"]
    gates = v2["gates"]
    gate_items = "".join(f"<li>{escape(m)}</li>" for m in gates.get("word_char", {}).get("messages", []))
    costs = {c["analyzer"]: c for c in v2["costs"]}
    cw, cwc = costs["word"], costs["word+char_wb"]
    figure = capture_figure(v2)

    # ---- verification (optional)
    ver = data.get("verification")
    ver_section = ""
    if ver:
        boot_rows = "".join(
            f"<tr><td>{escape(b['quantity'])}</td><td class='n'>{b['point_estimate']:+.4g}</td>"
            f"<td>[{b['ci95_low']:+.4g}, {b['ci95_high']:+.4g}]</td></tr>"
            for b in ver.get("bootstrap", [])
        )
        critic_items = "".join(
            f"<li><b>{escape(c['claim_id'])}：{escape(c['verdict_zh'])}。</b>{escape(c['correction_zh'])}</li>"
            for c in ver.get("critic", []) if c.get("correction_zh")
        )
        review_items = "".join(
            f"<li>{escape(r['summary_zh'])}</li>" for r in ver.get("code_review", [])
        ) or "<li>没有发现可复现的缺陷。</li>"
        ver_section = f"""
<h2 id="verify">独立复核</h2>
<p>{escape(ver['method_zh'])}</p>
<p>这里的独立复核指另外的代理重算和审查同一批结果，没有新增独立评估数据。以下保留复核意见；当前采用方案和后续安排以本页“结论与决定”为准。</p>
<h3>重算</h3>
<p>{escape(ver['recompute_zh'])}</p>
<h3>配对 bootstrap</h3>
<p>两个模型给同一批测试行打分，所以按行做配对重抽样，比较 word+char 减去 word 的差值。{escape(ver['bootstrap_note_zh'])}</p>
<table><tr><th>差值（word+char − word）</th><th class='n'>点估计</th><th>95% 区间</th></tr>{boot_rows}</table>
<h3>代码审查</h3>
<ul>{review_items}</ul>
<h3>对结论的反驳</h3>
<ul>{critic_items}</ul>"""

    body = f"""
<span class="tag">第 4 步</span><span class="tag">round 2</span><span class="tag">政策 v1 与 v2</span>
<h1>误报人工复核与字符 n-gram</h1>
<p class="lede">第二步发现自动处置在测试集上的精确率是 0.904，达不到 0.99。这一步调查标注口径和拼写变体两种可能的解释：人工复核部分误报，并比较字符 n-gram 特征。两项检查都不能单独确认首轮差距的原因。</p>
<div class="toc"><a href="#why">为什么做这一步</a><a href="#audit">误报人工复核</a><a href="#label-quality-reference">标签质量疑点与研究依据</a><a href="#what">字符 n-gram 改了什么</a><a href="#r1">首轮设计下的结果</a><a href="#v2">v2 设计下的结果</a>{'<a href="#verify">独立复核</a>' if ver else ''}<a href="#decision">结论与决定</a><a href="#repro">如何复现</a></div>

<h2 id="why">为什么做这一步</h2>
<p>第二步的诊断留下两个调查方向。词典命中后被标为 toxic 的比例，选择集是 0.76，测试集是 0.51；这可能涉及评论内容或标注口径差异，不能直接证明测试集标注更宽松。测试集的 OOV 比例约是选择集的两倍，提示可以尝试字符特征，但尚不能说明拼写变体造成了多少误报。</p>
<p>这一步开始时项目仍采用首轮设计，自动处置层要求 0.99。做到一半，第三步把设计改成了全部人工确认，所以字符特征又在 v2 下重新比较了一次。两组结果都保留，读的时候要分清它们回答的是哪个设计下的问题。</p>

<h2 id="audit">误报人工复核</h2>
<p>从首轮自动处置的 {audit['n_false_positive']} 条误报里，按“是否命中固定词典”和“是否有其他标签为真”分层，按比例抽了 {human['n']} 条。误报的定义与门槛一致：进了自动处置层，但触发它的标签在测试集里是 0。</p>
<table><tr><th>分层（词典命中 / 其他标签为真）</th><th class='n'>总体</th><th class='n'>样本</th></tr>{strata_rows}</table>
<p>先由 Claude 按一条写明的规则预读，再由项目负责人逐条读完，只在不同意的地方改判。这是看过预读结果的单人复核，不是独立标注者之间的一致性检验。判定规则：</p>
<ul><li><b>toxic</b>：攻击了某个人或群体，明说、讽刺、影射或针对读者的反问都算。</li>
<li><b>borderline</b>：粗口或无礼，但没有人被攻击，自嘲和引用也算这一类。</li>
<li><b>clean</b>：没有冒犯内容。</li></ul>
<p>人工改判了 {len(audit['changed'])} 条，方向有两种：讽刺和影射被改成 toxic，只是喊名字或只是提到身份词被改成 clean。</p>
<details><summary>改判的 {len(audit['changed'])} 条</summary>
<table><tr><th>文本</th><th>预读</th><th>人工</th><th>备注</th></tr>{change_rows}</table></details>
<p>最终结果：toxic {human['counts']['toxic']}，borderline {human['counts']['borderline']}，clean {human['counts']['clean']}。</p>
<p>下面是条件敏感性估计，不是重标后的正式测试成绩：假定未复核的原真阳性仍然正确，把样本中的未加权比例外推到全部原误报。原始 Jigsaw 标签保持不变；这批历史自动处置误报也不是当前人工审核价值的独立评估集。</p>
<table><tr><th>假设口径</th><th class='n'>条件估计的自动处置精确率</th><th>近似 95% 区间</th></tr>
<tr><td>按测试集标签</td><td class='n'>{f(1 - audit['n_false_positive'] / audit['n_auto_action'])}</td><td></td></tr>
<tr><td>假定判为 toxic 的原误报应算正确</td><td class='n'>{f(implied['toxic_only']['precision'])}</td><td>{ci(implied['toxic_only']['precision_ci95'])}</td></tr>
<tr><td>假定 toxic 和 borderline 都应算正确</td><td class='n'>{f(implied['toxic_or_borderline']['precision'])}</td><td>{ci(implied['toxic_or_borderline']['precision_ci95'])}</td></tr></table>
<p>在上述假设下，要达到 0.99，{audit['n_auto_action']:,} 条自动处置里最多只能有 {int(audit['n_auto_action'] * 0.01)} 条误报，换算到样本上至少要有 {human['rows_needed_toxic_for_floor']} 条被改计为正确。实际判为 toxic 的有 {human['counts']['toxic']} 条，合并 borderline 也只有 {human['counts']['toxic'] + human['counts']['borderline']} 条。区间由 Wilson 区间近似换算，未计入分层抽样、有限总体修正或人工判断的不确定性，不能视为完整的不确定性估计。</p>
<div class="box">按本次人工复核口径，样本中仍有 {human['counts']['clean']} 条被判为 clean，不能把这些误报都解释为测试集漏标。攻击对象和语境是值得继续检查的因素；这次复核没有证明词袋特征必然无法识别它们，也没有量化各因素对总体缺口的贡献。</div>

<h2 id="label-quality-reference">标签质量疑点与研究依据</h2>
<p>2026-09-23 补充。Daniel Borkan 等，Jigsaw，2019：<a href="https://arxiv.org/abs/1903.04561">Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification</a>。中文题意为“利用真实数据细致衡量文本分类中的非预期偏差”。发表于 WWW 2019 Companion，<a href="https://doi.org/10.1145/3308560.3317593">DOI</a>。</p>
<h3>论文提供的依据</h3>
<p>论文指出，毒性判断复杂且带有主观性；标注者偏见、用户构成和样本选择都可能造成模型学到不当关联。作者明确承认，跨群体的可靠测试标签是重要前提，并非所有应用都满足。这支持检查标签与抽样过程，不能直接证明本项目某条标签错误。见<a href="https://storage.googleapis.com/gweb-research2023-media/pubtools/5007.pdf#page=1">第 1 节</a>。</p>
<p>论文的三个 AUC 指标不依赖某个固定阈值。这里的群体指评论所提及的身份群体，背景是该群体之外的评论；阳性和阴性均按评估标签划分。见<a href="https://storage.googleapis.com/gweb-research2023-media/pubtools/5007.pdf#page=3">第 3.1 节</a>。</p>
<table><tr><th>指标</th><th>比较哪些样本</th><th>低值提示什么</th></tr>
<tr><td>Subgroup AUC</td><td>群体内部的阳性与阴性</td><td>群体内部的区分能力较弱</td></tr>
<tr><td>BPSN AUC</td><td>背景阳性与群体阴性</td><td>群体阴性相对背景阳性被排得过高，可能导致误报</td></tr>
<tr><td>BNSP AUC</td><td>背景阴性与群体阳性</td><td>群体阳性相对背景阴性被排得过低，可能导致漏报</td></tr></table>
<p>它们检查模型排序与标签的关系，不能单独鉴定标签质量，也不等于某个阈值下的 FPR 或 FDR。论文第 4.4 节还保留了标注者“难以判断”的选项，提醒我们不要把所有分歧都当成确定的标错。</p>
<h3>与本项目的关系</h3>
<p>本项目怀疑标签不足以支撑当前任务，有两类本地依据。第一，已有误报审计中，{human['counts']['toxic']} 条被复核为 toxic、{human['counts']['borderline']} 条为 borderline，说明原标签与本次判定口径存在分歧；但样本只来自历史误报，且由看过预读结果的一人复核，无法估计全测试集的标签错误率。第二，“值得送人工审核”与“已有毒性阳性标签”并不等价：含糊或需要语境的评论可能值得送审，最终却没有确认违规。这属于评估目标与标签含义的差异，不一定是原标签标错。</p>
<p>论文使用了另行构建、带身份标注的真实评论数据，不能把它的结论直接套到本项目的 Jigsaw 2018 测试行。当前身份词匹配只是代理分组，并不等同于论文的人工身份标注。</p>
<div class="box">记录结论：标签可靠性和任务适配性值得复核，但模型偏差、标注分歧与任务口径不一致需要分别判断。目前保留原始标签与现行验收门槛；本文仅作为研究依据记录，不新增 AUC 实现。独立人工评估集继续<a href="#future-review-evaluation">暂缓</a>。</div>

<h2 id="what">字符 n-gram 改了什么</h2>
<p>模型配置新增 <code>model.analyzer</code>：<code>word</code> 是原来的词级 TF-IDF；<code>char_wb</code> 是词边界内的 2 到 5 字符片段；<code>word+char_wb</code> 把两组特征并排。下游的六个逻辑回归、Platt 校准、阈值选择和路由都不变。对比用的配置只在这几行上与 baseline 不同。</p>

<h2 id="r1">首轮设计下的结果</h2>
<p class="prov">这三次运行来自未提交的工作区（{', '.join(escape(c['run_id']) for c in r1['costs'])}），数字可以从保存的运行目录复算，但不能从某个提交完整重现。</p>
<table><tr><th>运行</th><th>特征</th><th class='n'>自动处置精确率</th><th>95% 区间</th><th class='n'>条数</th><th class='n'>选择集精确率</th><th class='n'>人工层精确率</th><th class='n'>条数</th></tr>{r1_rows}</table>
<table><tr><th>标签 AP</th>{ap_head}</tr>{ap_rows}</table>
<p>排序在大部分标签上变好，identity_hate 提升最多，但自动处置的精确率区间互相重叠，没有一个接近 0.99。配对来看，word+char 去掉了基线 {pf['baseline_fp']} 条误报中的 {pf['baseline_fp_dropped']} 条，同时新放进 {pf['auto_rows_only_variant']} 条，其中 {pf['new_fp_added']} 条是新误报。人工复核里判为 clean 的行，word+char 仍送进自动处置的比例是 {escape(overlap.get('by_verdict', {}).get('clean', 'n/a'))}。</p>

<h2 id="v2">v2 设计下的结果</h2>
<p>两次运行都来自干净的提交 <code>{cw['git_commit'][:8]}</code>，配置只差特征。词特征这次运行与第三步那次运行的分层、队列精确率、各标签 AP 和模拟汇总完全相同；第三步那次来自未提交的工作区，这次说明它的结果可以从干净的提交复现。</p>
<h3>分层与工作量</h3>
<table><tr><th>特征</th><th>层级</th><th class='n'>条数</th><th class='n'>测试精确率</th><th>95% 区间</th><th class='n'>选择集精确率</th></tr>{tier_rows}</table>
<p>需要人工审核的评论从 {ww['n_requires_human_review']:,} 条增加到 {wwc['n_requires_human_review']:,} 条（占全部评论 {_pct(ww['review_fraction'], 2)} 到 {_pct(wwc['review_fraction'], 2)}），队列精确率从 {f(ww['precision'])} 降到 {f(wwc['precision'])}。</p>
<h3>召回：真阳性和高风险评论去了哪里</h3>
<table><tr><th>特征</th><th class='n'>标记</th><th class='n'>真阳性被标记</th><th class='n'>高风险被标记</th><th class='n'>高风险进优先档</th><th class='n'>高风险留在放行</th></tr>{recall_rows}</table>
<p>word+char 多抓到 {gain_total} 条真阳性，留在放行里的高风险评论从 {rw['high_risk_in_allow']['k']} 条降到 {rwc['high_risk_in_allow']['k']} 条。但它也多标记了 {k_wc - k_word:,} 条。</p>
<h3>相同标记量下的排序</h3>
<p>为了把“排序更好”和“标记更多”分开，两个模型都按最大标签概率排序，取前 k 条，看抓到多少。</p>
<figure><img src="figures/{figure}" alt="两幅折线图：横轴为按最大标签概率排序后标记的评论数，纵轴分别为抓到的真阳性数和高风险评论数。两条线几乎重合，word+char 略高；圆点标出两个模型实际的标记量。"><figcaption>两条曲线几乎重合，word+char 略高一点。word+char 的工作点更靠右，主要是因为它标记得更多。</figcaption></figure>
<table><tr><th class='n'>k</th><th class='n'>word 真阳性</th><th class='n'>word+char 真阳性</th><th class='n'>差</th><th class='n'>word 高风险</th><th class='n'>word+char 高风险</th><th class='n'>差</th></tr>{mv_rows}</table>
<p>真阳性和高风险评论要分开看。word+char 多出的 {gain_total} 条真阳性中，约 {gain_volume} 条只要让词模型也标记 {k_wc:,} 条就能得到，剩下约 {gain_total - gain_volume} 条来自排序，所以真阳性的提升大约七成来自多标记。高风险评论不一样：多出的 {hr_total} 条里只有约 {hr_volume} 条来自多标记，一半以上来自排序。换成漏检的角度看，标记 {k_word:,} 条时，词模型漏掉 {hr_miss_w} 条高风险评论，word+char 漏掉 {hr_miss_wc} 条，少了约 {100 * (hr_miss_w - hr_miss_wc) / hr_miss_w:.0f}%，多出的主要是 identity_hate。按最大概率排序与路由实际使用的分标签阈值不完全相同，这里是近似的分解；复核用路由本身重做，结果相差不到 3 条。</p>
<p>按标签看也是一样。在相同条数下，word+char 每个标签的精确率都更高：</p>
<table><tr><th>标签</th><th class='n'>word 的条数</th><th class='n'>word+char 的条数</th><th class='n'>word 在 word 条数</th><th class='n'>word+char 在 word 条数</th><th class='n'>word 在 word+char 条数</th><th class='n'>word+char 在 word+char 条数</th></tr>{ecp_rows}</table>
<p>word+char 在人工审核阈值处的条数更多，是因为同一条“选择集精确率 0.90”的规则，在排序更好的模型上会选出更多行。它不是一个可以随意调低的阈值：把词模型的阈值调到同样条数，toxic 和 insult 的精确率会跌破门槛。</p>
<h3>队列：同等审核量</h3>
<p>第三步报告里的模拟固定的是进入人工队列后的到达率，两个模型每小时收到同样多的审核任务。</p>
<table><tr><th>特征</th><th class='n'>审核量/h</th><th class='n'>危害代理值/人时</th><th class='n'>高危完成</th><th class='n'>高危未完成</th><th class='n'>高危等待 p90，分钟</th></tr>{eqr_rows}</table>
<p>这个口径下 word+char 略差。5 个种子分不出差别，但复核用 200 个种子重跑后，差距是系统性的，每人时处理的危害低 3% 到 5%：它的队列里每条评论的平均危害更低（1.79 对 1.88）。注意这个口径让 word+char 看到的评论流比词模型少约 11%，它回答的是“每个审核任务的价值”，不是“哪个模型在同样的评论上做得更好”。</p>
<h3>队列：同等评论流量</h3>
<p>更接近实际的比较是固定进来的评论流量。每个种子生成一条评论流，两个模型共用；每个模型只审核自己标记的评论，被它留在放行层的高风险评论算作未审核。流量取词模型在 60 / 108 / 180 条每小时审核量时对应的评论流量，{eq['n_seeds']} 个种子，按种子配对比较。下表是优先排序，均值 ± 样本标准差，数量按一个 8 小时班次计。</p>
<table><tr><th class='n'>评论/h</th><th>特征</th><th class='n'>审核任务</th><th class='n'>完成率</th><th class='n'>结束积压</th><th class='n'>危害处理/h</th><th class='n'>高风险留在放行</th><th class='n'>高风险在队列未完成</th><th class='n'>高风险未审核合计</th></tr>{eqi_rows}</table>
<table><tr><th class='n'>评论/h</th><th>排序</th><th class='n'>审核任务差</th><th class='n'>积压差</th><th class='n'>危害处理/h 差</th><th class='n'>高风险未审核差</th></tr>{diff_rows}</table>
<p>差值是 word+char 减词模型，± 为配对标准误。优先排序下，三个流量档位 word+char 都每小时处理更多危害，每个班次少漏 {-diff[rates[0]]['priority']['high_risk_not_reviewed']['mean']:.1f} 到 {max(-diff[r]['priority']['high_risk_not_reviewed']['mean'] for r in rates):.1f} 条高风险评论，因为它留在放行层的高风险评论更少。代价是审核量多约 12%：中间档位时它越过了 120 条每小时的容量，结束积压多 {mid['backlog_end']['mean']:.0f} 条。FIFO 排序在超载时结果反过来，高风险未审核反而多 {top_rate['fifo']['high_risk_not_reviewed']['mean']:.1f} 条，所以这份收益依赖 v2 的优先排序。</p>
<h3>回归门槛与成本</h3>
<p>用 v2 的回归门槛检查两份报告：词特征 {escape(gates.get('word', {}).get('summary') or 'n/a')}；word+char {escape(gates.get('word_char', {}).get('summary') or 'n/a')}。未通过的两项：</p>
<ul>{gate_items}</ul>
<p>这些分类回归门槛参照词模型的实测值设定：toxic 的下限是词模型的 0.663 减 0.02。word+char 的 toxic 读数低约 0.001，复核的重抽样里有 43% 会通过，说明这一项对抽样敏感；它在原报告中仍未通过。相同条数下的精确率更高，提示不能仅凭这一项把结果概括为分类整体变差。</p>
<p>severe_toxic 的预测量高估相对词模型更严重，复核的所有重抽样也都如此。这是本次数据上的聚合校准诊断，不能据此断言字符特征放大了训练集与测试集之间的阳性率差异。这两次运行中，该标签未通过自身阈值触发路由；现行回归门槛保持不变。</p>
<table><tr><th>特征</th><th class='n'>完整运行耗时，秒</th><th class='n'>模型文件，MB</th></tr>
<tr><td>word</td><td class='n'>{cw['seconds']}</td><td class='n'>{cw['model_mb']}</td></tr>
<tr><td>word+char</td><td class='n'>{cwc['seconds']}</td><td class='n'>{cwc['model_mb']}</td></tr></table>
{ver_section}
<h2 id="decision">结论与决定</h2>
<div class="box"><b>决定（2026-09-23，项目负责人）：暂不采用字符 n-gram，默认保持词特征。</b><code>model.analyzer</code> 开关留在代码里，默认值是 <code>word</code>。</div>
<p><b>它有没有效果：</b>有，但很小，而且集中在高风险评论上。相同标记量下，高风险漏检少约 15%，真阳性多约 2.5%，区间都不含 0。同一条评论流上，优先排序下每个班次少漏 3 到 5 条高风险评论。它没有解决第二步的问题：首轮设计下，自动处置精确率仍然在 0.90 左右。</p>
<p><b>为什么暂不采用：</b></p>
<ul><li><b>成本高。</b>完整运行时间约为 {cwc['seconds'] / cw['seconds']:.0f} 倍，模型文件约为 {cwc['model_mb'] / cw['model_mb']:.1f} 倍。这是单次运行的端到端时间，没有单独测每条评论的推理延迟。</li>
<li><b>审核量多约 12%。</b>在接近容量的评论流量下，队列会越过容量开始积压。要兑现它的收益，需要更多审核人力，或者重新选择阈值。</li>
<li><b>还需要评估阈值与容量的取舍。</b>候选的额外审核量和未通过的回归项尚未解决。若以后重启比较，应在开发数据上选择候选方案，另留独立数据评估，不能为了让候选通过而调整当前测试门槛。</li>
<li><b>证据只有一次训练、一份测试集。</b>区间以这两个训练好的模型为条件，没有估计重新训练带来的方差。</li></ul>
<p><b>之前的说法哪里不对：</b>最初的判断是“收益主要来自多标记，把词模型阈值调低也能得到”。复核不支持把这句话用于高风险评论：相同标记量下仍有排序收益；把词模型阈值调到同样条数，高风险只多抓约 17 到 20 条，而且会让两项精确率门槛不通过。因此，本次保留 word 同时考虑了计算成本、审核容量和证据范围；字符特征在这些运行中的收益仍如实保留。</p>
<p><b>后续安排：</b>当前继续使用 <code>word</code>，不启动新增独立数据采集或人工标注，也不修改现行门槛。独立数据验证仅保留为可选方向，暂缓且尚未启动。若以后决定开展，可再考虑是否将 word+char 纳入候选，并事先确定比较方法和验收要求。</p>
{DEFERRED_REVIEW_EVALUATION}
<h2 id="repro">如何复现</h2>
<pre><code>python scripts/audit_auto_action_fp.py --score analysis/auto_action_fp_audit
python scripts/run_pipeline.py --config configs/baseline.yaml
python scripts/run_pipeline.py --config configs/word_char.yaml
python scripts/compare_runs.py reports/&lt;word run&gt; reports/&lt;word+char run&gt; --equal-input --out analysis/v2
python record/collect_runs.py --features analysis
python record/render.py</code></pre>
<p class="prov">v2 运行 <code>{escape(cw['run_id'])}</code> 与 <code>{escape(cwc['run_id'])}</code>，commit <code>{cw['git_commit'][:12]}</code>，工作区干净。本页读取 <a href="features_run.json">features_run.json</a>，它由 <code>analysis/</code> 下的对比结果和人工复核结果汇总而来。</p>
"""
    page(STEPS[3][0], STEPS[3][1], body, STEPS[2], STEPS[4] if len(STEPS) > 4 else None)


def build_step5() -> None:
    if JEV is None or len(STEPS) < 5:
        return
    data = JEV
    protocol = data["protocol"]
    done = data["status"] == "completed_exploratory"
    status = "已完成探索性对照" if done else "对照尚未完成"
    if data["status"] == "blocked_registration":
        status = "注册受限，实测暂缓"
    access = data.get("access_blocker")
    access_note = (
        '<div class="box warn"><p>项目负责人注册时看到以下提示，'
        '目前无法完成注册，因此尚未尝试真实 Jev 调用。</p>'
        f"<blockquote>{escape(access['message'])}</blockquote>"
        f"<p>提示指向 <a href=\"{escape(access['information_url'], quote=True)}\">"
        "TypeSafe 官方动态</a>。此处记录用户遇到的访问限制；"
        "尚无模型效果结果，不能据此评价 Jev 的好坏。</p></div>"
    ) if access else ""
    count_rows = "".join(
        f"<tr><td>{escape(name)}</td><td class='n'>{counts['rows']:,}</td>"
        + "".join(f"<td class='n'>{counts[label]:,}</td>" for label in LABELS) + "</tr>"
        for name, counts in protocol["counts"].items()
    )
    runs = [(name, data.get(key)) for name, key in (("word", "baseline"), ("Jev", "jev"))]
    metric_rows = ""
    for name, run in runs:
        if run is None:
            metric_rows += f"<tr><td>{name}</td><td colspan='5'>尚无真实预测，不能计算</td></tr>"
            continue
        work = run["review_workload"]
        high = run["recall"]["high_risk_flagged"]
        metric_rows += (
            f"<tr><td>{name}</td><td class='n'>{work['n_requires_human_review']:,}</td>"
            f"<td class='n'>{100 * work['review_fraction']:.2f}%</td>"
            f"<td class='n'>{f(work['precision'])}</td>"
            f"<td class='n'>{high['k']} / {high['n']}</td>"
            f"<td class='n'>{run['recall']['high_risk_in_allow']['k']}</td></tr>"
        )
    per_label_rows = ""
    for label in LABELS:
        cells = []
        for _, run in runs:
            cells.extend([
                f(run["per_label"][label]["average_precision"]) if run else "待实测",
                f(run["brier"][label], 4) if run else "待实测",
            ])
        per_label_rows += f"<tr><td>{label}</td>" + "".join(
            f"<td class='n'>{value}</td>" for value in cells
        ) + "</tr>"
    threshold_rows = ""
    selection_rule = data.get("selection_rule", {})
    min_predictions = selection_rule.get("min_predicted_positives", 30)
    targets = selection_rule.get("precision_targets", {"priority_review": 0.95, "human_review": 0.90})
    for name, run in runs:
        if not run:
            continue
        for label, selected in run.get("threshold_selection", {}).items():
            for tier, target in targets.items():
                threshold = selected[f"at_{tier}"]
                reason = "已启用"
                if threshold["threshold"] is None:
                    needed = ceil(min_predictions * target)
                    reason = (
                        f"关闭：阳性 {selected['positives']} 条，少于至少所需的 {needed} 条"
                        if selected["positives"] < needed else
                        "关闭：没有分数阈值同时满足最低数量和精确率目标"
                    )
                threshold_rows += (
                    f"<tr><td>{name}</td><td>{label}</td><td>{tier}</td>"
                    f"<td class='n'>{f(threshold['threshold'])}</td>"
                    f"<td class='n'>{threshold['n_predicted_positive']:,}</td>"
                    f"<td class='n'>{f(threshold['precision'])}</td><td>{reason}</td></tr>"
                )
    threshold_section = (
        "<details><summary>阈值选择与关闭原因</summary>"
        "<p>下表为全体选择样本上的分标签阈值。身份词子组还需通过自己的阈值；"
        "其明细保存在 jev_run.json 的 thresholds.subgroup 中。</p>"
        "<table><tr><th>方法</th><th>标签</th><th>层级</th><th>阈值</th>"
        "<th>选择集预测阳性</th><th>选择集精确率</th><th>状态</th></tr>"
        + threshold_rows + "</table></details>"
    ) if threshold_rows else ""
    failure = data.get("failure")
    failure_note = (
        f"<p>停止阶段：<code>{escape(failure['stage'])}</code>。"
        f"{escape(failure['type'])}：{escape(failure['message'])}</p>"
    ) if failure else ""
    verification = data.get("verification")
    verification_section = (
        "<h2>实现验证</h2>"
        f"<p>全套测试 {verification['tests_passed']} 项通过，{verification['tests_skipped']} 项跳过。"
        f"{escape(verification['skips_zh'])} Ruff 与 mypy 均通过。</p>"
        f"<p>{escape(verification['full_baseline_note_zh'])} 运行目录："
        f"<code>{escape(verification['full_baseline_run'])}</code>。</p>"
        f"<p>{escape(verification['scope_zh'])}</p>"
    ) if verification else ""
    comparison = "<p>Jev 预测尚未完成，暂无等标记量差异、配对区间或两模型队列比较。</p>"
    capture = data.get("paired_capture")
    if capture:
        comparison = (
            f"<p>固定取前 {capture['k']:,} 条，按最大校准标签概率排序。word 抓到 "
            f"{capture['baseline_captured']} 条高风险评论，Jev 抓到 {capture['jev_captured']} 条。"
            f"差值为 {capture['difference_jev_minus_baseline']:+}，配对重抽样 95% 区间 "
            f"{ci(capture['ci95'], 1)}。这是排序诊断，不是含规则路由在相同审核预算下的实测结果。</p>"
        )
    eq = (data.get("comparison") or {}).get("equal_input")
    if eq:
        queue_rows = ""
        for name, entry in eq["runs"].items():
            for rate, stats in entry["by_input"].items():
                values = stats["priority"]
                queue_rows += (
                    f"<tr><td>{escape(name)}</td><td class='n'>{float(rate):.1f}</td>"
                    f"<td class='n'>{f(values['review_arrivals'], 1)}</td>"
                    f"<td class='n'>{f(values['backlog_end'], 1)}</td>"
                    f"<td class='n'>{f(values['high_risk_not_reviewed'], 1)}</td>"
                    f"<td class='n'>{f(values['harm_handled_per_hour'], 1)}</td></tr>"
                )
        comparison += (
            "<table><tr><th>方法</th><th>评论/h</th><th>审核任务/班次</th>"
            "<th>结束积压</th><th>高风险未审核/班次</th><th>危害代理处理/h</th></tr>"
            + queue_rows + "</table>"
            f"<p>{eq['n_seeds']} 个共同输入流种子，每班次 8 小时，表中均值 ± 样本标准差。"
            "未审核包括放行中的高风险评论和队列中未完成的高风险评论。模拟种子的误差不包含"
            "评论样本或重新训练的不确定性。</p>"
        )
    api = data.get("api")
    cost = "<p>尚未完成 Jev 调用，暂无调用成本或延迟测量。</p>"
    if api:
        cost = (
            f"<p>本次进程成功网络请求 {api.get('network_successes', 0):,} 次，"
            f"缓存命中 {api.get('cache_hits', 0):,} 次。成功网络响应累计输入 "
            f"{api.get('input_tokens', 0):,} tokens。延迟 p50 "
            f"{f(api.get('p50_latency_seconds'))} 秒，p95 {f(api.get('p95_latency_seconds'))} 秒。"
            "缓存重放不产生这些网络延迟；失败重试可能另计费用，实际账单未核对。</p>"
        )
    baseline = data.get("baseline")
    if baseline:
        cost += (
            f"<p>word 本次训练、校准和预测合计 {baseline['fit_calibration_prediction_seconds']:.1f} 秒；"
            f"其中阈值集与测试集的批量预测合计 {baseline['prediction_seconds']:.1f} 秒。"
            "这是本机批量耗时，不能直接与单条 API 延迟作倍数比较。</p>"
        )
    evidence = "".join(
        f"<li>{name}：<code>{escape(run['run'])}</code>，commit "
        f"<code>{escape((run.get('git_commit') or '')[:12])}</code>，"
        f"{'工作区含未提交改动' if run.get('git_dirty') else '工作区干净'}。</li>"
        for name, run in runs if run
    )
    body = f"""
<span class="tag">第 5 步</span><span class="tag">{status}</span>
<h1>Jev 与词模型：固定样本的探索性对照</h1>
<p class="lede">{escape(data['updated_utc'])} 更新。目标是在相同数据和审核政策下，检查语义判断能否改善六标签分类与审核排序。现有 baseline 是 TF-IDF 加逻辑回归，本轮没有训练 BERT。</p>
<div class="box"><b>当前结论：</b>{escape(data['conclusion_zh'])}</div>
{'<p>' + escape(data['execution_note_zh']) + '</p>' if data.get('execution_note_zh') else ''}
{access_note}
{failure_note}
<div class="toc"><a href="#design">实验设计</a><a href="#samples">固定样本</a><a href="#results">当前结果</a><a href="#comparison">公平比较</a><a href="#cost">成本与复现边界</a><a href="#repro">复现命令</a></div>
<h2 id="design">实验设计</h2>
<p>word 使用原 train 子集训练六个逻辑回归。Jev 使用固定版本 <code>{escape(protocol['model'])}</code>，一次调用分别询问六个标签，保留六个独立概率，不将它们归一化为总和 1。输入仅含评论正文与固定任务定义，不含标签、模型分数或人工审计判断。问题定义见 <a href="../configs/jev_questions.json">jev_questions.json</a>。</p>
<p>两种方法使用同一 calib 子集拟合 Platt 校准，同一 thresh 子集选择阈值，同一 test 子集评估。word 校准原始分类间隔，Jev 校准原始概率的 logit，裁剪范围为 0.000001 到 0.999999。若校准标签只有一类，保留原始概率并记录无法拟合。下游复用原有阈值选择、规则、人工确认与队列模拟。</p>
<p>两种方法的预训练条件不同，这是完整方法在本项目中的对照，不能把差异归因于某一项架构。所有处置仍由人工确认。</p>
<h2 id="samples">固定样本</h2>
<p>从原始划分中按 seed、split 和 ID 的 SHA-256 排序取样，不看标签。保留原始行序，记录各集合 ID 哈希；样本少也不换 seed。协议见 <a href="../analysis/jev/protocol.json">protocol.json</a>，选中的 ID 见 <a href="../analysis/jev/cohort.csv">cohort.csv</a>。</p>
<table><tr><th>用途</th><th>行数</th>{''.join(f'<th>{label} 阳性</th>' for label in LABELS)}</tr>{count_rows}</table>
<p>稀有标签的样本数限制结论强度。现有阈值规则至少需要 30 个预测阳性；90% 和 95% 的精确率目标分别至少需要 27 和 29 个真阳性。关闭某个标签的阈值不代表模型无法识别它；评论还可能通过其他标签或规则进入审核。</p>
<h2 id="results">当前结果</h2>
<table><tr><th>方法</th><th>需人工审核</th><th>审核比例</th><th>队列精确率</th><th>高风险送审/总数</th><th>高风险留在放行</th></tr>{metric_rows}</table>
<table><tr><th>标签</th><th>word AP</th><th>word Brier</th><th>Jev AP</th><th>Jev Brier</th></tr>{per_label_rows}</table>
<p>AP 越高越好，Brier 越低越好。表中正确性沿用原始 Jigsaw 标签，只是审核价值的代理指标。pilot 报告明确标为探索性，不能冒充完整语料上的 baseline 验收。</p>
{threshold_section}
<h2 id="comparison">相同标记量与相同评论流</h2>
{comparison}
<p>是否值得扩大实验，需要同时看高风险漏审差异的方向与区间、实际送审量、积压和完成量。不能仅凭总准确率或新增送审带来的召回提升更换默认模型。</p>
{verification_section}
<h2 id="cost">成本与复现边界</h2>
{cost}
<p>锁定模型版本、问题、样本与原始响应。缓存键绑定端点、版本、完整问题和文本；缺失、非法概率或版本不符会停止评估，不跳过困难样本。缓存重放能复算已保存预测，不能保证外部服务未来生成逐值相同的响应。</p>
<p>官方测试集已经反复查看，Jev 的预训练数据是否含此公开语料也未知。本轮不声称独立泛化结果，不比较 BERT 优劣，也不证明真实送审价值提升。独立人工评估继续暂缓。</p>
<h2 id="repro">复现命令</h2>
<pre><code># 准备固定样本并运行配对 word 对照，不调用 API
python scripts/run_jev_experiment.py --prepare-only
# 注册恢复并取得 API 访问后，配置 TYPESAFE_API_KEY 再执行真实请求
python scripts/run_jev_experiment.py --allow-network
# 只用缓存重放，缺失时停止
python scripts/run_jev_experiment.py
python record/render.py</code></pre>
<p>真实调用需要有效 API 访问，密钥只从进程环境读取，不写入配置或报告。<code>reports/jev-cache/</code> 不提交到 Git；跨机器复算需要另行取得该缓存或重新调用服务。</p>
<ul>{evidence}</ul>
<p class="prov">协议 SHA-256：<code>{escape(data['protocol_sha256'])}</code>。本页读取 <a href="jev_run.json">jev_run.json</a>，原始比较汇总保存在 <code>analysis/jev/</code>。</p>
<p>接口与版本依据：<a href="https://docs.typesafe.ai/primitives/noul">Noul</a>、<a href="https://docs.typesafe.ai/models">模型版本</a>、<a href="https://docs.typesafe.ai/model-jaggedness/jev-1.13">已知限制</a>。厂商关于性能和校准的描述不能替代本项目测量。</p>
"""
    page(STEPS[4][0], STEPS[4][1], body, STEPS[3], None)


if __name__ == "__main__":
    build_index()
    build_step1()
    build_step2()
    build_step3()
    build_step4()
    build_step5()
    print("written:", "index.html", *(s[0] for s in STEPS))
