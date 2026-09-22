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
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from review_router.ceilings import SCORED_TEST_COUNTS, ceiling_table  # noqa: E402
from scripts.render_results import fairness_gate_status  # noqa: E402

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
if HUMAN_RUN is not None:
    STEPS.append((
        "step3-human-review.html", "第 3 步：优先人工审核",
        "取消自动处置权限，按全部人工工作量重新比较排序效果",
    ))
    PENDING = ("第 4 步：独立数据验证", "冻结候选方案后，用新增数据验证质量与工作量")


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
<nav class="top" aria-label="项目步骤">{navigation}<span>{PENDING[0]}，待做</span></nav>
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
{'<p>以下是首轮结束时的计划。其中保留自动处置权限和 99% 要求的决定，已由第三步的人工确认方案替代；其余验证建议仍适用。</p>' if HUMAN_RUN is not None else ''}
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
<p class="lede">按步骤记录实验设计、测量结果和决定。前两页保留自动处置实验；第三页记录改成人工确认后的新流程与评估。</p>
<h2>当前项目在做什么</h2>
<p>系统将评论分为放行、普通人工审核、优先人工审核。高置信预测及规则标记的高风险评论进入优先档；任何处置都需要人工确认，两个人工档都占用审核容量。</p>
<p>本轮共评估 {workload["n_total"]:,} 条评论，{workload["n_requires_human_review"]:,} 条进入人工审核，占 {100 * workload["review_fraction"]:.2f}%。下一步要验证的是有限人力下的审核质量与完成量。</p>
<div class="box">99% 是首轮自动处置实验的历史要求。当前 95% 与 90% 是选择集上的分档参数；测试精确率如实报告，项目不再以达到 99% 来判定人工辅助方案是否有效。</div>
<h2>步骤</h2><ol class="steps">{steps}<li>{PENDING[0]}，待做。{PENDING[1]}。</li></ol>
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


def build_step3() -> None:
    run = HUMAN_RUN
    if run is None:
        return
    workload, sim = run["review_workload"], run["simulation"]
    policy = run["policy_snapshot"]
    gates = policy["gates"]["routing"]
    tier_rows = ""
    for tier in ("priority_review", HUMAN, ALLOW):
        stats = run["tiers"][tier]
        tier_rows += f"<tr><td>{tier}</td><td class='n'>{stats['n_predicted_positive']:,}</td><td class='n'>{100 * stats['coverage']:.2f}%</td><td class='n'>{f(stats['precision'])}</td><td>{ci(stats['precision_ci95'])}</td></tr>"
    comparisons = [
        ("超容量：危害代理值 / 审核人时", sim["harm_per_reviewer_hour"], gates["harm_per_reviewer_hour_vs_fifo_min"], False),
        ("近容量：已完成高危任务的等待 p90", sim["high_risk_wait_p90"], gates["high_risk_wait_p90_vs_fifo_max"], True),
        ("近容量：高危完成数量", sim["high_risk_handled"]["primary"], gates["high_risk_handled_vs_fifo_min"], False),
        ("超容量：高危完成数量", sim["high_risk_handled"]["thesis"], gates["high_risk_handled_vs_fifo_min"], False),
    ]
    comparison_rows = ""
    failed = 0
    for label, values, limit, upper in comparisons:
        ratio = values["router"] / values["fifo"] if values["router"] is not None and values["fifo"] else None
        status = "不可比较"
        if ratio is not None:
            ok = ratio <= limit if upper else ratio >= limit
            status = "通过" if ok else "未通过"
            failed += not ok
        comparison_rows += f"<tr><td>{label}</td><td class='n'>{f(values['router'])}</td><td class='n'>{f(values['fifo'])}</td><td class='n'>{f(ratio)}</td><td>{'≤' if upper else '≥'} {limit:g}</td><td>{status}</td></tr>"
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
<p>总人工需求为 {workload["n_requires_human_review"]:,} / {workload["n_total"]:,} 条，占 {100 * workload["review_fraction"]:.2f}%；合并人工队列精确率为 {f(workload["precision"])}，95% 区间 {ci(workload["precision_ci95"])}。任一真实标签为阳性就计为有审核价值。</p>
<p>运行声明的自动处置数为 {run["decision_contract"]["automatic_actions"]}，所有待审评论均要求人工确认。这里评估的是路由建议，未模拟人工复核的正确率或实际处罚结果。</p>
<h2>新验收标准与结果</h2>
<p>本轮以 FIFO 为参照：超容量下处理的危害代理值不降低；近容量下高危等待 p90 不增加；两个负载下的高危完成数量都不减少。等待只统计已完成任务，因此必须同时检查完成数量。下面的要求在看新结果前设定，是实测比值要求，不是统计非劣性证明。</p>
<table><tr><th>指标</th><th>优先排序</th><th>FIFO</th><th>比值</th><th>比值要求</th><th>结果</th></tr>{comparison_rows}</table>
<p>上述四项排序检查中有 {failed} 项未通过。没有再要求测试精确率达到 99%，但新的排序策略仍需接受这些容量与完成量检查。</p>
<p>与严重度排序的配对比较也需要保留。超容量时，优先排序的高危完成数量平均差值为 {severity_comparison['high_risk_handled']['mean_diff']:+.1f} 条，危害代理值/人时平均差值为 {severity_comparison['harm_per_reviewer_hour']['mean_diff']:+.3f}，方向均为优先排序减去严重度排序。优于 FIFO 并不说明它优于所有对照。</p>
<h2>四种排序的仿真结果</h2>
<p>容量为 {sim["assumptions"]["capacity_per_hour"]:g} 条/小时，{sim["assumptions"]["reviewers"]} 名审核员，每条 {sim["assumptions"]["handle_minutes"]:g} 分钟。表格为 {len(sim["assumptions"]["seeds"])} 个种子的均值 ± 标准差；等待单位为分钟。高危剩余包括仍在服务中的任务，积压只计尚未开始的任务。</p>
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
    page(STEPS[2][0], STEPS[2][1], body, STEPS[1], None)


if __name__ == "__main__":
    build_index()
    build_step1()
    build_step2()
    build_step3()
    print("written:", "index.html", *(s[0] for s in STEPS))
