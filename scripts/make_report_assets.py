#!/usr/bin/env python3
"""Generate report/generated/: LaTeX number macros, table bodies and vector figures.

Reads only saved result files (no torch, no GPU):
  results/analysis/summary.json, results/analysis/cells.csv,
  results/gate_coverage.json, slides/results/deck_data.json

Run:  uv run --with matplotlib --with numpy python scripts/make_report_assets.py
Every number in report/report.tex comes from numbers.tex or a generated table.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY = json.loads((ROOT / "results/analysis/summary.json").read_text())
GATE = json.loads((ROOT / "results/gate_coverage.json").read_text())
DECK = json.loads((ROOT / "slides/results/deck_data.json").read_text())
with open(ROOT / "results/analysis/cells.csv") as f:
    CELLS = list(csv.DictReader(f))

TEAL, SAFFRON, PLUM, RED, GREEN, GREY = "#0D7377", "#E07A3D", "#6B3A6B", "#C0392B", "#1E8449", "#6B7B85"
plt.rcParams.update(
    {
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "figure.dpi": 150,
    }
)


def f(x):
    return float(x) if x not in ("", None) else float("nan")


def short(m: str) -> str:
    return m.split("/")[-1].replace("Qwen2.5-", "Qwen2.5-")


def tex(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


# ---------------------------------------------------------------- data prep
STRICT = {(d["model_full"], d["task"], d["lang"]) for d in DECK["gate"] if d["tier"] == "strict"}
core = [
    r
    for r in CELLS
    if r["appendix"] == "False" and r["direction"] == "denoising" and (r["model"], r["task"], r["lang"]) in STRICT
]
by_cell = defaultdict(list)
for r in core:
    by_cell[(r["model"], r["task"], r["lang"])].append(r)


def mean(rows, k):
    v = [f(r[k]) for r in rows if r[k] != ""]
    return float(np.mean(v)) if v else float("nan")


def sd(rows, k):
    v = [f(r[k]) for r in rows if r[k] != ""]
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


core_cells = {k: v for k, v in by_cell.items()}
assert len(core_cells) == 24, len(core_cells)

# ---------------------------------------------------------------- macros
M: dict[str, str] = {}


def macro(name: str, val: str):
    M[name] = val


rho_means = {k: mean(v, "rho") for k, v in core_cells.items()}
n_pass = sum(1 for x in rho_means.values() if x >= 0.90)
n_above95 = sum(1 for x in rho_means.values() if x >= 0.95)
failing = sorted((k, x) for k, x in rho_means.items() if x < 0.90)
macro("NRuns", str(SUMMARY["n_runs"]))
macro("NGateCells", str(SUMMARY["gate_total"]))
macro("NStrict", str(SUMMARY["gate_passed"]))
macro("NFirstToken", str(SUMMARY["gate_first_token"]))
macro("NFail", str(SUMMARY["gate_total"] - SUMMARY["gate_passed"] - SUMMARY["gate_first_token"]))
macro("NCore", "24")
macro("HOnePass", str(n_pass))
macro("HOneAbove", str(n_above95))
macro("HOneMissA", f"{failing[0][1]:.2f}")
macro("HOneMissB", f"{failing[1][1]:.2f}")
macro("HOneMin", f"{min(rho_means.values()):.2f}")
macro("HOneMedian", f"{np.median(list(rho_means.values())):.3f}")
macro("HTwoNeg", str(sum(1 for k, v in core_cells.items() if mean(v, "mlp_r") < 0)))
h3 = SUMMARY["H3"]
macro("HThreeDiff", f"{h3['pooled_mean_diff']:+.3f}")
macro("HThreeLoNinety", f"{h3['ci90'][0]:+.3f}")
macro("HThreeHiNinety", f"{h3['ci90'][1]:+.3f}")
macro("HThreeLo", f"{h3['ci95'][0]:+.3f}")
macro("HThreeHi", f"{h3['ci95'][1]:+.3f}")
h4 = SUMMARY["H4"]
macro("HFourFert", f"{h4['rho_vs_tokens_per_word']['rho']:+.2f}")
macro("HFourFertLo", f"{h4['rho_vs_tokens_per_word']['ci_lo']:+.2f}")
macro("HFourFertHi", f"{h4['rho_vs_tokens_per_word']['ci_hi']:+.2f}")
macro("HFourLD", f"{h4['rho_vs_clean_ld']['rho']:+.2f}")
macro("HFourLDLo", f"{h4['rho_vs_clean_ld']['ci_lo']:+.2f}")
macro("HFourLDHi", f"{h4['rho_vs_clean_ld']['ci_hi']:+.2f}")
h5 = SUMMARY["H5"]
macro("HFiveMLPGain", f"{h5['mlp_r_gain_mean']:+.2f}")
macro("HFiveMLPLo", f"{h5['mlp_r_gain_ci95'][0]:+.2f}")
macro("HFiveMLPHi", f"{h5['mlp_r_gain_ci95'][1]:+.2f}")
macro("HFiveHeadGain", f"{h5['head_rho_gain_mean']:+.2f}")
macro("HFiveHeadLo", f"{h5['head_rho_gain_ci95'][0]:+.2f}")
macro("HFiveHeadHi", f"{h5['head_rho_gain_ci95'][1]:+.2f}")
macro("CostEAP", f"{h5['time_ratio_eap_over_attrp']:.1f}")
macro("CostAP", f"{h5['time_ratio_ap_over_attrp']:.0f}")
macro("MeanTAP", f"{DECK['cost']['mean_t_ap']:.0f}")
macro("MeanTAttr", f"{DECK['cost']['mean_t_attr']:.1f}")
macro("NTimed", str(DECK["cost"]["n_runs_timed"]))
sw = h5["step_sweep_seed0"]
macro("SweepA", f"{sw['5']['mean_head_rho']:.2f}")
macro("SweepB", f"{sw['10']['mean_head_rho']:.2f}")
macro("SweepC", f"{sw['20']['mean_head_rho']:.2f}")
macro("NoisingRho", f"{SUMMARY['noising_mean_rho']:.2f}")
macro("HindiFailLo", f"{DECK['hindi_fail_acc']['min']:.2f}")
macro("HindiFailHi", f"{DECK['hindi_fail_acc']['max']:.2f}")
macro("HindiFailN", str(DECK["hindi_fail_acc"]["n_cells"]))
g = DECK["eap_examples"]["gpt2_ioi"]
macro("GPTRho", f"{g['rho']:.2f}")
macro("GPTEapRho", f"{g['eap_rho']:.2f}")
macro("GPTMlpAttr", f"{g['mlp_r']:+.2f}")
macro("GPTMlpEap", f"{g['eap_mlp_r']:+.2f}")
macro("GPTLayerZeroAP", f"{DECK['mlp_layers']['gpt2_ioi']['ap'][0]:.2f}")
macro("GPTLayerZeroAttr", f"{DECK['mlp_layers']['gpt2_ioi']['attr'][0]:+.2f}")
macro("QwenMlpAttr", f"{DECK['eap_examples']['qwen05_ioi']['mlp_r']:+.2f}")
bl = DECK["bloom_ioi_mlp_r"]
macro("BloomMlpLo", f"{min(bl.values()):.3f}")
macro("BloomMlpHi", f"{max(bl.values()):.3f}")

# Cross-check the H5 head-rho gain against the raw table (guards against stale JSON).
gains = [mean(v, "eap_rho") - mean(v, "rho") for v in core_cells.values()]
assert abs(np.mean(gains) - h5["head_rho_gain_mean"]) < 0.01, (np.mean(gains), h5["head_rho_gain_mean"])
macro("HFiveHeadGainCheck", f"{np.mean(gains):+.2f}")

with open(OUT / "numbers.tex", "w") as fh:
    fh.write("% Generated by scripts/make_report_assets.py. Do not edit.\n")
    for k, v in M.items():
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------------------------------------------------------- tables
TASKS = ["ioi", "capitals", "agreement", "induction"]
rows = GATE["rows"]
order_models = [
    "gpt2",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "meta-llama/Llama-3.2-1B",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
]
tier_of = {(d["model_full"], d["task"], d["lang"]): d["tier"] for d in DECK["gate"]}
gate_by = {(r["model"], r["task"], r["lang"]): r for r in rows}

with open(OUT / "gate_table.tex", "w") as fh:
    for m in order_models:
        for (mm, task, lang), r in sorted(gate_by.items(), key=lambda kv: (TASKS.index(kv[0][1]), kv[0][2])):
            if mm != m:
                continue
            tier = tier_of[(mm, task, lang)]
            acc = f(r.get("clean_accuracy"))
            ld = f(r.get("clean_logit_diff"))
            st = f(r.get("single_token_answer_rate"))
            fmt = lambda x, d=2: "--" if np.isnan(x) else f"{x:.{d}f}"
            fh.write(
                f"{tex(short(mm))} & {tex(task)} & {tex(lang)} & {fmt(acc)} & {fmt(ld)} & {fmt(st)} & {tier.replace('_', '-')} \\\\\n"
            )

with open(OUT / "cells_table.tex", "w") as fh:
    for (m, t, l), v in sorted(
        core_cells.items(), key=lambda kv: (order_models.index(kv[0][0]), TASKS.index(kv[0][1]), kv[0][2])
    ):
        rr = mean(v, "rho_restricted")
        rr_s = "--" if np.isnan(rr) else f"{rr:.2f}"
        fh.write(
            f"{tex(short(m))} & {t} & {l} & {mean(v,'rho'):.3f} $\\pm$ {sd(v,'rho'):.3f} & {rr_s} & "
            f"{mean(v,'top14'):.2f} & {mean(v,'mlp_r'):+.2f} & {mean(v,'eap_rho'):.2f} & {mean(v,'eap_mlp_r'):+.2f} \\\\\n"
        )

noise = [
    r
    for r in CELLS
    if r["appendix"] == "False" and r["direction"] == "noising" and (r["model"], r["task"], r["lang"]) in STRICT
]
nb = defaultdict(list)
for r in noise:
    nb[(r["model"], r["task"], r["lang"])].append(r)
with open(OUT / "noising_table.tex", "w") as fh:
    for (m, t, l), v in sorted(
        nb.items(), key=lambda kv: (order_models.index(kv[0][0]), TASKS.index(kv[0][1]), kv[0][2])
    ):
        fh.write(f"{tex(short(m))} & {t} & {l} & {mean(v,'rho'):.3f} & {mean(v,'top14'):.2f} & {mean(v,'mlp_r'):+.2f} \\\\\n")

app = [r for r in CELLS if r["appendix"] == "True" and r["direction"] == "denoising"]
ab = defaultdict(list)
for r in app:
    ab[(r["model"], r["task"], r["lang"])].append(r)
with open(OUT / "robust_table.tex", "w") as fh:
    for (m, t, l), v in sorted(
        ab.items(), key=lambda kv: (order_models.index(kv[0][0]), TASKS.index(kv[0][1]), kv[0][2])
    ):
        fh.write(f"{tex(short(m))} & {t} & {l} & {mean(v,'rho'):.3f} & {mean(v,'top14'):.2f} & {mean(v,'mlp_r'):+.2f} \\\\\n")

# ---------------------------------------------------------------- figures
# 1. gate coverage
cols = [("ioi", "en"), ("ioi", "hi"), ("ioi", "hi_roman"), ("capitals", "en"), ("capitals", "hi"),
        ("agreement", "en"), ("agreement", "hi"), ("induction", "en"), ("induction", "hi")]
colour = {"strict": GREEN, "first_token": SAFFRON, "fail": RED, None: "#DDDDDD"}
fig, ax = plt.subplots(figsize=(6.6, 3.0))
for i, m in enumerate(order_models):
    for j, (t, l) in enumerate(cols):
        tier = tier_of.get((m, t, l))
        ax.add_patch(plt.Rectangle((j, len(order_models) - 1 - i), 0.92, 0.88, color=colour[tier]))
ax.set_xlim(0, len(cols))
ax.set_ylim(0, len(order_models))
ax.set_xticks([j + 0.46 for j in range(len(cols))])
ax.set_xticklabels([f"{t}\n{l.replace('_', ' ')}" for t, l in cols], fontsize=7)
ax.set_yticks([len(order_models) - 1 - i + 0.44 for i in range(len(order_models))])
ax.set_yticklabels([short(m) for m in order_models], fontsize=8)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREEN, SAFFRON, RED, "#DDDDDD")],
          labels=["pass (strict)", "pass (first-token only)", "fail", "not run"], ncol=4, fontsize=7,
          loc="upper center", bbox_to_anchor=(0.5, -0.17), frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_gate.pdf")
plt.close(fig)

# 2. rho per core cell
items = sorted(core_cells.items(), key=lambda kv: (TASKS.index(kv[0][1]), order_models.index(kv[0][0]), kv[0][2]))
fig, ax = plt.subplots(figsize=(6.6, 3.2))
xs = np.arange(len(items))
vals = [mean(v, "rho") for _, v in items]
errs = [sd(v, "rho") for _, v in items]
ax.bar(xs, vals, yerr=errs, color=[RED if k[2] == "hi" else TEAL for k, _ in items], capsize=1.5, width=0.75)
ax.axhline(0.90, color="k", lw=0.8, ls="--")
ax.text(len(items) - 0.4, 0.905, "0.90", ha="right", va="bottom", fontsize=7)
ax.set_xticks(xs)
ax.set_xticklabels([f"{short(k[0])} {k[1]}/{k[2]}" for k, _ in items], rotation=75, ha="right", fontsize=6)
ax.set_ylim(0.75, 1.01)
ax.set_ylabel(r"head Spearman $\rho$(AP, AttrP)")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=TEAL), plt.Rectangle((0, 0), 1, 1, color=RED)],
          labels=["English", "Hindi"], fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_rho_cells.pdf")
plt.close(fig)

# 3. Hindi vs English
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"width_ratios": [2.2, 1]})
models = ["bigscience/bloom-560m", "bigscience/bloom-1b1", "bigscience/bloom-1b7"]
for i, m in enumerate(models):
    for lang, colr, dx in (("en", TEAL, -0.12), ("hi", RED, 0.12)):
        v = [f(r["rho"]) for r in by_cell[(m, "agreement", lang)]]
        a1.scatter([i + dx] * len(v), v, s=10, color=colr, alpha=0.6, zorder=2)
        a1.hlines(np.mean(v), i + dx - 0.1, i + dx + 0.1, color=colr, lw=2, zorder=3)
a1.set_xticks(range(3))
a1.set_xticklabels([short(m) for m in models])
a1.set_ylabel(r"head $\rho$ (agreement)")
a1.legend(handles=[plt.Line2D([], [], marker="o", ls="", color=TEAL), plt.Line2D([], [], marker="o", ls="", color=RED)],
          labels=["English", "Hindi"], fontsize=7, frameon=False, loc="lower left")
a1.set_title("per seed; bars are means", fontsize=8)
a2.axvspan(-0.05, 0.05, color="#DDEEEE", zorder=0)
a2.axvline(0, color="k", lw=0.6)
lo95, hi95 = h3["ci95"]
lo90, hi90 = h3["ci90"]
a2.hlines(0, lo95, hi95, color=GREY, lw=1.2)
a2.hlines(0, lo90, hi90, color=PLUM, lw=3)
a2.plot([h3["pooled_mean_diff"]], [0], "o", color="k", ms=4)
a2.set_xlim(-0.08, 0.08)
a2.set_ylim(-1, 1)
a2.set_yticks([])
a2.set_xlabel(r"pooled $\rho_{hi}-\rho_{en}$")
a2.set_title(r"90% (thick) and 95% CI vs $\pm0.05$", fontsize=8)
a2.spines["left"].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig_hindi.pdf")
plt.close(fig)

# 4. GPT-2 IOI MLP by layer
ml = DECK["mlp_layers"]["gpt2_ioi"]
fig, ax = plt.subplots(figsize=(6.0, 2.5))
L = np.arange(len(ml["ap"]))
w = 0.27
ax.bar(L - w, ml["ap"], w, color=TEAL, label="AP")
ax.bar(L, ml["attr"], w, color=SAFFRON, label="AttrP")
ax.bar(L + w, ml["eap"], w, color=PLUM, label="EAP-IG")
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(L)
ax.set_xlabel("MLP layer (GPT-2 small, IOI, mean of 5 seeds)")
ax.set_ylabel("normalised effect")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_mlp_layers.pdf")
plt.close(fig)

# 5. EAP-IG trade-off
fig, ax = plt.subplots(figsize=(4.4, 3.4))
dx = [mean(v, "eap_rho") - mean(v, "rho") for v in core_cells.values()]
dy = [mean(v, "eap_mlp_r") - mean(v, "mlp_r") for v in core_cells.values()]
ax.scatter(dx, dy, s=14, color=PLUM, alpha=0.8)
ax.axhline(0, color="k", lw=0.5)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel(r"change in head $\rho$ (EAP-IG $-$ AttrP)")
ax.set_ylabel("change in MLP $r$ (EAP-IG $-$ AttrP)")
ax.text(-0.02, 1.7, "repairs MLP,\nbreaks heads", ha="right", va="top", fontsize=7, color=GREY) if False else None
fig.tight_layout()
fig.savefig(OUT / "fig_eap_tradeoff.pdf")
plt.close(fig)

for name in ("gate_table", "cells_table", "noising_table", "robust_table"):
    q = OUT / f"{name}.tex"
    t = q.read_text().rstrip()
    assert t.endswith("\\\\")
    q.write_text(t[:-2].rstrip() + "\n")

print("wrote", sorted(p.name for p in OUT.iterdir()))
print({k: M[k] for k in ("HOnePass", "HOneAbove", "HFiveHeadGain", "HFiveHeadGainCheck", "CostAP", "CostEAP")})
