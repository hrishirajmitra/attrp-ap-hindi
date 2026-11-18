#!/usr/bin/env python3
"""Final analysis (rules fixed in PREREGISTRATION_v2.md, Amendment 1).

Reads results/grid/*/comparison.json (+ results/gate_coverage.json) and writes
results/analysis/{cells.csv, summary.json, ANALYSIS.md, figures/*.png}.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.registry import MODELS, is_appendix

RNG = np.random.default_rng(0)
N_BOOT = 10000
MARGIN = 0.05


def load_cells(grid_dir: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(grid_dir.glob("*/comparison.json")):
        d = json.loads(f.read_text())
        c = d["cell"]
        desc = c.get("descriptors", {})
        row = {
            "model": c["model"], "task": c["task"], "lang": c["lang"],
            "direction": c["direction"], "seed": c["seed"],
            "rho": d["spearman_ap_attr"],
            "rho_restricted": d["restricted_spearman_ap_attr"]["rho"],
            "top14": d["topk_overlap_ap_attr"]["14"],
            "top14_signed": d.get("topk_overlap_signed_ap_attr", {}).get("14"),
            "mlp_r": d["mlp_pearson_ap_attr"]["pearson_r"],
            "verdict": d["verdict"]["primary"],
            "clean_ld": d["logit_diffs"]["clean"],
            "corr_ld": d["logit_diffs"]["corrupted"],
            "concentration": d.get("effect_concentration_top14"),
            "tokens_per_word": desc.get("tokens_per_word"),
            "seq_len": desc.get("seq_len"),
            "single_tok": c.get("batch_meta", {}).get("single_token_answer_rate"),
            "t_ap": d["wall_seconds"]["activation_patching"],
            "t_attr": d["wall_seconds"]["attribution_patching"],
            "t_eap": d["wall_seconds"].get("eap_ig"),
            "eap_rho": d.get("spearman_ap_eap_ig"),
            "eap_mlp_r": (d.get("mlp_pearson_ap_eap_ig") or {}).get("pearson_r"),
            "eap_top14": (d.get("topk_overlap_ap_eap_ig") or {}).get("14"),
        }
        for s, v in (d.get("eap_sweep") or {}).items():
            row[f"eap{s}_rho"] = v["spearman_ap_eap_ig"]
            row[f"eap{s}_mlp_r"] = v["mlp_pearson_r"]
            row[f"eap{s}_t"] = v["wall_seconds"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["appendix"] = [is_appendix(m, t) for m, t in zip(df.model, df.task)]
    return df


def boot_ci(vals: np.ndarray, level: float = 0.95, n: int = N_BOOT) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    idx = RNG.integers(0, len(vals), size=(n, len(vals)))
    means = vals[idx].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def hier_boot(pair_vals: list[np.ndarray], level: float, n: int = N_BOOT) -> tuple[float, float, float]:
    """Resample pairs, then seeds within pair; statistic = mean over pairs of mean diff."""
    pair_vals = [np.asarray(v, float)[~np.isnan(v)] for v in pair_vals if len(v)]
    if not pair_vals:
        return (float("nan"),) * 3
    est = float(np.mean([v.mean() for v in pair_vals]))
    k = len(pair_vals)
    out = np.empty(n)
    for i in range(n):
        pi = RNG.integers(0, k, size=k)
        out[i] = np.mean([RNG.choice(pair_vals[j], size=len(pair_vals[j]), replace=True).mean() for j in pi])
    a = (1 - level) / 2
    return est, float(np.quantile(out, a)), float(np.quantile(out, 1 - a))


def seed_mean(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    keys = ["model", "task", "lang", "direction"]
    g = df.groupby(keys)
    out = g[cols].mean().reset_index()
    out["n_seeds"] = g.size().values
    return out


def md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_(none)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) and not np.isnan(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def paired_lang_diffs(df: pd.DataFrame, a_lang: str, b_lang: str, col: str) -> dict:
    """For each (model, task) with both languages (denoising), seed-paired differences a - b."""
    d = df[(df.direction == "denoising")]
    out = {}
    for (m, t), g in d.groupby(["model", "task"]):
        A = g[g.lang == a_lang].set_index("seed")[col]
        B = g[g.lang == b_lang].set_index("seed")[col]
        common = A.index.intersection(B.index)
        if len(common) >= 2:
            out[(m, t)] = (A.loc[common] - B.loc[common]).to_numpy(float)
    return out


def ols_ci(X: np.ndarray, y: np.ndarray, n: int = N_BOOT) -> dict:
    """OLS with intercept; pair-bootstrap 95% CI for coefficients."""
    A = np.column_stack([np.ones(len(y)), X])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    bs = np.empty((n, A.shape[1]))
    for i in range(n):
        idx = RNG.integers(0, len(y), size=len(y))
        bs[i] = np.linalg.lstsq(A[idx], y[idx], rcond=None)[0]
    lo, hi = np.nanquantile(bs, 0.025, axis=0), np.nanquantile(bs, 0.975, axis=0)
    return {"beta": beta.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(), "n": int(len(y))}


def spearman_boot(x: np.ndarray, y: np.ndarray, n: int = 5000) -> dict:
    from scipy.stats import spearmanr

    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    if len(x) < 4:
        return {"rho": float("nan"), "n": int(len(x))}
    r = float(spearmanr(x, y)[0])
    bs = []
    for _ in range(n):
        idx = RNG.integers(0, len(x), size=len(x))
        if len(set(x[idx])) > 1 and len(set(y[idx])) > 1:
            bs.append(spearmanr(x[idx], y[idx])[0])
    return {"rho": r, "ci_lo": float(np.quantile(bs, 0.025)), "ci_hi": float(np.quantile(bs, 0.975)), "n": int(len(x))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--grid-dir", type=Path, default=ROOT / "results" / "grid")
    p.add_argument("--gate", type=Path, default=ROOT / "results" / "gate_coverage.json")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "analysis")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)

    df = load_cells(args.grid_dir)
    if df.empty:
        raise SystemExit("no results in grid dir")
    df.to_csv(args.out / "cells.csv", index=False)
    summary: dict = {"n_runs": int(len(df))}
    md: list[str] = ["# Analysis (rules: PREREGISTRATION_v2.md, Amendment 1)\n"]

    # ---- Gate coverage first -------------------------------------------------------------
    if args.gate.exists():
        g = json.loads(args.gate.read_text())
        rows = g["rows"]
        gt = pd.DataFrame(
            [
                {
                    "model": r["model"], "task": r["task"], "lang": r["lang"], "tier": r.get("tier"),
                    "acc": r.get("clean_accuracy"), "clean_ld": r.get("clean_logit_diff"),
                    "single_tok": r.get("single_token_answer_rate"),
                    "tok/word": (r.get("descriptors") or {}).get("tokens_per_word"),
                    "why": "; ".join(r.get("reasons") or []),
                }
                for r in rows
            ]
        )
        md.append("## 0. Competence gate coverage\n")
        md.append(
            f"{int((gt['tier'] == 'strict').sum())} strict-pass, "
            f"{int((gt['tier'] == 'first_token').sum())} first-token tier (single-token criterion waived), "
            f"{int((gt['tier'] == 'fail').sum())} failed, of {len(gt)} cells.\n"
        )
        md.append(md_table(gt))
        summary["gate_passed"] = int((gt["tier"] == "strict").sum())
        summary["gate_first_token"] = int((gt["tier"] == "first_token").sum())
        summary["gate_total"] = int(len(gt))

    tier_map = {(r["model"], r["task"], r["lang"]): r.get("tier", "strict") for r in (g["rows"] if args.gate.exists() else [])}
    df["tier"] = [tier_map.get((m, t, l), "strict") for m, t, l in zip(df.model, df.task, df.lang)]
    # Main tests: strict-tier core cells only. First-token-tier cells enter a sensitivity table.
    core = df[~df.appendix & (df.tier == "strict")]
    ft = df[~df.appendix & (df.tier == "first_token") & (df.direction == "denoising")]
    if len(ft):
        md.append("## 0b. First-token-tier cells (sensitivity set, not in main tests)\n")
        md.append(md_table(seed_mean(ft, ["rho", "top14", "mlp_r", "clean_ld", "tokens_per_word"])))
    den = core[core.direction == "denoising"]
    cm = seed_mean(den, ["rho", "rho_restricted", "top14", "mlp_r", "clean_ld", "tokens_per_word", "concentration"])
    sd = den.groupby(["model", "task", "lang"])["rho"].std().reset_index().rename(columns={"rho": "rho_sd"})
    cm = cm.merge(sd, on=["model", "task", "lang"], how="left")
    md.append("## 1. Agreement per core cell (denoising, mean over seeds)\n")
    md.append(md_table(cm[["model", "task", "lang", "n_seeds", "rho", "rho_sd", "rho_restricted", "top14", "mlp_r", "clean_ld", "tokens_per_word"]]))

    # ---- H1 / H2 -----------------------------------------------------------------------------
    h1_frac = float((cm.rho >= 0.90).mean())
    h2_frac = float((cm.mlp_r < 0).mean())
    summary["H1"] = {"fraction_cells_rho_ge_0.90": h1_frac, "n_cells": int(len(cm)),
                     "failing": cm[cm.rho < 0.90][["model", "task", "lang"]].to_dict("records"),
                     "holds": bool((cm.rho >= 0.90).all())}
    summary["H2"] = {"fraction_cells_mlp_r_lt_0": h2_frac, "holds": bool(h2_frac > 0.5)}
    verd = den.groupby(["model", "task", "lang"])["verdict"].agg(lambda s: s.value_counts().index[0]).reset_index()
    md.append("## 2. H1 and H2\n")
    md.append(f"- H1: {h1_frac:.0%} of {len(cm)} core cells have seed-mean head rho >= 0.90 "
              f"({'HOLDS' if summary['H1']['holds'] else 'DOES NOT HOLD'} in all cells).\n")
    md.append(f"- H2: MLP r(AP, AttrP) < 0 in {h2_frac:.0%} of cells "
              f"({'HOLDS' if summary['H2']['holds'] else 'DOES NOT HOLD'}).\n")
    md.append("Most common verdict per cell:\n\n" + md_table(verd))

    # ---- H3 --------------------------------------------------------------------------------
    md.append("## 3. H3: Hindi vs English (paired over seeds, core, denoising)\n")
    diffs = paired_lang_diffs(den, "hi", "en", "rho")
    if diffs:
        rows = []
        for (m, t), v in sorted(diffs.items()):
            lo, hi = boot_ci(v, 0.90)
            rows.append({"model": m, "task": t, "mean_rho_hi_minus_en": float(v.mean()), "ci90_lo": lo, "ci90_hi": hi, "n_seeds": len(v)})
        pt = pd.DataFrame(rows)
        est, lo90, hi90 = hier_boot(list(diffs.values()), 0.90)
        _, lo95, hi95 = hier_boot(list(diffs.values()), 0.95)
        verdict = (
            "EQUIVALENT within +/-0.05" if (lo90 > -MARGIN and hi90 < MARGIN)
            else "HINDI WORSE (CI entirely below -0.05)" if hi90 < -MARGIN
            else "HINDI BETTER (CI entirely above +0.05)" if lo90 > MARGIN
            else "INCONCLUSIVE (CI straddles the margin)"
        )
        summary["H3"] = {"pooled_mean_diff": est, "ci90": [lo90, hi90], "ci95": [lo95, hi95],
                         "margin": MARGIN, "verdict": verdict, "n_pairs": len(diffs),
                         "hindi_lower_ci95_upper_lt_0": bool(hi95 < 0)}
        md.append(md_table(pt))
        md.append(f"Pooled mean(rho_hi - rho_en) = {est:+.3f}; 90% CI [{lo90:+.3f}, {hi90:+.3f}]; "
                  f"95% CI [{lo95:+.3f}, {hi95:+.3f}] over {len(diffs)} pairs.\n\n**Verdict: {verdict}.**\n")
    else:
        md.append("_No matched en/hi pairs available._\n")

    # ---- H4 --------------------------------------------------------------------------------
    md.append("## 4. H4: tokenizer fertility and clean logit difference\n")
    cellm = cm.copy()
    h4 = {
        "rho_vs_tokens_per_word": spearman_boot(cellm.tokens_per_word.to_numpy(float), cellm.rho.to_numpy(float)),
        "rho_vs_clean_ld": spearman_boot(cellm.clean_ld.to_numpy(float), cellm.rho.to_numpy(float)),
    }
    pair_rows = []
    for (m, t), g in den.groupby(["model", "task"]):
        a, b = g[g.lang == "hi"], g[g.lang == "en"]
        if len(a) and len(b):
            pair_rows.append({
                "d_rho": a.rho.mean() - b.rho.mean(),
                "d_tpw": a.tokens_per_word.mean() - b.tokens_per_word.mean(),
                "d_cld": a.clean_ld.mean() - b.clean_ld.mean(),
            })
    if len(pair_rows) >= 5:
        pr = pd.DataFrame(pair_rows)
        h4["paired_ols_d_rho_on_[d_tpw,d_cld]"] = ols_ci(pr[["d_tpw", "d_cld"]].to_numpy(float), pr.d_rho.to_numpy(float))
        h4["paired_ols_intercept_is_language_effect_after_controls"] = True
    summary["H4"] = h4
    md.append("```json\n" + json.dumps(h4, indent=1) + "\n```\n")
    md.append("Interpretation guide: in the paired OLS the intercept is the Hindi-minus-English difference in rho that "
              "remains at equal fertility and equal clean logit difference.\n")

    # ---- script vs language ----------------------------------------------------------------
    md.append("## 5. Script vs language (IOI)\n")
    sl = {}
    for name, (a, b) in {"hi_minus_en": ("hi", "en"), "hi_roman_minus_en": ("hi_roman", "en"), "hi_minus_hi_roman": ("hi", "hi_roman")}.items():
        dd = {k: v for k, v in paired_lang_diffs(den[den.task == "ioi"], a, b, "rho").items()}
        if dd:
            est, lo, hi = hier_boot(list(dd.values()), 0.95)
            sl[name] = {"mean": est, "ci95": [lo, hi], "n_models": len(dd)}
    summary["script_vs_language"] = sl
    md.append("```json\n" + json.dumps(sl, indent=1) + "\n```\n")

    # ---- H5 / EAP-IG -----------------------------------------------------------------------
    md.append("## 6. H5: EAP-IG vs first-order AttrP\n")
    e = den.dropna(subset=["eap_mlp_r"])
    if len(e):
        d_mlp = (e.eap_mlp_r - e.mlp_r).to_numpy(float)
        d_head = (e.eap_rho - e.rho).to_numpy(float)
        h5 = {
            "mean_mlp_r_attrp": float(e.mlp_r.mean()), "mean_mlp_r_eap": float(e.eap_mlp_r.mean()),
            "mlp_r_gain_mean": float(d_mlp.mean()), "mlp_r_gain_ci95": list(boot_ci(d_mlp)),
            "head_rho_gain_mean": float(d_head.mean()), "head_rho_gain_ci95": list(boot_ci(d_head)),
            "time_ratio_eap_over_attrp": float((e.t_eap / e.t_attr).median()),
            "time_ratio_ap_over_attrp": float((e.t_ap / e.t_attr).median()),
        }
        h5["holds"] = bool(h5["mlp_r_gain_ci95"][0] > 0)
        sw = {}
        for s in ("3", "5", "10", "20"):
            if f"eap{s}_rho" in den:
                x = den[f"eap{s}_rho"].dropna()
                if len(x):
                    sw[s] = {"mean_head_rho": float(x.mean()), "mean_mlp_r": float(den[f"eap{s}_mlp_r"].dropna().mean()), "n": int(len(x))}
        h5["step_sweep_seed0"] = sw
        summary["H5"] = h5
        md.append("```json\n" + json.dumps(h5, indent=1) + "\n```\n")

    # ---- noising & appendix ----------------------------------------------------------------
    nz = core[core.direction == "noising"]
    if len(nz):
        nzm = seed_mean(nz, ["rho", "top14", "mlp_r"])
        md.append("## 7. Noising direction (core, seed mean)\n" + md_table(nzm))
        summary["noising_mean_rho"] = float(nzm.rho.mean())
    app = df[df.appendix & (df.direction == "denoising")]
    if len(app):
        am = seed_mean(app, ["rho", "top14", "mlp_r", "clean_ld"])
        md.append("## 8. Appendix: English robustness (Pythia, induction)\n" + md_table(am))

    # ---- figures ---------------------------------------------------------------------------
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    labels = [f"{r.model.split('/')[-1]}\n{r.task}/{r.lang}" for r in cm.itertuples()]
    ax.bar(range(len(cm)), cm.rho, yerr=cm.rho_sd.fillna(0), color=["#c0392b" if l.endswith("hi") or "roman" in l else "#2c7fb8" for l in [f"{r.lang}" for r in cm.itertuples()]], capsize=2)
    ax.axhline(0.9, ls="--", c="k", lw=0.8)
    ax.set_xticks(range(len(cm)), labels, rotation=90, fontsize=6)
    ax.set_ylabel("head Spearman rho (AP vs AttrP)")
    ax.set_title("Agreement per core cell (blue=en, red=hi/hi_roman)")
    fig.tight_layout()
    fig.savefig(args.out / "figures" / "rho_by_cell.png", dpi=160)
    plt.close(fig)

    if diffs:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for (m, t), v in diffs.items():
            en = cm[(cm.model == m) & (cm.task == t) & (cm.lang == "en")].rho.iloc[0]
            hi = cm[(cm.model == m) & (cm.task == t) & (cm.lang == "hi")].rho.iloc[0]
            ax.scatter(en, hi, s=25)
            ax.annotate(f"{m.split('/')[-1][:8]}/{t[:4]}", (en, hi), fontsize=5)
        lim = [min(cm.rho.min(), 0.5), 1.0]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.fill_between(lim, [x - MARGIN for x in lim], [x + MARGIN for x in lim], alpha=0.15)
        ax.set_xlabel("rho English")
        ax.set_ylabel("rho Hindi")
        fig.tight_layout()
        fig.savefig(args.out / "figures" / "en_vs_hi.png", dpi=160)
        plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col, lab in ((axs[0], "tokens_per_word", "tokens per word"), (axs[1], "clean_ld", "clean logit diff")):
        ax.scatter(cm[col], cm.rho, c=["#c0392b" if l != "en" else "#2c7fb8" for l in cm.lang], s=18)
        ax.set_xlabel(lab)
        ax.set_ylabel("rho")
    fig.tight_layout()
    fig.savefig(args.out / "figures" / "rho_vs_descriptors.png", dpi=160)
    plt.close(fig)

    if len(e):
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(e.mlp_r, e.eap_mlp_r, s=10, alpha=0.6)
        ax.plot([-1, 1], [-1, 1], "k--", lw=0.8)
        ax.set_xlabel("MLP r (AttrP)")
        ax.set_ylabel("MLP r (EAP-IG, 5 steps)")
        fig.tight_layout()
        fig.savefig(args.out / "figures" / "mlp_attrp_vs_eap.png", dpi=160)
        plt.close(fig)

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (args.out / "ANALYSIS.md").write_text("\n".join(md))
    print("\n".join(md)[:6000])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
