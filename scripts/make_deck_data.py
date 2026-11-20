#!/usr/bin/env python3
"""Export slides/results/deck_data.json from analysis artifacts (stdlib only)."""

from __future__ import annotations

import ast
import csv
import json
import statistics as st
import struct
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "slides" / "results" / "deck_data.json"


def load_npz(path: Path) -> dict[str, tuple[tuple[int, ...], list[float]]]:
    out: dict[str, tuple[tuple[int, ...], list[float]]] = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            raw = z.read(name)
            if raw[:6] != b"\x93NUMPY":
                continue
            major = raw[6]
            if major == 1:
                header_len = struct.unpack("<H", raw[8:10])[0]
                offset = 10
            else:
                header_len = struct.unpack("<I", raw[8:12])[0]
                offset = 12
            header = ast.literal_eval(raw[offset : offset + header_len].decode("latin1"))
            if header["descr"] != "<f8":
                continue
            count = 1
            for dim in header["shape"]:
                count *= dim
            data = raw[offset + header_len :]
            vals = list(struct.unpack(f"<{count}d", data[: 8 * count]))
            out[name[:-4]] = (tuple(header["shape"]), vals)
    return out


def mean(xs: list[float]) -> float:
    return float(st.mean(xs)) if xs else float("nan")


def short_model(name: str) -> str:
    return name.split("/")[-1]


def main() -> None:
    cells = list(csv.DictReader((ROOT / "results" / "analysis" / "cells.csv").open()))
    summary = json.loads((ROOT / "results" / "analysis" / "summary.json").read_text())
    gate = json.loads((ROOT / "results" / "gate_coverage.json").read_text())

    def f(row: dict, key: str) -> float | None:
        v = row.get(key, "")
        if v in ("", "nan", None):
            return None
        try:
            return float(v)
        except ValueError:
            return None

    # Strict-tier cells only for main tests (exclude first_token sensitivity set)
    strict_keys = {
        (r["model"], r["task"], r["lang"])
        for r in gate["rows"]
        if r.get("tier") == "strict"
    }
    # Core denoising cells: appendix=False AND strict gate
    core_den = [
        r
        for r in cells
        if r["direction"] == "denoising"
        and r["appendix"].lower() == "false"
        and (r["model"], r["task"], r["lang"]) in strict_keys
    ]
    # Group by (model, task, lang)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in core_den:
        grouped[(r["model"], r["task"], r["lang"])].append(r)

    h1_bars = []
    for (model, task, lang), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        rhos = [f(r, "rho") for r in rows]
        rhos = [x for x in rhos if x is not None]
        h1_bars.append(
            {
                "model": short_model(model),
                "model_full": model,
                "task": task,
                "lang": lang,
                "label": f"{short_model(model)}\n{task}/{lang}",
                "rho": mean(rhos),
                "n_seeds": len(rhos),
                "pass_h1": mean(rhos) >= 0.90 if rhos else False,
            }
        )

    # H3: BLOOM agreement en vs hi
    h3_pairs = []
    for size in ("bloom-560m", "bloom-1b1", "bloom-1b7"):
        full = f"bigscience/{size}"
        en = [f(r, "rho") for r in grouped.get((full, "agreement", "en"), [])]
        hi = [f(r, "rho") for r in grouped.get((full, "agreement", "hi"), [])]
        en = [x for x in en if x is not None]
        hi = [x for x in hi if x is not None]
        h3_pairs.append(
            {
                "model": size,
                "rho_en": mean(en),
                "rho_hi": mean(hi),
                "diff_hi_minus_en": mean(hi) - mean(en) if en and hi else float("nan"),
                "n_seeds": min(len(en), len(hi)),
            }
        )

    # Wall times over the core denoising runs (strict tier, appendix=False; one run = one seed)
    t_ap, t_attr, t_eap = [], [], []
    for r in core_den:
        a, b, c = f(r, "t_ap"), f(r, "t_attr"), f(r, "t_eap")
        if a and b and b > 0:
            t_ap.append(a)
            t_attr.append(b)
        if c and b and b > 0:
            t_eap.append(c)

    # Reported cost ratios come from summary.json (scripts/analyze_story.py, H5): the MEDIAN over
    # the same core denoising runs of the per-run time ratio t_ap/t_attr and t_eap/t_attr, so the
    # deck, ANALYSIS.md and WRITEUP.md quote one figure. The ratio of mean times is kept alongside
    # for reference only (it is what the mean seconds on slide 2 imply).
    h5 = summary.get("H5", {})
    means_ap = mean(t_ap) / mean(t_attr) if t_attr else float("nan")
    means_eap = mean(t_eap) / mean(t_attr) if t_attr and t_eap else float("nan")
    cost = {
        "ap_over_attrp": h5.get("time_ratio_ap_over_attrp", means_ap),
        "eap_over_attrp": h5.get("time_ratio_eap_over_attrp", means_eap),
        "ratio_definition": "median over core denoising runs of the per-run wall-time ratio (summary.json H5)",
        "ap_over_attrp_ratio_of_means": means_ap,
        "eap_over_attrp_ratio_of_means": means_eap,
        "n_runs_timed": len(t_attr),
        "mean_t_ap": mean(t_ap),
        "mean_t_attr": mean(t_attr),
        "mean_t_eap": mean(t_eap),
    }

    # Gate grid
    gate_rows = []
    for r in gate["rows"]:
        gate_rows.append(
            {
                "model": short_model(r["model"]),
                "model_full": r["model"],
                "task": r["task"],
                "lang": r["lang"],
                "tier": r.get("tier") or ("strict" if r.get("pass") else "fail"),
                "acc": r.get("clean_accuracy"),
                "clean_ld": r.get("clean_logit_diff"),
                "single_tok": r.get("single_token_answer_rate"),
                "why": "; ".join(r.get("reasons") or []) or "ok",
            }
        )

    n_strict = sum(1 for r in gate_rows if r["tier"] == "strict")
    n_first = sum(1 for r in gate_rows if r["tier"] == "first_token")
    n_fail = sum(1 for r in gate_rows if r["tier"] == "fail")

    # MLP layer series: mean over seeds for gpt2 IOI en denoising
    def mlp_series(model_safe: str, task: str = "ioi", lang: str = "en") -> dict:
        ap_acc: list[list[float]] = []
        at_acc: list[list[float]] = []
        eap_acc: list[list[float]] = []
        for seed in range(5):
            p = ROOT / "results" / "grid" / f"{model_safe}__{task}__{lang}__denoising__seed{seed}" / "scores.npz"
            if not p.exists():
                continue
            d = load_npz(p)
            ap_acc.append(d["ap_mlps"][1])
            at_acc.append(d["attr_mlps"][1])
            key = "eap5_mlps" if "eap5_mlps" in d else "eap_mlps"
            eap_acc.append(d[key][1])
        n_layers = len(ap_acc[0]) if ap_acc else 0
        return {
            "n_seeds": len(ap_acc),
            "n_layers": n_layers,
            "ap": [mean([row[i] for row in ap_acc]) for i in range(n_layers)],
            "attr": [mean([row[i] for row in at_acc]) for i in range(n_layers)],
            "eap": [mean([row[i] for row in eap_acc]) for i in range(n_layers)],
        }

    mlp_gpt2 = mlp_series("gpt2")
    mlp_qwen = mlp_series("Qwen__Qwen2.5-0.5B")
    mlp_pythia = mlp_series("EleutherAI__pythia-160m")

    # H5 / EAP exploratory: GPT-2 IOI seed-mean head rho and mlp_r
    def cell_means(model: str, task: str, lang: str) -> dict:
        rows = grouped.get((model, task, lang), [])
        return {
            "rho": mean([x for x in (f(r, "rho") for r in rows) if x is not None]),
            "eap_rho": mean([x for x in (f(r, "eap_rho") for r in rows) if x is not None]),
            "mlp_r": mean([x for x in (f(r, "mlp_r") for r in rows) if x is not None]),
            "eap_mlp_r": mean([x for x in (f(r, "eap_mlp_r") for r in rows) if x is not None]),
            "n": len(rows),
        }

    gpt2_ioi = cell_means("gpt2", "ioi", "en")
    qwen_ioi = cell_means("Qwen/Qwen2.5-0.5B", "ioi", "en")
    # BLOOM IOI (en): seed-mean MLP Pearson r(AP, AttrP) per size, quoted in the slide 7 notes
    bloom_ioi_mlp_r = {
        size: cell_means(f"bigscience/{size}", "ioi", "en")["mlp_r"]
        for size in ("bloom-560m", "bloom-1b1", "bloom-1b7")
    }

    # Clean-accuracy range of the failing Hindi IOI / Hindi capitals / romanised IOI gate cells
    hi_fail_acc = [
        r["acc"]
        for r in gate_rows
        if r["tier"] == "fail"
        and (r["task"], r["lang"]) in {("ioi", "hi"), ("capitals", "hi"), ("ioi", "hi_roman")}
        and r["acc"] is not None
    ]

    # Noising mean rho over core cells (strict tier, appendix=False), as in summary.json
    noising = [
        f(r, "rho")
        for r in cells
        if r["direction"] == "noising"
        and r["appendix"].lower() == "false"
        and (r["model"], r["task"], r["lang"]) in strict_keys
    ]
    noising = [x for x in noising if x is not None]

    # Fertility / clean LD for H4 backup
    h4_points = []
    for (model, task, lang), rows in grouped.items():
        rhos = [f(r, "rho") for r in rows]
        tpw = [f(r, "tokens_per_word") for r in rows]
        cld = [f(r, "clean_ld") for r in rows]
        rhos = [x for x in rhos if x is not None]
        tpw = [x for x in tpw if x is not None]
        cld = [x for x in cld if x is not None]
        h4_points.append(
            {
                "model": short_model(model),
                "task": task,
                "lang": lang,
                "rho": mean(rhos),
                "tokens_per_word": mean(tpw),
                "clean_ld": mean(cld),
                "is_hindi": lang == "hi",
            }
        )

    payload = {
        "meta": {
            "n_runs": summary.get("n_runs", len(cells)),
            "n_models": 9,
            "n_gate_cells": len(gate_rows),
            "n_strict": n_strict,
            "n_first_token": n_first,
            "n_fail": n_fail,
            "n_runnable": n_strict + n_first,
            "n_core_cells": len(grouped),
            "author": "Hrishiraj Mitra",
            "course": "Mechanistic Interpretability mini-project · IIIT Hyderabad",
        },
        "cost": cost,
        "h1": {
            "bars": h1_bars,
            "fraction_pass": summary.get("H1", {}).get("fraction_cells_rho_ge_0.90"),
            "n_cells": summary.get("H1", {}).get("n_cells"),
            "failing": summary.get("H1", {}).get("failing", []),
            "threshold": 0.90,
        },
        "h2": summary.get("H2", {}),
        "h3": {
            "pairs": h3_pairs,
            "pooled_mean_diff": summary.get("H3", {}).get("pooled_mean_diff"),
            "ci95": summary.get("H3", {}).get("ci95"),
            "verdict": summary.get("H3", {}).get("verdict"),
            "n_pairs": summary.get("H3", {}).get("n_pairs"),
            "margin": summary.get("H3", {}).get("margin", 0.05),
        },
        "h4": summary.get("H4", {}),
        "h4_points": h4_points,
        "h5": summary.get("H5", {}),
        "gate": gate_rows,
        "mlp_layers": {
            "gpt2_ioi": mlp_gpt2,
            "qwen05_ioi": mlp_qwen,
            "pythia160_ioi": mlp_pythia,
            "claim": (
                "Across 5 seeds, GPT-2 / Qwen2.5-0.5B / Pythia-160m&410m IOI: "
                "MLP layer 0 is where AP and AttrP disagree; remaining layers match."
            ),
        },
        "eap_examples": {
            "gpt2_ioi": gpt2_ioi,
            "qwen05_ioi": qwen_ioi,
        },
        "bloom_ioi_mlp_r": bloom_ioi_mlp_r,
        "hindi_fail_acc": {
            "min": min(hi_fail_acc),
            "max": max(hi_fail_acc),
            "n_cells": len(hi_fail_acc),
            "scope": "failing ioi/hi, capitals/hi and ioi/hi_roman cells in results/gate_coverage.json",
        },
        "noising_mean_rho": summary.get("noising_mean_rho", mean(noising)),
        "summary": summary,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
    print(
        f"H1 {sum(1 for b in h1_bars if b['pass_h1'])}/{len(h1_bars)} pass; "
        f"gate strict={n_strict} first={n_first} fail={n_fail}; "
        f"cost AP/AttrP={cost['ap_over_attrp']:.1f}x EAP/AttrP={cost['eap_over_attrp']:.1f}x"
    )
    print(
        f"GPT-2 IOI L0: AP={mlp_gpt2['ap'][0]:+.3f} Attr={mlp_gpt2['attr'][0]:+.3f} "
        f"EAP={mlp_gpt2['eap'][0]:+.3f}"
    )


if __name__ == "__main__":
    main()
