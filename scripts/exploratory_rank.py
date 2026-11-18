#!/usr/bin/env python3
"""POST-HOC exploratory analysis (not pre-registered), across seeds 0-4: rank agreement
restricted to heads that matter, to test whether all-head Spearman is dominated by near-zero noise."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

root = Path(sys.argv[1])
def agg(xs): return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)), "min": float(np.min(xs)), "max": float(np.max(xs))}
out = {}
for cond in ("abc", "flip"):
    acc = {}
    for s in range(5):
        z = np.load(root / f"full_{cond}_seed{s}" / "scores.npz")
        ap, at, ig = (z[k].ravel() for k in ("ap_heads", "attr_heads", "eap_heads"))
        acc.setdefault("frac_heads_|AP|<0.02", []).append(float((np.abs(ap) < 0.02).mean()))
        for thr in (0.02, 0.05):
            m = np.abs(ap) > thr
            acc.setdefault(f"n_heads_|AP|>{thr}", []).append(int(m.sum()))
            acc.setdefault(f"spearman_attr_|AP|>{thr}", []).append(float(spearmanr(ap[m], at[m])[0]))
            acc.setdefault(f"spearman_eap_|AP|>{thr}", []).append(float(spearmanr(ap[m], ig[m])[0]))
        for k in (10, 20):
            top = set(np.argsort(-np.abs(ap))[:k])
            acc.setdefault(f"top{k}_overlap_attr", []).append(len(top & set(np.argsort(-np.abs(at))[:k])) / k)
            acc.setdefault(f"top{k}_overlap_eap", []).append(len(top & set(np.argsort(-np.abs(ig))[:k])) / k)
    out[cond] = {k: agg(v) for k, v in acc.items()}
(root / "exploratory_rank.json").write_text(json.dumps(out, indent=2))
for c, d in out.items():
    print(c)
    for k, v in d.items():
        print(f"  {k}: {v['mean']:.3f} ± {v['sd']:.3f} [{v['min']:.2f}, {v['max']:.2f}]")
