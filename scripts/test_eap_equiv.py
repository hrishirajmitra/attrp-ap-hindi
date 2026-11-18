#!/usr/bin/env python3
"""Sanity: EAP-IG at alpha=0 (denoising) / alpha=0 (noising) must equal first-order AttrP."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.attribution_patch import attribution_patch_heads_and_mlps as attr
from src.data import generate_batch
from src.eap_ig import eap_ig_heads_and_mlps as eap
from src.modelload import load_model

m = load_model("gpt2", "cuda")
ok = True
for task, lang in (("ioi", "en"), ("agreement", "en")):
    b = generate_batch(m, task, lang, n=40, seed=0, device="cuda")
    for d in ("denoising", "noising"):
        a = attr(m, b, direction=d, micro_batch=20)
        e0 = eap(m, b, direction=d, micro_batch=20, alphas=[0.0])
        e5 = eap(m, b, direction=d, micro_batch=20, n_steps=5)
        e50 = eap(m, b, direction=d, micro_batch=20, n_steps=50)
        diff = max(np.abs(a["head_scores"] - e0["head_scores"]).max(), np.abs(a["mlp_scores"] - e0["mlp_scores"]).max())
        from scipy.stats import spearmanr
        r5 = spearmanr(a["head_scores"].ravel(), e5["head_scores"].ravel())[0]
        r50 = spearmanr(a["head_scores"].ravel(), e50["head_scores"].ravel())[0]
        print(f"{task} {d}: max|AttrP-EAP(alpha=0)|={diff:.2e}  rho(AttrP,EAP5)={r5:.3f} rho(AttrP,EAP50)={r50:.3f} "
              f"sum(AttrP)={a['head_scores'].sum()+a['mlp_scores'].sum():.3f} sum(EAP50)={e50['head_scores'].sum()+e50['mlp_scores'].sum():.3f}", flush=True)
        ok &= diff < 1e-5
print("EAP_EQUIV_OK" if ok else "EAP_EQUIV_FAILED")
