#!/usr/bin/env python3
"""Generate data/NATIVE_REVIEW.md: a concrete, fast review sheet (no torch needed)."""
import ast
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def consts(path: Path) -> dict:
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                out[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:  # noqa: BLE001
                pass
    return out


ioi = consts(ROOT / "src" / "data_ioi.py")
cap = consts(ROOT / "src" / "data_capitals.py")
rng = random.Random(7)

L = []
L += ["# Native-speaker review sheet (Hindi) - about 10 minutes", "",
      "**How to review:** read each line; do nothing if it is fine. Only write a short note next to a line that is wrong or unnatural "
      "(put `x` in the last column and the fix after it). Templated probe sentences may sound slightly stiff - that is OK; "
      "mark only things a native speaker would call *wrong* (grammar, gender, meaning) or *confusing*.", "",
      "Why it matters: results on Hindi IOI/capitals depend on these sentences. Note that every model failed the competence gate on "
      "Hindi IOI and capitals, so a template problem could be part of the reason - which is exactly what this review can tell us.", ""]

# ---- IOI ----
L += ["## 1. Hindi IOI (20 sample sentences)", "",
      "Probe prompt ends **before** the recipient (shown in **bold**); the model must predict that name.", "",
      "| # | Full sentence | x / fix |", "|--:|---|---|"]
names, places, objs = ioi["NAMES_HI"], ioi["PLACES_HI"], ioi["OBJECTS_HI"]
for i in range(20):
    t = ioi["TEMPLATES_HI"][i % 4]
    x, y, z = rng.sample(names, 3)
    io = y
    s = t.format(X=x, Y=y, Z=z, PLACE=rng.choice(places), OBJ=rng.choice(objs)) + f" **{io}** को दिया।"
    L.append(f"| {i+1} | {s} |  |")
L += ["", "**Known concern to judge:** the verb is always `दिया` (masc.). With feminine objects (किताब, अंगूठी, मिठाई) it should be `दी`. "
      "The verb is *after* the cut, so the model never sees it - but is the sentence acceptable as a probe? ( ) fine  ( ) must fix", "",
      "**Names** (common given names? mark rare/odd ones): " + ", ".join(names), "",
      "**Places:** " + ", ".join(places) + "  \n**Objects:** " + ", ".join(objs), "",
      "Templates:", ""]
for t in ioi["TEMPLATES_HI"]:
    L.append(f"- `{t}`")
L += ["", "Romanised variants (same structure, Latin script; tick if the romanisation is a normal way to write it):", ""]
for t in ioi["TEMPLATES_HI_ROMAN"]:
    L.append(f"- `{t}`")

# ---- capitals ----
L += ["", "## 2. Capitals (30 pairs): is the Hindi spelling the usual one?", "",
      "| Country | Capital | x / fix |", "|---|---|---|"]
for c, k in cap["PAIRS_HI"]:
    L.append(f"| {c} | {k} |  |")
L += ["", "Templates (`{COUNTRY}` filled in):", ""]
for t in cap["TEMPLATES_HI"]:
    L.append(f"- `{t}`")

# ---- MultiBLiMP ----
rows = list(csv.DictReader(open(ROOT / "data/multiblimp/hin_data.tsv", encoding="utf-8"), delimiter="\t"))


def div(a, b):
    a, b = a.split(), b.split()
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return (i, a, b) if 0 < i < min(len(a), len(b)) else None


samp = []
for r in rng.sample(rows, len(rows)):
    d = div(r["sen"], r["wrong_sen"])
    if d:
        samp.append(d)
    if len(samp) == 25:
        break
L += ["", "## 3. MultiBLiMP spot check (25 random pairs, from the published dataset)", "",
      "Left must be the grammatical one and right the ungrammatical one **at the word shown**. Mark any pair where the left is not clearly correct "
      "or both are acceptable.", "",
      "| # | Context (last words) | Correct | Wrong | x |", "|--:|---|---|---|---|"]
for n, (i, a, b) in enumerate(samp, 1):
    ctx = " ".join(a[max(0, i - 6):i])
    L.append(f"| {n} | …{ctx} | **{a[i]}** | {b[i]} |  |")

L += ["", "## Sign-off", "",
      "Overall: ( ) templates fine  ( ) fine after the fixes noted  ( ) not usable", "",
      "Reviewer: ______________  Date: ______________", ""]
(ROOT / "data" / "NATIVE_REVIEW.md").write_text("\n".join(L))
print("wrote data/NATIVE_REVIEW.md")
