# scripts/

Status of each script. "Final" means it produced files that the report or results depend on.

## Final study (multi-cell grid)

| Script | Purpose |
|---|---|
| `tokenizer_audit.py` | TransformerLens loadability and Hindi/English tokenisation per model (`results/TOKENIZER_NOTES.md`) |
| `gate.py` | Competence gate for one model's cells (run through `slurm/gate_model.sbatch`) |
| `merge_gate.py` | Merge per-model gate files into `results/gate_coverage.*` and lock `results/gated_cells.json` |
| `queue_init.py`, `worker.py` | Shared task queue over gated cells and the workers that pull from it |
| `run_all.py` | One (model, task, language, direction, seed) run: AP, AttrP and EAP-IG |
| `test_eap_equiv.py` | Check that EAP-IG at alpha = 0 equals AttrP (output: `results/eap_equiv_check.txt`) |
| `analyze_story.py` | Final analysis: hypotheses H1 to H5, tables, figures (`results/analysis/`) |
| `make_deck_data.py` | Export `slides/results/deck_data.json` for the deck |
| `make_report_assets.py` | Numbers, tables and figures for `report/report.tex` (light; needs matplotlib and numpy only) |
| `make_review_sheet.py` | Generate `data/NATIVE_REVIEW.md` |

## GPT-2 IOI pilot (v1)

`run_baseline.py`, `run_intervention.py`, `aggregate_seeds.py`, `exploratory_rank.py`, `write_report.py`, `validate_v1.py` (compares the new engine with the pilot scores).

## Early or auxiliary, no result files in this repository

| Script | Note |
|---|---|
| `make_cell_list.py` | Early array-job cell list; the final runs used `queue_init.py` |
| `smoke_ada.py`, `validate_engine.py` | Smoke test, and an optional cross-check against an external EAP library (it skipped: no torch on the machine it ran on; see `results/ENGINE_DECISION.md`) |
| `compare_languages.py`, `logit_lens.py`, `ada_workflow.sh` | Planned stage-2 analyses (language head-profile comparison, logit lens). Written but not part of the reported results |
