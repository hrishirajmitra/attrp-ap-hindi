# Attribution Patching vs Activation Patching — multi-cell extension

Mini-project **A**, extended: same question (does AttrP reproduce AP, and where does it fail?) across **English/Hindi**, **four tasks**, and **≤2B models**, with a competence gate matching tasks to models.

- v1 (done): GPT-2 small × IOI — [WRITEUP.md](WRITEUP.md), [PREREGISTRATION.md](PREREGISTRATION.md)
- v2 (grid): [PREREGISTRATION_v2.md](PREREGISTRATION_v2.md)

## Layout

```
mp/
├── PREREGISTRATION.md          # v1 locked protocol
├── PREREGISTRATION_v2.md       # multi-cell protocol + H1–H4
├── src/
│   ├── batch.py                # TaskBatch
│   ├── data.py                 # factory
│   ├── data_ioi.py / data_capitals.py / data_agreement.py / data_induction.py
│   ├── activation_patch.py     # AP (+ denoising/noising)
│   ├── attribution_patch.py    # AttrP
│   ├── eap_ig.py               # EAP-IG (+ step sweep)
│   ├── compare.py              # rho, restricted rho, top-k overlap
│   ├── registry.py             # models, tasks, gate thresholds
│   └── tokenize.py
├── scripts/
│   ├── tokenizer_audit.py
│   ├── gate.py                 # competence gate → coverage table
│   ├── make_cell_list.py       # expand gated cells → array jobs
│   ├── run_all.py              # --model --task --lang --direction
│   ├── aggregate_seeds.py      # v1 + grid + H3 paired test
│   └── ...
├── slurm/
│   ├── setup_env.sbatch
│   ├── run_gate.sbatch
│   ├── run_grid.sbatch         # SLURM array over gated cells
│   └── run_all.sbatch          # legacy GPT-2 IOI
└── results/
```

## Ada workflow (v2)

```bash
# See scripts/ada_workflow.sh for the full sequence.
sbatch ~/mp/slurm/setup_env.sbatch
NODE=$(cat ~/mp/.compute_node)

sbatch --nodelist=$NODE ~/mp/slurm/run_validate.sbatch   # engine sanity (+ optional EAP-IG)
sbatch --nodelist=$NODE ~/mp/slurm/run_gate.sbatch       # tokenizer + competence gate

# after gate: lock cells (no --allow-expected)
$VENV/bin/python scripts/make_cell_list.py --directions denoising noising
N=$(python3 -c "import json; print(json.load(open('results/gated_cells.json'))['n_jobs'])")
sbatch --nodelist=$NODE --array=0-$((N-1)) ~/mp/slurm/run_grid.sbatch   # stage 1

sbatch --nodelist=$NODE ~/mp/slurm/run_stage2.sbatch   # language compare + logit lens
```

Offline planning (no GPU/torch):

```bash
python3 scripts/gate.py --offline
python3 scripts/tokenizer_audit.py --offline
python3 scripts/make_cell_list.py --allow-expected --directions denoising
```

## Local smoke (CPU, tiny N)

```bash
python scripts/run_all.py --model gpt2 --task ioi --lang en \
  --n-prompts 4 --device cpu --out results/smoke --skip-eap
```

## Status

| Phase | Status |
|-------|--------|
| v1 GPT-2 IOI (5 seeds × abc/flip) | Done — see WRITEUP.md |
| Pipeline generalisation | Done |
| Offline expected gate / tokenizer notes | Done — `results/gate_coverage.*`, `results/tokenizer_audit.*` |
| Ada gate + grid runs | Pending (needs cluster) |
| Grid writeup numbers | Pending Ada results |
