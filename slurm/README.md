# slurm/

Job scripts for the Ada cluster (SLURM, one RTX 2080 Ti per job). Log paths use `%u` and scratch paths use `$USER`, so they work for any account; add your own `-A` and `--qos` flags on the command line.

## Used for the final study

| File | Purpose |
|---|---|
| `setup_env.sh`, `setup_env.sbatch`, `node_bootstrap.sh`, `build_wheelhouse.sbatch` | Build the Python environment on node-local scratch |
| `prefetch.sbatch`, `prefetch_models.sh` | Download model weights to node-local scratch |
| `gate_model.sbatch` | Competence gate, one job per model |
| `worker.sbatch` | Queue worker (task list from `scripts/queue_init.py`) |
| `analyze.sbatch` | CPU-only analysis (`scripts/analyze_story.py`) |
| `test_eap.sbatch` | EAP-IG equivalence check (`scripts/test_eap_equiv.py`) |

## GPT-2 pilot

`run_all.sbatch`, `run_baseline.sbatch`

## Earlier or auxiliary

`run_gate.sbatch`, `run_grid.sbatch` (first array-job design, replaced by the queue and workers), `run_validate.sbatch`, `smoke_validate.sbatch`, `smoke_models.sbatch`, `investigate.sbatch` (model-loading and memory checks), `run_stage2.sbatch` (planned language comparison and logit lens; not part of the reported results).
