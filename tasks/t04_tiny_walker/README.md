# t04_tiny_walker: Tiny Walker (dataset problem #4, Type I, train split)

Shortened, **CPU-only** proxy of problem #4. The task is to find a tiny policy (≤ 699 parameters, `policy.npz`
≤ 3,584 bytes) and the procedure that trains it, for gymnasium's **`BipedalWalker-v3`**. The metric is the mean
return over 100 private held-out terrains, so **higher is better**. Budget: a 600 s `train(ctx)` call on 16 CPU
cores. The task was added so that judge calibration is not fit only on CIFAR-style classification.

**Environment choice: plain BipedalWalker-v3, not Hardcore (deviation from the original).** On Hardcore with
30 min × 16 cores, nothing beat standing still:
- The starter learns to stand and gets about +5 with no falls.
- ARS variants learn to walk but fall at the first stumps or pits (−100), so their mean return is about 0 ± 100
  across terrains.
- The best variant (plain-terrain curriculum + softened fall penalty) scored 78.9 on one seed, but over seeds
  0-3 it gave −14.6, −2.9, +46.4 and −5.8, i.e. 5.8 ± 26, against the starter's 4.9 ± 0.7.

So Hardcore had no usable headroom. On plain BipedalWalker the same method family solves the task, so the
budget was cut to 10 min. At 30 min, ARS saturates at the ~333 ceiling on every seed; at 10 min it still
mostly solves the task, and sample efficiency matters.

## Layout
- `TASK.md`: the prompt. `task.yaml`: metadata, with `device: cpu`, `cpus: 16` (the real-eval chain then
  submits `realjudge/cpu_run.sbatch` on `compsci`) and `packages:`, which is used in Claude's implement prompt.
- `prepare_data.py`: idempotent and offline. It writes `$AR_ROOT/private/t04_tiny_walker/`:
  - `eval_seeds.npy`: int64 (100,), drawn once with seed 404 from [1e9, 2^31 − 1), distinct;
  - `meta.json`.

  A re-run with different content aborts. Then run `python scripts/make_decoy_private.py`; the decoy is random
  int64 values in the same range.
- `harness/t04lib.py`:
  - env id and mode table;
  - the parameter / size / dtype rules (`check_params`, `count_params`);
  - the `policy.py` AST scan (`scan_policy`);
  - the deterministic `rollout` and the multiprocessing evaluator.
- `harness/train_wrapper.py`: process 1.
  1. Snapshots `solution/policy.py` into `artifact/code/`.
  2. Imports `solution/train.py` as module `train`, with `solution/` on `sys.path`.
  3. Calls `train(ctx)` under a timer.
  4. Checks the returned dict and saves `artifact/policy.npz` with `np.savez`.

  It also records CPU time, but only for reaped children.
- `harness/evaluate.py`: process 2.
  1. Loads the npz (`allow_pickle=False`) and the snapshot.
  2. Applies the rules: params = array elements + `policy.py` numeric literals other than 0/1/2.
  3. Runs the 100 (smoke 20, cpu-smoke 8) held-out episodes in a pool of `n_cpus` processes.
  4. Re-plays episode 0 with a fresh `Policy` in-process to check determinism.
- `harness/check.py`: static scan. It also applies the `policy.py` AST rules as hard violations, and it rejects
  weight/data files (.npy/.npz/.pt/.json/...) shipped in `solution/`.
- `harness/run.sh <solution_dir> <out_dir> [--smoke|--cpu-smoke]`: runs offline.
  - It sets `OMP/OPENBLAS/MKL_NUM_THREADS=1` and `CUDA_VISIBLE_DEVICES=""`.
  - Hard caps for the training process are 690 s (full), 105 s (smoke) and 75 s (cpu-smoke); eval gets 300 s,
    120 s or 90 s.
- `harness/validate_cpu.sbatch`: validation inside `realjudge/jail.sh` (`JAIL=harness` by default, or `claude`).
- `starter/`: `policy.py` is a linear 24→4 map plus tanh (`theta`, 100 floats + 1 literal = 101 params,
  664 bytes). `train.py` runs basic antithetic random search (ARS-V1 style):
  - 2·n_cpus directions per iteration, σ 0.03, step 0.02, scaled by the return std;
  - one training terrain per direction;
  - no observation normalization and no reward shaping;
  - a fork pool of `n_cpus` workers.

  In practice it learns to **stand still** without falling (about +6). That is a naive but valid floor.

## Interface (details in harness/CONTRACT.md)
`policy.py`: `Policy(params)` with `reset()` and `act(obs: float32[24]) -> [4]`. It may import only numpy/math,
and it may use no RNG, file access, dynamic code, `id`/`hash`. Its numeric literals count as parameters.

`train.py`: `train(ctx) -> dict[str, np.ndarray]`. `ctx` holds:
- `mode`, `seed`, `n_cpus` (16 / ≤16 / 4);
- `time_budget_s` / `deadline` (600 / 60 / 30 s);
- `env_id`, `train_seed_range=(0, 1e9)`;
- `max_params`, `max_bytes`, `count_params(params)`, `scratch_dir`.

Libraries: numpy, gymnasium[box2d], torch (CPU), stdlib.

### Dependency install (shared venv)
`uv pip install --python $AR_ROOT/venv/bin/python "gymnasium[box2d]"` installed four packages:
- `gymnasium==1.4.0`
- `box2d==2.3.10`, from a **prebuilt manylinux wheel for py3.12**, so no swig/compile was needed;
- `pygame-ce==2.5.8`
- `farama-notifications==0.0.6`

`uv pip freeze` before and after differs only by these four lines. `python -m pytest -q tests` still passes.

Env speed is about 6k steps/s per core with random actions. Full-mode training does 40–46M env steps in 600 s
on 16 cores.

## Measured
Unless noted otherwise, all runs used `validate_cpu.sbatch` inside the **harness jail** (read-only FS, no
network), on `compsci` with 16 cores on fitz-44/36. Outputs are in `$AR_ROOT/t04_dev/finalP/`. CPU util is
`sacct` TotalCPU / (16 × elapsed).

| run | held-out mean return | episode sd / fell / ≥300 | train() wall | total wall (eval) | CPU util |
| --- | --- | --- | --- | --- | --- |
| **full, starter, seed 0** | **5.787** | 0.37 / 0% / 0% | 595.6 s | 598.7 s (2.3 s) | 88% (2:21:19 / 2:40:00) |
| **full, starter, seed 1** | **6.255** | 0.17 / 0% / 0% | 595.9 s | 598.9 s (2.2 s) | 87% |
| **full, starter, seed 2** | **5.931** | 0.13 / 0% / 0% | 595.2 s | 598.3 s (2.2 s) | 86% |
| full, probe, seed 0 | 299.43 | 44.1 / 5% / 95% | 595.4 s | 598.3 s | 81% |
| full, probe, seed 1 | 321.23 | 44.5 / 2% / 98% | 594.8 s | 597.3 s | 81% |
| full, probe, seed 2 | 320.59 | 1.2 / 0% / 100% | 594.6 s | 597.1 s | 81% |
| smoke, starter / probe | 3.83 / −86.6 | | 57.7 / 57.4 s | 59.2 / 58.2 s | 83% / 71% (reaped) |
| cpu-smoke, starter / probe, harness jail (4 cores) | 3.0 / −104.6 | | ~27 s | 29.9 / 28.6 s | ~86% (reaped) |
| cpu-smoke, starter / probe, **claude jail** (decoy seeds) | −7.2 / −92.7 | | ~27 s | 30.1 / 28.7 s | |

- The policy and npz sizes are as follows:
  - starter: 101 params, 664 bytes;
  - probe: 468 MLP weights + 48 normalization stats + 3 literals = **519 params**, **2,802 bytes**.
- **Floor = 5.991 ± 0.240** (mean ± sd, starter seeds 0-2), recorded in `task.yaml`. `ref: null`; the REF idea
  is in `scripts/make_anchor_ref.py`.

**Headroom probe** (`$AR_ROOT/t04_dev/probe_solution/`, outside starter/). It is ARS V2-t (Mania et al. 2018):
- running observation normalization (mean/std stored as `mu`/`isd`);
- a 24-16-4 tanh MLP;
- 48 directions per iteration, the top 16 used, σ 0.05, step 0.03;
- the −100 fall penalty replaced by 0 during training only.

Result: **313.75 ± 12.40**, a gap of **307.8**. That is 1,283× the starter's seed sd and **35× the pooled sd**,
so the task has headroom. The ceiling is around 330–335; 30-min runs reached 332–334.

The probe's seed sd (12.4) comes from occasional falls on a few terrains (seed 0: 5 of 100). The starter's sd
(0.24) is tiny because it never moves.

### Exploration that led here (dev runs, not in the floor/probe numbers; `$AR_ROOT/t04_dev/sweep{1,2,3}`)
- **Hardcore, 30 min:**
  - starter 4.9;
  - ARS MLP with the fall penalty removed: −72.6 (falls 100%);
  - ARS linear: 3.6;
  - ARS with the true reward: −7.2 / −9.7 (stands);
  - curriculum (30% plain) + no-fall: −1.9;
  - curriculum + fall −30: 78.9 (seed 0), then −14.6, −2.9, 46.4, −5.8 over 4 fresh seeds.

  The probe for Hardcore is kept at `$AR_ROOT/t04_dev/probe_hardcore_curriculum`.
- **Plain, 30 min:** starter 6.4; ARS linear 333.5; ARS MLP 332.4 (saturated).
- **Plain, 5 min:** ARS linear 313 / 290; ARS MLP 243 / 314. **Plain, 10 min:** ARS linear 271 / 328; ARS MLP
  312 / 330.

## Constraint-check tests (cpu-smoke, harness jail; `$AR_ROOT/t04_dev/bad2/`)
| deliberately bad / borderline solution | result |
| --- | --- |
| extra 650-element array returned (751 params) | **valid=false**: 751 parameters > 699 |
| 650 weights hard-coded as a list literal in `policy.py` | **valid=false**: literals counted, 751 > 699 |
| `np.random` noise in `act()` | **valid=false**: policy.py rule (.random), not evaluated |
| `train.py` does `np.load(".../private/t04_tiny_walker/eval_seeds.npy")` | **valid=false**: scan hard violation (the read itself succeeds in harness mode) |
| `train()` sleeps past the budget | **valid=false**: within_budget |
| 20 extra small arrays (501 params, 7,084 bytes) | **valid=false**: policy.npz > 3,584 bytes |
| `train()` rewrites `policy.py` | **valid=false**: policy.py modified during train / snapshot differs |
| float16 `theta` (control) | valid=true (464 bytes) |

A "non-deterministic" policy using `hash(object())` passed the replay check, because object addresses repeat
within a process. `id` and `hash` are now banned names in `policy.py`. Determinism otherwise rests on the ban
on RNG and imports.

## Caveats
- **The env is plain BipedalWalker-v3, not Hardcore, and the budget is 10 min, not 30** (see above). Solutions may
  still train on Hardcore if they want.
- The starter's floor is degenerate: it stands still. Any idea that does not learn to walk scores about 0 on
  real_score, and anything that walks reliably scores about 1. The probe is near the ceiling, so the spread
  among good ideas comes mostly from sample efficiency within 10 min and from fall rates.
- Seed noise for good methods is about ±12 to ±30 (bimodal: a seed that has not quite converged falls on a few
  terrains).
- **Private data is visible in harness mode**, as for every task. Reading the eval seeds through an obfuscated
  path would escape the static scan, and only the reviewer would catch it. Training on the eval terrains is only
  detectable this way. The seeds are random 31-bit ints disjoint from the training range, so accidental overlap
  is impossible.
- The constraints have edges:
  - "Parameters" = array elements + numeric literals other than 0/1/2 in `policy.py`.
  - The `policy.py` code itself does not count toward the 3.5 KiB; only `policy.npz` does.
  - Structural tricks in the code (e.g. a hand-written controller with few constants) are allowed; that is in
    the spirit of the original ("doesn't even have to be a neural network").
- `cpu_util_reaped` in `train_report.json` counts only reaped child processes, so use `sacct` for real CPU use.
  The probe is at about 81% because of straggler episodes in synchronous `pool.map` iterations.
