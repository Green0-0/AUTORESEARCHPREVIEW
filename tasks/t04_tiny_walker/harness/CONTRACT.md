# t04_tiny_walker — solution contract

CPU-only task. The environment is gymnasium's **`BipedalWalker-v3`** (box2d; the plain terrain, not Hardcore):
24-d observation (hull angle / angular velocity / velocities, joint angles and speeds, 2 leg-contact flags,
10 lidar readings), 4-d action in [-1, 1] (hip/knee torques). Reward: +~300 in total for walking to the far end,
minus a small torque cost; falling ends the episode with -100; episodes are truncated after 1,600 steps.
The (gently uneven) terrain is generated from the `env.reset(seed=...)` seed. Return ≥ 300 means "solved"
(the best tiny policies reach ~330).

You supply a tiny policy (fewer than 700 parameters, at most 3.5 KiB on disk) and the procedure that trains it.

## Files you must provide in `solution/`
### `policy.py` — the model (architecture code)
```python
class Policy:
    def __init__(self, params: dict[str, np.ndarray]): ...   # the dict your train() returned (from policy.npz)
    def reset(self) -> None: ...                              # called at the start of every episode
    def act(self, obs: np.ndarray) -> np.ndarray: ...         # obs float32 (24,) -> action (4,), clipped to [-1, 1]
```
Any architecture is allowed (linear, MLP, RNN / internal state reset in `reset()`, CPG, hand-designed
controller with tuned gains, ...). It must be **deterministic**: the same params and observations must give the
same actions (the harness replays one episode with a fresh `Policy` and requires an identical return).

Rules for `policy.py` (checked on the AST; any violation makes the run invalid):
- imports: only `numpy` and `math`;
- no file access, no RNG, no dynamic code: no `open`, `eval`, `exec`, `compile`, `__import__`, `getattr`, `id`, `hash`,
  `globals`, no `np.random` / `.random`, `np.load`, `fromfile`, ...;
- **every numeric literal other than 0, 1 and 2 counts as one parameter** (so constants cannot hide weights in
  the code; derive shapes from the arrays where possible); bytes literals are not allowed and non-docstring
  string literals are limited to 256 characters in total.
The harness snapshots `policy.py` **before** calling `train()` and evaluates that snapshot; `policy.py` must not
change during training.

### `train.py` — the training procedure
```python
def train(ctx: dict) -> dict[str, np.ndarray]
```
Returns the policy parameters. The harness saves them with `np.savez(artifact/policy.npz, **params)` (no
compression) and evaluation rebuilds `Policy(params)` in a separate process from that file. All learned state
(including observation-normalization statistics) must be in this dict. `solution/` is on `sys.path`, so
`train.py` can `from policy import Policy` and import helper modules from `solution/`.

`ctx` keys:
| key | value |
| --- | --- |
| `mode` | `"full"`, `"smoke"` or `"cpu-smoke"` |
| `n_cpus` | CPU cores you have: 16 (full), up to 16 (smoke), 4 (cpu-smoke). Use them all (e.g. a `multiprocessing` pool of `n_cpus` rollout workers). BLAS threads are pinned to 1 per process. |
| `time_budget_s`, `deadline` | 600 s (full), 60 s (smoke), 30 s (cpu-smoke); `deadline` is a `time.time()` value |
| `seed` | int (`AR_SEED`) |
| `env_id` | `"BipedalWalker-v3"` (create it with `gymnasium.make(ctx["env_id"])`) |
| `train_seed_range` | `(0, 10**9)`: **every `env.reset(seed=s)` used in training must have `s` in this range**. The held-out evaluation terrains use seeds outside it. |
| `max_params`, `max_bytes` | 699, 3584 |
| `count_params` | callable `params -> {"n_params", "npz_bytes", "params_ok", "size_ok", ...}`: exactly what the harness will count (arrays + `policy.py` literals) |
| `scratch_dir` | writable directory for temporary files (deleted after training) |

Libraries: `numpy`, `gymnasium` (+ `Box2D`), `torch` (CPU only; no GPU), the standard library.

### Budget
The whole `train()` call must finish within `time_budget_s` (+10 s grace), otherwise the run is **invalid**;
the process is hard-killed at 690 s (full). Stop on `ctx["deadline"]` with a margin (an episode can take up
to 1,600 steps, ~0.4 s on one core). Evaluation is separate and does not count.

### Size and parameter limits
- **Parameters:** total number of array elements in the returned dict + numeric literals in `policy.py`
  (see above) must be **≤ 699**. Every element counts as one parameter, whatever its dtype.
- **Size:** `artifact/policy.npz` (written by the harness with `np.savez`) must be **≤ 3,584 bytes**. Each array
  adds roughly 230 bytes of npz/zip/npy header overhead, so pack parameters into few arrays (e.g. one flat
  `theta`) if you are close to the limit.
- Allowed dtypes: `float32`, `float16`, `uint16` (e.g. bfloat16 bit patterns you decode in `policy.py`),
  `int8`, `uint8`, `int16`, `int32`. Keys must be Python identifiers. Values must be finite.

### Allowed
Any policy architecture and any training method: random search / ARS, evolution strategies, CMA-ES, policy
gradients, imitation of a scripted controller you write yourself, other terrains such as
`BipedalWalkerHardcore-v3` for training (same observation/action spaces), reward shaping during training, larger temporary models during training
(e.g. distilling a big policy into the tiny one), observation normalization, `torch` on CPU.

### Not allowed
- Training on the evaluation terrains: env seeds outside `ctx["train_seed_range"]`, or reading anything under
  `$AR_ROOT/private` (the eval seeds) or the harness's files.
- Pre-trained policies or weights from anywhere else; shipping weight / data files in `solution/` (all
  parameters must come out of `train()`); network access (there is none).
- Changing the environment's dynamics or the evaluation reward (shaping your *training* reward is fine).
- Writing anywhere except `scratch_dir`; modifying `policy.py` during training.

## What the harness checks (`metric.json` → `constraint_report`)
- `train.within_budget`: `train()` wall time ≤ budget + 10 s; `train.ok`: it returned a valid dict.
- `n_params ≤ 699` (`n_array_params` + `n_literal_params`), `npz_bytes ≤ 3584`, dtypes, finiteness.
- `policy_scan`: the `policy.py` AST rules above; snapshot unchanged during training.
- `deterministic`: a fresh `Policy` reproduces evaluation episode 0 exactly; all actions finite with shape (4,).
- `scan`: static scan of the solution (`harness/check.py`). Hard violations (network, private eval seeds, paths
  into the data tree, pre-trained policies, weight/data files shipped in `solution/`) invalidate the run.
  Review flags (file reads, subprocess, explicit `reset(seed=...)`) go to the reviewer.

**Metric:** `heldout_mean_return` (**higher is better**) = mean undiscounted episode return of the policy over
100 fixed held-out terrains of `BipedalWalker-v3` (env seeds drawn once, kept private), each run once
deterministically with the actions clipped to [-1, 1]. `constraint_report` also gives the return sd / median /
min / max, `frac_fell` and `frac_ge_300`. Any violation makes the run `valid: false`; `value` is still reported
(-1e9 if the evaluation itself failed).

## Running
```
harness/run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]
```
`--cpu-smoke`: 4 cores, 30 s training, 8 evaluation episodes, < 1.5 min (numbers are meaningless).
`--smoke`: 60 s training on up to 16 cores, 20 episodes. No flag: full mode (600 s on 16 cores, 100 episodes).
