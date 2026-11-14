# Build contracts (shared by all build agents)

These are the authoritative interfaces between modules. Behaviour comes from `class_notes/Autoresearch Preview Spec.md` (the "spec") and `IMPLEMENTATION_PLAN.md` (the "plan"). Where they disagree, **this file wins on interfaces** and **the spec wins on behaviour**, except where the plan explicitly lists a deviation.

## General rules
- Python 3.12. Always `source /home/users/kg349/AUTORESEARCHPREVIEW/env.sh`, which activates the venv at `/usr/xtmp/kg349/ar/venv` and has torch, vllm, openai, filelock, numpy, scipy, pandas, sklearn, pyyaml and pytest-able deps.
  - If you need a new package: `uv pip install <pkg>`, with env.sh sourced. **Never** pip-install into home.
- **The home dir is at its file quota (493k/500k files).** Write nothing bulky or many-file under `~`. Data, runs, caches and scratch go under `/usr/xtmp/kg349/ar/` (`$AR_ROOT`). Datasets are in `$AR_DATA` (= `$AR_ROOT/data`):
  - CIFAR-10: `cifar-10-batches-py/`
  - CIFAR-100: `cifar-100-python/`
  - both are the official python pickles.
- Only touch the files you own (listed in your brief). Don't `git commit`. Don't edit `IMPLEMENTATION_PLAN.md`, `CONTRACTS.md` or `env.sh`. Report any needed change in your final message instead.
- Tests: `pytest` under `tests/`, offline, fast (< 30 s each file), no GPU and no network. Install pytest into the venv if it's missing.
- **Cluster etiquette (hard rule):** never leave GPUs or CPUs idle.
  - GPU work only via `sbatch -p compsci-gpu --gres=gpu:a5000:1 -c 8 --mem=32G -t <tight>`. The job does the compute and exits.
  - No interactive GPU sessions, no sleeping or polling inside GPU jobs.
  - CPU-only work via `srun -p compsci -c 4 --mem=8G -t 00:20:00 ...` or on the login node if it's short.
  - Never use lab partitions (bhuwan, nlplab*, grisman, rudin, wiseman, skynet).
  - Log `nvidia-smi dmon -s um -d 5` in every GPU job you submit.

## Pipeline core

### Run directory
`$AR_ROOT/runs/<run_id>/` contains:
- `journal.json` (the state)
- `journal.lock`
- `events.jsonl` (append-only log of every transaction: `{"t":…, "worker":…, "op":…, "ids":[…]}`)
- `config.yaml` (frozen copy)
- `real/<idea_id>/` (real-eval workspaces)

### Config (`ar/config.py`, YAML in `configs/`)
```yaml
run_id: t19_optimized_s1
task_id: t19_data_ordering          # dir name under tasks/
arm: optimized                      # optimized | naive
seed: 1
task_prompt_file: tasks/t19_data_ordering/TASK.md   # repo-relative; its text is {{prompt}}
references: []                      # list of reference summaries (strings)
facts: []                           # list of strings; or facts_file: path to YAML list
llm:
  base_url: http://127.0.0.1:8000/v1
  model: qwen
  max_inflight: 64                  # global concurrency cap for this run's client
  purposes:                         # per call type
    ideation:  {thinking: true,  max_tokens: 16000, temperature: 0.8}
    summary:   {thinking: false, max_tokens: 400,   temperature: 0.3}
    tagging:   {thinking: false, max_tokens: 4000,  temperature: 0.3}
    assign:    {thinking: false, max_tokens: 800,   temperature: 0.2}
    judge:     {thinking: true,  max_tokens: 12000, temperature: 0.7}
sampler: {mode: thompson, lambda0: 0.1, lambda: 0.1, sigma: 0.1, gamma: 0.99}   # mode: thompson (optimized) | uniform (naive: same prompt pool, no learning) | fixed_generic (not used)
scoring: {s_min: 0.05, s_base: 0.5, calibration: {a: 0.0, b: 1.0}}             # s_cal = clip(a + b*s_raw, 0, 1)
ideation: {batch_size: 8, concurrent_batches: 6}
tagging: {trigger_count: 20, trigger_minutes: 10, batch_limit: 100, S: 10, r: 2, N: 4, K: 4}
judge:                              # judge strategy (plan §9.1); these defaults = the original J1 judge
  retries: 2
  strategy: basic                   # basic (J1) | basic_k (J2) | rubric (J3) | fewshot (J4) | ranking (J5)
  k: 5                              # basic_k: samples per judgment (judgment = their mean)
  facts: null                       # JUDGE-ONLY facts (list); null = use the top-level `facts` (legacy)
  facts_file: null                  # optional YAML list appended to the judge facts
  anchor_facts: false               # append the task's starter=0 / reference=1 facts (task.yaml floor/ref)
  fewshot_source: surrogate         # fewshot: surrogate (same-run earlier ideas + s_cal) | real (real_score)
  fewshot_n: 6                      # fewshot: examples per prompt, stratified over the score range
  fewshot_runs: []                  # fewshot real: extra run ids (same runs root, same task) used as examples
  rank_batch: 5                     # ranking: new ideas per call
  rank_max_wait_s: 60               # ranking: judge a partial batch once the oldest waiting idea is this old
  rank_max_idea_chars: 12000        # ranking: per-idea description cap (null = none)
claims: {timeout_minutes: 30}
stop: {max_hours: 24, max_ideas: null}
```
`ar.config.load(path) -> dict` validates the keys and fills defaults (the values above are the defaults). `ar.config.task_prompt(cfg) -> str`.

Judge facts: `ar.judging.judge_facts(cfg)` = (`judge.facts` if not null else top-level `facts`) + `judge.facts_file` + anchor facts. Ideators only ever see the top-level `facts`, so facts meant for the judge alone go in `judge.facts`. Validation: `judge.strategy`/`fewshot_source` must be known values; `ranking` needs reference ideas for `task_id` (`ar/anchors.py`); `anchor_facts` needs numeric floor/ref in `tasks/<task_id>/task.yaml`. Ranking calls use the LLM purpose `judge_rank` if `llm.purposes` defines it, else `judge`.

**Arm `naive`** is the same pipeline with `sampler.mode: fixed_generic`: every prompt is base Generic with no appends. Tagging, judging and F still run, so both arms produce identical records.

### Journal (`ar/journal.py`)
```python
class Journal:
    def __init__(self, run_dir: str | Path): ...
    @contextmanager
    def transaction(self, worker: str, op: str) -> Iterator[dict]:
        """filelock (journal.lock) -> load latest journal.json -> yield mutable state ->
        on normal exit: write journal.json.tmp, fsync, os.replace; append one events.jsonl line.
        On exception: discard changes, release lock, re-raise."""
    def read(self) -> dict:                       # latest snapshot (deep copy), no lock held after return
    @staticmethod
    def init(run_dir, config: dict) -> "Journal": # creates dir + empty state; refuses if journal exists
    # helpers (all take/return plain dicts; usable inside a transaction on `state`)
def new_idea_id(state) -> str                     # "i000123", monotonic
def claim(state, worker: str, stage: str, eligible: Callable[[dict], bool], limit: int, now: float, timeout_s: float) -> list[str]
    # marks up to `limit` eligible, unclaimed (or stale-claimed) non-failed ideas with idea["claim"]={"worker","stage","at"}; returns ids, oldest created first
def release(state, idea_id: str) -> None
def set_status(state, idea_id: str, status: str, now: float) -> None   # also records idea["times"][status]=now, clears claim
def fail(state, idea_id: str, error: str, now: float) -> None          # status "failed", error recorded
```
State schema:
```python
{
 "config": {...},
 "ideas": {id: Idea},          # insertion order = creation order
 "tags": {name: {"name", "description", "created_at", "ideas": [ids]}},
 "logs": {"refits": [], "batches": [], "tagging": [], "realjudge": []},
 "counters": {"next_idea": 0, "wave": 0, "scored_seq": 0},
}
Idea = {
 "id", "created_at": float, "wave": int, "arm",
 "prompt": {"config": SamplerConfig, "b_tilde": [12 floats] | None,
            "turns": [rendered str, ...], "draws": [DrawRecord...], "raw_outputs": [str per turn]},
 "description": str, "summary": str | None, "tags": [names],
 "surrogate_scores": [float raw, ...],        # 1 or 2 judgments (basic_k: each = mean of k samples)
 "judge_outputs": [str, ...],                 # one per judgment
 "judge_detail": [dict, ...],                 # non-basic strategies only, one per judgment:
                                              #   {"strategy", "samples"|"examples"|"batch","position", ...}
 "s_cal": float | None,                       # calibrated mean of surrogate_scores (what F uses)
 "F": float | None, "F_detail": {"b_t": {tag: float}, "s_first": float, "s_final": float, "confirmed": bool} | None,
 "scored_seq": int | None,                    # order in which F was frozen (sampler weights use this)
 "real_score": float | None, "real": {...} | None,
 "status": "generated|summarized|tagged|judged|scored|promoted|real_judged|failed",
 "times": {status: float}, "claim": {...} | None, "error": str | None, "retries": {stage: int},
}
```
Stages and owners:

| Stage | Picks up | Sets | Owner |
| --- | --- | --- | --- |
| `ideator` | — | ideas with status `generated`, then `summarized` | ideator |
| `tagger` | status `summarized` | status `tagged` | tagger |
| `judge` | status `tagged` | first score → `judged` → (confirmation if F > 0) → F frozen → `scored` | judge |
| `realjudge` | ideas chosen by script | `promoted` → `real_judged` | realjudge |

### Templating (`ar/templating.py`) and prompts (`ar/prompts.py`)
```python
@dataclass
class RenderContext:
    prompt: str
    references: list[str]
    facts: list[str]
    tags: list[tuple[str, str]]          # (name, description), pool creation order
    judged_ideas: list[dict]             # {"id","summary","s_cal","real_score"} creation order; "judged" = has surrogate score
def render(template: str, ctx: RenderContext, rng: np.random.Generator) -> tuple[str, list[dict]]
    # AST parse first, evaluate innermost-first; substituted text is never re-parsed.
    # DrawRecord = {"utility", "args", "value", "sampled_ids"(optional)}
```
`ar/prompts.py` holds every prompt string **verbatim** from the spec: the V1 ideator system prompt, the 7 bases (turn lists), the appends (top/bottom parts), Summary, the tagging system prompt, Generator/Merger/Writer Generic, Basic assignment, the judge system prompt and Basic surrogate. It also provides:
```python
BASES = ["Generic","BetterGeneric","BestSample1","BestSample2","RandSample","NewSampleCritique","TopicIdea"]
def assemble(config: SamplerConfig, ctx: RenderContext, rng) -> dict
    # -> {"system": str, "turns": [str, ...], "draws": [...]}; spec order for tops/bottoms; appends on turn 1 only
def summary_prompt(idea: str) -> str
def tagging_system(prompt: str, tags: list[tuple[str,str]]) -> str
def generator_prompt(summaries: list[str]) -> str
def merger_prompt(summaries: list[str], lists: list[str]) -> str
def writer_prompt(summaries: list[str], lists: list[str]) -> str
def assign_prompt(idea: str) -> str
def judge_system(prompt: str, facts: list[str], rng) -> str
def judge_prompt(idea: str) -> str                                   # J1/J2 "Basic"
def rubric_prompt(idea: str) -> str                                  # J3 "Arbitrary Rubric"
def fewshot_prompt(idea: str, examples: list[tuple[str, float]]) -> str   # J4 "Basic FS": "- summary (judging result: 0.42)"
def ranking_prompt(ideas: list[str], references: list[dict]) -> str  # J5: "Reference A (known judging result: 0.00):", "Idea k:"
```
Judge texts come from `class_notes/Autoresearch Draft Spec Graph Note.md` ("Basic", "Arbitrary Rubric", "Basic FS", "General System Prompt"; tests/test_judge_strategies.py checks them verbatim). "Ranking Based" has no prompt text in the spec; RANKING keeps its description sentence and adds the output format `Idea k: \boxed{x}`.

`ar/anchors.py`: `STARTER_IDEAS[task]` (short starter descriptions, known result 0), `reference_ideas()` (= scripts/make_anchor_ref.py REF_IDEAS, known result 1), `anchor_facts(task)`, `ranking_references(task)`.
`ar/judging.py` (shared by the judge worker and analysis/judge_study.py): `judge_facts(cfg)`, `select_examples(cands, n, rng, exclude_id, exclude_texts)` (stratified: sort by score, n equal-count bins, one random pick per bin, shown shuffled; near-duplicates = summary word-Jaccard >= 0.6 excluded), `ranking_batches(keys, b, R, seed)`, `messages_single(strategy, system, description, examples)`, `messages_ranking(system, descriptions, refs, max_chars)`.
Formatting of list utilities: one item per line as `- name: description` for tags; `- summary` for summaries; `- summary (surrogate: 0.62; real: 0.41)` for critiques, omitting `real` when absent. References and facts are `- text`.

### Sampler (`ar/sampler.py`) and scoring (`ar/scoring.py`)
```python
FEATURES = ["intercept","BetterGeneric","BestSample1","BestSample2","RandSample","NewSampleCritique","TopicIdea","NoTag","UseTag","NoIdea","References","RefIdeas"]
SamplerConfig = {"base": str, "tag_slot": None|"NoTag"|"UseTag", "NoIdea": bool, "References": bool, "RefIdeas": bool}
def featurize(cfg: SamplerConfig) -> np.ndarray                       # shape (12,)
def observations(state) -> tuple[np.ndarray, np.ndarray]              # X, y from ideas with F not None & not failed, ordered by scored_seq
def fit_posterior(X, y, lambda0, lam, sigma, gamma) -> tuple[np.ndarray, np.ndarray]   # mu (12,), Sigma (12,12)
def availability(state) -> dict   # {"tags": bool, "judged_ideas": bool, "references": bool}  (references from config)
def draw_batch(mu, Sigma, avail, n, rng, mode="thompson") -> list[tuple[SamplerConfig, list[float] | None]]
    # mode fixed_generic -> Generic, no appends, b_tilde None
# scoring.py
def parse_boxed(text: str) -> float | None          # last \boxed{...}, float, clamped [0,1] (J1-J4)
def parse_ranked(text: str, n: int) -> list[float | None]   # J5: score of "Idea 1".."Idea n"; None = missing
def calibrate(s_raw: float, cal: dict) -> float     # clip(a + b*s, 0, 1)
def compute_F(state, idea_id, s: float, s_min, s_base, b_t: dict | None = None) -> tuple[float, dict]
    # comparison set: ideas created before idea (created_at), status not failed, s_cal not None, excluding idea itself
    # b_t per tag = max s_cal among them holding t, else s_base; F = max_t max(0, s - b_t - s_min)
    # if b_t passed, reuse it (confirmation recompute against frozen b_t)
```

### LLM (`ar/llm.py`)
```python
@dataclass
class ChatResult: text: str; reasoning: str; finish_reason: str; prompt_tokens: int; completion_tokens: int
class LLM:
    def __init__(self, llm_cfg: dict): ...           # AsyncOpenAI to base_url; semaphore max_inflight
    async def chat(self, messages: list[dict], purpose: str) -> ChatResult
        # applies purposes[purpose] (thinking via extra_body chat_template_kwargs.enable_thinking),
        # splits out <think>...</think> (or reasoning_content) into .reasoning; retries transient errors (3x, backoff)
class MockLLM(LLM):   # deterministic offline fake for tests/e2e: answers per purpose
    # ideation -> short idea text; summary -> short summary; tagging generator/merger -> tag lists;
    # writer -> valid ```json {"new_tags":[...]} ```; assign -> some existing tag names; judge -> "... \boxed{0.xx}" (seeded random)
```
Token usage is appended to `$RUN/llm_usage.jsonl` (purpose, tokens, latency).

### Judge strategies (`ar/workers/judge.py`)
Every strategy produces one absolute score per idea per judgment, so the F mechanics are unchanged: first judgment → s_cal, F_first and frozen b_t (`judged`) → if F_first > 0 a confirmation judgment with the same strategy (few-shot: fresh example draw; ranking: the idea is released in `judged` and joins a later batch) → F against the frozen b_t (`scored`). `basic_k` makes k calls per judgment; `fewshot` selects examples at claim time (no examples yet → Basic prompt, `judge_detail.fallback = "basic"`); `ranking` claims up to `rank_batch` ideas (tagged and confirmation-pending mixed), waits at most `rank_max_wait_s` for a full batch, and re-asks only the ideas whose score is missing (in a smaller reshuffled batch) up to `retries` times before failing them. MockLLM answers every strategy's prompt shape (ranking: one `Idea k: \boxed{}` line per idea; rubric: criteria + final box).

### Workers (`ar/workers/*.py`)
Each worker module exposes `async def run(journal: Journal, cfg: dict, llm: LLM, stop: asyncio.Event) -> None`. It loops until `stop` is set: read state → claim work under a transaction → do the LLM calls **outside** the lock → write the results in a new transaction. LLM calls never happen while the lock is held.

### Orchestrator (`ar/orchestrator.py`)
`python -m ar.orchestrator --config configs/X.yaml [--config configs/Y.yaml ...] [--resume] [--mock]` runs all workers for each run in one asyncio process. It stops at `stop.max_hours` or `max_ideas`. It reaps stale claims, and on `--resume` it continues from the journal.

### Real judging (`ar/realjudge.py`, `realjudge/`)
Triggered by script, not automatically:
```
python -m ar.realjudge select  --run <run_id> --top K --random R [--stratified N]   # marks ideas "promoted", prints ids
python -m ar.realjudge submit  --run <run_id> --ids i0001 i0002                       # builds real/<id>/ and sbatch chain
python -m ar.realjudge ingest  --run <run_id>                                         # reads real/<id>/result.json -> journal
```
Per-idea Slurm chain (plan §2.1: no GPU while Claude works):
1. CPU job: Claude implements, then `run.sh --cpu-smoke`.
2. GPU job (afterok): `run.sh` full run only.
3. CPU job (afterany): one repair attempt if needed, which resubmits step 2 once; then the Claude review.
4. `result.json`.

`result.json` = `{"idea_id","metric","value","real_score","adherence","fidelity","valid","repaired","claude_usage":[...],"wall":{...}}`.
`real_score = clip((value - floor)/(ref - floor), 0, 1)` using `tasks/<task>/task.yaml`, set to 0 if adherence == 0. Larger-is-better is assumed, so flip the sign for any lower-is-better metric.

## Task harness contract (one dir per task: `tasks/<task_id>/`)
```
tasks/<task_id>/
  TASK.md            # the task prompt exactly as the pipeline sees it ({{prompt}}): shortened Type I/II task, proxy budget stated
  task.yaml          # id, type (I|II), split (train|test), budget_minutes, metric, higher_is_better, floor: null, ref: null, notes
  prepare_data.py    # idempotent; derives task-specific data into $AR_DATA/<task_id>/ (e.g. fixed noisy labels, fixed subsets)
  harness/
    CONTRACT.md      # what a solution must provide (files, functions, signatures), what it may/may not do. Claude reads this.
    run.sh           # run.sh <solution_dir> <out_dir> [--smoke | --cpu-smoke]  -> writes <out_dir>/metric.json
    *.py             # harness code (training wrapper if the task fixes it, eval, constraint checks)
  starter/           # minimal valid solution (simple baseline); must run end-to-end with run.sh
  README.md          # how it works, measured starter results (metric, wall time, GPU util), known caveats
```
- `metric.json` = `{"metric": str, "value": float, "valid": bool, "constraint_report": {...}, "wall_s": float, "mode": "full|smoke|cpu-smoke"}`. If a constraint is violated, `valid` is false and `value` is still reported.
- Modes:
  - full: the task budget on 1 A5000, enforced with `timeout`. Budget overrun means invalid.
  - `--smoke`: GPU, ≤ 2 min, tiny epochs.
  - `--cpu-smoke`: CPU only, tiny data slice, ≤ 3 min. Checks that the code runs end to end; Claude uses it on CPU nodes.
- **The solution never gets test labels.** The training step is given train data only, and the solution saves an artifact to `<out_dir>/artifact/`. Evaluation then loads that artifact in a separate step that reads the test data itself.
- **GPU efficiency:** the starter must keep the GPU busy, with > 80% SM utilization in `nvidia-smi dmon`. Preload the data to the GPU, do augmentation on the GPU, use AMP and large batches, and don't rely on a CPU DataLoader. The harness's own fixed parts (eval, fixed training loops) must be efficient too.
- The harness depends only on torch/torchvision/numpy and env vars (`AR_DATA`, `AR_SEED`, default 0). It must be self-contained, so it can be copied read-only into an idea dir.
- Anything the solution must not see (held-out architectures, test labels) lives outside the harness dir, under `$AR_ROOT/private/<task_id>/`, written by `prepare_data.py`.

## CPU-only tasks (added 2026-11-09, t04_tiny_walker)
- task.yaml may set `device: cpu|gpu` (default gpu), `cpus: N` (default 16) and `packages:` (extra allowed libraries, e.g. gymnasium[box2d]).
- For `device: cpu`, the real-eval full run goes to `realjudge/cpu_run.sbatch` on partition compsci with `-c N` and no GPU. It still writes `<out>/gpu_job.json` (with `"device": "cpu"`), so check_run, review and repair are shared.
- The "1 A5000" and "GPU utilization > 80%" rules do not apply to CPU tasks. There, keep all cores busy (measured with sacct).
- Harnesses may depend on the task's `packages:`, beyond torch, torchvision and numpy.
