You are implementing ONE research idea for an automated evaluation. Your working directory is the idea directory:

- `task.md` — the task (also reproduced below).
- `idea.md` — the idea to implement (also reproduced below).
- `harness/` — the fixed, read-only evaluation harness. **Read `harness/CONTRACT.md` first**: it defines exactly which files/functions your solution must provide, their signatures, and what a solution may and may not do.
- `solution/` — a minimal working starter solution. This is the ONLY place you may write.

## Rules (hard)
1. **Implement the idea faithfully.** Do not substitute a different method, do not "improve" it into something else, and do not fall back to a generic baseline. If a detail is unspecified, choose the simplest reasonable reading that stays true to the idea. If part of the idea is impossible under the contract, implement the closest faithful version and explain the deviation in `solution/NOTES.md`.
2. **Only write inside `solution/`.** Do not create, modify, chmod or delete anything in `harness/` or elsewhere outside `solution/` (scratch files go in `solution/` too and must be removed or kept small).
3. **No access to test data.** Never look for, load or infer test labels or held-out data. Do not read anything outside this idea directory: not `$AR_ROOT/private/`, not the raw dataset files under `$AR_DATA` (e.g. the CIFAR test pickles; the harness gives you the training data), not the project repository (its `tasks/` directory holds held-out material), and not other idea directories or runs. The solution gets training data only, through the interface described in `harness/CONTRACT.md`.
4. Respect the task's compute budget: the full run gets **{{BUDGET_MINUTES}} minutes on {{COMPUTE}}**, enforced by a timeout; overrunning it makes the result invalid. {{COMPUTE_NOTE}}
5. You have **no GPU** here. Test only with the CPU smoke mode:
   `bash harness/run.sh solution out_cpu_smoke --cpu-smoke`
   It must finish without errors and write `out_cpu_smoke/metric.json` with `"valid": true`. Fix any errors it reports. Do not try to run the full mode.
6. Only use packages already installed ({{PACKAGES}}). Do not install anything. No network downloads.
7. When you are done, write a short `solution/NOTES.md` (what you implemented, how it maps to the idea, any deviations), and finish with a one-paragraph summary.

## Task ({{TASK_ID}}, metric: {{METRIC}})
{{TASK_PROMPT}}

## Idea to implement
{{IDEA}}
