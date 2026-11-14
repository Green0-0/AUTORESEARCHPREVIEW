The evaluation of your solution failed: **{{FAILURE_KIND}}**.

You get exactly one repair attempt. Fix the cause in `solution/` so the full run succeeds and is valid. The same rules apply as before:
- keep implementing the **same idea faithfully** — do not switch to a different method or a generic baseline to make it pass;
- only write inside `solution/`; do not touch `harness/`; no access to test data;
- the full run has {{BUDGET_MINUTES}} minutes on {{COMPUTE_SHORT}} (timeout enforced — if the run was too slow, make it fit the budget);
- you have no GPU here: verify with `bash harness/run.sh solution out_cpu_smoke --cpu-smoke`, which must write `out_cpu_smoke/metric.json` with `"valid": true`.

Update `solution/NOTES.md` with what you changed and why.

## Failure details
{{ERROR_LOG}}
