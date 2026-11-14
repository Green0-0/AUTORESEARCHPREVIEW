"""analysis/judge_study.py end to end on a fake pool with MockLLM: every strategy x facts, thinking selection,
resumability (a rerun makes no new calls), LOO few-shot and ranking batch coverage."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ar import config as arconfig
from ar.journal import Journal

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def JS(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("judge_study_t", REPO / "analysis" / "judge_study.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    runs = tmp_path / "runs"
    for r, task in (("train_t07_naive_s2", "t07_activation_tuning"), ("train_t10_naive_s2", "t10_label_noise")):
        cfg = arconfig.from_dict({"run_id": r, "task_id": task, "seed": 1})
        j = Journal.init(runs / r, cfg)
        with j.transaction("t", "seed") as st:
            for n in range(7):
                iid = f"i{n:06d}"
                real = n < 5
                st["ideas"][iid] = {"id": iid, "created_at": 100.0 + n, "status": "real_judged" if real else "scored",
                                    "description": f"{task} idea {n}: method words {n} " * 20,
                                    "summary": f"summary {n} unique{n} tokens{n * 3} for {task}",
                                    "s_cal": 0.1 * n, "real_score": (0.2 * n if real else None),
                                    "real": ({"value": 0.5 + 0.05 * n} if real else None), "tags": ["a"]}
    monkeypatch.setattr(mod, "RUNS", runs)
    return mod


def _main(JS, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["judge_study.py", *argv])
    JS.main()


def test_judge_study_mock_end_to_end(JS, monkeypatch, tmp_path, capsys):
    out = tmp_path / "study"
    args = ["--out", str(out), "--pool", "train", "--mock", "--K", "2", "--R", "2", "--rank-batch", "3",
            "--fewshot-n", "2", "--B", "50", "--think-top", "1", "--think-k", "2"]
    _main(JS, monkeypatch, *args)
    s = json.loads((out / "summary.json").read_text())
    res = s["results"]
    for st in JS.STRATS:
        for f in ("facts", "nofacts"):
            r = res[f"{st}_{f}"]
            assert r["n_ok"] == 10 and r["calls"] > 0, (st, f)
            assert set(r["per_task_rho"]) == {"t07_activation_tuning", "t10_label_noise"}
            if f"{st}_{f}" != JS.BASELINE:
                assert "p_holm" in r["delta_vs_baseline"]
    assert len(s["meta"]["think_selected"]) >= 1
    assert all(k in res for k in s["meta"]["think_selected"])
    assert (out / "judge_study.md").read_text().startswith("# Judge study")
    calls = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]
    # J5: each idea appears in R=2 batches per facts setting; batches <= 3
    rank = [c for c in calls if c["cond"] == "J5_nofacts" and c["attempt"] == 1]
    seen = {}
    for c in rank:
        assert len(c["members"]) <= 3
        for m in c["members"]:
            seen[m] = seen.get(m, 0) + 1
    assert len(seen) == 10 and set(seen.values()) == {2}
    # J2: K=2 samples per idea per facts setting
    assert sum(1 for c in calls if c["cond"] == "J2_facts" and c["attempt"] == 1) == 20
    n_calls = len(calls)
    capsys.readouterr()
    _main(JS, monkeypatch, *args)                       # resume: everything cached
    assert len((out / "calls.jsonl").read_text().splitlines()) == n_calls
    assert "new calls 0" in capsys.readouterr().out
    _main(JS, monkeypatch, *args[:-6], "--phase", "analyze", "--B", "20", "--think-top", "1", "--think-k", "2")


def test_fewshot_real_is_leave_one_out(JS, tmp_path):
    items, journals = JS.load_pool("train")
    assert len(items) == 10 and {it["task"] for it in items} == {"t07_activation_tuning", "t10_label_noise"}

    class A:
        seed, fewshot_n, out = 0, 3, str(tmp_path / "o")
    Path(A.out).mkdir()
    st = JS.Study(A, items, journals)
    for it in items:
        ex = st.fs_real_examples(it)
        assert len(ex) == 3 and all(s != it["summary"] for s, _ in ex)
        same_task = {o["summary"] for o in items if o["task"] == it["task"]}
        assert all(s in same_task for s, _ in ex)
        sur = st.fs_surr_examples(it)                    # same run, created earlier only
        earlier = {i["summary"] for i in journals[it["run"]]["ideas"].values() if i["created_at"] < it["created_at"]}
        assert all(s in earlier for s, _ in sur)


def test_loto_in_summary_and_markdown(JS, monkeypatch, tmp_path):
    out = tmp_path / "study"
    _main(JS, monkeypatch, "--out", str(out), "--mock", "--K", "2", "--R", "1", "--rank-batch", "3",
          "--fewshot-n", "2", "--B", "20", "--think-top", "1", "--think-k", "2")
    lo = json.loads((out / "summary.json").read_text())["results"]["_loto"]
    tasks = {"t07_activation_tuning", "t10_label_noise"}
    assert set(lo["folds"]) == tasks and set(lo["per_task_rho"]["J3_facts"]) == tasks
    assert not any(c.endswith("_think") for c in lo["per_task_rho"])
    for f in lo["folds"].values():
        assert f["n_heldout"] == 5 and f["chosen"] in lo["per_task_rho"]
        assert f["heldout_rho"] == lo["per_task_rho"][f["chosen"]][next(t for t in tasks if lo["folds"][t] is f)]
    assert 0.0 <= lo["match_rate"] <= 1.0
    md = (out / "judge_study.md").read_text()
    assert "## LOTO" in md and "Per-task ρ by variant" in md


def test_loto_exposes_single_task_winner(JS):
    import numpy as np
    tasks = ["tA", "tB", "tC"]
    items = [{"task": t, "u": float(i)} for t in tasks for i in range(6)]
    u = np.array([it["u"] for it in items])
    good_a = np.where([it["task"] == "tA" for it in items], u, -u)        # perfect on tA only, inverted elsewhere
    steady = u + np.tile([0, 3, -2, 1, 0, -1], 3) * 0.6                 # decent everywhere
    vecs = {"X_facts": [good_a], "Y_nofacts": [steady], "Z_nofacts_think": [u]}
    lo = JS.loto(vecs, items, tasks)
    assert "Z_nofacts_think" not in lo["per_task_rho"]
    assert lo["overall_choice"] == "Y_nofacts"
    assert lo["folds"]["tB"]["chosen"] == "Y_nofacts" and lo["per_task_rho"]["X_facts"]["tA"] == 1.0
    assert lo["min_heldout_rho"] <= lo["mean_heldout_rho"] + 1e-12
    assert JS.loto(vecs, items[:6], ["tA"])["folds"] == {}
