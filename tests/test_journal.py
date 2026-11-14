import asyncio
import json
import multiprocessing as mp
import threading
import time

import pytest

from ar.journal import (Journal, claim, fail, new_idea_id, reap_stale, release,
                        set_status)


def _idea(iid, created_at, status="generated"):
    return {"id": iid, "created_at": created_at, "wave": 0, "arm": "optimized",
            "prompt": {}, "description": "x", "summary": None, "tags": [],
            "surrogate_scores": [], "judge_outputs": [], "s_cal": None, "F": None,
            "F_detail": None, "scored_seq": None, "real_score": None, "real": None,
            "status": status, "times": {status: created_at}, "claim": None,
            "error": None, "retries": {}}


def _add(state, t, status="generated"):
    iid = new_idea_id(state)
    state["ideas"][iid] = _idea(iid, t, status)
    return iid


@pytest.fixture
def j(tmp_path):
    return Journal.init(tmp_path / "run", {"run_id": "r"})


def _events(j):
    return [json.loads(l) for l in j.events_path.read_text().splitlines()]


def test_init_creates_and_refuses(tmp_path):
    j = Journal.init(tmp_path / "run", {"run_id": "r", "seed": 1})
    s = j.read()
    assert s["config"] == {"run_id": "r", "seed": 1}
    assert s["ideas"] == {} and s["tags"] == {}
    assert s["logs"] == {"refits": [], "batches": [], "tagging": [], "realjudge": []}
    assert s["counters"] == {"next_idea": 0, "wave": 0, "scored_seq": 0}
    assert (tmp_path / "run" / "config.yaml").exists()
    with pytest.raises(FileExistsError):
        Journal.init(tmp_path / "run", {"run_id": "other"})
    assert j.read()["config"]["run_id"] == "r"


def test_transaction_commit_and_event(j):
    with j.transaction("w1", "add") as s:
        a = _add(s, 1.0)
        b = _add(s, 2.0)
    assert (a, b) == ("i000000", "i000001")
    st = j.read()
    assert list(st["ideas"]) == [a, b]
    assert st["counters"]["next_idea"] == 2
    ev = _events(j)[-1]
    assert ev["worker"] == "w1" and ev["op"] == "add" and sorted(ev["ids"]) == [a, b]
    assert isinstance(ev["t"], float)
    assert not (j.run_dir / "journal.json.tmp").exists()
    # compact file (no indent)
    assert "\n" not in j.path.read_text().strip()


def test_transaction_exception_discards(j):
    with j.transaction("w", "add") as s:
        _add(s, 1.0)
    n_ev = len(_events(j))
    with pytest.raises(ValueError):
        with j.transaction("w", "bad") as s:
            _add(s, 2.0)
            s["ideas"]["i000000"]["status"] = "summarized"
            raise ValueError("boom")
    st = j.read()
    assert list(st["ideas"]) == ["i000000"]
    assert st["ideas"]["i000000"]["status"] == "generated"
    assert len(_events(j)) == n_ev
    # lock released
    with j.transaction("w", "after") as s:
        _add(s, 3.0)
    assert len(j.read()["ideas"]) == 2


def test_noop_transaction_writes_nothing(j):
    mtime = j.path.stat().st_mtime_ns
    n_ev = len(_events(j))
    with j.transaction("w", "poll") as s:
        claim(s, "w", "tagger", lambda i: True, 5, 0.0, 10.0)
    assert j.path.stat().st_mtime_ns == mtime
    assert len(_events(j)) == n_ev


def test_read_is_deep_copy(j):
    with j.transaction("w", "add") as s:
        _add(s, 1.0)
    r = j.read()
    r["ideas"]["i000000"]["status"] = "failed"
    r["ideas"].clear()
    assert j.read()["ideas"]["i000000"]["status"] == "generated"


def test_nested_transaction_raises(j):
    with pytest.raises(RuntimeError, match="nested"):
        with j.transaction("w", "a") as s:
            with j.transaction("w", "b"):
                pass
    # outer discarded, lock usable
    with j.transaction("w", "c") as s:
        _add(s, 1.0)
    assert len(j.read()["ideas"]) == 1


def test_new_idea_id_monotonic_skips_existing():
    s = {"ideas": {"i000001": {}}, "counters": {"next_idea": 0}}
    assert new_idea_id(s) == "i000000"
    assert new_idea_id(s) == "i000002"
    assert s["counters"]["next_idea"] == 3


def test_claim_order_limit_eligibility():
    s = {"ideas": {}, "counters": {"next_idea": 0}}
    ids = [_add(s, t, st) for t, st in [(1, "summarized"), (2, "generated"), (3, "summarized"),
                                         (4, "summarized"), (5, "summarized")]]
    elig = lambda i: i["status"] == "summarized"
    got = claim(s, "w1", "tagger", elig, 2, now=100.0, timeout_s=60.0)
    assert got == [ids[0], ids[2]]
    assert s["ideas"][ids[0]]["claim"] == {"worker": "w1", "stage": "tagger", "at": 100.0}
    # already claimed (fresh) -> skipped
    got2 = claim(s, "w2", "tagger", elig, 10, now=120.0, timeout_s=60.0)
    assert got2 == [ids[3], ids[4]]
    assert claim(s, "w3", "tagger", elig, 10, now=130.0, timeout_s=60.0) == []
    assert claim(s, "w3", "tagger", elig, 0, now=1e9, timeout_s=60.0) == []


def test_claim_returns_oldest_created_first_even_if_inserted_out_of_order():
    s = {"ideas": {}, "counters": {"next_idea": 0}}
    a = _add(s, 5.0)
    b = _add(s, 1.0)
    assert claim(s, "w", "x", lambda i: True, 10, 0.0, 1.0) == [b, a]
    assert list(s["ideas"]) == [a, b]  # insertion order untouched


def test_stale_claims_reclaimable_and_failed_never():
    s = {"ideas": {}, "counters": {"next_idea": 0}}
    a = _add(s, 1.0, "tagged")
    b = _add(s, 2.0, "tagged")
    elig = lambda i: i["status"] == "tagged"
    assert claim(s, "w1", "judge", elig, 10, now=0.0, timeout_s=1800) == [a, b]
    fail(s, b, "boom", now=1.0)
    assert s["ideas"][b]["status"] == "failed" and s["ideas"][b]["error"] == "boom"
    assert s["ideas"][b]["claim"] is None and s["ideas"][b]["times"]["failed"] == 1.0
    assert claim(s, "w2", "judge", elig, 10, now=1000.0, timeout_s=1800) == []
    assert claim(s, "w2", "judge", elig, 10, now=1800.0, timeout_s=1800) == []  # not older
    assert claim(s, "w2", "judge", elig, 10, now=1801.0, timeout_s=1800) == [a]
    assert s["ideas"][a]["claim"]["worker"] == "w2"
    # failed never claimed, even with eligible=True
    assert claim(s, "w3", "judge", lambda i: True, 10, now=1e9, timeout_s=1) == [a]


def test_release_set_status_reap():
    s = {"ideas": {}, "counters": {"next_idea": 0}}
    a = _add(s, 1.0, "summarized")
    b = _add(s, 2.0, "summarized")
    claim(s, "w", "tagger", lambda i: True, 10, now=10.0, timeout_s=100)
    release(s, a)
    assert s["ideas"][a]["claim"] is None
    set_status(s, b, "tagged", now=20.0)
    assert s["ideas"][b]["status"] == "tagged"
    assert s["ideas"][b]["times"] == {"summarized": 2.0, "tagged": 20.0}
    assert s["ideas"][b]["claim"] is None
    with pytest.raises(ValueError):
        set_status(s, b, "bogus", 1.0)
    claim(s, "w", "tagger", lambda i: True, 10, now=10.0, timeout_s=100)
    assert reap_stale(s, now=50.0, timeout_s=100) == []
    assert sorted(reap_stale(s, now=200.0, timeout_s=100)) == [a, b]


# ---------------------------------------------------------------- concurrency
def _mp_worker(run_dir, n, wid):
    j = Journal(run_dir)
    for k in range(n):
        with j.transaction(f"p{wid}", "inc") as s:
            s["counters"]["wave"] += 1
            iid = new_idea_id(s)
            s["ideas"][iid] = _idea(iid, time.time())


def test_multiprocess_no_lost_updates(j):
    P, M = 6, 40
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_mp_worker, args=(str(j.run_dir), M, w)) for w in range(P)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    s = j.read()
    assert s["counters"]["wave"] == P * M
    assert s["counters"]["next_idea"] == P * M
    assert list(s["ideas"]) == [f"i{k:06d}" for k in range(P * M)]
    assert len(_events(j)) == 1 + P * M


def test_fork_multiprocess_no_lost_updates(j):
    # forked children inherit the module registry; pid-keyed locks must still work
    with j.transaction("parent", "warm") as s:
        s["counters"]["scored_seq"] = 0
    P, M = 4, 25
    ctx = mp.get_context("fork")
    procs = [ctx.Process(target=_mp_worker, args=(str(j.run_dir), M, w)) for w in range(P)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
        assert p.exitcode == 0
    assert j.read()["counters"]["wave"] == P * M


def test_threads_no_lost_updates(j):
    T, M = 8, 25

    def work(w):
        jj = Journal(j.run_dir)  # separate instances share the per-path lock
        for _ in range(M):
            with jj.transaction(f"t{w}", "inc") as s:
                s["counters"]["wave"] += 1

    ths = [threading.Thread(target=work, args=(w,)) for w in range(T)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert j.read()["counters"]["wave"] == T * M


def test_asyncio_tasks_claim_disjoint(j):
    with j.transaction("seed", "add") as s:
        for t in range(60):
            _add(s, float(t), "summarized")

    claimed: list[str] = []

    async def worker(w):
        while True:
            with j.transaction(f"a{w}", "claim") as s:
                ids = claim(s, f"a{w}", "tagger", lambda i: i["status"] == "summarized",
                            3, time.time(), 1800)
            if not ids:
                return
            claimed.extend(ids)
            await asyncio.sleep(0)  # "LLM call" outside the lock
            with j.transaction(f"a{w}", "done") as s:
                for i in ids:
                    set_status(s, i, "tagged", time.time())
                s["counters"]["wave"] += 1

    async def main():
        await asyncio.gather(*(worker(w) for w in range(10)))

    asyncio.run(main())
    s = j.read()
    assert sorted(claimed) == sorted(s["ideas"]) and len(set(claimed)) == 60
    assert all(i["status"] == "tagged" and i["claim"] is None for i in s["ideas"].values())
    assert s["counters"]["wave"] == 20


def test_await_inside_transaction_detected(j):
    async def bad():
        with j.transaction("a", "x") as s:
            s["counters"]["wave"] += 1
            await asyncio.sleep(0.01)

    async def other():
        await asyncio.sleep(0.001)
        with j.transaction("b", "y") as s:
            s["counters"]["wave"] += 100

    async def main():
        return await asyncio.gather(bad(), other(), return_exceptions=True)

    res = asyncio.run(main())
    assert isinstance(res[1], RuntimeError)
    assert j.read()["counters"]["wave"] == 1


def test_perf_5000_ideas(j, capsys):
    big = "lorem ipsum dolor sit amet " * 80  # ~2 KB of text per field
    with j.transaction("seed", "bulk") as s:
        for t in range(5000):
            iid = _add(s, float(t), "scored")
            idea = s["ideas"][iid]
            idea["description"] = big
            idea["summary"] = big[:400]
            idea["prompt"] = {"config": {"base": "Generic"}, "b_tilde": [0.1] * 12,
                              "turns": [big, big], "draws": [{"utility": "tags", "args": [3],
                                                               "value": big[:300]}],
                              "raw_outputs": [big, big]}
            idea["judge_outputs"] = [big]
            idea["tags"] = ["a", "b", "c"]
    size_mb = j.path.stat().st_size / 1e6
    t0 = time.perf_counter()
    for _ in range(3):
        j.read()
    t_read = (time.perf_counter() - t0) / 3
    t0 = time.perf_counter()
    for k in range(3):
        with j.transaction("w", "touch") as s:
            s["counters"]["wave"] = k + 1
    t_tx = (time.perf_counter() - t0) / 3
    with capsys.disabled():
        print(f"\n[perf] 5000 ideas, {size_mb:.1f} MB: read {t_read*1000:.0f} ms, "
              f"full transaction (load+diff+save+event) {t_tx*1000:.0f} ms")
    assert t_tx < 5.0


# ---------------------------------------------------------------- cache correctness
def test_cached_commits_match_plain_serialization(j):
    expect = j.read()
    for k in range(30):
        with j.transaction("w", "mix") as s:
            if k % 3 == 0:
                _add(s, float(k))
            if k % 4 == 1 and s["ideas"]:
                first = next(iter(s["ideas"]))
                s["ideas"][first]["summary"] = f"s{k} é \"q\""
            if k % 5 == 2:
                s["logs"]["refits"].append({"n": k, "mu": [0.1] * 12})
                s["tags"][f"t{k}"] = {"name": f"t{k}", "description": "d", "created_at": k, "ideas": []}
            if k == 20:
                del s["ideas"]["i000003"]
            expect = json.loads(json.dumps(s))
        assert json.loads(j.path.read_text()) == expect
        assert list(json.loads(j.path.read_text())["ideas"]) == list(expect["ideas"])
    assert list(j.read()) == ["config", "ideas", "tags", "logs", "counters"]
    ev = _events(j)
    assert sum("i000003" in e["ids"] for e in ev) == 2  # creation + removal


def test_caller_mutation_after_commit_does_not_leak(j):
    with j.transaction("w", "add") as s:
        a = _add(s, 1.0)
        held = s["ideas"][a]
    held["status"] = "failed"  # mutate after commit (bug in a caller)
    with j.transaction("w", "noop") as s:
        assert s["ideas"][a]["status"] == "generated"
    with j.transaction("w", "touch") as s:
        s["counters"]["wave"] = 7
    assert j.read()["ideas"][a]["status"] == "generated"


def test_foreign_write_invalidates_cache(j):
    with j.transaction("w", "add") as s:
        _add(s, 1.0)
    # another process rewrites the journal and logs its event
    st = j.read()
    st["counters"]["wave"] = 99
    j.path.write_text(json.dumps(st))
    with open(j.events_path, "a") as f:
        f.write(json.dumps({"t": 0, "worker": "x", "op": "x", "ids": []}) + "\n")
    with j.transaction("w", "inc") as s:
        assert s["counters"]["wave"] == 99
        s["counters"]["wave"] += 1
    assert j.read()["counters"]["wave"] == 100


def test_exception_after_partial_mutation_keeps_cache_clean(j):
    with j.transaction("w", "add") as s:
        _add(s, 1.0)
    with pytest.raises(ValueError):
        with j.transaction("w", "nan") as s:
            s["ideas"]["i000000"]["s_cal"] = float("nan")  # not JSON-serializable (allow_nan=False)
    with j.transaction("w", "ok") as s:
        assert s["ideas"]["i000000"]["s_cal"] is None
        s["counters"]["wave"] = 1
    assert j.read()["ideas"]["i000000"]["s_cal"] is None
