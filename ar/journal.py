"""Run journal: one JSON state file guarded by a file lock (see CONTRACTS.md, "Journal").

Concurrency model
-----------------
* Across processes on one host: ``filelock.FileLock`` on ``journal.lock``.
* Across threads in one process: a per-path ``threading.Lock`` taken before the file lock.
* Across asyncio tasks in one thread: a transaction body is synchronous (no LLM calls under
  the lock), so tasks cannot interleave. Entering a second transaction on the same run
  from the thread that already holds it (nesting, or awaiting inside the ``with`` block and
  letting another task start one) raises ``RuntimeError`` instead of silently losing updates.

Writes are atomic: ``journal.json.tmp`` -> fsync -> ``os.replace`` (readers never see a torn
file, so ``read()`` needs no lock). Every committed transaction appends one line to
``events.jsonl``. A transaction that leaves the state unchanged writes nothing.

Speed: the process keeps a private cache of the last committed journal (text, parsed copy,
per-idea JSON strings), validated under the lock by the journal's stat plus the size of
``events.jsonl``. A transaction then costs one ``json.loads`` of the cached text, a dict
equality diff, serialization of only the changed ideas and the atomic write. A foreign write
(another process) invalidates the cache and the file is re-read.
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from filelock import FileLock

JOURNAL = "journal.json"
LOCK = "journal.lock"
EVENTS = "events.jsonl"
CONFIG = "config.yaml"

STATUSES = (
    "generated", "summarized", "tagged", "judged", "scored", "promoted", "real_judged", "failed",
)

# Per-process registry: resolved journal path -> {thread lock, owner thread, cache}.
_REG_GUARD = threading.Lock()
_REGISTRY: dict[tuple[int, str], dict[str, Any]] = {}
_MISSING = object()


def _entry(path: str) -> dict[str, Any]:
    key = (os.getpid(), path)  # pid in key: a forked child gets fresh locks and cache
    with _REG_GUARD:
        e = _REGISTRY.get(key)
        if e is None:
            e = {"tlock": threading.Lock(), "owner": None, "cache": None}
            _REGISTRY[key] = e
        return e


def empty_state(config: dict) -> dict:
    return {
        "config": config,
        "ideas": {},
        "tags": {},
        "logs": {"refits": [], "batches": [], "tagging": [], "realjudge": []},
        "counters": {"next_idea": 0, "wave": 0, "scored_seq": 0},
    }


def _dumps(obj: Any) -> str:
    # Compact, no indent: the file can reach tens of MB.
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    try:  # make the rename durable; not supported everywhere
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


class _Cache:
    """Last committed journal: text, a private parsed copy, and per-idea JSON strings.
    The private copy is never handed to callers, so caller-held references can't corrupt it."""

    __slots__ = ("text", "before", "idea_strs", "sig")

    def __init__(self, text: str, before: dict, idea_strs: dict[str, str]):
        self.text = text
        self.before = before
        self.idea_strs = idea_strs
        self.sig: tuple | None = None

    @classmethod
    def from_text(cls, text: str) -> "_Cache":
        before = json.loads(text)
        strs = {i: _dumps(v) for i, v in (before.get("ideas") or {}).items()}
        return cls(text, before, strs)

    def commit(self, state: dict) -> tuple[str, list[str]] | None:
        """Advance the cache to `state`. Returns (new file text, changed/removed idea ids),
        or None if nothing changed (cache untouched)."""
        if not isinstance(state, dict) or not isinstance(state.get("ideas"), dict):
            raise TypeError("journal state must be a dict with an 'ideas' dict")
        ideas = state["ideas"]
        b_ideas = self.before.get("ideas") or {}
        changed = [i for i, v in ideas.items() if b_ideas.get(i, _MISSING) != v]
        changed_set = set(changed)
        removed = [i for i in b_ideas if i not in ideas]
        rest_changed = {k for k in set(state) | set(self.before)
                        if k != "ideas" and state.get(k, _MISSING) != self.before.get(k, _MISSING)}
        if (not changed and not removed and not rest_changed
                and list(state) == list(self.before) and list(ideas) == list(b_ideas)):
            return None
        new_strs: dict[str, str] = {}
        for i, v in ideas.items():
            s = self.idea_strs.get(i) if i not in changed_set else None
            new_strs[i] = s if s is not None else _dumps(v)
        parts = []
        for k, v in state.items():
            if k == "ideas":
                body = ",".join(f"{_dumps(i)}:{s}" for i, s in new_strs.items())
                parts.append(f'"ideas":{{{body}}}')
            else:
                parts.append(f"{_dumps(k)}:{_dumps(v)}")
        text = "{" + ",".join(parts) + "}"
        before: dict = {}
        for k in state:
            if k == "ideas":
                before[k] = {i: (b_ideas[i] if (i in b_ideas and i not in changed_set)
                                 else json.loads(new_strs[i])) for i in ideas}
            elif k in rest_changed:
                before[k] = json.loads(_dumps(state[k]))
            else:
                before[k] = self.before[k]
        self.text, self.before, self.idea_strs = text, before, new_strs
        return text, changed + removed


class Journal:
    def __init__(self, run_dir: str | Path, lock_timeout: float = -1):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / JOURNAL
        self.lock_path = self.run_dir / LOCK
        self.events_path = self.run_dir / EVENTS
        self._key = str(self.path.resolve())
        self._lock_timeout = lock_timeout

    # ------------------------------------------------------------------ io
    def _load_text(self) -> str:
        with open(self.path, "r", encoding="utf-8") as f:
            return f.read()

    def _sig(self) -> tuple:
        """Cache validator: journal stat + events.jsonl size (every commit appends an event,
        so the pair changes on every write, even if an inode number is reused)."""
        st = os.stat(self.path)
        try:
            ev = os.stat(self.events_path).st_size
        except FileNotFoundError:
            ev = -1
        return (st.st_ino, st.st_size, st.st_mtime_ns, ev)

    def read(self) -> dict:
        """Latest committed snapshot (a fresh object; no lock held)."""
        last_err: Exception | None = None
        for attempt in range(5):
            try:
                return json.loads(self._load_text())
            except FileNotFoundError:
                raise
            except (json.JSONDecodeError, OSError) as e:  # paranoia (NFS); replace is atomic
                last_err = e
                time.sleep(0.05 * (attempt + 1))
        raise RuntimeError(f"cannot read {self.path}: {last_err}")

    # --------------------------------------------------------- transaction
    @contextmanager
    def transaction(self, worker: str, op: str) -> Iterator[dict]:
        """filelock (journal.lock) -> load latest journal.json -> yield mutable state ->
        on normal exit: write journal.json.tmp, fsync, os.replace; append one events.jsonl
        line. On exception: discard changes, release lock, re-raise."""
        if not self.path.exists():
            raise FileNotFoundError(f"no journal at {self.path}; use Journal.init()")
        e = _entry(self._key)
        me = threading.get_ident()
        if e["owner"] == me:
            raise RuntimeError(
                f"nested/interleaved transaction on {self.path} in the same thread "
                f"(op={op!r}); never await inside a transaction"
            )
        e["tlock"].acquire()
        try:
            e["owner"] = me
            flock = FileLock(str(self.lock_path), timeout=self._lock_timeout, thread_local=False)
            with flock:
                cache = e["cache"]
                e["cache"] = None  # invalid until this transaction finishes cleanly
                if cache is None or cache.sig != self._sig():
                    cache = _Cache.from_text(self._load_text())
                    cache.sig = self._sig()
                state = json.loads(cache.text)
                yield state  # on exception: cache stays None-or-unchanged; nothing written
                new = cache.commit(state)
                if new is None:  # no-op transaction: nothing to write or log
                    e["cache"] = cache
                    return
                text, ids = new
                _write_atomic(self.path, text)
                ev = {"t": time.time(), "worker": worker, "op": op, "ids": ids}
                with open(self.events_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                    f.flush()
                cache.sig = self._sig()
                e["cache"] = cache
        finally:
            e["owner"] = None
            e["tlock"].release()

    # ---------------------------------------------------------------- init
    @staticmethod
    def init(run_dir: str | Path, config: dict) -> "Journal":
        """Create run dir + empty state. Refuses if a journal already exists."""
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        j = Journal(run_dir)
        with FileLock(str(j.lock_path), thread_local=False):
            if j.path.exists():
                raise FileExistsError(f"journal already exists: {j.path} (use --resume)")
            _write_atomic(j.path, _dumps(empty_state(config)))
            cfg_path = run_dir / CONFIG
            if not cfg_path.exists():
                import yaml

                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
            with open(j.events_path, "a", encoding="utf-8") as f:
                f.write(_dumps({"t": time.time(), "worker": "init", "op": "init", "ids": []}) + "\n")
        return j


# ===================================================================== helpers
def new_idea_id(state: dict) -> str:
    """Next id "i000123" (monotonic). Bumps counters.next_idea; the caller inserts the idea."""
    c = state["counters"]
    while True:
        n = int(c.get("next_idea", 0))
        c["next_idea"] = n + 1
        iid = f"i{n:06d}"
        if iid not in state["ideas"]:
            return iid


def _claim_stale(idea: dict, now: float, timeout_s: float) -> bool:
    cl = idea.get("claim")
    if not cl:
        return True
    return now - float(cl.get("at", 0.0)) > timeout_s


def claim(state: dict, worker: str, stage: str, eligible: Callable[[dict], bool],
          limit: int, now: float, timeout_s: float) -> list[str]:
    """Claim up to `limit` eligible, non-failed ideas that are unclaimed or whose claim is
    older than `timeout_s`. Sets idea["claim"]={"worker","stage","at"}. Oldest created first."""
    if limit <= 0:
        return []
    # dict order is creation order; sort by created_at (stable) in case of manual edits.
    order = sorted(enumerate(state["ideas"].values()),
                   key=lambda p: (p[1].get("created_at", 0.0), p[0]))
    out: list[str] = []
    for _, idea in order:
        if idea.get("status") == "failed":
            continue
        if not _claim_stale(idea, now, timeout_s):
            continue
        if not eligible(idea):
            continue
        idea["claim"] = {"worker": worker, "stage": stage, "at": now}
        out.append(idea["id"])
        if len(out) >= limit:
            break
    return out


def release(state: dict, idea_id: str) -> None:
    state["ideas"][idea_id]["claim"] = None


def set_status(state: dict, idea_id: str, status: str, now: float) -> None:
    """Set status, record times[status]=now, clear the claim."""
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}")
    idea = state["ideas"][idea_id]
    idea["status"] = status
    idea.setdefault("times", {})[status] = now
    idea["claim"] = None


def fail(state: dict, idea_id: str, error: str, now: float) -> None:
    set_status(state, idea_id, "failed", now)
    state["ideas"][idea_id]["error"] = error


def reap_stale(state: dict, now: float, timeout_s: float) -> list[str]:
    """Clear claims older than timeout_s (orchestrator reaper). Returns the ids released."""
    out = []
    for iid, idea in state["ideas"].items():
        if idea.get("claim") and _claim_stale(idea, now, timeout_s):
            idea["claim"] = None
            out.append(iid)
    return out
