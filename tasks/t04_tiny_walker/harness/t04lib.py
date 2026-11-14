"""Shared harness code for t04_tiny_walker (Tiny Walker, dataset problem #4).

Environment, mode table, the parameter / size rules, the policy-file scan and the rollout used by evaluation.
Depends only on numpy, gymnasium[box2d] and the standard library.
"""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path

import numpy as np

ENV_ID = "BipedalWalker-v3"   # Hardcore had no headroom at this budget (see README)
OBS_DIM, ACT_DIM = 24, 4
MAX_PARAMS = 699                 # "fewer than 700 parameters"
MAX_BYTES = 3584                 # 3.5 KiB: artifact/policy.npz on disk
MAX_STR_CHARS = 256              # non-docstring string literals in policy.py
TRAIN_SEED_RANGE = (0, 10**9)    # training episodes must use seeds in [lo, hi); eval seeds lie in [1e9, 2^31)
GRACE_S = 10
ALLOWED_DTYPES = {"float32", "float16", "uint16", "int8", "uint8", "int16", "int32"}   # uint16 = bf16 bit patterns
POLICY_IMPORTS = {"numpy", "math", "__future__"}
BANNED_NAMES = {"open", "eval", "exec", "compile", "__import__", "globals", "vars", "input", "breakpoint",
                "getattr", "setattr", "__builtins__", "memoryview", "id", "hash"}
BANNED_ATTRS = {"random", "load", "loadtxt", "genfromtxt", "fromfile", "memmap", "fromregex", "DataSource",
                "ctypeslib", "lib", "testing", "f2py", "distutils", "__dict__", "__globals__", "__code__",
                "__subclasses__"}

# budget_s: the solution's train(ctx) call; n_cpus: worker processes it should use (= cores it gets)
MODES = {
    "full":      {"budget_s": 600,  "n_cpus": 16, "n_eval": 100},
    "smoke":     {"budget_s": 60,   "n_cpus": 16, "n_eval": 20},
    "cpu-smoke": {"budget_s": 30,   "n_cpus": 4,  "n_eval": 8},
}


def private_dir() -> Path:
    return Path(os.environ.get("AR_ROOT", "/usr/xtmp/kg349/ar")) / "private" / "t04_tiny_walker"


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        return os.cpu_count() or 1


def n_cpus_for(mode: str) -> int:
    return max(1, min(MODES[mode]["n_cpus"], available_cpus()))


def sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def make_env(env_id: str = ENV_ID):
    import gymnasium as gym
    return gym.make(env_id)


# ----------------------------------------------------------------------------------------- params / size rules
def check_params(params) -> tuple[dict, list[str]]:
    """Validate the dict train() returned. Returns (report, errors). Does not count policy.py literals."""
    errs = []
    if not isinstance(params, dict) or not params:
        return {"n_array_params": None}, ["train() must return a non-empty dict[str, np.ndarray]"]
    n = 0
    shapes = {}
    for k, v in params.items():
        if not (isinstance(k, str) and k.isidentifier()):
            errs.append(f"key {k!r} is not a python identifier")
            continue
        if not isinstance(v, np.ndarray):
            errs.append(f"{k}: {type(v).__name__} is not an np.ndarray")
            continue
        if v.dtype.name not in ALLOWED_DTYPES:
            errs.append(f"{k}: dtype {v.dtype} not allowed (allowed: {sorted(ALLOWED_DTYPES)})")
        elif v.dtype.kind == "f" and not np.all(np.isfinite(v)):
            errs.append(f"{k}: non-finite values")
        n += int(v.size)
        shapes[k] = [list(v.shape), v.dtype.name]
    return {"n_array_params": n, "arrays": shapes}, errs


def npz_bytes(params) -> int:
    buf = io.BytesIO()
    np.savez(buf, **params)
    return len(buf.getvalue())


def count_params(params, policy_path=None) -> dict:
    """What the harness will count: array elements (+ policy.py numeric literals) and the saved .npz size."""
    rep, errs = check_params(params)
    lit = scan_policy(policy_path)["n_literal_params"] if policy_path else 0
    n = (rep.get("n_array_params") or 0) + lit
    size = npz_bytes(params) if not errs else None
    return {"n_params": n, "n_array_params": rep.get("n_array_params"), "n_literal_params": lit,
            "npz_bytes": size, "params_ok": not errs and n <= MAX_PARAMS,
            "size_ok": size is not None and size <= MAX_BYTES, "errors": errs}


# ----------------------------------------------------------------------------------------- policy.py scan
def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                out.add(id(b[0].value))
    return out


def scan_policy(path) -> dict:
    """AST rules for policy.py: numeric literals other than 0, 1, 2 count as parameters; string literals are
    capped; only numpy/math imports; no file access, RNG or dynamic code."""
    src = Path(path).read_text()
    errs, lit_vals = [], []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"n_literal_params": 0, "errors": [f"policy.py syntax error: {e}"], "ok": False}
    docs = _docstring_nodes(tree)
    str_chars = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool) or v is None or v is Ellipsis:
                continue
            if isinstance(v, (int, float, complex)):
                if v not in (0, 1, 2):
                    lit_vals.append(v)
            elif isinstance(v, bytes):
                errs.append(f"line {node.lineno}: bytes literal not allowed")
            elif isinstance(v, str) and id(node) not in docs:
                str_chars += len(v)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in POLICY_IMPORTS:
                    errs.append(f"line {node.lineno}: import {m} (policy.py may import only numpy, math)")
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "numpy":
                for a in node.names:
                    if a.name in BANNED_ATTRS or a.name == "*":
                        errs.append(f"line {node.lineno}: from numpy import {a.name}")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errs.append(f"line {node.lineno}: {node.id} not allowed in policy.py")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            errs.append(f"line {node.lineno}: .{node.attr} not allowed in policy.py (no RNG / file access)")
        elif isinstance(node, ast.Global):
            errs.append(f"line {node.lineno}: global statement not allowed in policy.py")
    if str_chars > MAX_STR_CHARS:
        errs.append(f"string literals total {str_chars} chars > {MAX_STR_CHARS}")
    return {"n_literal_params": len(lit_vals), "literals": lit_vals[:50], "string_chars": str_chars,
            "source_bytes": len(src.encode()), "errors": errs, "ok": not errs}


def import_file(path, name="t04_policy"):
    """Import a file as module `name`, registered in sys.modules (so multiprocessing can pickle its functions)."""
    import sys
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------- rollout
def rollout(policy, env, seed: int) -> dict:
    """One deterministic episode: env.reset(seed=seed), policy.reset(), act until terminated/truncated."""
    obs, _ = env.reset(seed=int(seed))
    policy.reset()
    total, steps, finite = 0.0, 0, True
    while True:
        a = np.asarray(policy.act(np.asarray(obs, dtype=np.float32)), dtype=np.float64).reshape(-1)
        if a.shape != (ACT_DIM,) or not np.all(np.isfinite(a)):
            finite = False
            a = np.zeros(ACT_DIM) if a.shape != (ACT_DIM,) else np.nan_to_num(a)
        obs, r, term, trunc, _ = env.step(np.clip(a, -1.0, 1.0).astype(np.float32))
        total += float(r)
        steps += 1
        if term or trunc:
            break
    return {"seed": int(seed), "return": total, "steps": steps, "terminated": bool(term),
            "fell": bool(term and r <= -100), "action_ok": finite}


_W = {}


def _worker_init(policy_path, npz_path, env_id):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mod = import_file(policy_path)
    with np.load(npz_path, allow_pickle=False) as z:
        params = {k: z[k] for k in z.files}
    _W.update(mod=mod, params=params, env=make_env(env_id))


def _worker_episode(seed):
    pol = _W["mod"].Policy({k: v.copy() for k, v in _W["params"].items()})
    return rollout(pol, _W["env"], seed)


def evaluate_seeds(policy_path, npz_path, seeds, n_procs, env_id=ENV_ID) -> list[dict]:
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(n_procs, initializer=_worker_init, initargs=(str(policy_path), str(npz_path), env_id)) as pool:
        res = pool.map(_worker_episode, [int(s) for s in seeds], chunksize=1)
    return res


def write_json(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2))
