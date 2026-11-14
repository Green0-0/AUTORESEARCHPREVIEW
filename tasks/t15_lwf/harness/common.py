"""Shared helpers for the t15_lwf harness: paths, modes, data loading, aux-state accounting."""
import gc, importlib.util, io, os, sys, types
from pathlib import Path

import numpy as np
import torch

TASK_ID = "t15_lwf"
NUM_TASKS, CPT, NUM_CLASSES = 10, 10, 100
AUX_LIMIT_BYTES = 1 << 20          # 1 MiB
EXEMPT_FACTOR = 8                  # parameter-shaped copies allowed: <= 8x model parameter bytes
UNACCOUNTED_GPU_LIMIT = 8 << 20    # live CUDA bytes not reachable from the learner (hidden copies)

# mode -> (train examples per class, test examples per class or None=all, stream budget s, device)
MODES = {
    "full":      dict(train_per_class=None, test_per_class=None, budget_s=900.0, device="cuda"),
    "smoke":     dict(train_per_class=100,  test_per_class=None, budget_s=60.0,  device="cuda"),
    "cpu-smoke": dict(train_per_class=5,    test_per_class=10,   budget_s=90.0,  device="cpu"),
}


def data_dir() -> Path:
    return Path(os.environ.get("AR_DATA", "/usr/xtmp/kg349/ar/data"))


def private_dir() -> Path:
    root = os.environ.get("AR_ROOT")
    base = Path(root) if root else data_dir().parent
    return base / "private" / TASK_ID


def subset_per_class(y: np.ndarray, k, seed=0):
    if k is None:
        return np.arange(len(y))
    rng = np.random.RandomState(seed)
    idx = [rng.permutation(np.flatnonzero(y == c))[:k] for c in range(NUM_CLASSES)]
    return np.sort(np.concatenate(idx))


def load_solution(solution_dir: Path):
    solution_dir = Path(solution_dir).resolve()
    sys.path.insert(0, str(solution_dir))
    spec = importlib.util.spec_from_file_location("learner", solution_dir / "learner.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["learner"] = mod
    spec.loader.exec_module(mod)
    for name in ("build_model", "Learner", "predict"):
        if not hasattr(mod, name):
            raise AttributeError(f"solution/learner.py must define {name}")
    return mod


def solution_modules(solution_dir: Path):
    sd = str(Path(solution_dir).resolve())
    out = []
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if isinstance(m, types.ModuleType) and f and str(Path(f).resolve()).startswith(sd):
            out.append(m)
    return out


def serialized_size(obj) -> int:
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.tell()


# ---------------------------------------------------------------- reachability scan
def _storage_key(t: torch.Tensor):
    try:
        s = t.untyped_storage()
        return (t.device.type, t.device.index, s.data_ptr(), s.nbytes())
    except Exception:
        return (t.device.type, t.device.index, t.data_ptr(), t.numel() * t.element_size())


def collect_reachable(roots, max_objects=2_000_000):
    """Walk the object graph from `roots`; return list of (path, kind, obj) for tensors,
    ndarrays, bytes-like and long numeric python lists. Skips modules/classes/functions' code."""
    found, seen, stack, n = [], set(), [(r, p) for p, r in roots], 0
    while stack and n < max_objects:
        obj, path = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid); n += 1
        if isinstance(obj, torch.Tensor):
            found.append((path, "tensor", obj))
            if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
                stack.append((obj.grad, path + ".grad"))
            continue
        if isinstance(obj, np.ndarray):
            found.append((path, "ndarray", obj)); continue
        if isinstance(obj, (bytes, bytearray, memoryview)):
            if len(obj) > 4096:
                found.append((path, "bytes", obj))
            continue
        if isinstance(obj, (str, int, float, bool, complex, type(None), type, types.FunctionType,
                            types.BuiltinFunctionType, types.MethodWrapperType, torch.device, torch.dtype)):
            continue
        if isinstance(obj, types.ModuleType):
            continue  # module globals are scanned explicitly via solution_modules()
        if isinstance(obj, types.MethodType):
            stack.append((obj.__self__, path + ".__self__")); continue
        if isinstance(obj, dict):
            for k, v in obj.items():
                stack.append((v, f"{path}[{k!r}]" if isinstance(k, (str, int)) else path + "[?]"))
                if not isinstance(k, (str, int)):
                    stack.append((k, path + "[key]"))
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            if len(obj) > 1024 and all(isinstance(v, (int, float)) for v in list(obj)[:64]):
                found.append((path, "pylist", obj))
            for i, v in enumerate(obj):
                stack.append((v, f"{path}[{i}]"))
            continue
        if isinstance(obj, torch.Generator):
            continue
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            stack.append((d, path))
        for s in getattr(type(obj), "__slots__", ()) or ():
            if isinstance(s, str) and hasattr(obj, s):
                stack.append((getattr(obj, s), f"{path}.{s}"))
        if isinstance(obj, torch.optim.Optimizer):
            stack.append((obj.state, path + ".state"))
    return found


def account(learner, solution_dir, extra_exclude=()):
    """Return accounting dict for the learner's persistent state (call between tasks)."""
    gc.collect()
    model = getattr(learner, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("learner.model must be an nn.Module")
    params = list(model.parameters())
    param_keys = {_storage_key(p) for p in params}
    param_shapes = {tuple(p.shape) for p in params}
    param_bytes = sum(p.numel() * p.element_size() for p in params)
    roots = [("learner", learner)] + [(f"module:{m.__name__}", vars(m)) for m in solution_modules(solution_dir)]
    found = collect_reachable(roots)
    storages = {}
    for path, kind, obj in found:
        if kind == "tensor":
            key = _storage_key(obj)
            nbytes = key[3]
            image_like = obj.dim() >= 3 and tuple(obj.shape[-3:]) in ((3, 32, 32), (32, 32, 3))
            exempt = key in param_keys or (tuple(obj.shape) in param_shapes and not image_like)
            cls = "model_param" if key in param_keys else ("param_shaped" if exempt else "aux")
            dev = obj.device.type
        elif kind == "ndarray":
            key, nbytes, cls, dev = ("np", obj.__array_interface__["data"][0], obj.nbytes), obj.nbytes, "aux", "cpu"
        elif kind == "bytes":
            key, nbytes, cls, dev = ("b", id(obj)), len(obj), "aux", "cpu"
        else:
            key, nbytes, cls, dev = ("l", id(obj)), 8 * len(obj), "aux", "cpu"
        if key in extra_exclude:
            continue
        prev = storages.get(key)
        if prev is None or (prev["cls"] != "model_param" and cls == "model_param"):
            storages[key] = dict(path=path, nbytes=nbytes, cls=cls, dev=dev)
    by = {"model_param": 0, "param_shaped": 0, "aux": 0}
    for v in storages.values():
        by[v["cls"]] += v["nbytes"]
    cuda_reached = sum(v["nbytes"] for v in storages.values() if v["dev"] == "cuda")
    top_aux = sorted((v for v in storages.values() if v["cls"] == "aux"), key=lambda v: -v["nbytes"])[:8]
    return dict(model_param_bytes=param_bytes, param_shaped_bytes=by["param_shaped"],
                scanned_aux_bytes=by["aux"], cuda_reached_bytes=cuda_reached,
                top_aux=[(v["path"], v["nbytes"]) for v in top_aux])
