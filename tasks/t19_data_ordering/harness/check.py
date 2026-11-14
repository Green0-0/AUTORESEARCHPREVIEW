"""Mechanical constraint checks on the solution source (cheap, static).
Hard failures: missing scheduler.py / Scheduler class. Everything else is reported as warnings for the reviewer."""
import re
from pathlib import Path

SUSPICIOUS = {
    r"\bprivate\b|test_x|test_y|cifar-100-python": "references test / private data",
    r"gc\.get_objects|inspect\.stack|sys\._getframe|_h\b|\._h\.": "introspection / harness internals (model, optimizer access)",
    r"\bstate_dict\b|\.parameters\(\)|param_groups|\.grad\b": "touches model parameters / optimizer",
    r"\btorch\.manual_seed\b|cuda\.manual_seed": "sets global torch seeds (harness uses its own generators; allowed but note)",
    r"\bctx\.(images|labels)\s*\[[^\]]*\]\s*=": "writes into ctx tensors",
}


def static_check(solution_dir):
    sol = Path(solution_dir)
    rep = {"ok": True, "errors": [], "warnings": [], "files": []}
    f = sol / "scheduler.py"
    if not f.exists():
        rep["ok"] = False
        rep["errors"].append("solution/scheduler.py missing")
        return rep
    for p in sorted(sol.rglob("*.py")):
        rep["files"].append(str(p.relative_to(sol)))
        src = p.read_text(errors="replace")
        for pat, why in SUSPICIOUS.items():
            if re.search(pat, src):
                rep["warnings"].append(f"{p.name}: {why} (/{pat}/)")
    if not re.search(r"^class\s+Scheduler\b", f.read_text(errors="replace"), re.M):
        rep["ok"] = False
        rep["errors"].append("scheduler.py defines no top-level `class Scheduler`")
    return rep
