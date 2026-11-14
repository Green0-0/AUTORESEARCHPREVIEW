"""Surrogate-score parsing, calibration and selective improvement F (pure functions).

Spec: "Judging" and "Objective: selective improvement"; interfaces per CONTRACTS.md.
"""
from __future__ import annotations

import re

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_LEAD = re.compile(r"^\s*(" + _NUM + r")\s*(\\?%)?")
_FRAC = re.compile(r"^\s*\\[dt]?frac\s*\{\s*(" + _NUM + r")\s*\}\s*\{\s*(" + _NUM + r")\s*\}")


def _last_boxed_content(text: str) -> str | None:
    start = text.rfind("\\boxed")
    while start != -1:
        i = start + len("\\boxed")
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == "{":
            depth, j = 0, i
            while j < len(text):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[i + 1:j]
                j += 1
            return None  # unterminated last box
        start = text.rfind("\\boxed", 0, start)
    return None


def _clamp(v: float) -> float:
    return min(1.0, max(0.0, v))


def _box_value(content: str | None) -> float | None:
    """Parse the content of one \\boxed{...}: leading float, percent or \\frac; clamped to [0,1]."""
    if content is None:
        return None
    content = content.replace("\\text{", "").replace("$", "")
    m = _FRAC.match(content)
    if m:
        den = float(m.group(2))
        if den == 0:
            return None
        return _clamp(float(m.group(1)) / den)
    m = _LEAD.match(content)
    if not m:
        return None
    if re.match(r"[A-Za-z]", content[m.end():]):   # e.g. the template placeholder "0.xx" echoed back
        return None
    v = float(m.group(1))
    if v != v:  # NaN guard
        return None
    if m.group(2):
        v /= 100.0
    return _clamp(v)


def parse_boxed(text: str) -> float | None:
    """Float inside the LAST \\boxed{...}, clamped to [0,1]; None if missing/unparseable.

    Accepts a leading float (e.g. "0.7", " .65 ", "0.70/1" -> 0.70), a percent ("70%" -> 0.7)
    and \\frac{a}{b}. Used by the basic (J1/J2), rubric (J3) and few-shot (J4) judges."""
    if not text:
        return None
    return _box_value(_last_boxed_content(text))


def _all_boxes(text: str) -> list[tuple[int, int, str]]:
    """Every terminated \\boxed{...} as (start, end, content), in text order."""
    out, pos = [], 0
    while True:
        start = text.find("\\boxed", pos)
        if start == -1:
            return out
        i = start + len("\\boxed")
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text) or text[i] != "{":
            pos = start + 1
            continue
        depth, j = 0, i
        while j < len(text):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth != 0:
            return out  # unterminated
        out.append((start, j + 1, text[i + 1:j]))
        pos = j + 1


_IDEA_LABEL = re.compile(r"\bidea\s*(?:#|no\.?|number)?\s*(\d+)", re.I)
_LEAD_NUM = re.compile(r"^[\s>*_#\-\[(|]*(\d+)\s*[.):\]|]")
_PLAIN_LINE = re.compile(r"^[\s>*_#\-\[(|]*idea\s*#?\s*(\d+)[\s*_\])|]*[:=\-\u2013\u2014]+[\s*_]*(" + _NUM + r")\s*(%)?", re.I | re.M)


def parse_ranked(text: str, n: int) -> list[float | None]:
    """Ranking judge (J5): one score per new idea "Idea 1".."Idea n" -> list of n floats in [0,1] (None = not found).

    Passes, each only filling ideas still missing; within a pass a LATER answer overrides an earlier one
    (models often restate the final scores at the end):
      1. a \\boxed{} whose own line carries the label before it ("Idea 3: \\boxed{0.4}", "**Idea 3** - \\boxed{..}",
         "3. \\boxed{..}"); the label closest to the box wins;
      2. a box with no label on its line takes the nearest "Idea k" in the 300 characters before it;
      3. "Idea k: 0.4" lines without a box;
      4. if no box carried any label and there are exactly n boxes, they are assigned in order.
    """
    out: list[float | None] = [None] * n
    if not text or n <= 0:
        return out
    boxes = _all_boxes(text)
    labelled = set()
    found1: dict[int, float] = {}
    for k, (start, _end, content) in enumerate(boxes):
        v = _box_value(content)
        line = text[text.rfind("\n", 0, start) + 1:start]
        ids = _IDEA_LABEL.findall(line)
        if ids:
            idx = int(ids[-1])
        else:
            m = _LEAD_NUM.match(line)
            if not m:
                continue
            idx = int(m.group(1))
        labelled.add(k)
        if v is not None and 1 <= idx <= n:
            found1[idx] = v
    for idx, v in found1.items():
        out[idx - 1] = v
    if all(x is not None for x in out):
        return out
    found2: dict[int, float] = {}
    for k, (start, _end, content) in enumerate(boxes):
        if k in labelled:
            continue
        v = _box_value(content)
        ids = _IDEA_LABEL.findall(text[max(0, start - 300):start])
        if v is not None and ids and 1 <= int(ids[-1]) <= n and out[int(ids[-1]) - 1] is None:
            found2[int(ids[-1])] = v
    for idx, v in found2.items():
        out[idx - 1] = v
    found3: dict[int, float] = {}
    for m in _PLAIN_LINE.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= n and out[idx - 1] is None:
            v = float(m.group(2))
            found3[idx] = _clamp(v / 100.0 if m.group(3) else v)
    for idx, v in found3.items():
        out[idx - 1] = v
    if not labelled and not found2 and not found3 and len(boxes) == n:
        vals = [_box_value(c) for _, _, c in boxes]
        out = [o if o is not None else v for o, v in zip(out, vals)]
    return out


def calibrate(s_raw: float, cal: dict) -> float:
    a = float((cal or {}).get("a", 0.0))
    b = float((cal or {}).get("b", 1.0))
    return _clamp(a + b * float(s_raw))


def comparison_set(state: dict, idea_id: str) -> list[dict]:
    """Ideas created strictly before idea_id (created_at), not failed, with s_cal, excluding itself."""
    me = state["ideas"][idea_id]
    t0 = me["created_at"]
    return [i for iid, i in state["ideas"].items()
            if iid != idea_id and i.get("status") != "failed" and i.get("s_cal") is not None
            and i.get("created_at") is not None and i["created_at"] < t0]


def compute_b_t(state: dict, idea_id: str, s_base: float) -> dict:
    tags = list(dict.fromkeys(state["ideas"][idea_id].get("tags") or []))
    comp = comparison_set(state, idea_id)
    b_t = {}
    for t in tags:
        vals = [float(i["s_cal"]) for i in comp if t in (i.get("tags") or [])]
        b_t[t] = max(vals) if vals else float(s_base)
    return b_t


def compute_F(state: dict, idea_id: str, s: float, s_min: float, s_base: float,
              b_t: dict | None = None) -> tuple[float | None, dict]:
    """F = max_t max(0, s - b_t - s_min). If b_t is given it is reused (frozen) verbatim.

    Returns (F, {"b_t": {tag: float}, "s": s, "best_tag": tag|None}). An idea with no tags
    (and no b_t passed) gets F=None (spec: no F, not an observation)."""
    if b_t is None:
        b_t = compute_b_t(state, idea_id, s_base)
    else:
        b_t = {k: float(v) for k, v in b_t.items()}
    if not b_t:
        return None, {"b_t": {}, "s": float(s), "best_tag": None}
    best_tag, F = None, 0.0
    for t, b in b_t.items():
        g = max(0.0, float(s) - b - float(s_min))
        if g > F:
            best_tag, F = t, g
    return F, {"b_t": b_t, "s": float(s), "best_tag": best_tag}
