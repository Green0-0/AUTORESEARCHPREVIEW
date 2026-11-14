"""Template utility parser/evaluator (spec "Template utilities"; CONTRACTS.md "Templating").

A template is parsed into an AST first and then evaluated innermost-first, once per prompt.
Substituted content (task prompt, tags, summaries, helper variables) is inserted as plain text
and never re-parsed, so text containing ``{{prompt}}`` or unbalanced braces passes through
literally.

Syntax: ``{{name}}`` or ``{{name;arg1;arg2...}}``. Arguments may contain nested utilities,
e.g. ``{{references;{{rangeint;1;3}}}}``. Inside a utility, ``;`` separates arguments.

List-utility semantics: integer N = at most N items; decimal N (contains '.') = that fraction
of the available items, rounded to nearest, minimum 1. Capped at the number available, sampled
without replacement, printed in creation order (one item per line). No arg = all items.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Union

import numpy as np

__all__ = ["RenderContext", "TemplateError", "parse", "render", "list_count", "Node", "Text"]


class TemplateError(ValueError):
    """Malformed template, unknown utility or bad utility arguments."""


@dataclass
class RenderContext:
    prompt: str = ""
    references: list[str] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)
    tags: list[tuple[str, str]] = field(default_factory=list)   # (name, description), pool creation order
    judged_ideas: list[dict] = field(default_factory=list)      # {"id","summary","s_cal","real_score"}, creation order


# ----------------------------------------------------------------------------- AST
@dataclass
class Text:
    text: str


@dataclass
class Node:
    name: str
    args: list[list["Part"]]   # each arg is a sequence of parts
    pos: int = 0


Part = Union[Text, Node]


def parse(template: str) -> list[Part]:
    """Parse a template into a list of Text/Node parts. Raises TemplateError if unbalanced."""
    i, n = 0, len(template)
    # stack of open nodes: each entry is (pos, list_of_args) where args is list[list[Part]]
    root: list[Part] = []
    stack: list[tuple[int, list[list[Part]]]] = []
    buf: list[str] = []

    def cur() -> list[Part]:
        return stack[-1][1][-1] if stack else root

    def flush() -> None:
        if buf:
            cur().append(Text("".join(buf)))
            buf.clear()

    while i < n:
        if template.startswith("{{", i):
            flush()
            stack.append((i, [[]]))
            i += 2
        elif template.startswith("}}", i):
            if not stack:
                raise TemplateError(f"unmatched '}}}}' at position {i}")
            flush()
            pos, args = stack.pop()
            name_parts = args[0]
            if any(isinstance(p, Node) for p in name_parts):
                raise TemplateError(f"utility name cannot contain a utility (position {pos})")
            name = "".join(p.text for p in name_parts).strip()
            if not name:
                raise TemplateError(f"empty utility at position {pos}")
            cur().append(Node(name, args[1:], pos))
            i += 2
        elif template[i] == ";" and stack:
            flush()
            stack[-1][1].append([])
            i += 1
        else:
            buf.append(template[i])
            i += 1
    if stack:
        raise TemplateError(f"unclosed '{{{{' at position {stack[-1][0]}")
    flush()
    return root


# ----------------------------------------------------------------------------- helpers
def _one_line(s: str) -> str:
    return re.sub(r"\s*\n\s*", " ", str(s)).strip()


# Deviation from the spec (overnight 2026-11-08): a *fractional* N (e.g. NoIdea's
# {{idea_summaries;{{rangefloat;0.1;1}}}}) scales with the run size, so after ~1000 judged ideas the
# prompt would exceed the 32k context and long prompts already thrash the KV cache. Fractional draws
# are therefore capped at MAX_FRACTION_ITEMS items. Integer N and "no N" (full list) are unchanged.
MAX_FRACTION_ITEMS = 30


def list_count(arg: str | None, available: int) -> int:
    """Number of items to sample for a list utility given its (evaluated) N argument."""
    if available <= 0:
        return 0
    if arg is None:
        return available
    a = arg.strip()
    try:
        if "." in a:
            frac = float(a)
            if not math.isfinite(frac) or frac < 0:
                raise ValueError
            k = max(1, int(math.floor(frac * available + 0.5)))   # round half up
            k = min(k, MAX_FRACTION_ITEMS)
        else:
            k = int(a)
            if k < 0:
                raise ValueError
    except ValueError:
        raise TemplateError(f"bad list size N={arg!r}") from None
    return min(k, available)


def _sample_indices(n_avail: int, k: int, rng) -> list[int]:
    if k >= n_avail:
        return list(range(n_avail))
    if rng is None:
        raise TemplateError("random draw needed but no rng supplied")
    return sorted(int(x) for x in rng.choice(n_avail, size=k, replace=False))


def _fmt_score(x) -> str:
    return f"{float(x):.2f}"


def _num(s: str, kind, util: str):
    try:
        return kind(s.strip())
    except ValueError:
        raise TemplateError(f"{util}: bad numeric argument {s!r}") from None


# ----------------------------------------------------------------------------- evaluation
LIST_UTILS = ("references", "facts", "tags", "idea_summaries", "idea_summaries_and_critiques")
SCALAR_UTILS = ("rangeint", "rangefloat", "choice", "prompt")


class _Evaluator:
    def __init__(self, ctx: RenderContext, rng, variables: dict[str, str] | None):
        self.ctx = ctx
        self.rng = rng
        self.variables = variables or {}
        self.draws: list[dict] = []

    def parts(self, parts: list[Part]) -> str:
        return "".join(p.text if isinstance(p, Text) else self.node(p) for p in parts)

    def node(self, node: Node) -> str:
        # innermost first: evaluate every argument before this node
        args = [self.parts(a).strip() for a in node.args]
        name = node.name
        if name in self.variables:
            if args:
                raise TemplateError(f"variable {{{{{name}}}}} takes no arguments")
            return str(self.variables[name])
        fn = getattr(self, "u_" + name, None)
        if fn is None or name not in LIST_UTILS + SCALAR_UTILS:
            raise TemplateError(f"unknown template utility {{{{{name}}}}} at position {node.pos}")
        return fn(args)

    def _need_rng(self, util):
        if self.rng is None:
            raise TemplateError(f"{util}: random draw needed but no rng supplied")
        return self.rng

    # scalar utilities
    def u_rangeint(self, args):
        if len(args) != 2:
            raise TemplateError(f"rangeint takes 2 arguments, got {len(args)}")
        a, b = _num(args[0], int, "rangeint"), _num(args[1], int, "rangeint")
        if b < a:
            raise TemplateError(f"rangeint: empty range [{a}, {b}]")
        v = int(self._need_rng("rangeint").integers(a, b + 1))
        self.draws.append({"utility": "rangeint", "args": args, "value": v})
        return str(v)

    def u_rangefloat(self, args):
        if len(args) != 2:
            raise TemplateError(f"rangefloat takes 2 arguments, got {len(args)}")
        a, b = _num(args[0], float, "rangefloat"), _num(args[1], float, "rangefloat")
        if b < a:
            raise TemplateError(f"rangefloat: empty range [{a}, {b}]")
        v = round(float(self._need_rng("rangefloat").uniform(a, b)), 4)
        self.draws.append({"utility": "rangefloat", "args": args, "value": v})
        return f"{v:.4f}"   # always contains '.', so a nested list utility reads it as a fraction

    def u_choice(self, args):
        if not args:
            raise TemplateError("choice needs at least one option")
        j = int(self._need_rng("choice").integers(0, len(args)))
        self.draws.append({"utility": "choice", "args": args, "value": args[j]})
        return args[j]

    def u_prompt(self, args):
        if args:
            raise TemplateError("prompt takes no arguments")
        return self.ctx.prompt

    # list utilities
    def _list(self, util, args, items: list, ids: list, fmt) -> str:
        if len(args) > 1:
            raise TemplateError(f"{util} takes at most 1 argument, got {len(args)}")
        arg = args[0] if args else None
        k = list_count(arg, len(items))
        idx = _sample_indices(len(items), k, self.rng) if k else []
        self.draws.append({"utility": util, "args": args, "value": len(idx),
                           "sampled_ids": [ids[i] for i in idx]})
        return "\n".join(fmt(items[i]) for i in idx)

    def u_references(self, args):
        r = self.ctx.references
        return self._list("references", args, r, list(range(len(r))), lambda s: f"- {_one_line(s)}")

    def u_facts(self, args):
        f = self.ctx.facts
        return self._list("facts", args, f, list(range(len(f))), lambda s: f"- {_one_line(s)}")

    def u_tags(self, args):
        t = self.ctx.tags
        return self._list("tags", args, t, [x[0] for x in t],
                          lambda x: f"- {_one_line(x[0])}: {_one_line(x[1])}")

    def u_idea_summaries(self, args):
        j = self.ctx.judged_ideas
        return self._list("idea_summaries", args, j, [x["id"] for x in j],
                          lambda x: f"- {_one_line(x.get('summary') or '')}")

    def u_idea_summaries_and_critiques(self, args):
        j = self.ctx.judged_ideas

        def fmt(x):
            s = x.get("s_cal")
            crit = f"surrogate: {_fmt_score(s) if s is not None else 'n/a'}"
            if x.get("real_score") is not None:
                crit += f"; real: {_fmt_score(x['real_score'])}"
            return f"- {_one_line(x.get('summary') or '')} ({crit})"

        return self._list("idea_summaries_and_critiques", args, j, [x["id"] for x in j], fmt)


def render(template: str, ctx: RenderContext, rng: np.random.Generator | None,
           variables: dict[str, str] | None = None) -> tuple[str, list[dict]]:
    """Render `template`. Returns (text, draws). `variables` (extension) binds zero-arg
    placeholders such as {{idea}} to literal text; they take precedence over utilities."""
    ev = _Evaluator(ctx, rng, variables)
    out = ev.parts(parse(template))
    return out, ev.draws
