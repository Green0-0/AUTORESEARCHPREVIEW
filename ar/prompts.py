"""Prompt strings, copied verbatim from class_notes/"Autoresearch Preview Spec.md", plus assembly
helpers (CONTRACTS.md "Templating and prompts"). tests/test_prompts.py checks every string
against the spec's code blocks.
"""
from __future__ import annotations

from ar.templating import RenderContext, render

# ----------------------------------------------------------------------------- ideation
IDEATOR_SYSTEM = """Your role is to act as an ideator in an autoresearch pipeline whose job is to come up with and advance ideas that tackle the user prompt:
{{prompt}}

Additionally, reference these known facts from previous research rounds:
{{facts;1000}}

Carefully analyze data, consider scenarios, utilize logic, and review criteria/constraints when coming up with ideas. Also be creative and think outside the box, don't be afraid to grasp at distant concepts and tie them together."""

SUMMARY = """Here is an idea:
{{idea}}

Summarize the idea in at most 100 words: what it does, and how it differs from standard approaches. Output the summary and nothing else."""

_BEST_SAMPLE_T1 = "Come up with {{rangeint;2;5}} new ideas, one after another. They should all be meaningfully different."

BASE_PROMPTS: dict[str, list[str]] = {
    "Generic": [
        "Come up with a new idea. Output the full idea description document and nothing else.",
    ],
    "BetterGeneric": [
        "Come up with a new idea, focusing on novel approaches and creative ideas that are also logically sound and well motivated. Output the full idea description document and nothing else.",
    ],
    "BestSample1": [
        _BEST_SAMPLE_T1,
        "Take the best idea, discard everything else, and write the full idea description document and nothing else.",
    ],
    "BestSample2": [
        _BEST_SAMPLE_T1,
        "Take the best parts from each idea (whenever possible) and write a new singular full idea; output its description document and nothing else.",
    ],
    "RandSample": [
        "Come up with 5 new ideas, one after another. Number them. They should all be meaningfully different.",
        "Take idea #{{rangeint;1;5}}, discard everything else, and write the full idea description document and nothing else.",
    ],
    "NewSampleCritique": [
        "Come up with {{rangeint;2;5}} new ideas, one after another, and also a brief critique of each afterwards. Each idea should be meaningfully different.",
        "Write a new idea, taking inspiration from the previous ideas and attempting to overcome their weaknesses. Discard the previous ideas, and write the full idea description document for the new idea and nothing else.",
    ],
    "TopicIdea": [
        "Come up with {{rangeint;10;20}} new topics/concepts/fields, that could be used to tackle the problem, one after another. Each topic should be meaningfully different.",
        "Pick the most promising and write a new idea. Output the full idea description document and nothing else.",
    ],
}
BASES = ["Generic", "BetterGeneric", "BestSample1", "BestSample2", "RandSample", "NewSampleCritique", "TopicIdea"]

# appends: {"top": str | None, "bottom": str}
APPENDS: dict[str, dict[str, str | None]] = {
    "References": {
        "top": "Here are some scientific papers that may or may not be relevant:\n{{references;{{rangeint;1;3}}}}",
        "bottom": "Take inspiration from any of the papers if possible/relevant.",
    },
    "RefIdeas": {
        "top": "Here are some previously generated ideas for reference, with their evaluation results:\n{{idea_summaries_and_critiques;{{rangeint;1;3}}}}",
        "bottom": "While ensuring your ideas are unique, take inspiration and reference from the previously generated ideas.",
    },
    "NoTag": {
        "top": None,
        "bottom": "Ensure any ideas do not relate with the following tags {{tags;{{rangefloat;0.1;1}}}}.",
    },
    "UseTag": {
        "top": None,
        "bottom": "Ensure any ideas relate to the following tags in some way: {{tags;{{rangeint;1;5}}}}",
    },
    "NoIdea": {
        "top": None,
        "bottom": "Ensure any idea does not relate with the following ideas {{idea_summaries;{{rangefloat;0.1;1}}}}.",
    },
}

# ----------------------------------------------------------------------------- tagging
TAGGING_SYSTEM = """Your role is to act as a tagging agent whose job is to manage tags for ideas within an autoresearch pipeline for the prompt:
{{prompt}}

A tag is simply a title with a short descriptor that is general and may apply to many ideas sharing a common trait.

Here is the current list of tags (new ones are added over time):
{{tags}}"""

GENERATOR_GENERIC = """Here is a new list of ideas that are being considered:
{{ideas}}

Output all the tags missing from the current list that are needed to describe the new ideas. Ensure that tags are reasonably general while being descriptive of an idea's properties."""

MERGER_GENERIC = """Here is a new list of ideas that are being considered:
{{ideas}}

Here are {{K}} proposed lists of tags, to be added to the existing tag list to correctly tag the new ideas:
{{K_lists_of_proposals}}

Deduplicate, merge and refine these into a single comprehensive set of tags."""

# Built line by line: two schema lines carry trailing spaces in the spec.
WRITER_GENERIC = "\n".join([
    "Here is a new list of ideas that are being considered:",
    "{{ideas}}",
    "",
    "Here is a list of proposed tags: {{K_lists_of_proposals}}",
    "",
    "Output a json (wrapped in markdown code block beginning with ```json and ending with ```) whose schema is shaped as follows:",
    "```json",
    '{"new_tags": ',
    "[",
    '{"name":"new_tag_1_name_here", "description":"new_tag_1_description_here"}, ',
    '{"name":"new_tag_2_name_here", "description":"new_tag_2_description_here"} ... ',
    "]",
    "}",
    "```",
    'And so on, for all tags. Follow the schema of having a list named "new_tags" and items with "name" and "description" exactly.',
    "",
    "Begin outputting the json now, starting with the markdown code block header.",
])

BASIC_ASSIGNMENT = """Here is a new idea being considered: {{idea}}

Output the list of tags by name, exact, from the source list"""

# ----------------------------------------------------------------------------- judging
JUDGE_SYSTEM = """Your role is to act as a judge who predicts the performance of ideas (from 0 to 1, where 0 is a complete fail and 1 is a complete success, and 0.5 is halfway there) arising from an autoresearch pipeline whose job is to tackle this user prompt:
{{prompt}}

Additionally, reference these known facts from previous research rounds:
{{facts;1000}}"""

BASIC_SURROGATE = r"""Here is a new idea:
{{idea}}

Output the expected performance of the idea as a float between 0.0 and 1.0 within \boxed{}"""

# Judge strategies J2-J5 (plan §9.1). Texts are verbatim from class_notes/"Autoresearch Draft Spec Graph
# Note.md" ("Basic", "Arbitrary Rubric", "Basic FS"); J2 reuses BASIC_SURROGATE (= spec "Basic"). The spec
# gives "Ranking Based" only as a description ("Given two or more new ideas and some number of reference
# ideas, judge each with the goal of correctly ranking everything relatively in mind"), so RANKING is built
# from that sentence plus the one-line-per-idea output format its parser needs.
ARBITRARY_RUBRIC = r"""Here is a new idea:
{{idea}}

Create a relevant rubric, that, when averaged, will yield an accurate grade (float between 0.0 and 1.0) for the performance of the idea.

Output the final averaged performance derived from the rubric within \boxed{}"""

BASIC_FS = r"""Here is a new idea: {{idea}}

Here are some previous ideas, each with their judging result:
{{idea_and_result_lists}}.

Referencing the previous judged results to ensure consistency in judging, output the expected performance of the new idea as a float between 0.0 and 1.0 within \boxed{}"""

RANKING = r"""Here are some reference ideas, each with its known judging result:
{{reference_ideas}}

Here are {{n}} new ideas:
{{new_ideas}}

Judge each new idea with the goal of correctly ranking everything (the new ideas and the reference ideas) relatively in mind. Then output the expected performance of every new idea as a float between 0.0 and 1.0, one line per new idea, exactly in this form (from Idea 1 to Idea {{n}}):
Idea 1: \boxed{0.xx}"""

RANK_SEP = "\n\n---\n\n"

# ----------------------------------------------------------------------------- assembly
TOP_ORDER = ["References", "RefIdeas"]


def _cfg_get(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def _check_available(name: str, ctx: RenderContext) -> None:
    need = {"References": ctx.references, "RefIdeas": ctx.judged_ideas, "NoIdea": ctx.judged_ideas,
            "NoTag": ctx.tags, "UseTag": ctx.tags}[name]
    if not need:
        raise ValueError(f"append {name} chosen but its requirement is not met (sampler must not pick it)")


def assemble(config, ctx: RenderContext, rng) -> dict:
    """Render the ideation conversation for a SamplerConfig.

    Turn 1 = tops (References, RefIdeas) + base turn 1 + bottoms (References, RefIdeas, tag slot,
    NoIdea), joined by blank lines. Appends go on turn 1 only; later base turns are rendered too.
    Returns {"system", "turns", "draws"}; each draw additionally carries "where"
    ("system" | "turn1" | "turn2").
    """
    base = _cfg_get(config, "base")
    if base not in BASE_PROMPTS:
        raise ValueError(f"unknown base prompt {base!r}")
    tag_slot = _cfg_get(config, "tag_slot")
    if tag_slot not in (None, "NoTag", "UseTag"):
        raise ValueError(f"bad tag_slot {tag_slot!r}")

    chosen_tb = [n for n in TOP_ORDER if _cfg_get(config, n, False)]
    bottom_names = chosen_tb + ([tag_slot] if tag_slot else []) + (["NoIdea"] if _cfg_get(config, "NoIdea", False) else [])
    for n in bottom_names:
        _check_available(n, ctx)

    tops = [APPENDS[n]["top"] for n in chosen_tb]
    bottoms = [APPENDS[n]["bottom"] for n in bottom_names]
    turn_templates = list(BASE_PROMPTS[base])
    turn_templates[0] = "\n\n".join(tops + [turn_templates[0]] + bottoms)

    draws: list[dict] = []
    system, d = render(IDEATOR_SYSTEM, ctx, rng)
    draws += [{**x, "where": "system"} for x in d]
    turns = []
    for k, t in enumerate(turn_templates, start=1):
        text, d = render(t, ctx, rng)
        turns.append(text)
        draws += [{**x, "where": f"turn{k}"} for x in d]
    return {"system": system, "turns": turns, "draws": draws}


# ----------------------------------------------------------------------------- helpers
_EMPTY = RenderContext(prompt="")


def _fill(template: str, **variables) -> str:
    """Substitute helper placeholders (e.g. {{idea}}) as inert text; no random utilities."""
    out, _ = render(template, _EMPTY, None, variables={k: str(v) for k, v in variables.items()})
    return out


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {' '.join(str(s).split())}" for s in items)


def _format_lists(lists: list[str]) -> str:
    return "\n\n".join(f"List {i}:\n{str(l).strip()}" for i, l in enumerate(lists, start=1))


def summary_prompt(idea: str) -> str:
    return _fill(SUMMARY, idea=idea)


def tagging_system(prompt: str, tags: list[tuple[str, str]]) -> str:
    out, _ = render(TAGGING_SYSTEM, RenderContext(prompt=prompt, tags=list(tags)), None)
    return out


def generator_prompt(summaries: list[str]) -> str:
    return _fill(GENERATOR_GENERIC, ideas=_bullets(summaries))


def merger_prompt(summaries: list[str], lists: list[str]) -> str:
    return _fill(MERGER_GENERIC, ideas=_bullets(summaries), K=len(lists),
                 K_lists_of_proposals=_format_lists(lists))


def writer_prompt(summaries: list[str], lists: list[str]) -> str:
    return _fill(WRITER_GENERIC, ideas=_bullets(summaries), K_lists_of_proposals="\n" + _format_lists(lists))


def assign_prompt(idea: str) -> str:
    return _fill(BASIC_ASSIGNMENT, idea=idea)


def judge_system(prompt: str, facts: list[str], rng) -> str:
    out, _ = render(JUDGE_SYSTEM, RenderContext(prompt=prompt, facts=list(facts)), rng)
    return out


def judge_prompt(idea: str) -> str:
    return _fill(BASIC_SURROGATE, idea=idea)


def rubric_prompt(idea: str) -> str:
    return _fill(ARBITRARY_RUBRIC, idea=idea)


def fewshot_prompt(idea: str, examples: list[tuple[str, float]]) -> str:
    """examples = [(summary, judging result in [0,1]), ...] in presentation order.
    One line per example: ``- <summary> (judging result: 0.42)``."""
    lines = "\n".join(f"- {' '.join(str(s).split())} (judging result: {float(v):.2f})" for s, v in examples)
    return _fill(BASIC_FS, idea=idea, idea_and_result_lists=lines)


def ranking_prompt(ideas: list[str], references: list[dict]) -> str:
    """ideas = new idea descriptions, labelled Idea 1..n in this order; references = [{"label","description",
    "score"}], shown as ``Reference <label> (known judging result: 0.00):``."""
    refs = RANK_SEP.join(f"Reference {r['label']} (known judging result: {float(r['score']):.2f}):\n{str(r['description']).strip()}"
                         for r in references)
    new = RANK_SEP.join(f"Idea {k}:\n{str(d).strip()}" for k, d in enumerate(ideas, start=1))
    return _fill(RANKING, reference_ideas=refs, n=len(ideas), new_ideas=new)
