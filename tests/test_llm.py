import asyncio
import json
import re
import types

import httpx
import openai
import pytest

from ar.llm import LLM, MockLLM, TransientLLMError, split_reasoning


# ------------------------------------------------------------------ splitting
@pytest.mark.parametrize("content,field,ans,reas", [
    ("plain answer", None, "plain answer", ""),
    ("<think>\nhmm\n</think>\n\nthe answer", None, "the answer", "hmm"),
    ("hmm only\n</think>\nthe answer", None, "the answer", "hmm only"),          # template opened <think>
    ("<think>\nlong reasoning cut off", None, "", "long reasoning cut off"),     # truncated
    ("answer", "sep reasoning", "answer", "sep reasoning"),                      # reasoning_content field
    ("<think>a</think>x <think>b</think> final", None, "final", "a</think>x <think>b"),
    ("", None, "", ""),
    (None, None, "", ""),
])
def test_split_reasoning(content, field, ans, reas):
    a, r = split_reasoning(content, field)
    assert a == ans
    assert r == reas


# ------------------------------------------------------------- fake client
class FakeCompletions:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def create(self, **kw):
        self.calls.append(kw)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.01)
            item = self.script.pop(0) if self.script else {"content": "ok"}
            if isinstance(item, BaseException):
                raise item
            msg = types.SimpleNamespace(content=item.get("content"),
                                        reasoning_content=item.get("reasoning_content"))
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg, finish_reason=item.get("finish", "stop"))],
                usage=types.SimpleNamespace(prompt_tokens=11, completion_tokens=22))
        finally:
            self.active -= 1


def make(script, **kw):
    comp = FakeCompletions(script)
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp))
    cfg = {"model": "qwen-test", "max_inflight": kw.pop("max_inflight", 4),
           "purposes": {"judge": {"thinking": True, "max_tokens": 123, "temperature": 0.6},
                        "summary": {"thinking": False, "max_tokens": 50, "temperature": 0.3}}}
    return LLM(cfg, client=client, backoff_s=0.0, **kw), comp


def test_params_and_reasoning_field(tmp_path):
    usage = tmp_path / "llm_usage.jsonl"
    llm, comp = make([{"content": "<think>r</think>A"},
                      {"content": "B", "reasoning_content": "sep"},
                      {"content": "<think>cut", "finish": "length"}], usage_path=usage)

    async def go():
        a = await llm.chat([{"role": "user", "content": "x"}], "judge")
        b = await llm.chat([{"role": "user", "content": "x"}], "summary")
        c = await llm.chat([{"role": "user", "content": "x"}], "judge")
        return a, b, c
    a, b, c = asyncio.run(go())
    assert (a.text, a.reasoning, a.finish_reason) == ("A", "r", "stop")
    # reasoning lives in the reasoning.jsonl sidecar; raw keeps only a pointer + the answer
    assert a.raw.startswith("<think>[reasoning in reasoning.jsonl call_id=") and a.raw.endswith("chars=1]</think>A")
    assert (b.text, b.reasoning) == ("B", "sep") and b.raw.endswith("chars=3]</think>B")
    assert c.text == "" and c.reasoning == "cut" and c.finish_reason == "length"
    assert a.prompt_tokens == 11 and a.completion_tokens == 22
    k0, k1 = comp.calls[0], comp.calls[1]
    assert k0["model"] == "qwen-test" and k0["max_tokens"] == 123 and k0["temperature"] == 0.6
    assert k0["extra_body"] == {"chat_template_kwargs": {"enable_thinking": True}}
    assert k1["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    lines = [json.loads(l) for l in usage.read_text().splitlines()]
    assert len(lines) == 3 and all(l["ok"] for l in lines)
    assert lines[0]["purpose"] == "judge" and lines[2]["finish_reason"] == "length"
    assert llm.stats["judge"]["length"] == 1
    # defaults filled for purposes not in cfg
    assert llm.params("ideation")["thinking"] is True


def test_retries_transient_then_success(tmp_path):
    req = httpx.Request("POST", "http://x/v1/chat/completions")
    llm, comp = make([openai.APIConnectionError(request=req), TransientLLMError("boom"),
                      openai.InternalServerError("500", response=httpx.Response(500, request=req), body=None),
                      {"content": "fine"}], usage_path=tmp_path / "u.jsonl")
    r = asyncio.run(llm.chat([{"role": "user", "content": "x"}], "summary"))
    assert r.text == "fine" and r.attempts == 4 and len(comp.calls) == 4
    assert llm.stats["summary"]["retries"] == 3
    lines = [json.loads(l) for l in (tmp_path / "u.jsonl").read_text().splitlines()]
    assert [l["ok"] for l in lines] == [False, False, False, True]


def test_retries_exhausted_and_nontransient():
    llm, comp = make([TransientLLMError("a")] * 4)
    with pytest.raises(TransientLLMError):
        asyncio.run(llm.chat([{"role": "user", "content": "x"}], "summary"))
    assert len(comp.calls) == 4  # 1 + 3 retries
    llm, comp = make([ValueError("bad request")])
    with pytest.raises(ValueError):
        asyncio.run(llm.chat([{"role": "user", "content": "x"}], "summary"))
    assert len(comp.calls) == 1
    with pytest.raises(KeyError):
        asyncio.run(llm.chat([], "nonexistent"))


def test_semaphore_caps_inflight():
    llm, comp = make([], max_inflight=3)

    async def go():
        await asyncio.gather(*[llm.chat([{"role": "user", "content": str(i)}], "summary")
                               for i in range(12)])
    asyncio.run(go())
    assert comp.max_active == 3 and len(comp.calls) == 12


# ------------------------------------------------------------------- mock
TAG_SYS = ("Your role is to act as a tagging agent ...\n\nHere is the current list of tags "
           "(new ones are added over time):\n- alpha: first tag\n- beta_two: second tag\n")


def test_mock_answers(tmp_path):
    m = MockLLM(seed=3, usage_path=tmp_path / "u.jsonl")

    async def go():
        out = {}
        out["judge"] = [await m.chat([{"role": "system", "content": "s"},
                                      {"role": "user", "content": "Here is a new idea:\nX"}], "judge")
                        for _ in range(20)]
        out["writer"] = await m.chat([{"role": "system", "content": TAG_SYS},
                                      {"role": "user", "content": "Output a json (wrapped ...)"}], "tagging")
        out["gen"] = await m.chat([{"role": "system", "content": TAG_SYS},
                                   {"role": "user", "content": "Here is a new list of ideas"}], "tagging")
        out["assign"] = await m.chat([{"role": "system", "content": TAG_SYS},
                                      {"role": "user", "content": "Here is a new idea being considered: X"}],
                                     "assign")
        out["idea"] = await m.chat([{"role": "system", "content": "s"},
                                    {"role": "user", "content": "go"}], "ideation")
        out["sum"] = await m.chat([{"role": "user", "content": "Here is an idea:\nfoo bar\n\nSummarize"}],
                                  "summary")
        return out
    o = asyncio.run(go())
    from ar.scoring import parse_boxed
    vals = [parse_boxed(r.text) for r in o["judge"]]
    assert all(v is not None and 0.3 <= v <= 0.9 for v in vals)
    assert len(set(vals)) > 10            # repeated identical request -> independent draws
    assert o["judge"][0].reasoning        # judge thinking is on by default -> <think> split off
    js = re.findall(r"```json\s*(.*?)```", o["writer"].text, re.S)[-1]
    tags = json.loads(js)["new_tags"]
    assert tags and all(set(t) == {"name", "description"} for t in tags)
    assert not {t["name"] for t in tags} & {"alpha", "beta_two"}
    assert o["gen"].text.startswith("Proposed tags")
    names = [l.strip() for l in o["assign"].text.splitlines() if l.strip()]
    assert names and set(names) <= {"alpha", "beta_two"}
    assert o["idea"].text and "<think>" not in o["idea"].text and o["idea"].reasoning
    assert o["sum"].text.startswith("Summary:")
    assert (tmp_path / "u.jsonl").exists()


def test_mock_deterministic_and_faults():
    async def seq(m):
        return [(await m.chat([{"role": "user", "content": f"q{i}"}], "judge")).text for i in range(5)]
    assert asyncio.run(seq(MockLLM(seed=1))) == asyncio.run(seq(MockLLM(seed=1)))
    assert asyncio.run(seq(MockLLM(seed=1))) != asyncio.run(seq(MockLLM(seed=2)))
    m = MockLLM(seed=0, length_p=1.0)
    r = asyncio.run(m.chat([{"role": "user", "content": "x"}], "ideation"))
    assert r.finish_reason == "length" and r.text == "" and r.reasoning
    m = MockLLM(seed=0, fail_p=1.0)
    with pytest.raises(TransientLLMError):
        asyncio.run(m.chat([{"role": "user", "content": "x"}], "summary"))
    assert len(m.calls) == 4
