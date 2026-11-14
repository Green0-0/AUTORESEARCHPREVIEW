"""Async OpenAI-compatible LLM client for the vLLM endpoints, plus a deterministic offline fake.

See CONTRACTS.md, "LLM". One ``LLM`` instance per run: every call goes through one
semaphore (``max_inflight``), applies the per-purpose settings from ``llm.purposes``
(thinking on/off via ``extra_body.chat_template_kwargs.enable_thinking``, ``max_tokens``,
``temperature``), splits the reasoning from the answer, retries transient errors, and
appends one usage line per call to ``llm_usage.jsonl`` when a path is given.

Reasoning splitting handles every shape vLLM/Qwen produce:
* a separate ``reasoning_content`` / ``reasoning`` field (vLLM with a reasoning parser);
* inline ``<think>...</think>answer``;
* ``...</think>answer`` with no opening tag (template already emitted ``<think>``);
* an unclosed ``<think>...`` (truncated at max_tokens): all reasoning, empty answer.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_PURPOSES = {
    "ideation": {"thinking": True, "max_tokens": 16000, "temperature": 0.8},
    "summary": {"thinking": False, "max_tokens": 400, "temperature": 0.3},
    "tagging": {"thinking": False, "max_tokens": 4000, "temperature": 0.3},
    "assign": {"thinking": False, "max_tokens": 800, "temperature": 0.2},
    "judge": {"thinking": True, "max_tokens": 12000, "temperature": 0.7},
}


@dataclass
class ChatResult:
    text: str                      # the answer, reasoning removed, stripped
    reasoning: str                 # the reasoning ("" when none)
    finish_reason: str             # "stop" | "length" | ...
    prompt_tokens: int
    completion_tokens: int
    raw: str = ""                  # full raw output (inline <think> kept; separate field re-wrapped)
    latency: float = 0.0           # seconds for the successful attempt
    attempts: int = 1              # 1 + transient retries used


class TransientLLMError(Exception):
    """Raised by fakes/tests to signal a retryable failure."""


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


def split_reasoning(content: str | None, reasoning_field: str | None = None) -> tuple[str, str]:
    """-> (answer, reasoning). See module docstring for the handled shapes."""
    content = content or ""
    reasoning_parts = []
    if reasoning_field:
        reasoning_parts.append(reasoning_field.strip())
    close = content.rfind(_THINK_CLOSE)
    if close >= 0:
        head, answer = content[:close], content[close + len(_THINK_CLOSE):]
        o = head.find(_THINK_OPEN)
        if o >= 0:
            pre, head = head[:o], head[o + len(_THINK_OPEN):]
            # text before <think> is unusual; keep it with the answer rather than lose it
            answer = pre + answer if pre.strip() else answer
        reasoning_parts.append(head.strip())
    else:
        o = content.find(_THINK_OPEN)
        if o >= 0:  # unclosed: truncated in the middle of reasoning
            answer = content[:o]
            reasoning_parts.append(content[o + len(_THINK_OPEN):].strip())
        else:
            answer = content
    reasoning = "\n\n".join(p for p in reasoning_parts if p)
    return answer.strip(), reasoning


def _is_transient(e: BaseException) -> bool:
    if isinstance(e, (TransientLLMError, asyncio.TimeoutError, ConnectionError, TimeoutError)):
        return True
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
                      openai.InternalServerError)):
        return True
    if isinstance(e, openai.APIStatusError):
        return getattr(e, "status_code", 0) in (408, 409, 429) or getattr(e, "status_code", 0) >= 500
    return False


class LLM:
    def __init__(self, llm_cfg: dict, usage_path: str | Path | None = None, *,
                 client: Any = None, retries: int = 3, backoff_s: float = 2.0,
                 timeout_s: float = 3600.0):
        """llm_cfg = cfg["llm"]. ``usage_path`` (e.g. $RUN/llm_usage.jsonl; may also be given as
        ``llm_cfg["usage_path"]``) gets one JSON line per call. ``client`` overrides the AsyncOpenAI client (tests)."""
        llm_cfg = dict(llm_cfg or {})
        self.cfg = llm_cfg
        self.base_url = llm_cfg.get("base_url", "http://127.0.0.1:8000/v1")
        self.model = llm_cfg.get("model", "qwen")
        self.max_inflight = int(llm_cfg.get("max_inflight", 64))
        self.purposes = {k: dict(v) for k, v in DEFAULT_PURPOSES.items()}
        for k, v in (llm_cfg.get("purposes") or {}).items():
            self.purposes.setdefault(k, {}).update(v or {})
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)
        self.timeout_s = float(timeout_s)
        usage_path = usage_path or llm_cfg.get("usage_path")   # orchestrator passes it in llm_cfg
        self.usage_path = Path(usage_path) if usage_path else None
        self._sem = asyncio.Semaphore(self.max_inflight)
        self._client = client
        self.stats: dict[str, dict[str, int]] = {}   # purpose -> {"calls","length","errors","retries"}

    # ------------------------------------------------------------------ client
    @property
    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            # retries are ours (logged, with backoff); the SDK's own are disabled.
            self._client = AsyncOpenAI(base_url=self.base_url, api_key="EMPTY",
                                       timeout=self.timeout_s, max_retries=0)
        return self._client

    def params(self, purpose: str) -> dict:
        if purpose not in self.purposes:
            raise KeyError(f"unknown LLM purpose {purpose!r}")
        return self.purposes[purpose]

    async def _request(self, messages: list[dict], purpose: str, p: dict) -> dict:
        """One raw call -> {"content","reasoning","finish_reason","prompt_tokens","completion_tokens"}."""
        r = await self.client.chat.completions.create(
            model=self.model, messages=messages,
            max_tokens=int(p.get("max_tokens", 4000)),
            temperature=float(p.get("temperature", 0.7)),
            extra_body={"chat_template_kwargs": {"enable_thinking": bool(p.get("thinking", False))}},
        )
        ch = r.choices[0]
        msg = ch.message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        if reasoning is None:
            extra = getattr(msg, "model_extra", None) or {}
            reasoning = extra.get("reasoning_content") or extra.get("reasoning")
        u = getattr(r, "usage", None)
        return {"content": msg.content or "", "reasoning": reasoning or "",
                "finish_reason": ch.finish_reason or "",
                "prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0)}

    # -------------------------------------------------------------------- chat
    async def chat(self, messages: list[dict], purpose: str) -> ChatResult:
        p = self.params(purpose)
        st = self.stats.setdefault(purpose, {"calls": 0, "length": 0, "errors": 0, "retries": 0})
        attempt = 0
        t0 = time.time()
        while True:
            attempt += 1
            try:
                async with self._sem:   # backoff sleeps happen outside the semaphore
                    t0 = time.time()
                    out = await self._request(messages, purpose, p)
                break
            except Exception as e:  # noqa: BLE001
                lat = time.time() - t0
                transient = _is_transient(e)
                self._log_usage({"purpose": purpose, "ok": False, "attempt": attempt,
                                 "latency": round(lat, 3), "error": repr(e)[:300],
                                 "transient": transient})
                if not transient or attempt > self.retries:
                    st["errors"] += 1
                    raise
                st["retries"] += 1
                delay = self.backoff_s * (2 ** (attempt - 1)) * (0.5 + random.random())
                await asyncio.sleep(delay)
        lat = time.time() - t0
        text, reasoning = split_reasoning(out["content"], out["reasoning"])
        # Reasoning traces (often 4-16k tokens) are kept OUT of the journal: they go to a sidecar
        # reasoning.jsonl next to llm_usage.jsonl, and `raw` carries only a pointer + the answer.
        # This keeps journal.json in the tens of MB so lock-held writes stay fast.
        raw = out["content"]
        if reasoning:
            call_id = uuid.uuid4().hex[:12]
            self._log_reasoning({"call_id": call_id, "purpose": purpose, "reasoning": reasoning})
            raw = f"{_THINK_OPEN}[reasoning in reasoning.jsonl call_id={call_id} chars={len(reasoning)}]{_THINK_CLOSE}{text}"
        st["calls"] += 1
        if out["finish_reason"] == "length":
            st["length"] += 1
        self._log_usage({"purpose": purpose, "ok": True, "attempt": attempt,
                         "latency": round(lat, 3), "finish_reason": out["finish_reason"],
                         "prompt_tokens": out["prompt_tokens"],
                         "completion_tokens": out["completion_tokens"],
                         "thinking": bool(p.get("thinking", False)),
                         "reasoning_chars": len(reasoning), "answer_chars": len(text)})
        return ChatResult(text=text, reasoning=reasoning, finish_reason=out["finish_reason"],
                          prompt_tokens=out["prompt_tokens"],
                          completion_tokens=out["completion_tokens"],
                          raw=raw, latency=lat, attempts=attempt)

    def _log_reasoning(self, rec: dict) -> None:
        if self.usage_path is None:
            return
        try:
            with open(self.usage_path.parent / "reasoning.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), **rec}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _log_usage(self, rec: dict) -> None:
        if self.usage_path is None:
            return
        rec = {"t": time.time(), **rec}
        try:
            self.usage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.usage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass  # usage logging must never kill a worker

    async def aclose(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            try:
                await self._client.close()
            except Exception:  # noqa: BLE001
                pass


# ============================================================================ mock
_TAG_VOCAB = [
    ("curriculum_ordering", "Orders or schedules training examples by difficulty or another criterion"),
    ("loss_reweighting", "Changes per-example loss weights during training"),
    ("data_augmentation", "Adds or modifies input augmentations"),
    ("regularization", "Adds explicit regularization terms or constraints"),
    ("architecture_change", "Modifies the network structure"),
    ("optimizer_schedule", "Changes the optimizer or learning-rate schedule"),
    ("ensembling", "Combines several models or snapshots"),
    ("self_distillation", "Uses the model's own predictions as soft targets"),
    ("label_cleaning", "Detects and corrects or drops suspicious labels"),
    ("sample_selection", "Chooses subsets of examples to train on"),
    ("representation_learning", "Adds auxiliary objectives to shape features"),
    ("early_exit", "Allows prediction from intermediate layers"),
    ("synthetic_data", "Generates new training examples"),
    ("adversarial_robustness", "Targets robustness to perturbations without adversarial training"),
    ("meta_learning", "Learns a component of the training procedure"),
    ("normalization", "Changes normalization layers or statistics"),
]
_WORDS = ("gradient spectral entropy curriculum margin prototype mixup cosine momentum teacher "
          "cluster boundary confidence temperature anneal sparse contrastive replay pruning "
          "bootstrap kernel ranking memory schedule").split()

_TAG_LINE = re.compile(r"^- ([^:\n]+?):\s", re.M)
_RANK_IDEA = re.compile(r"^Idea (\d+):$", re.M)


class MockLLM(LLM):
    """Deterministic offline fake. Answers by purpose (and, for tagging, by prompt shape):
    ideation -> short idea (with <think> when thinking is on); summary -> short summary;
    tagging generator/merger -> tag lists; writer -> ```json {"new_tags": [...]}```;
    assign -> names present in the tag pool of the system prompt; judge -> \\boxed{x},
    x ~ U(0.3, 0.9). Outputs depend only on (seed, purpose, messages, k-th repeat of that
    exact request), so results do not depend on asyncio scheduling and a repeated request
    (e.g. a confirmation judge call) is an independent draw.

    Fault injection for tests: ``fail_p`` (transient error probability), ``judge_garbage_p``
    (judge answer without \\boxed), ``length_p`` (truncate inside <think>), ``latency_s``."""

    def __init__(self, llm_cfg: dict | None = None, usage_path: str | Path | None = None, *,
                 seed: int = 0, fail_p: float = 0.0, judge_garbage_p: float = 0.0,
                 length_p: float = 0.0, latency_s: float = 0.0, backoff_s: float = 0.0,
                 **kw):
        super().__init__(llm_cfg or {}, usage_path, client=object(), backoff_s=backoff_s, **kw)
        self.seed = int((llm_cfg or {}).get("mock_seed", seed))
        self.fail_p = fail_p
        self.judge_garbage_p = judge_garbage_p
        self.length_p = length_p
        self.latency_s = latency_s
        self._seen: dict[str, int] = {}
        self.calls: list[tuple[str, list[dict]]] = []   # (purpose, messages) log for tests

    def _rng(self, messages, purpose) -> random.Random:
        key = json.dumps([self.seed, purpose, messages], sort_keys=True, ensure_ascii=False)
        h = hashlib.sha256(key.encode()).hexdigest()
        k = self._seen.get(h, 0)
        self._seen[h] = k + 1
        return random.Random(int(hashlib.sha256(f"{h}:{k}".encode()).hexdigest()[:16], 16))

    async def _request(self, messages: list[dict], purpose: str, p: dict) -> dict:
        self.calls.append((purpose, [dict(m) for m in messages]))
        rng = self._rng(messages, purpose)
        if self.latency_s:
            await asyncio.sleep(self.latency_s * rng.random())
        if self.fail_p and rng.random() < self.fail_p:
            raise TransientLLMError("mock transient failure")
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        thinking = bool(p.get("thinking", False))
        answer = self._answer(purpose, system, user, messages, rng)
        finish = "stop"
        content = answer
        if thinking:
            content = f"<think>\nLet me reason about this ({rng.choice(_WORDS)}).\n</think>\n\n{answer}"
            if self.length_p and rng.random() < self.length_p:
                content = "<think>\nLet me reason at great length about " + " ".join(
                    rng.choice(_WORDS) for _ in range(30))
                finish = "length"
        n_prompt = sum(len(m.get("content", "")) for m in messages) // 4
        return {"content": content, "reasoning": "", "finish_reason": finish,
                "prompt_tokens": n_prompt, "completion_tokens": max(1, len(content) // 4)}

    # ---------------------------------------------------------------- answers
    def _answer(self, purpose, system, user, messages, rng) -> str:
        if purpose == "ideation":
            n_turn = sum(1 for m in messages if m["role"] == "user")
            w = " ".join(rng.choice(_WORDS) for _ in range(6))
            return (f"Idea (turn {n_turn}): {w}.\n\nMethod: combine {rng.choice(_WORDS)} with "
                    f"{rng.choice(_WORDS)} scheduling; expected gain from {rng.choice(_WORDS)} "
                    f"structure. Token {rng.randrange(10**6):06d}.")
        if purpose == "summary":
            body = user.split("Here is an idea:", 1)[-1].strip().splitlines()
            first = body[0][:160] if body else "an idea"
            return f"Summary: {first} Differs by using {rng.choice(_WORDS)}."
        if purpose.startswith("judge"):
            if self.judge_garbage_p and rng.random() < self.judge_garbage_p:
                return "I think this idea is fairly promising, maybe around sixty percent."
            ids = [int(k) for k in _RANK_IDEA.findall(user)]
            if ids:   # ranking judge (J5): one labelled line per new idea, after some prose
                lines = [f"Idea {k}: \\boxed{{{rng.uniform(0.3, 0.9):.3f}}}" for k in ids]
                return "Ranking the new ideas between the two references.\n\n" + "\n".join(lines)
            if "Create a relevant rubric" in user:   # rubric judge (J3)
                crit = [rng.uniform(0.2, 1.0) for _ in range(3)]
                body = "\n".join(f"- criterion {n + 1} ({rng.choice(_WORDS)}): {c:.2f}" for n, c in enumerate(crit))
                return f"Rubric:\n{body}\n\nAverage:\n\n\\boxed{{{sum(crit) / 3:.3f}}}"
            x = rng.uniform(0.3, 0.9)
            return f"The idea is plausible under the budget.\n\n\\boxed{{{x:.3f}}}"
        pool = [n.strip() for n in _TAG_LINE.findall(system.split("Here is the current list of tags", 1)[-1])] \
            if "Here is the current list of tags" in system else []
        if purpose == "assign":
            if not pool:
                return "No matching tags."
            k = min(len(pool), rng.randint(1, 3))
            return "\n".join(rng.sample(pool, k))
        if purpose == "tagging":
            missing = [t for t in _TAG_VOCAB if t[0] not in pool]
            if "```json" in user or "Output a json" in user:
                k = min(len(missing), rng.randint(2, 4)) if missing else 0
                new = rng.sample(missing, k) if k else []
                js = json.dumps({"new_tags": [{"name": n, "description": d} for n, d in new]})
                return f"```json\n{js}\n```"
            k = min(len(missing), rng.randint(1, 4)) if missing else 0
            items = rng.sample(missing, k) if k else []
            head = "Merged tags:" if "Deduplicate" in user else "Proposed tags:"
            return head + "\n" + "\n".join(f"- {n}: {d}" for n, d in items)
        return f"OK ({purpose})."
