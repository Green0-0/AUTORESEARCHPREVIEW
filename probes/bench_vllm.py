"""Throughput probe for the vLLM replicas.

Fires batches of ideation-shaped chat requests at each endpoint concurrently,
at several concurrency levels, and reports output tokens/s. Both replicas are
driven at the same time, so no GPU sits idle during the probe.

usage: python bench_vllm.py --ports 8000 8001 --out results.json
"""
import argparse, asyncio, json, time
from openai import AsyncOpenAI

SYSTEM = (
    "Your role is to act as an ideator in an autoresearch pipeline whose job is to come up with "
    "and advance ideas that tackle the user prompt:\n{task}\n\nAdditionally, reference these known "
    "facts from previous research rounds:\n\n\nCarefully analyze data, consider scenarios, utilize "
    "logic, and review criteria/constraints when coming up with ideas. Also be creative and think "
    "outside the box, don't be afraid to grasp at distant concepts and tie them together."
)
TASK = (
    "Begin with CIFAR-10 where 40% of training labels have been replaced uniformly at random (the "
    "noisy label file is fixed and provided). The clean labels are not available. Find novel training "
    "procedures that maximize clean test top-1 accuracy. Using any external pretrained weights is "
    "banned. Training should not take longer than 15 minutes on 1 A5000."
)
USER = ("Come up with {n} new ideas, one after another. They should all be meaningfully different. "
        "Request id {i}.")


async def one(client, model, i, max_tokens, thinking):
    t0 = time.time()
    r = await client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM.format(task=TASK)},
                  {"role": "user", "content": USER.format(n=2 + i % 4, i=i)}],
        max_tokens=max_tokens, temperature=0.8,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    return r.usage.prompt_tokens, r.usage.completion_tokens, time.time() - t0


async def level(clients, model, conc, n_per_client, max_tokens, thinking):
    sem = {id(c): asyncio.Semaphore(conc) for c in clients}

    async def run(c, i):
        async with sem[id(c)]:
            return await one(c, model, i, max_tokens, thinking)

    t0 = time.time()
    res = await asyncio.gather(*[run(c, i) for c in clients for i in range(n_per_client)],
                               return_exceptions=True)
    wall = time.time() - t0
    ok = [r for r in res if not isinstance(r, Exception)]
    errs = [repr(r)[:200] for r in res if isinstance(r, Exception)]
    out_tok = sum(r[1] for r in ok)
    return {
        "concurrency_per_replica": conc, "replicas": len(clients), "thinking": thinking,
        "max_tokens": max_tokens, "requests_ok": len(ok), "errors": len(errs),
        "error_samples": errs[:3], "wall_s": round(wall, 1),
        "out_tok_per_s_total": round(out_tok / wall, 1),
        "out_tok_per_s_per_replica": round(out_tok / wall / len(clients), 1),
        "mean_out_tokens": round(out_tok / max(1, len(ok)), 1),
        "mean_latency_s": round(sum(r[2] for r in ok) / max(1, len(ok)), 1),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", nargs="+", type=int, default=[8000, 8001])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    clients = [AsyncOpenAI(base_url=f"http://127.0.0.1:{p}/v1", api_key="x", timeout=1800)
               for p in a.ports]
    plan = [(16, 32, 1024, False), (32, 64, 1024, False), (64, 128, 1024, False),
            (32, 64, 4096, True)]
    results = []
    for conc, n, mt, th in plan:
        r = await level(clients, a.model, conc, n, mt, th)
        print(json.dumps(r), flush=True)
        results.append(r)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    asyncio.run(main())
