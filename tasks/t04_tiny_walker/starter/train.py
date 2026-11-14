"""Starter: basic random search (antithetic, ARS-V1 style) on the linear policy in policy.py.

Each iteration samples N = 2 * n_cpus Gaussian directions, evaluates theta +/- sigma*delta on one fresh
training terrain per direction (both signs share the terrain), and steps along the reward-difference-weighted
mean direction, scaled by the std of the 2N returns. No observation normalization, no reward shaping, no
direction selection. Rollouts run in a pool of n_cpus worker processes. Returns the final theta.
"""
import multiprocessing as mp
import time

import numpy as np

from policy import Policy

OBS, ACT = 24, 4
SIGMA, ALPHA = 0.03, 0.02
_ENV = {}


def _init(env_id):
    import gymnasium as gym
    _ENV["env"] = gym.make(env_id)


def _episode(theta, seed):
    env = _ENV["env"]
    pol = Policy({"theta": theta})
    obs, _ = env.reset(seed=int(seed))
    total, steps = 0.0, 0
    while True:
        obs, r, term, trunc, _ = env.step(np.clip(pol.act(obs), -1, 1).astype(np.float32))
        total += r
        steps += 1
        if term or trunc:
            return total, steps


def _pair(args):
    theta, delta, seed = args
    rp, sp = _episode(theta + SIGMA * delta, seed)
    rm, sm = _episode(theta - SIGMA * delta, seed)
    return rp, rm, sp + sm


def train(ctx):
    rng = np.random.default_rng(ctx["seed"])
    lo, hi = ctx["train_seed_range"]
    n_dir = 2 * ctx["n_cpus"]
    theta = np.zeros(ACT * (OBS + 1), np.float32)
    # leave room for one more iteration (an iteration is at most ~2 s with 16 workers)
    stop = ctx["deadline"] - (5 if ctx["mode"] == "full" else 3)
    steps_total, it = 0, 0
    with mp.get_context("fork").Pool(ctx["n_cpus"], initializer=_init, initargs=(ctx["env_id"],)) as pool:
        while time.time() < stop:
            deltas = rng.standard_normal((n_dir, theta.size)).astype(np.float32)
            seeds = rng.integers(lo, hi, n_dir)
            res = pool.map(_pair, [(theta, deltas[i], seeds[i]) for i in range(n_dir)], chunksize=1)
            rp = np.array([r[0] for r in res]); rm = np.array([r[1] for r in res])
            steps_total += sum(r[2] for r in res)
            sd = np.concatenate([rp, rm]).std() + 1e-8
            theta = theta + (ALPHA / (n_dir * sd)) * ((rp - rm) @ deltas).astype(np.float32)
            it += 1
            if it % 50 == 0:
                print(f"it {it} t {time.time() - ctx['deadline'] + ctx['time_budget_s']:.0f}s "
                      f"steps {steps_total} mean return {np.concatenate([rp, rm]).mean():.1f}", flush=True)
    print(f"done: {it} iterations, {steps_total} env steps", flush=True)
    return {"theta": theta.astype(np.float32)}
