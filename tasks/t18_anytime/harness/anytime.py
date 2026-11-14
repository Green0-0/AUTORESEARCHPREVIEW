"""Interruption distribution, FLOP counting and scoring for t18_anytime.

Interruption budget B (FLOPs per image) ~ LogUniform[B_MIN, B_MAX].
A model emits predictions at cumulative per-image costs c_1 <= c_2 <= ... <= c_S.
An example interrupted at budget B is scored with the latest prediction with c_k <= B;
if c_1 > B there is no prediction and it counts as wrong.
Because costs are input-independent (checked), the expectation over B is computed exactly:
    E[acc] = sum_k acc_k * P(c_k <= B < c_{k+1}),  c_{S+1} = +inf.
This equals the limit of averaging over infinitely many seeded interruption draws.
FLOPs are those counted by torch.utils.flop_counter.FlopCounterMode (conv, matmul, attention, ...;
one multiply-add = 2 FLOPs; elementwise / norm / pooling ops are not counted).
"""
import math

import torch
from torch.utils.flop_counter import FlopCounterMode

B_MIN = 2.0e6   # FLOPs per image
B_MAX = 2.0e9   # FLOPs per image
MAX_STAGES = 32
NUM_CLASSES = 100


def survival(c):
    """P(B >= c) for B ~ LogUniform[B_MIN, B_MAX]."""
    if c <= B_MIN:
        return 1.0
    if c >= B_MAX:
        return 0.0
    return (math.log(B_MAX) - math.log(c)) / (math.log(B_MAX) - math.log(B_MIN))


def stage_weights(costs):
    """w_k = P(stage k is the latest prediction completed before the interruption)."""
    s = [survival(c) for c in costs] + [0.0]
    return [max(0.0, s[k] - s[k + 1]) for k in range(len(costs))]


def expected_accuracy(costs, accs):
    return float(sum(w * a for w, a in zip(stage_weights(costs), accs)))


def _check_logits(out, bsz):
    if not torch.is_tensor(out) or out.dim() != 2 or out.shape[0] != bsz or out.shape[1] != NUM_CLASSES:
        raise ValueError(f"each emitted prediction must be a tensor [B,{NUM_CLASSES}], got "
                         f"{getattr(out, 'shape', type(out))}")


def count_stage_flops(model, x):
    """Run the anytime generator on batch x under FlopCounterMode.
    Returns (per-image cumulative FLOPs after each emitted prediction, list of argmax predictions)."""
    bsz = x.shape[0]
    costs, preds = [], []
    with torch.no_grad():
        counter = FlopCounterMode(display=False)
        with counter:
            it = iter(model.anytime(x))
            while True:
                try:
                    out = next(it)
                except StopIteration:
                    break
                total = counter.get_total_flops()       # everything computed so far, incl. this stage
                _check_logits(out, bsz)
                costs.append(total / bsz)
                preds.append(out.argmax(1))
                if len(costs) > MAX_STAGES:
                    raise ValueError(f"more than {MAX_STAGES} predictions emitted")
    return costs, preds


def run_stages(model, x):
    """Plain (uncounted) anytime pass; returns list of argmax predictions per stage."""
    bsz = x.shape[0]
    preds = []
    with torch.no_grad():
        for out in model.anytime(x):
            _check_logits(out, bsz)
            preds.append(out.argmax(1))
            if len(preds) > MAX_STAGES:
                raise ValueError(f"more than {MAX_STAGES} predictions emitted")
    return preds
