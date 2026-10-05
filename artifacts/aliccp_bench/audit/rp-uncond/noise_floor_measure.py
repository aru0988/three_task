"""Measure the fp32 noise floor of PromptStats ratio_std for a constant-direction (U-like) arm.

Mimics rp_uncond: m constant (expand), alpha pinned, scale = ||h||/sqrt(d) detached, delta = alpha*scale*m,
stats computed in fp64 from fp32 tensors (same as RP.PromptStats.update). Real dims d=64, batch 2000.
"""
import math
import sys

import torch

sys.path.insert(0, r"D:\MPT-Rec-three_task\MPT-Rec-exp-aliccp-rp-unconditional")

from aliccp_benchmark import residual_prompt as RP
from aliccp_benchmark import rp_pinned as RPP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)
torch.manual_seed(0)

d = 64
alpha = torch.tensor(RPP.PINNED_ALPHA, dtype=torch.float32, device=device)
# constant direction: rows identical via expand (as in the planned implementation)
m1 = torch.tanh(torch.randn(1, d, device=device))
stats = RP.PromptStats()
n_batches = 250
for b in range(n_batches):
    batch = 2000
    m = m1.expand(batch, d)
    reps = []
    for scale_norm in (0.05, 0.5, 5.0):
        h = torch.randn(batch, d, device=device)
        h = h / h.norm(dim=-1, keepdim=True) * scale_norm
        h = h * (1 + 0.3 * torch.rand(batch, 1, device=device))
        reps.append(h)
    deltas = []
    for h in reps:
        s = h.detach().norm(dim=-1, keepdim=True) / math.sqrt(d)
        deltas.append(alpha * s * m)
    geff = RP.effective_gate(m, alpha)
    stats.update(deltas, reps, geff)
res = stats.result()
print("ratio_mean:", res["streams"]["ratio_mean"])
print("ratio_std :", res["streams"]["ratio_std"])
print("geff_std  :", res["gate"]["geff_std"])
print("geff_min  :", res["gate"]["geff_min"])
print("ratio_max :", res["streams"]["ratio_max"])
print("ratio_std / 1e-9 =", [float(v) / 1e-9 for v in res["streams"]["ratio_std"]])

# ---- per-batch (2000-row) ratio max-minus-min, matching UA5(d) probe ----
print("\n-- per-batch 2000-row ratio spread (UA5-d style) --")
worst = 0.0
for b in range(8):
    batch = 2000
    m = m1.expand(batch, d)
    h = torch.randn(batch, d, device=device)
    h = h / h.norm(dim=-1, keepdim=True) * 5.0
    h = h * (1 + 0.3 * torch.rand(batch, 1, device=device))
    s = h.detach().norm(dim=-1, keepdim=True) / math.sqrt(d)
    delta = alpha * s * m
    dd = delta.detach().double().cpu()
    hh = h.detach().double().cpu()
    ratio = dd.norm(dim=1) / hh.norm(dim=1).clamp_min(1e-12)
    spread = float(ratio.max() - ratio.min())
    worst = max(worst, spread)
    print(f"batch {b}: spread = {spread!r}")
print("worst batch spread:", worst, "-> /1e-9 =", worst / 1e-9)
