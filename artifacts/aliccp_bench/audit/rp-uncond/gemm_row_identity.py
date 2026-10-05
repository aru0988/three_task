"""Decide the exact prompt_deltas form: bitwise equality of gen output across shape paths."""
import torch
import torch.nn as nn

for device in (torch.device("cuda"), torch.device("cpu")):
    torch.manual_seed(0)
    gen = nn.Sequential(nn.Linear(80, 16), nn.ReLU(), nn.Linear(16, 64)).to(device).eval()
    c = torch.randn(80, device=device)
    with torch.no_grad():
        m1 = torch.tanh(gen(c.unsqueeze(0)))                      # (1,64)
        for B in (2, 6, 7, 64, 2000):
            mb = torch.tanh(gen(c.unsqueeze(0).expand(B, -1)))    # (B,64) identical rows
            rows_ok = torch.equal(mb, mb[0:1].expand_as(mb))
            cross_ok = torch.equal(mb[0], m1[0])
            print(f"{device} B={B}: rows_bitidentical={rows_ok} row0==single-row {cross_ok}")
        # huge/zero input invariance path: (1,80) fixed shape only depends on c
        z = torch.tanh(gen(c.unsqueeze(0)))
        print(f"{device}: repeat single-row bit-identical: {torch.equal(m1, z)}")
