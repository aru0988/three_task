"""One-way, task-influenced U; the original G/S computation is untouched."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


class UniversalExpert(nn.Module):
    def __init__(self, widths, rep_dim=128, hidden=256, seed=1685480946):
        super().__init__()
        self.widths = list(widths)
        self.input_dim = sum(widths)
        self.mask_rng = torch.Generator().manual_seed(20261009)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Sequential(nn.Linear(self.input_dim + len(widths), hidden),
                                         nn.ReLU(), nn.Linear(hidden, rep_dim))
            self.decoder = nn.Linear(rep_dim, self.input_dim)
        self.register_buffer('mean', torch.zeros(self.input_dim))
        self.register_buffer('scale', torch.ones(self.input_dim))

    def set_normalizer(self, mean, std):
        self.mean.copy_(mean.detach())
        self.scale.copy_(std.detach().clamp_min(.1))

    def normalize(self, x):
        return ((x.detach() - self.mean) / self.scale).clamp(-10, 10)

    def sample_mask(self, n):
        # Exact fixed count, independently for each sample; private CPU RNG.
        rank = torch.rand(n, len(self.widths), generator=self.mask_rng).argsort(dim=1)
        mask = torch.zeros(n, len(self.widths), dtype=torch.bool)
        return mask.scatter_(1, rank[:, :max(1, round(.3 * len(self.widths)))], True)

    def expand_mask(self, mask):
        return torch.repeat_interleave(mask, torch.tensor(self.widths, device=mask.device), dim=1)

    def masked_input(self, x, mask):
        mask = mask.to(x.device)
        return torch.cat([self.normalize(x).masked_fill(self.expand_mask(mask), 0), mask.float()], 1)

    def forward(self, x):
        return self.encoder(torch.cat([self.normalize(x), x.new_zeros(len(x), len(self.widths))], 1))

    def losses(self, x, general, mask=None):
        mask = (self.sample_mask(len(x)) if mask is None else mask).to(x.device)
        target = self.normalize(x)
        predicted = self.decoder(self.encoder(self.masked_input(x, mask)))
        # Equal field weighting, rather than overweighting 4-D categorical fields.
        field_error = torch.stack([v.mean(1) for v in (predicted - target).square().split(self.widths, 1)], 1)
        field_zero = torch.stack([v.mean(1) for v in target.square().split(self.widths, 1)], 1)
        reconstruction = (field_error * mask).sum() / mask.sum()
        zero = (field_zero * mask).sum() / mask.sum()
        clean = self(x)
        std = clean.std(0, unbiased=False)
        variance = F.relu(.5 - torch.sqrt(clean.var(0, unbiased=False) + 1e-4)).mean()
        gu = general.detach()
        uc = (clean - clean.mean(0)) / std.clamp_min(.01)
        gc = (gu - gu.mean(0)) / gu.std(0, unbiased=False).clamp_min(.01)
        decorrelation = (uc.T @ gc / max(len(x), 1)).square().mean()
        total = reconstruction + .01 * decorrelation + .01 * variance
        return dict(total=total, reconstruction=reconstruction, zero=zero,
                    decorrelation=decorrelation, variance=variance)


class UniversalStage1(nn.Module):
    """Use original train manager; append auxiliary objective at its loss boundary.

    get_l2_reg is that manager's existing additional-loss hook. The base L2 term
    remains unchanged; only this wrapper adds the explicitly logged SSL terms.
    """
    def __init__(self, base, universal):
        super().__init__()
        self.base = base
        self.universal = universal
        self.loss_history = []
        self.auxiliary = None

    def forward(self, x, alpha=1):
        out = self.base(x, alpha)
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                embedded = self.base.embedding_network(x)
                general = self.base.shared_expert_network(embedded)
            self.auxiliary = self.universal.losses(embedded, general)
            self.loss_history.append({k: float(v.detach()) for k, v in self.auxiliary.items()})
        return out

    def get_l2_reg(self):
        return self.base.get_l2_reg() + self.auxiliary['total']

    def predict(self, x):
        return self.base.predict(x)

    def cluster_predict(self, x):
        return self.base.cluster_predict(x)


class ResidualHead(nn.Module):
    def __init__(self, original, rep_dim):
        super().__init__()
        self.core = copy.deepcopy(original)
        with torch.random.fork_rng(devices=[]):
            self.projection = nn.Linear(rep_dim, rep_dim, bias=False)
            with torch.no_grad():
                self.projection.weight.copy_(torch.eye(rep_dim))
        self.logit = nn.Parameter(torch.tensor(math.log(.1 / .9)))

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs, extra):
        h = self.core
        env = h.env_embedding_network(h.new_env_idx).squeeze(0)
        weights = F.softmax(h.projection_network(dnn_input) @ torch.stack(env_embs, 1) / h.temperature, -1)
        specific = (torch.stack(spec_reps, 2) @ weights.unsqueeze(2)).squeeze(-1)
        fused = (torch.stack([specific * env, gen_rep], 2) @ h.gate_network(dnn_input).unsqueeze(2)).squeeze(-1)
        fused = fused + self.logit.sigmoid() * self.projection(extra.detach())
        return h.tower_network(fused).reshape(-1)

    def get_l2_reg(self):
        return self.core.get_l2_reg()
