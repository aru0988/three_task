"""Matched B/U components for Universal Representation experiments."""

import math

import torch
from torch import nn

from multitaskrec.model import ReverseLayerF
from universal_experiment.model import UniversalStage1


def stage2_parameters(head, universal, base, *, freeze_u):
    """Freeze the base and return the exact optimizer parameter list."""
    for parameter in base.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in universal.parameters():
        parameter.requires_grad_(not freeze_u)
        parameter.grad = None
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    parameters = list(head.parameters())
    if not freeze_u:
        parameters.extend(universal.parameters())
    return parameters


def make_stage1_model(base, universal, arm, read_policy):
    """Construct one B/U Stage-1 arm without accepting historical controls."""
    if arm not in ("B", "U"):
        raise ValueError(f"paired runner supports only B/U, got {arm!r}")
    if read_policy not in ("no_read", "read_detached"):
        raise ValueError(f"unknown read policy {read_policy!r}")
    if arm == "B":
        return base
    if universal is None:
        raise ValueError("U arm requires a UniversalExpert")
    if read_policy == "no_read":
        return UniversalStage1(base, universal)
    return ReadDetachedStage1(base, universal)


class ReadDetachedStage1(nn.Module):
    """Let old-task heads read U without sending their gradients into U."""

    def __init__(self, base, universal):
        super().__init__()
        self.base = base
        self.universal = universal
        rep_dim = base.env_embedding_network.embedding_dim
        self.projections = nn.ModuleList()
        self.logits = nn.ParameterList()
        with torch.random.fork_rng(devices=[]):
            for _ in range(base.num_tasks):
                projection = nn.Linear(rep_dim, rep_dim, bias=False)
                with torch.no_grad():
                    projection.weight.copy_(torch.eye(rep_dim))
                self.projections.append(projection)
                self.logits.append(nn.Parameter(torch.tensor(math.log(.1 / .9))))
        self.loss_history = []
        self.auxiliary = None

    def forward(self, x, alpha=1):
        base = self.base
        dnn_input = base.embedding_network(x)
        gen_rep = base.shared_expert_network(dnn_input)
        gen_preds = [base.tower_networks[i](gen_rep).squeeze()
                     for i in range(base.num_tasks)]
        gate_outs = [gate(dnn_input) for gate in base.gate_networks]
        extra = self.universal(dnn_input).detach()

        fused_preds = []
        for i in range(base.num_tasks):
            spec_rep = base.specific_expert_networks[i](dnn_input)
            env = base.env_embedding_network(base.env_indices[i])
            all_reps = torch.stack([spec_rep * env, gen_rep], dim=2)
            fused = torch.matmul(all_reps, gate_outs[i].unsqueeze(dim=2)).squeeze()
            fused = fused + self.logits[i].sigmoid() * self.projections[i](extra)
            fused_preds.append(base.tower_networks[i](fused).squeeze())

        env_pred = base.env_classifier(ReverseLayerF.apply(gen_rep, alpha))
        if self.training and torch.is_grad_enabled():
            devices = ([next(base.parameters()).device.index]
                       if next(base.parameters()).is_cuda else [])
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                embedded = base.embedding_network(x)
                general = base.shared_expert_network(embedded)
            self.auxiliary = self.universal.losses(embedded, general)
            self.loss_history.append({k: float(v.detach())
                                      for k, v in self.auxiliary.items()})
        return {"gen_preds": gen_preds, "fused_preds": fused_preds,
                "env_pred": env_pred}

    def predict(self, x):
        return self.forward(x)["fused_preds"]

    def cluster_predict(self, x):
        return self.predict(x)

    def get_l2_reg(self):
        return self.base.get_l2_reg() + self.auxiliary["total"]
