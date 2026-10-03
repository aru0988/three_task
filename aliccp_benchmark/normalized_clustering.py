"""处理臂机制：`cluster_2` 的逐任务损失尺度校正（rank/quantile 归一化后再 argmin）。

定位（预注册见 docs/superpowers/specs/2026-10-03-aliccp-stage1-normalized-clustering-design.md）：
scale-correction ablation——只把"跨任务逐样本损失直接比较"改为"先在每任务内做确定性
rank/quantile 归一化、再比较"，修 AliCCP 上因 CTR/CVR 损失尺度不可比（65×）导致的
B4 环境坍缩（raw argmin 退化为 CVR 标签函数）。无超参数、无学习、不消耗 RNG；
不改 `multitaskrec/*`（默认关闭时本模块不可达）。

语义钉死（§2.3；测试强制）：
- 输入 = 当次聚类事件的逐样本损失 (N, 2) float32 CPU（与仓库 `cluster_2` 完全同 ops 产出）；
- 逐任务 1-based average-rank（并列取平均）→ q = (rank − 1)/(N − 1) ∈ [0, 1]（float64）；
- 全并列 → 恒 0.5；N=1 → 恒 0；raw/q 在 argmin 前必须全有限（否则 AssertionError）；
- 分配 = torch.argmin(q, dim=1)（并列取先索引，确定性），int64、长度 N、CPU。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import protocol
from .bench import ClusteringArm, RecordingMPTRecTrainManager

CLUSTERING_NAME = "rank_normalized"
_DIAG_QUANTILES = (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0)


def per_task_rank01(losses: torch.Tensor) -> torch.Tensor:
    """逐任务 rank/quantile 变换（average-rank 归一化到 [0, 1]；float64 输出）。

    对任意严格单调的逐任务变换不变；全并列列恒 0.5；N==1 恒 0。确定性、无 RNG。
    """
    l = losses.double()
    n = l.shape[0]
    cols = []
    for k in range(l.shape[1]):
        uniq, inv, counts = torch.unique(l[:, k], return_inverse=True, return_counts=True)
        cum = torch.cumsum(counts, dim=0)
        avg_rank = (cum - counts).double() + (counts.double() + 1.0) / 2.0  # 1-based 平均秩
        ranks = avg_rank[inv]
        cols.append((ranks - 1.0) / max(n - 1, 1))
    return torch.stack(cols, dim=1)


def _quantile_summary(values: torch.Tensor) -> dict:
    v = values.double()
    out = {"mean": float(v.mean()), "std": float(v.std(unbiased=False))}
    for q in _DIAG_QUANTILES:
        out[f"q{q:g}"] = float(torch.quantile(v, q))
    return out


@torch.no_grad()
def rank_normalized_cluster_2(model, train_loader, device) -> tuple[torch.Tensor, dict]:
    """`cluster_2` 的尺度校正版本：同 ops 复算逐样本损失 → 逐任务 rank 归一化 → argmin。

    返回 (assignment, diagnostics)；不改变训练状态之外的任何行为，不消耗 RNG。
    """
    model.eval()
    loss_func = nn.BCELoss(reduction="none")
    chunks = []
    for y_0, y_1, _, features in train_loader:
        for key in features:
            features[key] = features[key].to(device)
        pred = model.cluster_predict(features)
        loss_0 = loss_func(pred[0].cpu(), y_0.float())
        loss_1 = loss_func(pred[1].cpu(), y_1.float())
        chunks.append(torch.stack([loss_0, loss_1], dim=1))
    raw = torch.cat(chunks, dim=0)  # (N, 2) float32 CPU（与 multitaskrec/train.py cluster_2 同 ops）
    if not bool(torch.isfinite(raw).all()):
        raise AssertionError("聚类损失向量含非有限值（raw）")
    normalized = per_task_rank01(raw)
    if not bool(torch.isfinite(normalized).all()):
        raise AssertionError("归一化损失含非有限值")
    assignment = torch.argmin(normalized, dim=1)
    diagnostics = {
        "raw_loss_quantiles": {
            task: _quantile_summary(raw[:, k]) for k, task in enumerate(("ctr_task0", "cvr_task1"))
        },
        "normalized_loss_quantiles": {
            task: _quantile_summary(normalized[:, k]) for k, task in enumerate(("ctr_task0", "cvr_task1"))
        },
        "env_0": int((assignment == 0).sum()),
        "env_1": int((assignment == 1).sum()),
        "env_0_share": float((assignment == 0).double().mean()),
    }
    return assignment, diagnostics


class RankNormalizedClusteringMPTRecTrainManager(RecordingMPTRecTrainManager):
    """处理臂 manager：仅覆写 `cluster_2` 的赋值规则（尺度校正 → argmin）。

    事件 schema 与基类逐键一致（epoch/diff_num/env_0/env_1，B4 判定口径不变）；
    额外的分位数诊断进 `cluster_diagnostics`（落 meta，不参与训练）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_diagnostics: list = []

    def cluster_2(self):
        old_env_ids = self.env_ids.clone()
        result, diagnostics = rank_normalized_cluster_2(self.model, self.train_loader, self.device)
        counts = torch.bincount(result, minlength=protocol.NUM_ENVS)
        epoch = len(self.val_epoch_aucs) + 1  # cluster 在该 epoch 的 val 评估之前调用（与基类记录一致）
        self.cluster_events.append(
            {
                "epoch": epoch,
                "diff_num": int((old_env_ids != result).sum()),
                "env_0": int(counts[0]),
                "env_1": int(counts[1]),
            }
        )
        self.cluster_diagnostics.append({"epoch": epoch, **diagnostics})
        return result


RANK_NORMALIZED_ARM = ClusteringArm(name=CLUSTERING_NAME, manager_factory=RankNormalizedClusteringMPTRecTrainManager)
