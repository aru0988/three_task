"""census/aliccp 逐值一致的共享指标：AUC 与 env 一致率。

仅收录经等价测试证明数值一致的实现（benchmark/tests/test_metrics.py 黄金常量）。
机制公式（GateStats/RepStats、新任务 gate 均值）、各数据集门禁判定保持各自实现，
不在此合并（spec 差异表）。
"""
from __future__ import annotations

import torch
from sklearn.metrics import roc_auc_score


def auc(y_true, y_hat) -> float:
    """roc_auc_score 的薄封装：接受 tensor 或 array，先在 CPU 上取值。"""
    yt = y_true if isinstance(y_true, torch.Tensor) else torch.as_tensor(y_true)
    yh = y_hat if isinstance(y_hat, torch.Tensor) else torch.as_tensor(y_hat)
    return float(roc_auc_score(yt.int().cpu().numpy(), yh.detach().cpu().numpy()))


def env_accuracy(env_pred, env_ids) -> float:
    """env_pred (N, num_envs) 的 argmax 与 env_ids 的一致率；长度不等时按 min 截断。"""
    pred = env_pred.argmax(dim=1)
    ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
    n = min(len(pred), len(ids))
    return float((pred[:n].cpu() == ids[:n].cpu()).float().mean())
