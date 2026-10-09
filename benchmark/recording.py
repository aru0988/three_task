"""census/aliccp 共用的记录层 manager：逐 epoch val AUC（+可选 env_acc）、cluster 事件、best_epoch()。

只挂记录钩子，不改变 multitaskrec.MPTRecTrainManager 的任何计算行为（与两个旧实现逐值一致，
见 benchmark/tests/test_recording.py 黄金常量与旧实现对照测试）。两个数据集差异做成显式配置：

- record_env_acc：census=True，每个 val epoch 追加一次全训练集 env 一致率（M1）；
  aliccp=False，其 env_acc 由阶段 1 结束后 env_accuracy_probe 在前 200 个 batch 上计算。
- cluster_epoch_offset：cluster 事件中 epoch 的记法。cluster_2 在该 epoch 的 val 评估之前调用，
  census 记 `len(val_epoch_aucs)`（偏移 0），aliccp 记 `len(val_epoch_aucs)+1`（偏移 1）。
"""
from __future__ import annotations

import torch

from benchmark.metrics import env_accuracy
from multitaskrec.train import MPTRecTrainManager

NUM_ENVS = 2


class RecordingMPTRecTrainManager(MPTRecTrainManager):
    """记录层：val AUC 曲线、cluster 事件、best_epoch()。不改变训练行为。"""

    def __init__(self, *args, record_env_acc=False, cluster_epoch_offset=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_env_acc = record_env_acc
        self.cluster_epoch_offset = cluster_epoch_offset
        self.val_epoch_aucs: list[list[float]] = []
        self.env_accs: list[float] = []
        self.cluster_records: list[dict] = []

    def evaluation_two_task(self, data_loader):
        auc_pair = super().evaluation_two_task(data_loader)
        if data_loader is self.val_loader:
            self.val_epoch_aucs.append([float(a) for a in auc_pair])
            if self.record_env_acc:
                self.env_accs.append(self.train_env_acc())
        return auc_pair

    def cluster_2(self):
        previous = self.env_ids.clone()
        updated = super().cluster_2()
        counts = torch.bincount(updated, minlength=NUM_ENVS).tolist()
        self.cluster_records.append({"epoch": len(self.val_epoch_aucs) + self.cluster_epoch_offset,
                                     "diff_num": int((previous != updated).sum()),
                                     "env_counts": [int(c) for c in counts]})
        return updated

    @torch.no_grad()
    def train_env_acc(self) -> float:
        """M1：训练集上 env_pred 与当前 env_ids 的一致率（父类每轮开头重设 train()，无副作用）。"""
        self.model.eval()
        log_probs, ids = [], []
        for step, batch in enumerate(self.train_loader):
            features = {key: value.to(self.device) for key, value in batch[-1].items()}
            log_probs.append(self.model(features)["env_pred"].cpu())
            ids.append(self.env_ids[self.batch_size * step: self.batch_size * (step + 1)])
        return env_accuracy(torch.cat(log_probs), torch.cat(ids))

    def best_epoch(self) -> int:
        """与仓库选点规则一致：sum(auc_val) 首次取到最大值者（1-based，spec 7.5）。"""
        sums = [a + b for a, b in self.val_epoch_aucs]
        return int(max(range(len(sums)), key=lambda i: sums[i])) + 1
