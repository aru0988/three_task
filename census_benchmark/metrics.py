"""AUC、机制指标 M1/M3/M4、A/B 门禁判定（spec 7、8）。不 import fvcore，不做 FLOPs。"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from census_benchmark import protocol as P


def auc(y_true: torch.Tensor, y_hat: torch.Tensor) -> float:
    return float(roc_auc_score(y_true.int().cpu(), y_hat.detach().cpu()))


def env_accuracy(env_pred_logprob: torch.Tensor, env_ids: torch.Tensor) -> float:
    """M1：env_pred 与当前 env_ids 的一致率（梯度反转生效时应趋近 1/NUM_TASKS = 0.5）。"""
    return float((env_pred_logprob.argmax(dim=1).cpu() == env_ids.cpu()).float().mean())


class GateStats:
    """M3：各 gate 在样本维度的平均权重（流式累加，显存 O(1)）。

    gate_networks[i] 输出 2 维（specific 分支 / general 分支）且过 softmax → 两维互补，
    故 result() 报告**每个任务 specific 分支**的样本均值（= 1 − general 分支均值），长度 = num_tasks；
    这正是 B3「未坍缩到单一分支」判据所需的量（0.05 / 0.95 即两端的坍缩边界）。
    """

    def __init__(self, num_tasks: int = P.NUM_TASKS):
        self.total = torch.zeros(num_tasks, 2, dtype=torch.float64)
        self.count = 0

    def update(self, gate_outs: list[torch.Tensor]) -> None:
        for i, gate_out in enumerate(gate_outs):
            # 累加器恒在 CPU：先在原设备归约，再显式 .cpu()，否则 CPU 累加器接 CUDA 张量会 device mismatch
            self.total[i] += gate_out.detach().double().sum(dim=0).cpu()
        self.count += gate_outs[0].shape[0]

    def result(self) -> list[float]:
        return (self.total[:, 0] / max(self.count, 1)).tolist()


class RepStats:
    """M4：cos(gen_rep, spec_rep_i) 均值 + gen_rep 各维标准差均值（防常量表征）。

    用累加和 / 平方和代替"按固定 seed 抽样后保存中间张量"：全量、显存 O(1)、不受抽样影响
    （spec 7.2 实现约束以显存峰值为目的，流式累加达成同一目的且更强）。
    """

    def __init__(self, num_tasks: int = P.NUM_TASKS):
        self.cos_sum = torch.zeros(num_tasks, dtype=torch.float64)
        self.gen_sum = self.gen_sq_sum = None
        self.count = 0

    def update(self, gen_rep: torch.Tensor, spec_reps: list[torch.Tensor]) -> None:
        gen = gen_rep.detach().double()
        # 同 GateStats：归约留在原设备，结果显式 .cpu()，三个累加器恒为 CPU（result() 的返回类型不变）
        gen_sum = gen.sum(dim=0).cpu()
        gen_sq_sum = (gen * gen).sum(dim=0).cpu()
        self.gen_sum = gen_sum if self.gen_sum is None else self.gen_sum + gen_sum
        self.gen_sq_sum = gen_sq_sum if self.gen_sq_sum is None else self.gen_sq_sum + gen_sq_sum
        for i, spec in enumerate(spec_reps):
            self.cos_sum[i] += F.cosine_similarity(gen, spec.detach().double(), dim=1).sum().cpu()
        self.count += gen.shape[0]

    def result(self) -> dict:
        n = max(self.count, 1)
        mean = self.gen_sum / n
        std = (self.gen_sq_sum / n - mean * mean).clamp_min(0).sqrt()
        return {"cos_gen_spec": (self.cos_sum / n).tolist(), "gen_std": float(std.mean())}


@torch.no_grad()
def evaluate_newtask(newtask, backbone, loader, device, *, mechanism: bool = False) -> dict:
    """评估新任务头。mechanism=True 时额外算 M3/M4（只在 val 上开一次）。"""
    newtask.eval(); backbone.eval()                 # 三件套之 2：backbone 恒为 eval
    ys, preds = [], []
    gate = GateStats() if mechanism else None
    rep = RepStats() if mechanism else None
    for _, _, y, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)   # 三件套之 3：no_grad 抽取
        pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
        ys.append(y); preds.append(pred.detach())
        if mechanism:
            gate.update([backbone.gate_networks[i](dnn_input) for i in range(P.NUM_TASKS)])
            rep.update(gen_rep, spec_reps)
    out = {"auc": auc(torch.cat(ys), torch.cat(preds))}
    if mechanism:
        out["gate_mean"] = gate.result(); out.update(rep.result())
    return out


def judge(*, backbone_sha_equal: bool, grads_all_none: bool, split_ok: bool, split_stats_ok: bool,
          env_ids_ok: bool, auc_val_income: float, auc_val_marital: float, auc_test_education: float,
          auc_val_education_best: float, gate_mean: list[float], env_shares: list[float]) -> dict:
    """A1/A2/A4/A5 + B1–B4（spec 8）。A3 按需触发，由调用方另行写入，不进入 overall_pass。"""

    def verdict(passed: bool, **detail) -> dict:
        return {"pass": bool(passed), "detail": detail}

    report = {
        "A1": verdict(backbone_sha_equal and grads_all_none,
                      backbone_sha_equal=backbone_sha_equal, grads_all_none=grads_all_none),
        "A2": verdict(split_ok, split_fingerprint_consistent=split_ok),
        "A4": verdict(split_stats_ok, disjoint_and_complete=split_stats_ok),
        "A5": verdict(env_ids_ok, env_ids_sha256_matches_stage1=env_ids_ok),
        "B1": verdict(min(auc_val_income, auc_val_marital, auc_test_education) >= P.AUC_FLOOR,
                      auc_val_income=auc_val_income, auc_val_marital=auc_val_marital,
                      auc_test_education=auc_test_education, floor=P.AUC_FLOOR),
        "B2": verdict(abs(auc_val_education_best - auc_test_education) <= P.VAL_TEST_GAP,
                      gap=abs(auc_val_education_best - auc_test_education), limit=P.VAL_TEST_GAP),
        "B3": verdict(all(P.GATE_MIN <= w <= P.GATE_MAX for w in gate_mean), gate_mean=gate_mean),
        "B4": verdict(all(share >= P.ENV_SHARE_MIN for share in env_shares), env_shares=env_shares),
    }
    report["overall_pass"] = all(item["pass"] for item in report.values())
    report["failures"] = [key for key, item in report.items() if isinstance(item, dict) and not item["pass"]]
    return report
