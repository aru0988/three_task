"""AUC、机制指标 M1/M3/M4、A/B 门禁判定（spec 7、8）。不 import fvcore，不做 FLOPs。"""
from __future__ import annotations

import math

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


class NullRouteStats:
    """M7：null 候选（router 权重最后一列）的样本均值与 top1 占比。

    逐样本左结合累加（而非按批 tensor 归约）：单批调用与任意分批流式调用结果**逐位相同**，
    且与十进制字面量的左结合求和在双精度下一致——M7 需可逐位复现，不依赖归约内核的求和顺序。
    累加器因此恒为 Python float（CPU），任意设备（含 CUDA）的输入都只先归约到 CPU 再累加。
    """

    def __init__(self):
        self.weight_sum = 0.0
        self.top1_count = 0
        self.count = 0

    def update(self, weights: torch.Tensor) -> None:
        null_weights = weights.detach()[:, -1].cpu().tolist()          # float32 → Python float 无损
        null_is_top1 = (weights.detach().argmax(dim=1) == weights.shape[1] - 1).cpu().tolist()
        for weight, is_top1 in zip(null_weights, null_is_top1):
            self.weight_sum += weight
            self.top1_count += int(is_top1)
        self.count += len(null_weights)

    def result(self) -> dict:
        n = max(self.count, 1)
        return {"null_mean": self.weight_sum / n, "null_top1_rate": self.top1_count / n}


class NullRouteDiagnostics:
    """C 类：null 质量分布（std/var/分位数）与 router 熵、逐样本方差。

    移植自姊妹线 `aliccp_benchmark/metrics.py`（`exp/aliccp-stage2-null-expert` `6224c0f`），口径逐字一致，
    仅命名空间不同。诊断值只要求同机同输入确定性（**不**承担 M7 的逐位复现契约）：逐样本值按批序累积，
    result() 时一次性在 float64 上计算。只在 val 上使用，test 不参与任何机制诊断（预注册 §5）。
    """

    QUANTILES = (10, 25, 50, 75, 90)

    def __init__(self):
        self.null_values = []          # 逐样本保留（Census val 49881 行 → 内存可忽略）
        self.entropy_sum = 0.0
        self.entropy_sq_sum = 0.0
        self.route_var_sum = 0.0
        self.count = 0

    def update(self, weights: torch.Tensor) -> None:
        w = weights.detach().cpu().double()
        self.null_values.extend(w[:, -1].tolist())
        # 0 权重贡献恰为 0（clamp 只防 log(0)，乘回原 w）；softmax 输出严格为正，clamp 仅为防御
        entropy = -(w.clamp_min(1e-12).log() * w).sum(dim=1)
        self.entropy_sum += float(entropy.sum())
        self.entropy_sq_sum += float((entropy * entropy).sum())
        self.route_var_sum += float(w.var(dim=1, unbiased=False).sum())
        self.count += w.shape[0]

    def result(self) -> dict:
        n = max(self.count, 1)
        values = torch.tensor(self.null_values, dtype=torch.float64) if self.null_values else torch.zeros(1, dtype=torch.float64)
        ent_mean = self.entropy_sum / n
        ent_var = max(self.entropy_sq_sum / n - ent_mean * ent_mean, 0.0)
        out = {
            "null_std": float(values.std(unbiased=False)),
            "null_var": float(values.var(unbiased=False)),
            "route_entropy_mean": ent_mean,
            "route_entropy_std": math.sqrt(ent_var),
            "route_var_mean": self.route_var_sum / n,
        }
        for q in self.QUANTILES:
            out[f"null_q{q:02d}"] = float(torch.quantile(values, q / 100))
        return out


def null_supervision_stats(null_values, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """分层与相关性（C 类）：education 标签分层 null 均值、corr(null, pred)、corr(null, |err|)、预测离散度。

    移植自姊妹线 `aliccp_benchmark/metrics.py`（`6224c0f`）的 `null_supervision_stats`，标签名由 BSI 改为
    education（键 `null_mean_edu_pos/neg`）。只在 val 上使用（预注册 §5）；常量向量（零方差）→ 相关无定义，
    返回 None，如实记录不解读。
    """
    values = torch.tensor(list(null_values), dtype=torch.float64)
    y = y_true.detach().cpu().double().flatten()
    pred = y_pred.detach().cpu().double().flatten()

    def _pearson(a: torch.Tensor, b: torch.Tensor):
        a_c, b_c = a - a.mean(), b - b.mean()
        denom = float((a_c * a_c).sum() * (b_c * b_c).sum()) ** 0.5
        if denom == 0.0:
            return None
        return float((a_c * b_c).sum()) / denom

    pos, neg = y > 0.5, y <= 0.5
    pos_mean = float(values[pos].mean()) if bool(pos.any()) else None
    neg_mean = float(values[neg].mean()) if bool(neg.any()) else None
    return {
        "null_mean_edu_pos": pos_mean,
        "null_mean_edu_neg": neg_mean,
        "null_label_gap": (pos_mean - neg_mean) if (pos_mean is not None and neg_mean is not None) else None,
        "corr_null_pred": _pearson(values, pred),
        "corr_null_abs_err": _pearson(values, (y - pred).abs()),
        "pred_std": float(pred.std(unbiased=False)),
    }


# ---- Null Expert 臂的预注册接受标准（spec 2026-09-29-stage2-null-expert-design.md 5；看到结果前写死）----
# 对照基线 = 同 checkpoint、同 split、同 model seed 的既有 run 20260929-1735-s20260929-m1685480945-short-904f8d0
NULL_ARM_BASELINE_TEST_AUC = 0.8500685307
NULL_ARM_AUC_MIN = 0.8521                        # 主判据：AUC-Test-Education ≥ 0.8521
NULL_ARM_TOP1_MIN, NULL_ARM_TOP1_MAX = 0.05, 0.95  # 机制判据：null_top1_rate ∈ [0.05, 0.95]（两端含等号）


def null_arm_verdict(test_auc: float, null_top1_rate: float) -> dict:
    """Null Expert 臂的机械判定（spec 5）：主判据与机制判据**同时**满足才算通过。"""
    checks = {"auc": bool(test_auc >= NULL_ARM_AUC_MIN),
              "top1": bool(NULL_ARM_TOP1_MIN <= null_top1_rate <= NULL_ARM_TOP1_MAX)}
    return {"test_auc": test_auc, "null_top1_rate": null_top1_rate,
            "baseline_test_auc": NULL_ARM_BASELINE_TEST_AUC, "auc_min": NULL_ARM_AUC_MIN,
            "top1_range": [NULL_ARM_TOP1_MIN, NULL_ARM_TOP1_MAX],
            "checks": checks, "pass": bool(checks["auc"] and checks["top1"])}


@torch.no_grad()
def evaluate_newtask(newtask, backbone, loader, device, *, mechanism: bool = False) -> dict:
    """评估新任务头。mechanism=True 时额外算 M3/M4（只在 val 上开一次）；启用 Null Expert 时追加 M7。"""
    newtask.eval(); backbone.eval()                 # 三件套之 2：backbone 恒为 eval
    ys, preds = [], []
    gate = GateStats() if mechanism else None
    rep = RepStats() if mechanism else None
    null = NullRouteStats() if mechanism and getattr(newtask, "use_null_expert", False) else None
    for _, _, y, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)   # 三件套之 3：no_grad 抽取
        pred = newtask(dnn_input, gen_rep, spec_reps, env_embs)
        ys.append(y); preds.append(pred.detach())
        if mechanism:
            gate.update([backbone.gate_networks[i](dnn_input) for i in range(P.NUM_TASKS)])
            rep.update(gen_rep, spec_reps)
            if null is not None:
                null.update(newtask.routing_weights(dnn_input, env_embs))
    out = {"auc": auc(torch.cat(ys), torch.cat(preds))}
    if mechanism:
        out["gate_mean"] = gate.result(); out.update(rep.result())
        if null is not None:
            out.update(null.result())
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
