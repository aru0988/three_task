"""指标与门禁判定（spec 9/10）。纯函数，便于单测；不依赖 GPU。"""
from __future__ import annotations

import math

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from . import protocol

GATE_MIN, GATE_MAX = protocol.GATE_MIN, protocol.GATE_MAX
AUC_FLOORS = (protocol.AUC_FLOOR_CTR, protocol.AUC_FLOOR_CVR, protocol.AUC_FLOOR_BSI)


def auc_score(y_true, y_hat) -> float:
    yt = y_true.int().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true).astype(int)
    yh = y_hat.detach().cpu().numpy() if isinstance(y_hat, torch.Tensor) else np.asarray(y_hat, dtype=float)
    return float(roc_auc_score(yt, yh))


def env_accuracy(env_pred, env_ids) -> float:
    """env_pred: (N, num_envs) 概率/对数概率 → argmax 与 env_ids 的一致率。"""
    pred = env_pred.argmax(dim=1)
    ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
    n = min(len(pred), len(ids))
    return float((pred[:n] == ids[:n]).float().mean())


def evaluate_a_gates(facts: dict) -> dict:
    """A 类协议正确性门禁（spec 10）。A3 为按需复跑，首轮 SKIP。"""
    a1_ok = facts["backbone_sha_before"] == facts["backbone_sha_after"] and facts["backbone_grads_none"]
    a2_ok = facts["prefix_sha_ok"] and facts["fingerprint_sha_match"]
    a4_ok = facts["len_ok"] and facts["counts_ok"]
    a5_ok = facts["env_ids_sha_match"]
    a6_ok = facts["backbone_sha_matches_stage1"] and facts["stage1_id_recorded"]
    return {
        "A1": {
            "verdict": "PASS" if a1_ok else "FAIL",
            "detail": f"sha_before==sha_after:{facts['backbone_sha_before'] == facts['backbone_sha_after']}, "
                      f"grads_none:{facts['backbone_grads_none']}",
        },
        "A2": {
            "verdict": "PASS" if a2_ok else "FAIL",
            "detail": f"prefix_sha_ok:{facts['prefix_sha_ok']}, fingerprint_sha_match:{facts['fingerprint_sha_match']}",
        },
        "A3": {"verdict": "SKIP", "detail": "按需复跑（spec 10，不阻塞首轮）"},
        "A4": {
            "verdict": "PASS" if a4_ok else "FAIL",
            "detail": f"len_ok:{facts['len_ok']}, counts_ok:{facts['counts_ok']}",
        },
        "A5": {"verdict": "PASS" if a5_ok else "FAIL", "detail": f"env_ids_sha_match:{facts['env_ids_sha_match']}"},
        "A6": {
            "verdict": "PASS" if a6_ok else "FAIL",
            "detail": f"backbone_sha_matches_stage1:{facts['backbone_sha_matches_stage1']}, "
                      f"stage1_id_recorded:{facts['stage1_id_recorded']}",
        },
    }


def evaluate_b_gates(facts: dict) -> dict:
    """B 类活性与机制门禁（spec 10；smoke 只记录不判定）。"""
    ctr_ok = facts["auc_val_ctr"] >= protocol.AUC_FLOOR_CTR
    cvr_ok = facts["auc_val_cvr"] >= protocol.AUC_FLOOR_CVR
    bsi_ok = facts["auc_test_bsi"] >= protocol.AUC_FLOOR_BSI
    b1_ok = ctr_ok and cvr_ok and bsi_ok

    gap = abs(facts["auc_val_bsi_best"] - facts["auc_test_bsi"])
    b2_ok = gap <= protocol.VAL_TEST_GAP_BSI

    gate_mean = facts["gate_mean"]
    b3_ok = bool(gate_mean) and all(GATE_MIN <= float(g) <= GATE_MAX for g in gate_mean)

    events = facts["cluster_events"]
    train_size = facts.get("train_size", 0)
    if not events:
        b4_verdict, b4_detail = "N/A", "无 cluster 调用（0 次，未判定）"
    else:
        share = protocol.ENV_SHARE_MIN * train_size
        shares = [(e["env_0"], e["env_1"]) for e in events]
        b4_ok = all(e0 >= share and e1 >= share for e0, e1 in shares)
        b4_verdict = "PASS" if b4_ok else "FAIL"
        b4_detail = f"env 占比下限 {protocol.ENV_SHARE_MIN:.0%}；events={shares}"

    return {
        "B1": {
            "verdict": "PASS" if b1_ok else "FAIL",
            "detail": f"CTR {facts['auc_val_ctr']:.4f}>={protocol.AUC_FLOOR_CTR}:{ctr_ok}, "
                      f"CVR {facts['auc_val_cvr']:.4f}>={protocol.AUC_FLOOR_CVR}:{cvr_ok}, "
                      f"BSI {facts['auc_test_bsi']:.4f}>={protocol.AUC_FLOOR_BSI}:{bsi_ok}",
        },
        "B2": {"verdict": "PASS" if b2_ok else "FAIL", "detail": f"|val-test|={gap:.4f} <= {protocol.VAL_TEST_GAP_BSI}"},
        "B3": {"verdict": "PASS" if b3_ok else "FAIL", "detail": f"gate_mean={list(gate_mean)} ∈ [{GATE_MIN}, {GATE_MAX}]"},
        "B4": {"verdict": b4_verdict, "detail": b4_detail},
    }


def hard_pass(a_gates: dict, enforce_b: bool, b_gates: dict | None = None) -> bool:
    """A 类必须全 PASS（A3 SKIP 视为通过）；enforce_b 时 B 类必须 PASS 或 N/A。"""
    a_ok = all(g["verdict"] in ("PASS", "SKIP") for g in a_gates.values())
    if not enforce_b:
        return a_ok
    return a_ok and b_gates is not None and all(g["verdict"] in ("PASS", "N/A") for g in b_gates.values())


# ---- Null Expert 处理臂（跨数据集复现；spec: docs/superpowers/specs/2026-10-03-aliccp-stage2-null-expert-design.md）----
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

    诊断值只要求同机同输入确定性（不承担 M7 的逐位复现契约）：逐样本值按批序累积，
    result() 时一次性在 float64 上计算。只在 val 上使用，test 不参与任何机制诊断（spec §7）。
    """

    QUANTILES = (10, 25, 50, 75, 90)

    def __init__(self):
        self.null_values = []          # 逐样本保留（val 500k 行 → 内存可忽略）
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


class SourceGateStats:
    """源任务 gate 均值（C 类）：backbone gate_networks[i] 输出的 specific 分支在样本维度的均值。

    与 CensusIncome 臂 GateStats 同口径：gate 输出 2 维（specific / general）且过 softmax，
    两维互补，这里报告 specific 分支均值（= 1 − general 分支均值），长度 = num_tasks。
    """

    def __init__(self, num_tasks: int = protocol.NUM_TASKS):
        self.total = torch.zeros(num_tasks, 2, dtype=torch.float64)
        self.count = 0

    def update(self, gate_outs) -> None:
        for i, gate_out in enumerate(gate_outs):
            # 累加器恒在 CPU：先在原设备归约，再显式 .cpu()，否则 CPU 累加器接 CUDA 张量会 device mismatch
            self.total[i] += gate_out.detach().double().sum(dim=0).cpu()
        self.count += gate_outs[0].shape[0]

    def result(self) -> list:
        return (self.total[:, 0] / max(self.count, 1)).tolist()


def null_supervision_stats(null_values, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """分层与相关性（C 类）：BSI 标签分层 null 均值、corr(null, pred)、corr(null, |err|)、预测离散度。

    只在 val 上使用（spec §7）；BSI 正例率 ~99%，分层近乎退化，如实记录不解读。
    常量向量（零方差）→ 相关无定义，返回 None。
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
        "null_mean_bsi_pos": pos_mean,
        "null_mean_bsi_neg": neg_mean,
        "null_label_gap": (pos_mean - neg_mean) if (pos_mean is not None and neg_mean is not None) else None,
        "corr_null_pred": _pearson(values, pred),
        "corr_null_abs_err": _pearson(values, (y - pred).abs()),
        "pred_std": float(pred.std(unbiased=False)),
    }


# ---- Null Expert 臂的预注册接受标准（spec 2026-10-03-aliccp-stage2-null-expert-design.md §5；看到结果前写死）----
# 主判据（PI 指定 +0.0055，严于协议 §10.1 的 +0.005）：绝对最小值 = 基线 0.5988392178311113 + 0.0055
NULL_ARM_BASELINE_TEST_AUC = 0.5988392178311113
NULL_ARM_BASELINE_VAL_AUC = 0.5781533414372665
NULL_ARM_AUC_MIN = 0.6043392178311113
NULL_ARM_PROTOCOL_AUC_MIN = 0.6038392178311113   # 协议 10.1 口径（+0.005），仅次要记录，不用于判定
NULL_ARM_TOP1_MIN, NULL_ARM_TOP1_MAX = 0.05, 0.95


def null_arm_verdict(test_auc: float, val_auc_best: float, null_top1_rate: float) -> dict:
    """Null Expert 臂的机械判定（spec §5）：主判据三条与机制判据**同时**满足才算通过。"""
    checks = {
        "auc": bool(test_auc >= NULL_ARM_AUC_MIN),
        "val_direction": bool(val_auc_best > NULL_ARM_BASELINE_VAL_AUC),
        "top1": bool(NULL_ARM_TOP1_MIN <= null_top1_rate <= NULL_ARM_TOP1_MAX),
    }
    protocol_checks = {
        "auc": bool(test_auc >= NULL_ARM_PROTOCOL_AUC_MIN),
        "val_direction": bool(val_auc_best > NULL_ARM_BASELINE_VAL_AUC),
    }
    return {
        "test_auc": test_auc, "val_auc_best": val_auc_best, "null_top1_rate": null_top1_rate,
        "baseline_test_auc": NULL_ARM_BASELINE_TEST_AUC, "baseline_val_auc": NULL_ARM_BASELINE_VAL_AUC,
        "delta_test_auc": test_auc - NULL_ARM_BASELINE_TEST_AUC,
        "delta_val_auc": val_auc_best - NULL_ARM_BASELINE_VAL_AUC,
        "auc_min": NULL_ARM_AUC_MIN, "protocol_auc_min": NULL_ARM_PROTOCOL_AUC_MIN,
        "top1_range": [NULL_ARM_TOP1_MIN, NULL_ARM_TOP1_MAX],
        "checks": checks, "protocol_checks": protocol_checks,
        "pass": bool(all(checks.values())),
    }
