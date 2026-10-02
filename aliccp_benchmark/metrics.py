"""指标与门禁判定（spec 9/10）。纯函数，便于单测；不依赖 GPU。"""
from __future__ import annotations

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


# ---- 特定混合固定衰减对照臂（post-hoc 归因控制；spec: docs/superpowers/specs/2026-10-03-aliccp-stage2-specific-attenuation-control-design.md）----

# 系数推导（spec §1.3）：唯一来源 = 前臂 run metrics.json 的 null_mean 全精度字段（非四舍五入）
NULL_RUN_ID = "20261003-0221-p2M-v500k-t1M-m1688723512-short-6224c0f-nullx"
BASELINE_RUN_ID = "20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9"
NULL_RUN_NULL_MEAN = 0.3027766269669533
SPEC_ATTENUATION_COEF = 1.0 - NULL_RUN_NULL_MEAN            # = 0.6972233730330467（文档字面量）
SPEC_ATTENUATION_COEF_F32 = 0.6972233653068542              # float32 铸造有效因子（记录备查，判定不依赖）
# 前臂与基线记录值（逐位取自各自 run 的 metrics.json）
BASELINE_TEST_AUC = 0.5988392178311113
BASELINE_VAL_AUC = 0.5781533414372665
NULL_TEST_AUC = 0.6126857532817985
NULL_VAL_AUC = 0.5919398413138859
NULL_DELTA_TEST_AUC = 0.013846535450687258
NULL_DELTA_VAL_AUC = 0.013786499876619396
# 预注册接受标准（spec §5；看到结果前写死）
ATTN_ARM_REPRO_MIN = 0.70                       # "most" 的预注册定义
ATTN_ARM_AUC_MIN_REPRO = 0.6085317926465923     # = BASELINE_TEST_AUC + 0.70 × NULL_DELTA_TEST_AUC（判定权威字面量）
ATTN_ARM_AUC_MIN_PI = 0.6043392178311113        # PI provisional +0.0055 口径（同前臂字面量；单独记录，不参与分类）
ATTN_ARM_PARTIAL_MIN = 0.30                     # 记录带下界（descriptive，不判定）


def attenuation_arm_verdict(test_auc: float, val_auc_best: float) -> dict:
    """衰减对照臂机械判定（spec §5）：reproduces_most（AUC 字面量，闭区间）× val_direction（严格 >）。

    两条同时满足 → ATTENUATION_SUPPORTED（衰减解释增益主体）；否则 ATTENUATION_FALSIFIED。
    PI provisional 口径单独记录（checks.pi_utility），不参与分类。
    """
    delta_test = test_auc - BASELINE_TEST_AUC
    delta_val = val_auc_best - BASELINE_VAL_AUC
    ratio_test = delta_test / NULL_DELTA_TEST_AUC
    ratio_val = delta_val / NULL_DELTA_VAL_AUC
    checks = {
        "reproduces_most": bool(test_auc >= ATTN_ARM_AUC_MIN_REPRO),
        "val_direction": bool(val_auc_best > BASELINE_VAL_AUC),
        "pi_utility": bool(test_auc >= ATTN_ARM_AUC_MIN_PI),
    }
    supported = bool(checks["reproduces_most"] and checks["val_direction"])
    if checks["reproduces_most"]:
        band = "MOST"
    elif ratio_test >= ATTN_ARM_PARTIAL_MIN:
        band = "PARTIAL"
    elif ratio_test >= 0.0:
        band = "MINOR"
    else:
        band = "NEGATIVE"
    return {
        "test_auc": test_auc,
        "val_auc_best": val_auc_best,
        "baseline_test_auc": BASELINE_TEST_AUC,
        "baseline_val_auc": BASELINE_VAL_AUC,
        "null_arm_test_auc": NULL_TEST_AUC,
        "null_arm_val_auc": NULL_VAL_AUC,
        "delta_test_auc": delta_test,
        "delta_val_auc": delta_val,
        "null_delta_test_auc": NULL_DELTA_TEST_AUC,
        "null_delta_val_auc": NULL_DELTA_VAL_AUC,
        "reproduction_ratio_test": ratio_test,
        "reproduction_ratio_val": ratio_val,
        "auc_min_repro": ATTN_ARM_AUC_MIN_REPRO,
        "auc_min_pi": ATTN_ARM_AUC_MIN_PI,
        "repro_min": ATTN_ARM_REPRO_MIN,
        "checks": checks,
        "band": band,
        "classification": "ATTENUATION_SUPPORTED" if supported else "ATTENUATION_FALSIFIED",
    }


# ---- C 类诊断（只记录；口径与 null 臂对照兼容）----
class SourceGateStats:
    """源任务 gate 均值（C 类）：backbone gate_networks[i] 输出的 specific 分支在样本维度的均值。

    与 null 臂同口径（gate 输出 2 维 specific/general 且过 softmax，报告 specific 分支均值，长度 = num_tasks）。
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


def _f64(values) -> torch.Tensor:
    return values.detach().cpu().double().flatten()


def pred_dispersion(preds) -> dict:
    """预测离散度（float64 终算；std/var 用 unbiased=False，与 null 臂 pred_std 同公式）。"""
    p = _f64(preds)
    out = {
        "pred_mean": float(p.mean()),
        "pred_std": float(p.std(unbiased=False)),
        "pred_var": float(p.var(unbiased=False)),
        "pred_min": float(p.min()),
        "pred_max": float(p.max()),
    }
    for q in (10, 25, 50, 75, 90):
        out[f"pred_q{q:02d}"] = float(torch.quantile(p, q / 100))
    return out


def pearson_corr(a, b):
    """Pearson 相关（float64）；任一常量向量（零方差）→ None。"""
    x, y = _f64(a), _f64(b)
    xc, yc = x - x.mean(), y - y.mean()
    denom = float((xc * xc).sum() * (yc * yc).sum()) ** 0.5
    if denom == 0.0:
        return None
    return float((xc * yc).sum()) / denom


def average_rank(values: torch.Tensor) -> torch.Tensor:
    """平均秩（并列取平均排名，与 scipy.stats.rankdata(method='average') 同口径）。"""
    arr = _f64(values).numpy()
    _, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    group_avg = (cum - counts) + (counts + 1) / 2.0
    return torch.from_numpy(group_avg[inverse])


def spearman_corr(a, b):
    return pearson_corr(average_rank(a), average_rank(b))


def logit_clip(preds, eps: float = 1e-12) -> torch.Tensor:
    """sigmoid 输出 → logit；两端 eps 截断（防御 float32 下溢到精确 0/1）。"""
    p = _f64(preds).clamp(eps, 1.0 - eps)
    return (p / (1.0 - p)).log()


def paired_delta_stats(a, b) -> dict:
    """逐样本配对差 a − b（同序）的分布统计（float64）。"""
    d = _f64(a) - _f64(b)
    out = {
        "delta_mean": float(d.mean()),
        "delta_std": float(d.std(unbiased=False)),
        "delta_min": float(d.min()),
        "delta_max": float(d.max()),
        "frac_pos": float((d > 0).double().mean()),
        "frac_neg": float((d < 0).double().mean()),
    }
    for q in (10, 50, 90):
        out[f"delta_q{q:02d}"] = float(torch.quantile(d, q / 100))
    return out


def cross_arm_stats(pred_a, pred_b) -> dict:
    """同序两臂预测对照：logit-Pearson / pred-Pearson / pred-Spearman + paired delta。"""
    out = {
        "pearson_logit": pearson_corr(logit_clip(pred_a), logit_clip(pred_b)),
        "pearson_pred": pearson_corr(pred_a, pred_b),
        "spearman_pred": spearman_corr(pred_a, pred_b),
    }
    out.update(paired_delta_stats(pred_a, pred_b))
    return out
