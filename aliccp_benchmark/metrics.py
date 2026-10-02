"""指标与门禁判定（spec 9/10）。纯函数，便于单测；不依赖 GPU。"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from multitaskrec.model import SPEC_ATTENUATION_INIT, SPEC_ATTENUATION_MIN

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


# ---- 可学习标量衰减臂（预注册：docs/superpowers/specs/2026-10-03-aliccp-stage2-learnable-attenuation-design.md §5）----

# 先例运行与记录值（逐位取自各自 run metrics.json；仅作对照记录，不重跑）
BASELINE_RUN_ID = "20261003-0130-p2M-v500k-t1M-m1688723512-short-b2e17f9"
NULL_RUN_ID = "20261003-0221-p2M-v500k-t1M-m1688723512-short-6224c0f-nullx"
CONTROL_RUN_ID = "20261003-0322-p2M-v500k-t1M-m1688723512-short-f88dca4-sattn"
BASELINE_TEST_AUC = 0.5988392178311113
BASELINE_VAL_AUC = 0.5781533414372665
NULL_TEST_AUC = 0.6126857532817985
NULL_VAL_AUC = 0.5919398413138859
NULL_DELTA_TEST_AUC = 0.013846535450687258
NULL_DELTA_VAL_AUC = 0.013786499876619396
CONTROL_TEST_AUC = 0.6115453624066993
CONTROL_VAL_AUC = 0.5914326183692061
CONTROL_DELTA_TEST_AUC = 0.012706144575588052
CONTROL_DELTA_VAL_AUC = 0.013279276931939643
# post-hoc 系数（固定对照臂冻结值）：**仅**作 §5.4 descriptive 距离参照；本臂代码路径不得使用
POSTHOC_COEF_F64 = 0.6972233730330467
POSTHOC_COEF_F32 = 0.6972233653068542
# 预注册构造与门禁常量（spec §1.3/§5.1）
LEARNABLE_COEF_MIN = SPEC_ATTENUATION_MIN     # = 0.05（model.py 单一事实来源）
LEARNABLE_RAW_INIT = SPEC_ATTENUATION_INIT    # = 1.0（恒等初始化）
LEARNABLE_PARAM_NAME = "spec_attenuation_raw"
LEARNABLE_CHANGE_MIN = 0.01                   # M1/M3/D1 的"实质性"阈值
LEARNABLE_INTERIOR_MARGIN = 0.01              # M3 内点余量（相对 c_min 与 1.0）
LEARNABLE_GRAD_MIN = 1e-12                    # M2 数值非零阈值（严格大于）
LEARNABLE_GRAD_WINDOW = 10                    # M2 前 N 步窗口
DEFAULT_TRAINABLE_PARAMS = 8129               # 默认臂实测（对照臂 run 记录；测试以真实配置独立复算）
LEARNABLE_TRAINABLE_PARAMS = 8130             # = 8129 + 1（恰一个标量）
# PI 口径（与前臂同一字面量；spec §5.2）
PI_DELTA_TEST_AUC = 0.0055
PI_TEST_AUC_MIN = 0.6043392178311113          # = BASELINE_TEST_AUC + 0.0055
PROTOCOL_DELTA_TEST_AUC = 0.005               # 协议 §10.1 级口径（单独记录）

LEARNABLE_CLASSIFICATIONS = (
    "INTEGRITY_FAIL",
    "DEGENERATE_FLOOR",
    "IDENTITY_PRESERVED",
    "GRADIENT_WINDOW_ANOMALY",
    "LEARNED_ATTENUATION_SUPPORTED",
    "LEARNED_ATTENUATION_NO_UTILITY",
)


def learnable_attenuation_verdict(
    *,
    test_auc: float,
    val_auc_best: float,
    c_final: float,
    grad_abs_max_first10: float,
    trainable_params: int,
    trainable_params_default: int,
    trainable_param_names,
    buffer_keys,
    a_class_ok: bool,
) -> dict:
    """可学习衰减臂机械判定（spec §5.1–§5.4）：M1–M5 机制门 × U1 效用门 × D1/D2 方向门 + 六类分类。

    c_final 必须来自 best-epoch 重载后的 state_dict（即实际产生 test AUC 的头）。
    M4 按预注册字面量（== 8130）判定；trainable_params_default 仅记录（run 内独立构造默认头实测）。
    posthoc 距离与重现比属 descriptive，不参与分类。
    """
    names = list(trainable_param_names)
    checks = {
        "M1_change": bool(abs(c_final - LEARNABLE_RAW_INIT) >= LEARNABLE_CHANGE_MIN),
        "M2_gradient": bool(grad_abs_max_first10 > LEARNABLE_GRAD_MIN),
        "M3_interior": bool(LEARNABLE_COEF_MIN + LEARNABLE_INTERIOR_MARGIN <= c_final <= 1.0 - LEARNABLE_INTERIOR_MARGIN),
        "M4_params": bool(
            trainable_params == LEARNABLE_TRAINABLE_PARAMS
            and names.count(LEARNABLE_PARAM_NAME) == 1
            and sorted(buffer_keys) == ["new_env_idx"]
        ),
        "M5_freeze": bool(a_class_ok),
        "U1_pi_utility": bool(test_auc >= PI_TEST_AUC_MIN and val_auc_best > BASELINE_VAL_AUC),
        "protocol_utility": bool(test_auc - BASELINE_TEST_AUC >= PROTOCOL_DELTA_TEST_AUC and val_auc_best > BASELINE_VAL_AUC),
        "D1_direction": bool(c_final <= 1.0 - LEARNABLE_CHANGE_MIN),
        "D2_val_direction": bool(val_auc_best > BASELINE_VAL_AUC),
    }
    mechanism_pass = all(checks[k] for k in ("M1_change", "M2_gradient", "M3_interior", "M4_params", "M5_freeze"))
    if not (checks["M5_freeze"] and checks["M4_params"]):
        classification = "INTEGRITY_FAIL"
    elif c_final <= LEARNABLE_COEF_MIN + LEARNABLE_INTERIOR_MARGIN:
        classification = "DEGENERATE_FLOOR"
    elif abs(c_final - LEARNABLE_RAW_INIT) < LEARNABLE_CHANGE_MIN:
        classification = "IDENTITY_PRESERVED"
    elif not checks["M2_gradient"]:
        classification = "GRADIENT_WINDOW_ANOMALY"
    elif checks["U1_pi_utility"]:
        classification = "LEARNED_ATTENUATION_SUPPORTED"
    else:
        classification = "LEARNED_ATTENUATION_NO_UTILITY"

    delta_test = test_auc - BASELINE_TEST_AUC
    delta_val = val_auc_best - BASELINE_VAL_AUC
    return {
        "test_auc": test_auc,
        "val_auc_best": val_auc_best,
        "baseline_test_auc": BASELINE_TEST_AUC,
        "baseline_val_auc": BASELINE_VAL_AUC,
        "delta_test_auc": delta_test,
        "delta_val_auc": delta_val,
        "c_final": c_final,
        "c_min": LEARNABLE_COEF_MIN,
        "raw_init": LEARNABLE_RAW_INIT,
        "grad_abs_max_first10": grad_abs_max_first10,
        "trainable_params": trainable_params,
        "trainable_params_default": trainable_params_default,
        "trainable_params_expected": LEARNABLE_TRAINABLE_PARAMS,
        "checks": checks,
        "mechanism_pass": mechanism_pass,
        "utility_pass": checks["U1_pi_utility"],
        "descriptive": {
            "reproduction_ratio_test_vs_null": delta_test / NULL_DELTA_TEST_AUC,
            "reproduction_ratio_test_vs_control": delta_test / CONTROL_DELTA_TEST_AUC,
            "reproduction_ratio_val_vs_null": delta_val / NULL_DELTA_VAL_AUC,
            "reproduction_ratio_val_vs_control": delta_val / CONTROL_DELTA_VAL_AUC,
            "posthoc_distance_f64": abs(c_final - POSTHOC_COEF_F64),
            "posthoc_distance_f32": abs(c_final - POSTHOC_COEF_F32),
        },
        "classification": classification,
    }


# ---- C 类诊断（只记录；口径与 null/固定对照臂记录值兼容）----
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


class SourceGateStats:
    """源任务 gate 均值（C 类）：backbone gate_networks[i] 输出的 specific 分支在样本维度的均值。

    与 null/固定对照臂同口径（gate 输出 2 维 specific/general 且过 softmax，报告 specific 分支均值，
    长度 = num_tasks）；float64 逐批累加，累加器恒在 CPU。
    """

    def __init__(self, num_tasks: int = protocol.NUM_TASKS):
        self.total = torch.zeros(num_tasks, 2, dtype=torch.float64)
        self.count = 0

    def update(self, gate_outs) -> None:
        for i, gate_out in enumerate(gate_outs):
            # 先在原设备归约，再显式 .cpu()，否则 CPU 累加器接 CUDA 张量会 device mismatch
            self.total[i] += gate_out.detach().double().sum(dim=0).cpu()
        self.count += gate_outs[0].shape[0]

    def result(self) -> list:
        return (self.total[:, 0] / max(self.count, 1)).tolist()
