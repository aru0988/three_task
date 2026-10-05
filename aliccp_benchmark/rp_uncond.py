"""AliCCP 阶段 2 残差 Prompt 的**无条件固定条件对照**消融（unconditional fixed-condition control；
本分支 exp/aliccp-stage2-residual-prompt-unconditional-control）。

唯一事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-unconditional-control-design.md

消融语义（预注册 §2）：把逐样本条件方向 `P(x_i)` 换成**与样本无关的常量方向**——生成器（同一架构、
同一 2384 可训练参数预算、同一隔离 RNG 初始化路径）**喂入冻结常量条件向量 c**（§2.1：
`torch.Generator(manual_seed(20261006))` 单次 `randn(80)`，float32，sha256 `0d45cc46…`；非参数
buffer、随 checkpoint 落盘）；α 与 C_p 完全相同的钉死值 `+0.07101669907569885`（非参数 buffer、
优化器排除）；范数标定/注入流/委托逐字沿用。C_p 臂 = `f08ae6e` F_c 的**逐位重放**（REP_Cp 前置条件）。

  * `C_p`（variant `residual-prompt-pinned`，`rp_pinned.py` 原样运行）：条件 = 自身 `dnn_input`；
  * `U`（variant `residual-prompt-uncond`，本模块）：条件 = 常量 c（与样本无关；批内扩维输入，
    与 C_p 常量条件路径同一构造 ⇒ 预注册 §4-I8 逐位等价）。

判定（预注册 §5）：`analyze_runs` 只读三 run（B/C_p/U）记录 → 前置条件（ID/REP_B/REP_Cp/A/UA/CC/CV）
→ 因果判定树（`SAMPLE_CONDITIONING_SUPPORTED` / `SAMPLE_CONDITIONING_NOT_SUPPORTED` / `INVALID`，
主量 `GAP_U = test(C_p) − test(U)`）+ 两臂二级分类（`POSITIVE_IMPROVEMENT` /
`NO_CLEAR_IMPROVEMENT` / `CLEAR_DEGRADATION`）。U 臂门禁 = UA1–UA11（§5.2；含样本不变性 UA5 与
恒定门控 UA8）；学习活性带不适用（a priori，同 f08ae6e §1.4）。历史 +0.0055 判定单列保留。
不主张新颖性。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from aliccp_benchmark import residual_prompt as RP
from aliccp_benchmark import rp_pinned as RPP

# ---- 臂与接线（预注册 §2）----
VARIANT_UNCOND = "residual-prompt-uncond"                  # U：钉死 α 无条件/全局条件
RUN_ID_SUFFIX_UNCOND = "-rpu"
CLI_VARIANTS = (RP.BASELINE_VARIANT, RP.VARIANT, RPP.VARIANT_PINNED, VARIANT_UNCOND)
ALL_IMPLEMENTED_VARIANTS = CLI_VARIANTS + (RPP.VARIANT_PINNED_SHUFFLED,)


def arm_suffix_for(variant: str) -> str:
    """臂后缀分派（uncond ⇒ -rpu；其余经 `rp_pinned.run_id_suffix_for` 逐字沿用）。"""
    if variant == VARIANT_UNCOND:
        return RUN_ID_SUFFIX_UNCOND
    return RPP.run_id_suffix_for(variant)


# ---- 冻结判据常量（预注册 §1/§2/§5；只允许在看到结果之前修改）----
PINNED_ALPHA = RPP.PINNED_ALPHA                # fixed signed α（与 C_p 逐位相同）
UNCOND_CONST_SEED = 20261006                   # 常量条件向量种子（冻结）
COND_VECTOR_SHAPE = (80,)
COND_VECTOR_SHA256 = "0d45cc4611cebde9675858f1afc633a91b3e9114d6ddc02899a5d97f34babf58"
MATERIAL_DELTA = 0.001                         # 最小实质差异（= 用户指定；二级分类正边界同值）
SECONDARY_POSITIVE_MIN = 0.001                 # 二级分类：Δtest ≥ +0.001 ⇒ POSITIVE_IMPROVEMENT（闭）
SECONDARY_NEGATIVE_MAX = -0.02                 # Δtest ≤ −0.02 ⇒ CLEAR_DEGRADATION（闭）
RATIO_REL_DIFF_DISCLOSURE_TOL = 0.05           # 有效幅度披露阈值（预声明；描述性，不作门禁）

# ---- 结构机制门禁常量（预注册 §5.2）----
EXPECTED_TRAINABLE_PROMPT_KEYS = RPP.EXPECTED_TRAINABLE_PROMPT_KEYS
EXPECTED_STATE_EXTRA_KEYS = tuple(sorted(("prompt_gate", "uncond_condition")
                                         + EXPECTED_TRAINABLE_PROMPT_KEYS))    # state_dict 新增 6 键
EXPECTED_NEW_PARAMS_TOTAL = RPP.EXPECTED_NEW_PARAMS_TOTAL   # 2384（与 C_p 逐项相同）
EXPECTED_HEAD_PARAMS = RPP.EXPECTED_HEAD_PARAMS             # 8129
EXPECTED_COND_NUMEL = 80
EXPECTED_ALPHA_NUMEL = 1
UNCOND_STD_TOL = 1e-6                          # UA5(d)/UA8 等值性界（§12 C1b-2 校正后）
RATIO_SPREAD_TOL = RPP.RATIO_SPREAD_TOL        # UA7：三流 ratio_mean spread（S2）
PIN_BOUND_TOL = RP.BOUND_TOL                   # UA6：ratio_max ≤ |α| + 1e-6（S1）
PRED_STD_MIN_RATIO = RP.PRED_STD_MIN_RATIO     # UA9：pred_std ≥ 0.5 × ref_pred_std

# ---- 钉死身份（预注册 §1.5；与 verify_uncond_prerun.py 逐位一致）----
STAGE1_ID = RPP.STAGE1_ID
STAGE1_BACKBONE_SHA = RPP.STAGE1_BACKBONE_SHA
STAGE1_ENV_IDS_SHA = RPP.STAGE1_ENV_IDS_SHA
STAGE1_FINGERPRINT_SHA = RPP.STAGE1_FINGERPRINT_SHA
MODEL_SEED_PINNED = RPP.MODEL_SEED_PINNED
BASELINE_RUN_ID = RPP.BASELINE_RUN_ID
CORRECT_RUN_ID = RPP.CORRECT_RUN_ID
SHUF_HIST_RUN_ID = RPP.SHUF_HIST_RUN_ID
PINNED_CORRECT_RUN_ID = "20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp"
PINNED_SHUFFLED_RUN_ID = "20261005-1032-p2M-v500k-t1M-m1688723740-short-1c13841-rpps"
REF_HEAD_SHA = RPP.REF_HEAD_SHA
MECHANISM_REL = RPP.MECHANISM_REL
MECHANISM_PIN_LF_SHA = RPP.MECHANISM_PIN_LF_SHA
BASELINE_RECORD = RPP.BASELINE_RECORD
CORRECT_RECORD = RPP.CORRECT_RECORD
SHUF_HIST_RECORD = RPP.SHUF_HIST_RECORD
PINNED_CORRECT_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5798886656441761, "test_auc_bsi": 0.5947650925865714,
    "gate_mean": [0.775494625, 0.224505265625],
    "per_epoch_val": [0.46759930720014714, 0.49016156687227064, 0.5156547237007145,
                      0.5480714746135289, 0.5798886656441761],
    "variant": RPP.VARIANT_PINNED, "classification": "VALID_NEGATIVE",
    "alpha_final": PINNED_ALPHA,
}
PINNED_SHUFFLED_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5817369266195509, "test_auc_bsi": 0.5966263705980125,
    "gate_mean": [0.7925029375, 0.207497140625],
    "per_epoch_val": [0.4657800526815393, 0.48481854615132286, 0.5134964621264595,
                      0.5511490407183938, 0.5817369266195509],
    "variant": RPP.VARIANT_PINNED_SHUFFLED, "classification": "VALID_NEGATIVE",
    "alpha_final": PINNED_ALPHA,
}
PINNED_CORRECT_NEWTASK_SHA = "264aedbb9f3f605e5a8e5dd9b2aee8c1945d2dcf0388be722541d2a84f4062b8"
PINNED_CORRECT_PROMPT_SHA = "77f028993e8b1ea85210ac1b8d08f44698433264c36997bb850e10789d62e64b"
DELTA_TEST_CORRECT = 0.008103113327467715      # 学习 correct 臂 test 增量（钉死，逐位）
DELTA_VAL_CORRECT = 0.008622497456618139
DELTA_TEST_SHUFFLED_HIST = 0.0014787611000699474
HISTORICAL_U1_THRESHOLD = 0.0055               # 历史效用门（§5.6 单列保留，不适用于本实验主判定）
CONTEXT = {                                    # 上下文对照（只读数；§1.2 审计值）
    "learnable_correct": {"run_id": CORRECT_RUN_ID, "delta_test": DELTA_TEST_CORRECT,
                          "delta_val": DELTA_VAL_CORRECT, "alpha_final": PINNED_ALPHA,
                          "classification": "VALID_POSITIVE"},
    "learnable_shuffled": {"run_id": SHUF_HIST_RUN_ID, "delta_test": DELTA_TEST_SHUFFLED_HIST,
                           "delta_val": 0.0010618265892405887,
                           "alpha_final": 0.010569889098405838,
                           "classification": "MECHANISM_FAIL"},
    "pinned_correct": {"run_id": PINNED_CORRECT_RUN_ID, "delta_test": -0.002677172368479308,
                       "delta_val": -0.00104604355490312, "alpha_final": PINNED_ALPHA,
                       "classification": "VALID_NEGATIVE"},
    "pinned_shuffled": {"run_id": PINNED_SHUFFLED_RUN_ID, "delta_test": -0.000815894357038216,
                        "delta_val": 0.000802217420471707, "alpha_final": PINNED_ALPHA,
                        "classification": "VALID_NEGATIVE"},
    "budget_chain": {
        "five_epoch": {"delta_test": DELTA_TEST_CORRECT, "delta_val": DELTA_VAL_CORRECT},
        "ten_epoch": {"delta_test": 0.0074396653390377265,
                      "delta_val": 0.0048363447472773435},
        "twenty_epoch": {"delta_test": 0.004155967377889924,
                         "delta_val": 0.0023134736258212385,
                         "status": "NOT_PERSIST", "claim": True, "ablation_eligible": True},
        "budget_trend_label": "MONOTONE_NARROWING",
    },
    "alpha_pinned_verdict": {"gap": -0.001861278011441092,
                             "delta_val_gap": -0.001848260975374827,
                             "verdict": "CONDITION_ALIGNMENT_NOT_SUPPORTED",
                             "subreason": "GAP_BELOW_MATERIALITY"},
}
PRECONDITION_PRIORITY = ("IDENTITY_MISMATCH", "BASELINE_REPRODUCTION_FAILED",
                         "CROSS_EXPERIMENT_REPLAY_FAILED", "PROTOCOL_INVALID",
                         "PINNED_ALPHA_INVALID", "UNCONDITION_INVALID", "MECHANISM_FAIL",
                         "COMPARATOR_MISMATCH", "CONSTANT_VECTOR_MISMATCH")


# =====================================================================================
# §2.1 常量条件向量推导（逐字冻结；独立复核按此规格重实现）
# =====================================================================================

def draw_uncond_condition(input_size: int = EXPECTED_COND_NUMEL,
                          seed: int = UNCOND_CONST_SEED) -> torch.Tensor:
    """冻结规程：专用 torch.Generator（不触碰全局 RNG）+ 单次 randn，float32。"""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randn(int(input_size), generator=generator, dtype=torch.float32)


def cond_sha256(tensor: torch.Tensor) -> str:
    """张量摘要（dtype + shape + bytes；与核验/复核脚本同一约定）。"""
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode()
                          + t.numpy().tobytes()).hexdigest()


def cond_stats(tensor: torch.Tensor) -> dict:
    d = tensor.detach().cpu().double()
    return {"mean": float(d.mean()), "std": float(d.std(unbiased=False)),
            "min": float(d.min()), "max": float(d.max()), "norm": float(d.norm())}


# =====================================================================================
# §2.2 无条件头（唯一重写 = 条件输入 → 常量 c；α/公式/注入/委托逐字继承）
# =====================================================================================

class UnconditionalResidualPromptNewTask(RPP.PinnedAlphaResidualPromptNewTask):
    """钉死 α 无条件/全局条件头：生成器条件输入替换为常量 buffer `uncond_condition`。

    - 构造路径逐位同 `PinnedAlphaResidualPromptNewTask`（共享参数/RNG 端点/生成器初始化不变）；
    - 唯一重写 = `prompt_deltas`：`m = tanh(gen(c.unsqueeze(0).expand(B, -1)))`（批内扩维输入，
      与 C_p 常量条件路径同一构造 ⇒ §4-I8 逐位等价）；deltas 公式 `α · (||h||/√d) · m` 逐字不变；
    - `forward`/`super().forward` 委托/α 语义/范数标定零改动。
    """

    def __init__(self, *args, uncond_condition: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        dim = int(self.prompt_generator[0].in_features)
        if uncond_condition is None:
            uncond_condition = draw_uncond_condition(dim)
        self.register_buffer("uncond_condition",
                             uncond_condition.detach().clone().float())

    def prompt_deltas(self, dnn_input: torch.Tensor, reps: list[torch.Tensor]):
        """常量条件版（§2.2）：输入值不参与计算（仅用其 batch 大小做扩维）；行间逐位相同（CUDA 运行路径）。"""
        batch = int(dnn_input.shape[0])
        cond = self.uncond_condition.to(dnn_input.device).unsqueeze(0).expand(batch, -1)
        m = torch.tanh(self.prompt_generator(cond))          # 批内扩维输入（同 C_p 常量条件调用规模）
        deltas = []
        for h in reps:
            if h.shape[-1] != m.shape[-1]:
                raise ValueError(f"表征维度 {h.shape[-1]} 与 prompt 方向 {m.shape[-1]} 不一致，拒绝出数")
            scale = h.detach().norm(dim=-1, keepdim=True) / math.sqrt(m.shape[-1])
            deltas.append(self.prompt_gate * scale * m)
        return deltas, m


def build_uncond_newtask(*, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                         hidden: int = RP.PROMPT_HIDDEN, pinned_alpha: float = PINNED_ALPHA):
    return UnconditionalResidualPromptNewTask(
        input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=tower_dnn_hidden_units,
        reg_dnn=reg_dnn, device=device, hidden=hidden, pinned_alpha=pinned_alpha)


# =====================================================================================
# §5.2 UA1–UA5/UA11 运行期审计（继承 PA1/PA2 记录；新增常量向量与不变性探针）
# =====================================================================================

class UncondPromptAudit(RPP.PinnedPromptAudit):
    """PA1/PA2 记录（继承）+ 常量向量证明（重抽/隔离/摘要）+ 首 batch 样本不变性探针（UA5）。"""

    def __init__(self, *args, optimizer=None, **kwargs):
        super().__init__(*args, optimizer=optimizer, **kwargs)
        gate = self.newtask.prompt_gate
        self.pin["alpha_numel"] = int(gate.numel())
        cond = self.newtask.uncond_condition.detach().cpu()
        dim = int(cond.numel())
        torch_state = torch.get_rng_state()
        py_state = random.getstate()
        again = draw_uncond_condition(dim)
        rng_isolation = (torch.equal(torch.get_rng_state(), torch_state)
                         and random.getstate() == py_state)
        self._invariance: dict | None = None
        self.uncond = {
            "const_seed": UNCOND_CONST_SEED,
            "cond_sha256": cond_sha256(cond),
            "cond_numel": dim,
            "cond_dim": int(cond.shape[0]),
            "cond_stats": cond_stats(cond),
            "regen_bit_identical": bool(torch.equal(again, cond)),
            "rng_isolation_ok": bool(rng_isolation),
        }

    @torch.no_grad()
    def init_forward_check(self, dnn_input, gen_rep, spec_reps, env_embs) -> None:
        """super() 的零 α/激活探针 + UA5 不变性探针（真实首 train batch；幂等）。"""
        if self.init_forward is not None:
            return
        super().init_forward_check(dnn_input, gen_rep, spec_reps, env_embs)
        reps = [gen_rep, *spec_reps]
        deltas_real, m_real = self.newtask.prompt_deltas(dnn_input, reps)
        rows_ok = bool(torch.equal(m_real, m_real[0:1].expand_as(m_real)))
        rows_diff = float((m_real - m_real[0:1]).abs().max()) if m_real.shape[0] else 0.0
        _, m_zero = self.newtask.prompt_deltas(torch.zeros_like(dnn_input), reps)
        zero_ok = bool(torch.equal(m_zero, m_real))
        zero_diff = float((m_zero - m_real).abs().max()) if m_real.numel() else 0.0
        spread = 0.0
        n_zero_rows = 0
        for delta, h in zip(deltas_real, reps):
            dd = torch.atleast_2d(delta.detach().double())
            hh = torch.atleast_2d(h.detach().double())
            h_norm = hh.norm(dim=1)
            n_zero_rows += int((h_norm == 0).sum())
            ratio = dd.norm(dim=1) / h_norm.clamp_min(1e-12)
            nonzero = h_norm > 0
            if bool(nonzero.any()):
                spread = max(spread, float(ratio[nonzero].max() - ratio[nonzero].min()))
        self._invariance = {
            "direction_bit_identical_across_batch_rows": rows_ok,
            "direction_max_abs_diff_across_rows": rows_diff,
            "direction_bit_identical_under_zero_input": zero_ok,
            "direction_max_abs_diff_zero_input": zero_diff,
            "ratio_rows_max_minus_min": spread,
            "ratio_rows_nonzero_only": True,   # 零范数行（||h||=0 ⇒ δ=0 ⇒ ratio 记 0）不入 spread（§12 C1b-4）
            "n_rows": int(m_real.shape[0]),
            "n_zero_rep_rows": n_zero_rows,
        }

    def result(self, alpha_final: float) -> dict:
        base = super().result(alpha_final)
        base["uncond"] = dict(self.uncond)
        base["uncond"]["invariance"] = dict(self._invariance) if self._invariance else None
        return base


# =====================================================================================
# §5.2 UA1–UA11 门禁（判据先于结果写死，不得事后修改）
# =====================================================================================

def uncond_mechanism_gates(*, probe: dict, val_stats: dict, params: dict, reference: dict,
                           expected_reference_auc: float = RP.BASELINE_AUC_VAL,
                           expected_reference_pred_std: float = RP.REFERENCE_PRED_STD) -> dict:
    """M0 + UA1–UA11（预注册 §5.2）。

    - M0：参照头身份（in-run 复算 val AUC/pred_std == 钉死值）。
    - UA1：α 钉死身份（buffer/非参数/requires_grad False/不在 named_parameters；构造与训练后 == 钉死值）。
    - UA2：优化器排除（不含 α 且覆盖全部参数；逐 epoch 探针无 α 梯度且 α 恒为钉死值）。
    - UA3：头身份（共享参数逐位/RNG 端点/新增 state_dict 键恰 6：4 参数 + α buffer + c buffer）。
    - UA4：注入纯度（零 α 逐位恒等 + 恢复逐位）+ 激活（钉死 α 下与参照头有差）。
    - UA5：样本不变性（真实首 batch：行间逐位 + 零输入逐位 + c 重抽逐位 + 逐行 ratio 恒定）。
    - UA6：范数界（S1：ratio_max ≤ |α| + 1e-6）。
    - UA7：跨流一致（S2：三流 ratio_mean spread ≤ 1e-6）。
    - UA8：逐样本有效门控恒定（真无条件性实测：geff_std/ratio_std ≤ 1e-6 ∧ geff_min > 0）。
    - UA9：无坍缩（pred_std ≥ 0.5 × ref_pred_std）。
    - UA10：预算/核算（4 可训练键 == 2384/8129；α numel 1；c numel 80；新增键 == 6）。
    - UA11：梯度有效（逐 epoch 生成器梯度记录齐全、全非零有限）。
    """
    construction = probe.get("construction") or {}
    init_forward = probe.get("init_forward") or {}
    records = probe.get("grad_probe") or []
    pin = probe.get("pin") or {}
    uncond = probe.get("uncond") or {}
    invariance = uncond.get("invariance") or {}
    alpha_final = float(probe.get("alpha_final") or 0.0)
    streams = val_stats.get("streams") or {}
    gate = val_stats.get("gate") or {}
    dispersion = val_stats.get("dispersion") or {}

    reference = reference or {}
    reference_dispersion = reference.get("pred_dispersion") or {}
    ref_auc = reference.get("val_auc")
    ref_std = reference_dispersion.get("pred_std")
    m0_pass = bool(ref_auc is not None and ref_std is not None
                   and abs(float(ref_auc) - expected_reference_auc) <= RP.REFERENCE_IDENTITY_TOL
                   and abs(float(ref_std) - expected_reference_pred_std) <= RP.REFERENCE_IDENTITY_TOL)

    ua1_pass = bool(pin.get("is_parameter") is False
                    and pin.get("requires_grad") is False
                    and pin.get("in_named_parameters") is False
                    and pin.get("alpha_at_construction") == PINNED_ALPHA
                    and alpha_final == PINNED_ALPHA
                    and pin.get("alpha_after_training") == PINNED_ALPHA
                    and pin.get("pin_unchanged_after_training") is True)
    ua2_pass = bool(pin.get("optimizer_excludes_prompt_gate") is True
                    and pin.get("optimizer_covers_named_parameters") is True
                    and records
                    and all(r.get("alpha_grad_norm") is None for r in records)
                    and all(float(r.get("alpha") or 0.0) == PINNED_ALPHA for r in records))
    ua3_pass = bool(construction.get("shared_params_bit_identical")
                    and construction.get("global_rng_endpoint_identical")
                    and construction.get("extra_keys") == sorted(EXPECTED_STATE_EXTRA_KEYS))
    ua4_pass = bool(init_forward.get("zero_alpha_bit_identical")
                    and init_forward.get("alpha_restored_exact")
                    and float(init_forward.get("zero_alpha_max_abs_diff") or 0.0) == 0.0
                    and init_forward.get("pinned_differs_from_reference"))
    spread_val = invariance.get("ratio_rows_max_minus_min")
    ua5_pass = bool(invariance.get("direction_bit_identical_across_batch_rows") is True
                    and invariance.get("direction_bit_identical_under_zero_input") is True
                    and uncond.get("regen_bit_identical") is True
                    and spread_val is not None and float(spread_val) <= UNCOND_STD_TOL)
    ratio_max = [float(r) for r in (streams.get("ratio_max") or [1e9])]
    ua6_pass = all(r <= abs(PINNED_ALPHA) + PIN_BOUND_TOL for r in ratio_max)
    ratio_mean = [float(r) for r in (streams.get("ratio_mean") or [])]
    spread = (max(ratio_mean) - min(ratio_mean)) if ratio_mean else float("inf")
    ua7_pass = bool(ratio_mean) and spread <= RATIO_SPREAD_TOL
    ratio_std = [float(r) for r in (streams.get("ratio_std") or [1e9])]
    ua8_pass = bool(float(gate.get("geff_std") if gate.get("geff_std") is not None else 1e9)
                    <= UNCOND_STD_TOL
                    and float(gate.get("geff_min") or 0.0) > 0.0
                    and all(r <= UNCOND_STD_TOL for r in ratio_std))
    pred_std = float(dispersion.get("pred_std") or 0.0)
    ua9_pass = bool(pred_std > 0.0 and ref_std is not None and float(ref_std) > 0.0
                    and pred_std >= PRED_STD_MIN_RATIO * float(ref_std))
    names = sorted(str(item["name"]) for item in (params.get("new_param_list") or []))
    ua10_pass = bool(names == sorted(EXPECTED_TRAINABLE_PROMPT_KEYS)
                     and int(params.get("new_params_total", -1)) == EXPECTED_NEW_PARAMS_TOTAL
                     and int(params.get("head_params", -1)) == EXPECTED_HEAD_PARAMS
                     and pin.get("alpha_numel") == EXPECTED_ALPHA_NUMEL
                     and uncond.get("cond_numel") == EXPECTED_COND_NUMEL
                     and (construction.get("extra_keys") == sorted(EXPECTED_STATE_EXTRA_KEYS)))
    ua11_pass = bool(records and all(
        r.get("generator_grad_norm") is not None
        and math.isfinite(float(r["generator_grad_norm"]))
        and float(r["generator_grad_norm"]) > 0.0 for r in records))

    return {
        "M0": {"pass": m0_pass,
               "rule": f"|ref_val_auc - {expected_reference_auc}| <= {RP.REFERENCE_IDENTITY_TOL} AND "
                       f"|ref_pred_std - {expected_reference_pred_std}| <= {RP.REFERENCE_IDENTITY_TOL}",
               "observed": {"ref_val_auc": ref_auc, "ref_pred_std": ref_std}},
        "UA1": {"pass": ua1_pass,
                "rule": "prompt_gate 非参数 buffer ∧ requires_grad False ∧ 不在 named_parameters ∧ "
                        "构造/训练后 == PINNED_ALPHA（精确）",
                "observed": {"is_parameter": pin.get("is_parameter"),
                             "requires_grad": pin.get("requires_grad"),
                             "in_named_parameters": pin.get("in_named_parameters"),
                             "alpha_at_construction": pin.get("alpha_at_construction"),
                             "alpha_after_training": pin.get("alpha_after_training"),
                             "alpha_final": alpha_final, "pinned_alpha": PINNED_ALPHA}},
        "UA2": {"pass": ua2_pass,
                "rule": "optimizer 不含 α ∧ 覆盖全部 named_parameters ∧ 逐 epoch 探针 alpha_grad_norm None ∧ "
                        "α 恒为 PINNED_ALPHA",
                "observed": {"optimizer_excludes_prompt_gate": pin.get("optimizer_excludes_prompt_gate"),
                             "optimizer_covers_named_parameters": pin.get("optimizer_covers_named_parameters"),
                             "optimizer_params_total": pin.get("optimizer_params_total"),
                             "probe_alpha_grad_norms": [r.get("alpha_grad_norm") for r in records]}},
        "UA3": {"pass": ua3_pass,
                "rule": "shared bit-identical AND rng endpoint identical AND extra state keys exact "
                        "(4 generator params + prompt_gate buffer + uncond_condition buffer)",
                "observed": {"shared_params_bit_identical": construction.get("shared_params_bit_identical"),
                             "global_rng_endpoint_identical": construction.get("global_rng_endpoint_identical"),
                             "extra_keys": construction.get("extra_keys")}},
        "UA4": {"pass": ua4_pass,
                "rule": "zero-alpha forward bit-identical to reference ∧ alpha restored exact ∧ "
                        "pinned-alpha forward differs from reference (active)",
                "observed": {"zero_alpha_bit_identical": init_forward.get("zero_alpha_bit_identical"),
                             "zero_alpha_max_abs_diff": init_forward.get("zero_alpha_max_abs_diff"),
                             "alpha_restored_exact": init_forward.get("alpha_restored_exact"),
                             "pinned_max_abs_diff": init_forward.get("pinned_max_abs_diff"),
                             "pinned_differs_from_reference": init_forward.get("pinned_differs_from_reference")}},
        "UA5": {"pass": ua5_pass,
                "rule": "direction rows bit-identical ∧ zero-input direction bit-identical ∧ "
                        "cond regen bit-identical ∧ per-row ratio spread <= 1e-6（预注册 §5.2；§12 C1b-2/3）",
                "observed": dict(invariance, regen_bit_identical=uncond.get("regen_bit_identical"))},
        "UA6": {"pass": ua6_pass, "rule": f"every stream ratio_max <= |PINNED_ALPHA| + {PIN_BOUND_TOL}",
                "observed": {"ratio_max": ratio_max, "alpha_abs": abs(PINNED_ALPHA)}},
        "UA7": {"pass": ua7_pass, "rule": f"three-stream ratio_mean spread <= {RATIO_SPREAD_TOL}",
                "observed": {"ratio_mean": ratio_mean, "spread": spread}},
        "UA8": {"pass": ua8_pass,
                "rule": f"geff_std <= {UNCOND_STD_TOL} AND geff_min > 0 AND every stream ratio_std <= {UNCOND_STD_TOL}",
                "observed": {"geff_std": gate.get("geff_std"), "geff_min": gate.get("geff_min"),
                             "geff_max": gate.get("geff_max"), "ratio_std": ratio_std}},
        "UA9": {"pass": ua9_pass, "rule": f"pred_std > 0 AND pred_std >= {PRED_STD_MIN_RATIO} * ref_pred_std",
                "observed": {"pred_std": pred_std, "ref_pred_std": ref_std}},
        "UA10": {"pass": ua10_pass,
                 "rule": f"4 exact trainable prompt keys AND new_params_total == {EXPECTED_NEW_PARAMS_TOTAL} "
                         f"AND head_params == {EXPECTED_HEAD_PARAMS} AND alpha numel == 1 AND cond numel == 80 "
                         f"AND state extra keys exact",
                 "observed": {"names": names, "new_params_total": params.get("new_params_total"),
                              "head_params": params.get("head_params"),
                              "alpha_numel": pin.get("alpha_numel"),
                              "cond_numel": uncond.get("cond_numel"),
                              "state_extra_keys": sorted(EXPECTED_STATE_EXTRA_KEYS)}},
        "UA11": {"pass": ua11_pass,
                 "rule": "per-epoch generator grad records present, all finite and > 0",
                 "observed": {"generator_grad_norms": [r.get("generator_grad_norm") for r in records]}},
    }


def uncond_arm_verdict(*, auc_test, auc_val, probe, val_stats, params, reference,
                       protocol_ok=True, variant=VARIANT_UNCOND) -> dict:
    """UA1–UA11 + U1/U2 判定与结局分类（预注册 §5.2 分类映射；判据先于结果写死）。"""
    gates = uncond_mechanism_gates(probe=probe, val_stats=val_stats, params=params,
                                   reference=reference)
    delta_test = float(auc_test) - RP.BASELINE_AUC_TEST
    delta_val = float(auc_val) - RP.BASELINE_AUC_VAL
    u1_pass = bool(delta_test >= RP.AUC_TEST_DELTA_MIN)
    u2_pass = bool(delta_val > RP.AUC_VAL_DIRECTION_MIN)
    u_pass = bool(u1_pass and u2_pass)

    if not gates["M0"]["pass"]:
        classification, subreason = "MECHANISM_FAIL", "REFERENCE_IDENTITY"
    elif not (gates["UA1"]["pass"] and gates["UA2"]["pass"] and gates["UA10"]["pass"]):
        classification, subreason = "MECHANISM_FAIL", "UNCONDITION_PIN_INVALID"
    elif not (gates["UA3"]["pass"] and gates["UA4"]["pass"]):
        classification, subreason = "MECHANISM_FAIL", "INVALID_IMPLEMENTATION"
    elif not gates["UA5"]["pass"]:
        classification, subreason = "MECHANISM_FAIL", "NOT_TRULY_UNCONDITIONAL"
    elif not gates["UA8"]["pass"]:
        classification, subreason = "MECHANISM_FAIL", "UNCONDITIONAL_GATE_NOT_CONSTANT"
    elif not gates["UA11"]["pass"]:
        classification, subreason = "MECHANISM_FAIL", "GRADIENT_INVALID"
    elif not gates["UA6"]["pass"]:
        classification, subreason = "MECHANISM_FAIL", "NORM_BOUND_VIOLATION"
    elif not gates["UA7"]["pass"]:
        classification, subreason = "MECHANISM_FAIL", "CROSS_STREAM_INCONSISTENT"
    elif not gates["UA9"]["pass"]:
        classification, subreason = "MECHANISM_FAIL", "PREDICTION_COLLAPSE"
    elif not protocol_ok:
        classification, subreason = "MECHANISM_FAIL", "PROTOCOL_INVALID"
    elif u_pass:
        classification, subreason = "VALID_POSITIVE", None
    else:
        classification, subreason = "VALID_NEGATIVE", None

    arm = dict(gates)
    arm["U"] = {"pass": u_pass,
                "U1": {"pass": u1_pass, "rule": f"delta_test >= +{RP.AUC_TEST_DELTA_MIN}",
                       "observed": {"auc_test": float(auc_test), "delta_test": delta_test,
                                    "baseline_auc_test": RP.BASELINE_AUC_TEST}},
                "U2": {"pass": u2_pass, "rule": "delta_val > 0",
                       "observed": {"auc_val": float(auc_val), "delta_val": delta_val,
                                    "baseline_auc_val": RP.BASELINE_AUC_VAL}}}
    arm["protocol_10_1"] = {"delta_test_ge_0.005": bool(delta_test >= 0.005), "val_positive": u2_pass}
    arm["classification"] = classification
    arm["subreason"] = subreason
    arm["pass"] = classification == "VALID_POSITIVE"
    arm["variant"] = variant
    return arm


# =====================================================================================
# §5.4/§5.5/§5.6 判定（纯函数 + 只读分析器）
# =====================================================================================

def causal_verdict(*, gap_u: float, delta_val_gap_u: float, g_u: float,
                   preconditions: dict | None = None) -> dict:
    """预注册 §5.4 因果判定树（先于结果写死）；preconditions 按插入顺序为首个失败项给出 subreason。"""
    gap_u, delta_val_gap_u, g_u = float(gap_u), float(delta_val_gap_u), float(g_u)
    components = {
        "gap_u": gap_u, "delta_val_gap_u": delta_val_gap_u,
        "gap_material_positive": gap_u >= MATERIAL_DELTA,
        "validation_agreement": delta_val_gap_u > 0.0,
        "unconditional_matches_or_exceeds_conditioned": gap_u <= 0.0,
    }
    for name, failed in (preconditions or {}).items():
        if failed:
            return {"verdict": "INVALID", "subreason": name, "components": components}
    if gap_u >= MATERIAL_DELTA and delta_val_gap_u > 0.0:
        subreason = ("CONDITIONING_ESSENTIAL" if g_u < SECONDARY_POSITIVE_MIN
                     else "PARTIAL_CONDITIONING_CONTRIBUTION")
        return {"verdict": "SAMPLE_CONDITIONING_SUPPORTED", "subreason": subreason,
                "components": components}
    return {"verdict": "SAMPLE_CONDITIONING_NOT_SUPPORTED",
            "subreason": ("GAP_BELOW_MATERIALITY" if gap_u < MATERIAL_DELTA
                          else "VALIDATION_DISAGREEMENT"),
            "components": components}


def secondary_classification(delta_test: float) -> str:
    """预注册 §5.5 二级分类（边界先于结果写死；±闭端按预注册）。"""
    delta_test = float(delta_test)
    if delta_test >= SECONDARY_POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if delta_test <= SECONDARY_NEGATIVE_MAX:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def headroom_assessment(*, delta_test: float, delta_val: float, arm_record: dict,
                        prompt_doc: dict) -> dict:
    """预注册 §5.5：NO_CLEAR_IMPROVEMENT 的机械 headroom（仅当落入该区间时适用）。

    活性口径（U 臂适配，a priori）：|α| > 0 ∧ 三流 ratio_mean > 0 ∧ geff_max > 0
    （`geff_std ≈ 0` 是真无条件性证明 UA8，不构成失活）。
    """
    if secondary_classification(delta_test) != "NO_CLEAR_IMPROVEMENT":
        return {"applicable": False}
    gate = prompt_doc.get("gate") or {}
    streams = prompt_doc.get("val_stats") or {}
    ratio_mean = [float(r) for r in (streams.get("ratio_mean") or [])]
    active = (abs(float(prompt_doc.get("alpha_final") or 0.0)) > 0.0
              and bool(ratio_mean) and all(r > 0.0 for r in ratio_mean)
              and float(gate.get("geff_max") or 0.0) > 0.0)
    epochs_run = len(arm_record.get("per_epoch") or [])
    right_censored = (arm_record.get("best_epoch") == arm_record.get("epochs") == epochs_run)
    return {
        "applicable": True,
        "mechanism_activity": "PIN_ACTIVE" if active else "INACTIVE",
        "validation_direction": "POSITIVE" if float(delta_val) > 0.0 else "NON_POSITIVE",
        "epoch_trajectory": "RIGHT_CENSORED_STILL_IMPROVING" if right_censored else "EARLY_STOPPED",
        "gap_to_positive_threshold": SECONDARY_POSITIVE_MIN - float(delta_test),
        "retained_ratio_vs_historical_correct": (float(delta_test) / DELTA_TEST_CORRECT)
        if DELTA_TEST_CORRECT > 0 else None,
    }


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _a_class_ok(gate_doc: dict) -> bool:
    gates = gate_doc.get("gates") or {}
    return all(gates.get(g, {}).get("verdict") in ("PASS", "SKIP")
               for g in ("A1", "A2", "A3", "A4", "A5", "A6"))


UNCOND_GATE_NAMES = ("M0",) + tuple(f"UA{i}" for i in range(1, 12))
PINNED_GATE_NAMES = ("M0",) + tuple(f"PA{i}" for i in range(1, 9))


def _rule_reference(o: dict) -> bool:
    return bool(o.get("ref_val_auc") is not None and o.get("ref_pred_std") is not None
                and abs(float(o["ref_val_auc"]) - RP.BASELINE_AUC_VAL) <= RP.REFERENCE_IDENTITY_TOL
                and abs(float(o["ref_pred_std"]) - RP.REFERENCE_PRED_STD)
                <= RP.REFERENCE_IDENTITY_TOL)


def _rule_alpha_identity(o: dict) -> bool:
    return bool(o.get("is_parameter") is False and o.get("requires_grad") is False
                and o.get("in_named_parameters") is False
                and o.get("alpha_at_construction") == PINNED_ALPHA
                and o.get("alpha_after_training") == PINNED_ALPHA
                and o.get("alpha_final") == PINNED_ALPHA
                and o.get("pinned_alpha") == PINNED_ALPHA)


def _rule_optimizer_excluded(o: dict) -> bool:
    return bool(o.get("optimizer_excludes_prompt_gate") is True
                and o.get("optimizer_covers_named_parameters") is True
                and (o.get("probe_alpha_grad_norms") or []) != []
                and all(g is None for g in (o.get("probe_alpha_grad_norms") or [])))


def _rule_purity_activation(o: dict) -> bool:
    return bool(o.get("zero_alpha_bit_identical") and o.get("alpha_restored_exact")
                and float(o.get("zero_alpha_max_abs_diff") or 0.0) == 0.0
                and o.get("pinned_differs_from_reference"))


def _rule_norm_bound(o: dict) -> bool:
    return all(float(r) <= abs(PINNED_ALPHA) + PIN_BOUND_TOL for r in (o.get("ratio_max") or [1e9]))


def _rule_stream_spread(o: dict) -> bool:
    means = [float(r) for r in (o.get("ratio_mean") or [])]
    return bool(means) and (max(means) - min(means)) <= RATIO_SPREAD_TOL


def _rule_no_collapse(o: dict) -> bool:
    return bool(float(o.get("pred_std") or 0.0) > 0.0 and o.get("ref_pred_std") is not None
                and float(o["ref_pred_std"]) > 0.0
                and float(o["pred_std"]) >= PRED_STD_MIN_RATIO * float(o["ref_pred_std"]))


_PINNED_GATE_RULES = {
    "M0": _rule_reference, "PA1": _rule_alpha_identity, "PA2": _rule_optimizer_excluded,
    "PA3": lambda o: bool(o.get("shared_params_bit_identical")
                          and o.get("global_rng_endpoint_identical")
                          and o.get("extra_keys") == sorted(RPP.EXPECTED_STATE_EXTRA_KEYS)),
    "PA4": _rule_purity_activation, "PA5": _rule_norm_bound, "PA6": _rule_stream_spread,
    "PA7": _rule_no_collapse,
    "PA8": lambda o: bool(sorted(str(n) for n in (o.get("names") or []))
                          == sorted(EXPECTED_TRAINABLE_PROMPT_KEYS)
                          and int(o.get("new_params_total", -1)) == EXPECTED_NEW_PARAMS_TOTAL
                          and int(o.get("head_params", -1)) == EXPECTED_HEAD_PARAMS),
}


def _rederive_pinned_gate_pass(arm: dict, gate: str) -> bool:
    """由记录 `observed` 值机械重推 PA 门禁（不信任记录布尔；判据 == `pinned_mechanism_gates`）。"""
    o = ((arm.get(gate) or {}).get("observed")) or {}
    rule = _PINNED_GATE_RULES.get(gate)
    if rule is None:
        raise AssertionError(f"未知门禁 {gate}")
    return rule(o)


def _rederive_uncond_gate_pass(arm: dict, gate: str) -> bool:
    """由记录 `observed` 值机械重推 UA 门禁（不信任记录布尔；判据 == `uncond_mechanism_gates`）。"""
    o = ((arm.get(gate) or {}).get("observed")) or {}
    rules = {
        "M0": _rule_reference, "UA1": _rule_alpha_identity, "UA2": _rule_optimizer_excluded,
        "UA3": lambda o: bool(o.get("shared_params_bit_identical")
                              and o.get("global_rng_endpoint_identical")
                              and o.get("extra_keys") == sorted(EXPECTED_STATE_EXTRA_KEYS)),
        "UA4": _rule_purity_activation, "UA6": _rule_norm_bound, "UA7": _rule_stream_spread,
        "UA9": _rule_no_collapse,
    }
    if gate in rules:
        return rules[gate](o)
    if gate == "UA5":
        spread = o.get("ratio_rows_max_minus_min")
        return bool(o.get("direction_bit_identical_across_batch_rows") is True
                    and o.get("direction_bit_identical_under_zero_input") is True
                    and o.get("regen_bit_identical") is True
                    and spread is not None and float(spread) <= UNCOND_STD_TOL)
    if gate == "UA8":
        return bool(float(o.get("geff_std") if o.get("geff_std") is not None else 1e9)
                    <= UNCOND_STD_TOL
                    and float(o.get("geff_min") or 0.0) > 0.0
                    and all(float(r) <= UNCOND_STD_TOL for r in (o.get("ratio_std") or [1e9])))
    if gate == "UA10":
        return bool(sorted(str(n) for n in (o.get("names") or []))
                    == sorted(EXPECTED_TRAINABLE_PROMPT_KEYS)
                    and int(o.get("new_params_total", -1)) == EXPECTED_NEW_PARAMS_TOTAL
                    and int(o.get("head_params", -1)) == EXPECTED_HEAD_PARAMS
                    and o.get("alpha_numel") == EXPECTED_ALPHA_NUMEL
                    and o.get("cond_numel") == EXPECTED_COND_NUMEL
                    and o.get("state_extra_keys") == sorted(EXPECTED_STATE_EXTRA_KEYS))
    if gate == "UA11":
        norms = o.get("generator_grad_norms") or []
        return bool(norms) and all(g is not None and math.isfinite(float(g)) and float(g) > 0.0
                                   for g in norms)
    raise AssertionError(f"未知门禁 {gate}")


def analyze_runs(*, root, baseline_run, arm_cond_run, arm_uncond_run) -> dict:
    """只读 B/C_p/U + 五个历史对照 run 记录 → 预注册 §5 全量判定（不改任何 run 产物）。"""
    root = Path(root)
    base = _read_json(root / "runs" / baseline_run / "metrics.json")
    base_gate = _read_json(root / "runs" / baseline_run / "gate_report.json")
    cp = _read_json(root / "runs" / arm_cond_run / "metrics.json")
    cp_gate = _read_json(root / "runs" / arm_cond_run / "gate_report.json")
    cp_prompt = _read_json(root / "runs" / arm_cond_run / "prompt_report.json")
    up = _read_json(root / "runs" / arm_uncond_run / "metrics.json")
    up_gate = _read_json(root / "runs" / arm_uncond_run / "gate_report.json")
    up_prompt = _read_json(root / "runs" / arm_uncond_run / "prompt_report.json")
    ctx_base = _read_json(root / "runs" / BASELINE_RUN_ID / "metrics.json")
    ctx_corr = _read_json(root / "runs" / CORRECT_RUN_ID / "metrics.json")
    ctx_corr_prompt = _read_json(root / "runs" / CORRECT_RUN_ID / "prompt_report.json")
    ctx_shuf = _read_json(root / "runs" / SHUF_HIST_RUN_ID / "metrics.json")
    ctx_pc = _read_json(root / "runs" / PINNED_CORRECT_RUN_ID / "metrics.json")
    ctx_ps = _read_json(root / "runs" / PINNED_SHUFFLED_RUN_ID / "metrics.json")

    # ---- §5.1-ID：identity ----
    expected_ref_path = root / "runs" / BASELINE_RUN_ID / "newtask.pt"
    cp_ref_recorded = (cp_prompt.get("reference_dispersion") or {}).get("newtask_checkpoint")
    up_ref_recorded = (up_prompt.get("reference_dispersion") or {}).get("newtask_checkpoint")
    ref_sha = _sha256_file(expected_ref_path) if expected_ref_path.is_file() else None
    runs = (base, cp, up)
    identity = {
        "stage1_same": base["stage1_id"] == cp["stage1_id"] == up["stage1_id"],
        "stage1_pinned": base["stage1_id"] == STAGE1_ID,
        "seed_same_pinned": base["model_seed"] == cp["model_seed"] == up["model_seed"]
        == MODEL_SEED_PINNED,
        "epochs": base["epochs"] == cp["epochs"] == up["epochs"] == 5,
        "patience": base["patience"] == cp["patience"] == up["patience"] == 2,
        "tag": base["tag"] == cp["tag"] == up["tag"] == "short",
        "dirty_false": all(not r.get("git", {}).get("dirty") for r in runs),
        "baseline_variant": base.get("variant") == RP.BASELINE_VARIANT,
        "baseline_no_suffix": not base["run_id"].endswith(
            (RP.RUN_ID_SUFFIX, RPP.RUN_ID_SUFFIX_CORRECT, RPP.RUN_ID_SUFFIX_SHUFFLED,
             RUN_ID_SUFFIX_UNCOND)),
        "arm_cond_variant": cp.get("variant") == RPP.VARIANT_PINNED,
        "arm_cond_suffix": cp["run_id"].endswith(RPP.RUN_ID_SUFFIX_CORRECT),
        "arm_uncond_variant": up.get("variant") == VARIANT_UNCOND,
        "arm_uncond_suffix": up["run_id"].endswith(RUN_ID_SUFFIX_UNCOND),
        "arms_ref_path_pinned": (cp_ref_recorded is not None and up_ref_recorded is not None
                                 and Path(str(cp_ref_recorded)) == expected_ref_path
                                 and Path(str(up_ref_recorded)) == expected_ref_path),
        "arms_ref_sha": ref_sha == REF_HEAD_SHA,
        "stage1_shas": all(r.get("backbone_sha256_loaded") == STAGE1_BACKBONE_SHA
                           and r.get("env_ids_sha256") == STAGE1_ENV_IDS_SHA
                           and r.get("fingerprint_sha256") == STAGE1_FINGERPRINT_SHA
                           for r in runs),
    }

    # ---- §5.1-REP_B：B 与历史基线记录逐位一致 ----
    base_pe = [e["val_auc_bsi"] for e in base["per_epoch"]]
    baseline_newtask = root / "runs" / baseline_run / "newtask.pt"
    reproduction_b = {
        "per_epoch_val": base_pe == BASELINE_RECORD["per_epoch_val"],
        "best_epoch": base["best_epoch"] == BASELINE_RECORD["best_epoch"],
        "best_val": base["best_val_auc_bsi"] == BASELINE_RECORD["best_val_auc_bsi"],
        "test": base["test_auc_bsi"] == BASELINE_RECORD["test_auc_bsi"],
        "gate_mean": base["gate_mean"] == BASELINE_RECORD["gate_mean"],
        "newtask_sha": baseline_newtask.is_file() and _sha256_file(baseline_newtask) == REF_HEAD_SHA,
    }

    # ---- §5.1-REP_Cp：C_p 与 f08ae6e F_c 记录逐位一致（跨实验重放）----
    cp_pe = [e["val_auc_bsi"] for e in cp["per_epoch"]]
    cp_newtask = root / "runs" / arm_cond_run / "newtask.pt"
    replay_cp = {
        "per_epoch_val": cp_pe == PINNED_CORRECT_RECORD["per_epoch_val"],
        "best_epoch": cp["best_epoch"] == PINNED_CORRECT_RECORD["best_epoch"],
        "best_val": cp["best_val_auc_bsi"] == PINNED_CORRECT_RECORD["best_val_auc_bsi"],
        "test": cp["test_auc_bsi"] == PINNED_CORRECT_RECORD["test_auc_bsi"],
        "gate_mean": cp["gate_mean"] == PINNED_CORRECT_RECORD["gate_mean"],
        "newtask_sha": cp_newtask.is_file() and _sha256_file(cp_newtask) == PINNED_CORRECT_NEWTASK_SHA,
        "alpha_final": cp_prompt.get("alpha_final") == PINNED_ALPHA,
    }

    # ---- §5.1-A ----
    protocol_a = {"baseline": _a_class_ok(base_gate), "arm_cond": _a_class_ok(cp_gate),
                  "arm_uncond": _a_class_ok(up_gate)}

    # ---- §5.1-UA：U 的 UA1–UA11 与 C_p 的 PA1–PA8（由记录 observed 值机械重推，不信任记录布尔）----
    cp_arm = cp.get("rp_arm") or {}
    up_arm = up.get("rp_arm") or {}
    mechanism = {
        "arm_cond": {g: _rederive_pinned_gate_pass(cp_arm, g) for g in PINNED_GATE_NAMES},
        "arm_uncond": {g: _rederive_uncond_gate_pass(up_arm, g) for g in UNCOND_GATE_NAMES},
        "recorded_pass_consistent": {
            "arm_cond": all(bool((cp_arm.get(g) or {}).get("pass"))
                            == _rederive_pinned_gate_pass(cp_arm, g) for g in PINNED_GATE_NAMES),
            "arm_uncond": all(bool((up_arm.get(g) or {}).get("pass"))
                              == _rederive_uncond_gate_pass(up_arm, g)
                              for g in UNCOND_GATE_NAMES),
        },
    }

    # ---- §5.1-CC：历史对照链 + α 钉死来源 ----
    comparator = {
        "baseline_values": (ctx_base["run_id"] == BASELINE_RUN_ID
                            and ctx_base["test_auc_bsi"] == BASELINE_RECORD["test_auc_bsi"]
                            and ctx_base["best_val_auc_bsi"] == BASELINE_RECORD["best_val_auc_bsi"]
                            and [e["val_auc_bsi"] for e in ctx_base["per_epoch"]]
                            == BASELINE_RECORD["per_epoch_val"]
                            and ctx_base["gate_mean"] == BASELINE_RECORD["gate_mean"]),
        "correct_values": (ctx_corr["run_id"] == CORRECT_RUN_ID
                           and ctx_corr["test_auc_bsi"] == CORRECT_RECORD["test_auc_bsi"]
                           and ctx_corr["best_val_auc_bsi"] == CORRECT_RECORD["best_val_auc_bsi"]
                           and [e["val_auc_bsi"] for e in ctx_corr["per_epoch"]]
                           == CORRECT_RECORD["per_epoch_val"]
                           and ctx_corr["gate_mean"] == CORRECT_RECORD["gate_mean"]
                           and ctx_corr.get("variant") == CORRECT_RECORD["variant"]
                           and (ctx_corr.get("rp_arm") or {}).get("classification")
                           == CORRECT_RECORD["classification"]),
        "shuf_hist_values": (ctx_shuf["run_id"] == SHUF_HIST_RUN_ID
                             and ctx_shuf["test_auc_bsi"] == SHUF_HIST_RECORD["test_auc_bsi"]
                             and ctx_shuf["best_val_auc_bsi"] == SHUF_HIST_RECORD["best_val_auc_bsi"]
                             and [e["val_auc_bsi"] for e in ctx_shuf["per_epoch"]]
                             == SHUF_HIST_RECORD["per_epoch_val"]
                             and ctx_shuf.get("variant") == SHUF_HIST_RECORD["variant"]
                             and (ctx_shuf.get("rp_arm") or {}).get("classification")
                             == SHUF_HIST_RECORD["classification"]),
        "pinned_correct_values": (ctx_pc["run_id"] == PINNED_CORRECT_RUN_ID
                                  and ctx_pc["test_auc_bsi"] == PINNED_CORRECT_RECORD["test_auc_bsi"]
                                  and ctx_pc["best_val_auc_bsi"] == PINNED_CORRECT_RECORD["best_val_auc_bsi"]
                                  and [e["val_auc_bsi"] for e in ctx_pc["per_epoch"]]
                                  == PINNED_CORRECT_RECORD["per_epoch_val"]
                                  and ctx_pc["gate_mean"] == PINNED_CORRECT_RECORD["gate_mean"]
                                  and ctx_pc.get("variant") == PINNED_CORRECT_RECORD["variant"]
                                  and (ctx_pc.get("rp_arm") or {}).get("classification")
                                  == PINNED_CORRECT_RECORD["classification"]),
        "pinned_shuffled_values": (ctx_ps["run_id"] == PINNED_SHUFFLED_RUN_ID
                                   and ctx_ps["test_auc_bsi"] == PINNED_SHUFFLED_RECORD["test_auc_bsi"]
                                   and ctx_ps["best_val_auc_bsi"] == PINNED_SHUFFLED_RECORD["best_val_auc_bsi"]
                                   and [e["val_auc_bsi"] for e in ctx_ps["per_epoch"]]
                                   == PINNED_SHUFFLED_RECORD["per_epoch_val"]
                                   and ctx_ps["gate_mean"] == PINNED_SHUFFLED_RECORD["gate_mean"]
                                   and ctx_ps.get("variant") == PINNED_SHUFFLED_RECORD["variant"]
                                   and (ctx_ps.get("rp_arm") or {}).get("classification")
                                   == PINNED_SHUFFLED_RECORD["classification"]),
        "pin_source_alpha": ctx_corr_prompt.get("alpha_final") == PINNED_ALPHA,
        "arms_alpha_equal_pin": (cp_prompt.get("alpha_final") == PINNED_ALPHA
                                 and up_prompt.get("alpha_final") == PINNED_ALPHA),
        "correct_deltas": (ctx_corr["test_auc_bsi"] - ctx_base["test_auc_bsi"]
                           == CORRECT_RECORD["test_auc_bsi"] - BASELINE_RECORD["test_auc_bsi"]
                           and ctx_corr["best_val_auc_bsi"] - ctx_base["best_val_auc_bsi"]
                           == CORRECT_RECORD["best_val_auc_bsi"] - BASELINE_RECORD["best_val_auc_bsi"]),
        "shuf_hist_delta": (ctx_shuf["test_auc_bsi"] - ctx_base["test_auc_bsi"]
                            == SHUF_HIST_RECORD["test_auc_bsi"] - BASELINE_RECORD["test_auc_bsi"]),
        "replayed_g_c": (cp["test_auc_bsi"] - base["test_auc_bsi"]
                         == PINNED_CORRECT_RECORD["test_auc_bsi"] - BASELINE_RECORD["test_auc_bsi"]),
    }

    # ---- §5.1-CV：常量向量完整性（独立重抽）----
    up_uncond = up_prompt.get("uncond") or {}
    regen = draw_uncond_condition(EXPECTED_COND_NUMEL)
    constant_vector = {
        "cond_sha_recorded": up_uncond.get("cond_sha256") == COND_VECTOR_SHA256,
        "cond_regen_bit_identical": up_uncond.get("regen_bit_identical") is True,
        "cond_shape_numel": (up_uncond.get("cond_numel") == EXPECTED_COND_NUMEL
                             and up_uncond.get("cond_dim") == COND_VECTOR_SHAPE[0]),
        "cond_independent_redraw": cond_sha256(regen) == COND_VECTOR_SHA256,
    }

    # ---- §5.3 效用分量 ----
    base_test, base_val = base["test_auc_bsi"], base["best_val_auc_bsi"]
    cp_test, cp_val = cp["test_auc_bsi"], cp["best_val_auc_bsi"]
    up_test, up_val = up["test_auc_bsi"], up["best_val_auc_bsi"]
    gap_u = cp_test - up_test
    delta_val_gap_u = cp_val - up_val
    g_c, g_u = cp_test - base_test, up_test - base_test
    delta_val_c, delta_val_u = cp_val - base_val, up_val - base_val
    retention_u = (g_u / g_c) if g_c > 0 else None
    cp_ratio = [float(r) for r in (cp_prompt.get("val_stats") or {}).get("ratio_mean") or []]
    up_ratio = [float(r) for r in (up_prompt.get("val_stats") or {}).get("ratio_mean") or []]
    ratio_mean_c = float(sum(cp_ratio) / len(cp_ratio)) if cp_ratio else None
    ratio_mean_u = float(sum(up_ratio) / len(up_ratio)) if up_ratio else None
    ratio_rel_diff = (abs(ratio_mean_c - ratio_mean_u) / ratio_mean_c
                      if ratio_mean_c and ratio_mean_u is not None and ratio_mean_c > 0 else None)

    preconditions = {
        "IDENTITY_MISMATCH": not all(identity.values()),
        "BASELINE_REPRODUCTION_FAILED": not all(reproduction_b.values()),
        "CROSS_EXPERIMENT_REPLAY_FAILED": not all(replay_cp.values()),
        "PROTOCOL_INVALID": not all(protocol_a.values()),
        "PINNED_ALPHA_INVALID": not (mechanism["arm_cond"]["PA1"] and mechanism["arm_cond"]["PA2"]
                                     and mechanism["arm_cond"]["PA8"]),
        "UNCONDITION_INVALID": not (mechanism["arm_uncond"]["UA1"] and mechanism["arm_uncond"]["UA2"]
                                    and mechanism["arm_uncond"]["UA10"]),
        "MECHANISM_FAIL": not all(v for status in mechanism.values() for v in status.values()),
        "COMPARATOR_MISMATCH": not all(comparator.values()),
        "CONSTANT_VECTOR_MISMATCH": not all(constant_vector.values()),
    }
    causal = causal_verdict(gap_u=gap_u, delta_val_gap_u=delta_val_gap_u, g_u=g_u,
                            preconditions=preconditions)

    components = causal["components"]
    components.update({
        "baseline_test": base_test, "baseline_val": base_val,
        "arm_cond_test": cp_test, "arm_cond_val": cp_val,
        "arm_uncond_test": up_test, "arm_uncond_val": up_val,
        "g_c": g_c, "g_u": g_u,
        "retained_ratio_u_vs_cond": retention_u,
        "delta_val_c": delta_val_c, "delta_val_u": delta_val_u,
        "retention_c_vs_hist_correct": g_c / DELTA_TEST_CORRECT if DELTA_TEST_CORRECT else None,
        "retention_u_vs_hist_correct": g_u / DELTA_TEST_CORRECT if DELTA_TEST_CORRECT else None,
        "G_c_ge_historical_0.0055": g_c >= HISTORICAL_U1_THRESHOLD,
        "G_u_ge_historical_0.0055": g_u >= HISTORICAL_U1_THRESHOLD,
        "ratio_mean_cond": ratio_mean_c, "ratio_mean_uncond": ratio_mean_u,
        "ratio_relative_diff": ratio_rel_diff,
        "ratio_disclosure_triggered": (ratio_rel_diff is not None
                                       and ratio_rel_diff > RATIO_REL_DIFF_DISCLOSURE_TOL),
    })

    secondary = {"arm_cond": secondary_classification(g_c),
                 "arm_uncond": secondary_classification(g_u)}
    headroom = {"arm_cond": headroom_assessment(delta_test=g_c, delta_val=delta_val_c,
                                                arm_record=cp, prompt_doc=cp_prompt),
                "arm_uncond": headroom_assessment(delta_test=g_u, delta_val=delta_val_u,
                                                  arm_record=up, prompt_doc=up_prompt)}
    result = {
        "analysis": "rp_uncond_compare",
        "root": str(root), "baseline_run": baseline_run,
        "arm_cond_run": arm_cond_run, "arm_uncond_run": arm_uncond_run,
        "context_baseline_run": BASELINE_RUN_ID, "context_correct_run": CORRECT_RUN_ID,
        "context_shuffled_run": SHUF_HIST_RUN_ID,
        "context_pinned_correct_run": PINNED_CORRECT_RUN_ID,
        "context_pinned_shuffled_run": PINNED_SHUFFLED_RUN_ID,
        "pinned_alpha": PINNED_ALPHA, "uncond_const_seed": UNCOND_CONST_SEED,
        "cond_vector_sha256": COND_VECTOR_SHA256,
        "identity": {"checks": identity, "pass": all(identity.values())},
        "reproduction_b": {"checks": reproduction_b, "pass": all(reproduction_b.values())},
        "replay_cp": {"checks": replay_cp, "pass": all(replay_cp.values())},
        "protocol_a": {**protocol_a, "pass": all(protocol_a.values())},
        "mechanism": {"checks": mechanism,
                      "pass": all(v for name in ("arm_cond", "arm_uncond")
                                  for v in mechanism[name].values())},
        "comparator": {"checks": comparator, "pass": all(comparator.values())},
        "constant_vector": {"checks": constant_vector, "pass": all(constant_vector.values())},
        "preconditions": preconditions,
        "components": components,
        "verdict": causal["verdict"], "subreason": causal["subreason"],
        "secondary": secondary, "headroom": headroom,
        "context": {"learnable_arms": CONTEXT["learnable_correct"],
                    "learnable_shuffled": CONTEXT["learnable_shuffled"],
                    "pinned_correct": CONTEXT["pinned_correct"],
                    "pinned_shuffled": CONTEXT["pinned_shuffled"],
                    "budget_chain": CONTEXT["budget_chain"],
                    "alpha_pinned_verdict": CONTEXT["alpha_pinned_verdict"],
                    "historical_u1_threshold": HISTORICAL_U1_THRESHOLD},
        "per_epoch": {"baseline": [e["val_auc_bsi"] for e in base["per_epoch"]],
                      "arm_cond": [e["val_auc_bsi"] for e in cp["per_epoch"]],
                      "arm_uncond": [e["val_auc_bsi"] for e in up["per_epoch"]]},
        "records": {name: {"run_id": doc["run_id"], "commit": doc.get("commit"), "git": doc.get("git"),
                           "best_epoch": doc.get("best_epoch"),
                           "epochs_run": len(doc.get("per_epoch") or []),
                           "stopped_early": len(doc.get("per_epoch") or []) < doc.get("epochs", 0),
                           "wall_seconds": doc.get("wall_seconds"),
                           "peak_vram_mb": doc.get("peak_vram_mb"),
                           "hard_pass": doc.get("hard_pass")}
                   for name, doc in (("baseline", base), ("arm_cond", cp), ("arm_uncond", up))},
        "inherited_b_gates": {name: {g: (doc.get("gates", {}).get(g) or {}).get("verdict")
                                     for g in ("B1", "B2", "B3", "B4")}
                              for name, doc in (("baseline", base_gate), ("arm_cond", cp_gate),
                                                ("arm_uncond", up_gate))},
        "a_gates": {name: {g: (doc.get("gates", {}).get(g) or {}).get("verdict")
                           for g in ("A1", "A2", "A3", "A4", "A5", "A6")}
                    for name, doc in (("baseline", base_gate), ("arm_cond", cp_gate),
                                      ("arm_uncond", up_gate))},
    }
    out_path = root / "runs" / arm_uncond_run / "rp_uncond_compare.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# =====================================================================================
# CLI（纯分析；恰一次）
# =====================================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="AliCCP 无条件固定条件对照消融分析（预注册 §5；只读三 run 记录）")
    parser.add_argument("--root", type=str, default="artifacts/aliccp_bench")
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--arm-cond-run", type=str, required=True)
    parser.add_argument("--arm-uncond-run", type=str, required=True)
    args = parser.parse_args(argv)
    result = analyze_runs(root=Path(args.root), baseline_run=args.baseline_run,
                          arm_cond_run=args.arm_cond_run, arm_uncond_run=args.arm_uncond_run)
    comp = result["components"]
    print(f"verdict={result['verdict']} subreason={result['subreason']} "
          f"secondary={result['secondary']}")
    print(f"GAP_U={comp['gap_u']!r} G_c={comp['g_c']!r} G_u={comp['g_u']!r} "
          f"R_u={comp['retained_ratio_u_vs_cond']!r} dval_gap_u={comp['delta_val_gap_u']!r}")
    print(f"validation_agreement={comp['validation_agreement']} "
          f"unconditional_matches_or_exceeds_conditioned="
          f"{comp['unconditional_matches_or_exceeds_conditioned']}")
    print(f"report -> {Path(args.root) / 'runs' / args.arm_uncond_run / 'rp_uncond_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
