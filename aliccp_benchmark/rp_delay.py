"""AliCCP 阶段 2 残差 Prompt 的延迟解冻学习动力学消融（delayed-unfreeze dynamics；本分支
exp/aliccp-stage2-residual-prompt-alpha-dynamics）。

唯一事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-delayed-unfreeze-dynamics-design.md

消融语义（预注册 §2）：把 prompt 模块（可学习标量门控 `prompt_gate` α + 生成器 4 张量）在
**构造值**上冻结恰好 `DELAY_EPOCHS = 1` 个 epoch（α = 0.0 精确；生成器 = 隔离 RNG 初始权重），
在 `UNFREEZE_EPOCH = 2` 起点经 `unfreeze_prompt_module` 解冻（requires_grad 恢复 True；优化器
自首个有效梯度起与学习臂同构）。延迟界（k=1）由六条已提交学习臂的逐 epoch α 探针唯一确定：
probe(ep1).α == 0.0 ∧ probe(ep1).generator_grad == 0.0（精确），probe(ep2).α != 0 ∧
probe(ep2).generator_grad > 0（跨 seed、跨预算普遍）。

结构性事实（引理，预注册 §1.4）：α ≡ 0 ⇒ delta ≡ ±0.0 ⇒ 前向与参照头逐位一致（S3），且
∂delta/∂θ_gen ∝ α = 0 ⇒ 生成器在冻结期梯度精确为零（任何优化器算术下都不动）。因此 D 臂
epoch 1 的整条训练流与配对基线 B **逐位相同**（FROZEN_ID 前置条件：epoch-1 train_loss/val
三方逐位恒等 == 钉死常量）。

判定（预注册 §5）：B/D 双 run（本分支新跑）→ 前置条件（ID / REP_B / FROZEN_ID / A / DD1–DD9 /
CC）→ 因果判定树（`DYNAMICS_SUPPORTED` / `DYNAMICS_NOT_SUPPORTED` / `INVALID`）+ 二级分类
（`POSITIVE_IMPROVEMENT` / `NO_CLEAR_IMPROVEMENT` / `CLEAR_DEGRADATION`）。`half_gap_met`
（回收固定 α 与学习 α 差距的至少一半）为**预声明描述统计**，不进判定树；历史 `+0.0055` 单列保留。
不主张新颖性。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

from aliccp_benchmark import protocol as P
from aliccp_benchmark import residual_prompt as RP
from aliccp_benchmark import rp_pinned as RPP

# ---- 臂与接线（预注册 §2.3）----
VARIANT_DELAY = "residual-prompt-delay"
RUN_ID_SUFFIX_DELAY = "-rpd"
ALL_VARIANTS = tuple(RPP.ALL_VARIANTS) + (VARIANT_DELAY,)


def arm_suffix_for(variant: str) -> str:
    """臂后缀分派（delay ⇒ -rpd；其余沿用 rp_pinned 分派，含 baseline/学习/钉死臂）。"""
    if variant == VARIANT_DELAY:
        return RUN_ID_SUFFIX_DELAY
    return RPP.run_id_suffix_for(variant)


# ---- 冻结判据常量（预注册 §2/§5；只允许在看到结果之前修改）----
DELAY_EPOCHS = 1                 # 延迟 = 恰好"α 离开 0"的那一个 epoch（六条学习臂一致）
UNFREEZE_EPOCH = DELAY_EPOCHS + 1
MATERIAL_DELTA = 0.001           # 最小实质差异（= 用户指定；二级分类边界同值）
SECONDARY_POSITIVE_MIN = 0.001   # 二级分类：ΔD ≥ +0.001 ⇒ POSITIVE_IMPROVEMENT（闭）
SECONDARY_NEGATIVE_MAX = -0.02   # ΔD ≤ −0.02 ⇒ CLEAR_DEGRADATION（闭）
HALF_GAP_FRACTION = 0.5          # 半程回收比例（预声明描述统计；闭端 ≥）
REFERENCE_IDENTITY_TOL = RP.REFERENCE_IDENTITY_TOL   # 1e-9（M0）
BOUND_TOL = RP.BOUND_TOL                             # 1e-6（DD5）
RATIO_BAND = RP.RATIO_BAND                           # (0.005, 0.5)（DD6；与学习臂 G5 同值）
PRED_STD_MIN_RATIO = RP.PRED_STD_MIN_RATIO           # 0.5（DD8）
EXPECTED_NEW_PARAMS_TOTAL = RP.EXPECTED_NEW_PARAMS_TOTAL   # 2385（α 仍为参数身份；DD9）
EXPECTED_HEAD_PARAMS = RP.EXPECTED_HEAD_PARAMS             # 8129

# ---- 判定标签（先于结果写死）----
VERDICT_SUPPORTED = "DYNAMICS_SUPPORTED"
VERDICT_NOT_SUPPORTED = "DYNAMICS_NOT_SUPPORTED"
VERDICT_INVALID = "INVALID"
SUBREASON_BELOW_MATERIALITY = "BELOW_MATERIALITY"
SUBREASON_VALIDATION_DISAGREEMENT = "VALIDATION_DISAGREEMENT"
INVALID_PRIORITY = ("IDENTITY_MISMATCH", "BASELINE_REPRODUCTION_FAILED",
                    "FROZEN_PHASE_IDENTITY_FAILED", "PROTOCOL_INVALID", "MECHANISM_FAIL",
                    "COMPARATOR_MISMATCH")
SECONDARY_LABELS = ("POSITIVE_IMPROVEMENT", "NO_CLEAR_IMPROVEMENT", "CLEAR_DEGRADATION")

# ---- 钉死身份（预注册 §1.5；与 verify_delay_prerun.py 逐位一致）----
STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
STAGE1_BACKBONE_SHA = "e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c"
STAGE1_ENV_IDS_SHA = "5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0"
STAGE1_FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
MODEL_SEED_PINNED = 1688723740
SHORT_EPOCHS, SHORT_PATIENCE = 5, 2
BASELINE_RUN_ID = RPP.BASELINE_RUN_ID
LEARNABLE_RUN_ID = RPP.CORRECT_RUN_ID
PINNED_RUN_ID = "20261005-1029-p2M-v500k-t1M-m1688723740-short-7d26918-rpp"
SHUF_HIST_RUN_ID = RPP.SHUF_HIST_RUN_ID
PINNED_SHUFFLED_RUN_ID = "20261005-1032-p2M-v500k-t1M-m1688723740-short-1c13841-rpps"
REF_HEAD_SHA = RPP.REF_HEAD_SHA
REF_HEAD_REL = f"artifacts/aliccp_bench/runs/{BASELINE_RUN_ID}/newtask.pt"

# ---- 钉死参照常量（预注册 §1.5；只读复算来源见 verify_delay_prerun.py）----
BASELINE_AUC_TEST = RP.BASELINE_AUC_TEST                # 0.5974422649550507
BASELINE_AUC_VAL = RP.BASELINE_AUC_VAL                  # 0.5809347091990792
REFERENCE_PRED_STD = RP.REFERENCE_PRED_STD              # 0.005217193225189258
BASELINE_PER_EPOCH_VAL = list(RPP.BASELINE_RECORD["per_epoch_val"])
BASELINE_GATE_MEAN = list(RPP.BASELINE_RECORD["gate_mean"])
BASELINE_PER_EPOCH_LOSS = [0.08193688414408826, 0.05193358083860949, 0.04973645433795173,
                           0.04873575337347574, 0.04805051837593783]
BASELINE_EP1_TRAIN_LOSS = BASELINE_PER_EPOCH_LOSS[0]    # 0.08193688414408826（FROZEN_ID）
BASELINE_EP1_VAL = BASELINE_PER_EPOCH_VAL[0]            # 0.4645711559431739（FROZEN_ID）

DELTA_TEST_LEARNABLE = RPP.DELTA_TEST_CORRECT           # +0.008103113327467715
DELTA_VAL_LEARNABLE = RPP.DELTA_VAL_CORRECT             # +0.008622497456618139
LEARNABLE_PROMPT_REPORT_SHA = RPP.CORRECT_PROMPT_REPORT_SHA   # 185df4d0…
LEARNABLE_ALPHA_FINAL = RPP.PINNED_ALPHA                # 0.07101669907569885（L 的 alpha_final）

PINNED_CORRECT_RECORD = {                               # C_p = 7d26918-rpp（钉死 α correct 臂）
    "run_id": PINNED_RUN_ID,
    "test_auc_bsi": 0.5947650925865714,
    "best_val_auc_bsi": 0.5798886656441761,
    "best_epoch": 5,
    "gate_mean": [0.775494625, 0.224505265625],
    "per_epoch_val": [0.46759930720014714, 0.49016156687227064, 0.5156547237007145,
                      0.5480714746135289, 0.5798886656441761],
    "variant": "residual-prompt-pinned",
    "classification": "VALID_NEGATIVE",
    "alpha_final": RPP.PINNED_ALPHA,
}
DELTA_TEST_PINNED = PINNED_CORRECT_RECORD["test_auc_bsi"] - BASELINE_AUC_TEST   # −0.002677172368479308
DELTA_VAL_PINNED = PINNED_CORRECT_RECORD["best_val_auc_bsi"] - BASELINE_AUC_VAL  # −0.00104604355490312
GAP = DELTA_TEST_LEARNABLE - DELTA_TEST_PINNED          # +0.010780285695947023
HALF_GAP_TARGET = DELTA_TEST_PINNED + HALF_GAP_FRACTION * GAP   # +0.0027129704794942035

# 历史 shuffled 臂记录（CC 只读核对用；不含 α 轨迹断言）
SHUF_HIST_RECORD = {"test_auc_bsi": 0.5989210260551207, "best_val_auc_bsi": 0.5819965357883198,
                    "best_epoch": 5, "variant": "residual-prompt-shuffled",
                    "classification": "MECHANISM_FAIL"}
SHUF_HIST_PROMPT_SHA = "28636895e9f5ce7be7ce3908aaa80c0e4c7213f633f4087086159f4e1e60c95e"
PINNED_SHUFFLED_RECORD = {"test_auc_bsi": 0.5966263705980125, "best_val_auc_bsi": 0.5817369266195509,
                          "best_epoch": 5, "variant": "residual-prompt-pinned-shuffled",
                          "classification": "VALID_NEGATIVE"}
LEARNABLE_METRICS_SHA = "f5619bb0dead85e2ae601e9525ba5a5e4ce9c7302297e02152a2aeda733cdc7f"
PINNED_METRICS_SHA = "f0a884ce6c59e0b37657cec4de3fc012d53539e175571d1ee70aa9d98ced14e9"
PINNED_NEWTASK_SHA = "264aedbb9f3f605e5a8e5dd9b2aee8c1945d2dcf0388be722541d2a84f4062b8"
PINNED_PROMPT_REPORT_SHA = "77f028993e8b1ea85210ac1b8d08f44698433264c36997bb850e10789d62e64b"

# ---- 六条学习臂 α 轨迹钉死（预注册 §2.2 证据表；与 verify_delay_prerun.py 双副本同步）----
LEARNABLE_TRAJECTORIES = {
    "013e105-rpg": {
        "run": "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg", "budget": "short",
        "prompt_report_sha": "185df4d0f9910508aeff84ed780e7df4fe8ad4fd6b8451ff429a74927d333c1e",
        "alphas": [0.0, 0.016439981758594513, 0.037462275475263596, 0.050921376794576645,
                   0.06157483905553818], "alpha_final": 0.07101669907569885},
    "f2ccec2-rpg": {
        "run": "20261004-0431-p2M-v500k-t1M-m1688723740-long-f2ccec2-rpg", "budget": "long",
        "prompt_report_sha": "ba1d41a777de4ec7713132a5946b994b871c4f4b1682f02e152ad23ff83d7bb8",
        "metrics_sha": "9b225f6e943f82e9af14641c5251ad3892381febb00bb598d104258a692eb513",
        "alphas": [0.0, 0.016439981758594513, 0.037462275475263596, 0.050921376794576645,
                   0.06157483905553818, 0.07101669907569885, 0.07947666943073273,
                   0.08645110577344894, 0.09248566627502441, 0.09788396954536438],
        "alpha_final": 0.1025848239660263},
    "eafc336-rpg": {
        "run": "20261005-0829-p2M-v500k-t1M-m1688723740-xlong-eafc336-rpg", "budget": "xlong",
        "prompt_report_sha": "d5a15cc479bdff605087fe8c5cda11556cd8ede5489bc36996026bcb70c0879c",
        "metrics_sha": "d32a0fbff15d6bb560d64721f738604beb93b9f17b671364bcd4485308c35615",
        "alpha_final": 0.12710827589035034},
    "c17b100-rpg.s3": {
        "run": "20261005-0642-p2M-v500k-t1M-m1688738016-short-c17b100-rpg", "budget": "short",
        "prompt_report_sha": "daeb0d560b3c9c3bc8d9507579d43cdb6c75db317d31464b662e2dcaa511b5de",
        "metrics_sha": "d69f5092eeee96efc959e4520cffbbf208dd2c1b912518bf542cf8ab58fc9259",
        "alphas": [0.0, -0.046634916216135025, -0.04634027183055878, -0.04700079560279846,
                   -0.04794004559516907],
        "alpha_final": -0.04903412237763405},
    "c17b100-rpg.s4": {
        "run": "20261005-0646-p2M-v500k-t1M-m1688749593-short-c17b100-rpg", "budget": "short",
        "prompt_report_sha": "bc55349f07c26ec90078d43eaee5eb78388aa17396deaf970eb90dafdb28b51e",
        "metrics_sha": "2dcb008fa2a73528861d0f46bad772c49af66bbfeb75ffdd8093512a7d59aba0",
        "alphas": [0.0, 0.019704077392816544, 0.023464083671569824, 0.032033320516347885,
                   0.04252833500504494],
        "alpha_final": 0.053119830787181854},
    "c17b100-rpg.s5": {
        "run": "20261005-0649-p2M-v500k-t1M-m1688762746-short-c17b100-rpg", "budget": "short",
        "prompt_report_sha": "69cc23b070a22666594b09fcd30897da69a4b6246961cf15b02998148629d565",
        "metrics_sha": "96ce5a9a7966f7ad3a8530fa77254a1933bf986fd0cee59eb841ac77054f9619",
        "alphas": [0.0, -0.026758631691336632, -0.02457469142973423, -0.02771887741982937,
                   -0.03476540744304657],
        "alpha_final": -0.04552163928747177},
}

# ---- 上下文常量（只读；§5.3 报告项）----
FIVE_SEED_MEAN_DELTA = 0.005910402347593546
U_ARM_DELTA = -0.0026189914978292927
BUDGET_CHAIN = {"five_epoch": DELTA_TEST_LEARNABLE, "ten_epoch": 0.0074396653390377265,
                "twenty_epoch": 0.004155967377889924, "budget_trend_label": "MONOTONE_NARROWING"}


# =====================================================================================
# §2.3 延迟解冻头
# =====================================================================================

def _prompt_module_sha(newtask) -> str:
    """生成器 4 张量（键排序）的内容 sha256（构造/解冻两点比对用；不含 α）。"""
    digest = hashlib.sha256()
    for name, param in sorted(newtask.prompt_generator.state_dict().items()):
        digest.update(name.encode())
        digest.update(P.sha256_tensor(param).encode())
    return digest.hexdigest()


class DelayedUnfreezeResidualPromptNewTask(RP.ResidualPromptNewTask):
    """延迟解冻残差 prompt 头：α 与生成器构造即冻结（requires_grad=False），解冻前不接收梯度。

    构造路径逐位同 `ResidualPromptNewTask`（共享参数/生成器初始化/隔离 RNG 端点不变）；
    `prompt_gate` 保持 **nn.Parameter** 身份（与学习臂同一 state_dict 键、同一优化器覆盖、同一
    参数计数 2385），仅将 requires_grad 置 False；生成器 4 张量同理。`prompt_deltas` 与
    `forward` 逐字继承 ⇒ 公式 `α · (||h||/√d) · m` 不变，唯一差异 = 学习期被延迟。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_gate.requires_grad_(False)               # α：参数身份、冻结可学习性
        for param in self.prompt_generator.parameters():
            param.requires_grad_(False)                      # 生成器：冻结期结构惰性（α=0 ⇒ 梯度 0）


def build_delay_newtask(*, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                        hidden: int = RP.PROMPT_HIDDEN):
    if int(hidden) < 1:
        raise ValueError(f"hidden 必须 ≥ 1（收到 {hidden}）")
    return DelayedUnfreezeResidualPromptNewTask(
        input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=tower_dnn_hidden_units,
        reg_dnn=reg_dnn, device=device, hidden=hidden)


def unfreeze_prompt_module(newtask) -> dict:
    """解冻：恰一次地把 α 与生成器 requires_grad 置 True（校验齐全；违反即拒绝出数）。"""
    gate = newtask.prompt_gate
    if not isinstance(gate, nn.Parameter):
        raise ValueError("prompt_gate 必须是 nn.Parameter（与学习臂同一参数身份）")
    if bool(gate.requires_grad):
        raise ValueError("重复解冻：prompt_gate.requires_grad 已为 True")
    if float(gate.detach()) != 0.0:
        raise ValueError(f"解冻要求 α 精确等于构造值 0.0（当前 {float(gate.detach())!r}）")
    gen_params = list(newtask.prompt_generator.parameters())
    if len(gen_params) != 4:
        raise ValueError(f"生成器参数数必须为 4（实际 {len(gen_params)}）")
    if any(bool(p.requires_grad) for p in gen_params):
        raise ValueError("重复解冻：生成器存在 requires_grad=True 的参数")
    gate.requires_grad_(True)
    for param in gen_params:
        param.requires_grad_(True)
    return {"alpha_at_unfreeze": float(gate.detach()),
            "requires_grad_after_unfreeze": {"alpha": bool(gate.requires_grad),
                                             "generator": [bool(p.requires_grad)
                                                           for p in gen_params]}}


class DelayPromptAudit(RP.PromptAudit):
    """DD1–DD4 运行期审计：构造身份（继承 G1 等价）+ 冻结期结构记录 + 解冻事件（恰一次）。

    - 构造记录：α 为 Parameter 且 requires_grad=False ∧ 生成器 4 张量 requires_grad=False ∧
      优化器覆盖全部 named_parameters（含 α，对象同一性）∧ 生成器构造 sha；
    - 冻结签名（epoch-1 探针，继承的 grad_probe）：α == 0.0 ∧ alpha_grad_norm is None ∧
      generator_grad_norm == 0.0；
    - `unfreeze(epoch)`：仅 epoch == UNFREEZE_EPOCH 且未解冻过可调用；记录解冻前后 α 值、
      生成器 sha 相等、requires_grad 恢复、RNG 端点不变。
    """

    def __init__(self, *args, optimizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        gate = self.newtask.prompt_gate
        gen_params = list(self.newtask.prompt_generator.parameters())
        if optimizer is None:
            optimizer_covers = optimizer_total = None
        else:
            opt_params = [p for group in optimizer.param_groups for p in group["params"]]
            optimizer_covers = ({id(p) for p in opt_params}
                                == {id(p) for _, p in self.newtask.named_parameters()})
            optimizer_total = len(opt_params)
        self._unfrozen = False
        self.delay = {
            "delay_epochs": DELAY_EPOCHS,
            "unfreeze_epoch": UNFREEZE_EPOCH,
            "alpha_at_construction": float(gate.detach()),
            "alpha_requires_grad_at_construction": bool(gate.requires_grad),
            "generator_requires_grad_at_construction": [bool(p.requires_grad)
                                                        for p in gen_params],
            "generator_sha_at_construction": _prompt_module_sha(self.newtask),
            "alpha_at_unfreeze": None,
            "generator_sha_at_unfreeze": None,
            "unfrozen_once": False,
            "requires_grad_after_unfreeze": None,
            "optimizer_covers_named_parameters": optimizer_covers,
            "optimizer_params_total": optimizer_total,
            "unfreeze_rng_endpoint_unchanged": None,
        }

    def unfreeze(self, epoch: int) -> None:
        """runner 于 epoch == UNFREEZE_EPOCH 起点调用恰一次（§6 运行规程）。"""
        if int(epoch) != UNFREEZE_EPOCH:
            raise ValueError(f"解冻只允许发生在 epoch {UNFREEZE_EPOCH}（收到 {epoch}）")
        if self._unfrozen:
            raise ValueError("重复解冻：unfreeze 已调用过（恰一次纪律）")
        state_before = torch.get_rng_state()
        record = unfreeze_prompt_module(self.newtask)
        self._unfrozen = True
        self.delay.update({
            "alpha_at_unfreeze": record["alpha_at_unfreeze"],
            "generator_sha_at_unfreeze": _prompt_module_sha(self.newtask),
            "unfrozen_once": True,
            "requires_grad_after_unfreeze": record["requires_grad_after_unfreeze"],
            "unfreeze_rng_endpoint_unchanged": bool(torch.equal(state_before,
                                                               torch.get_rng_state())),
        })

    def result(self, alpha_final: float) -> dict:
        out = super().result(alpha_final)
        out["delay"] = dict(self.delay)
        return out


# =====================================================================================
# §5.2 D 臂结构机制门禁（DD1–DD9；由 observed 值机械重算）
# =====================================================================================

def dd_gate_observed(*, probe: dict, val_stats: dict, params: dict, reference: dict) -> tuple:
    """由 run 记录中的 observed 值机械重算 DD1–DD9（不信任记录布尔）。返回 (passes, aux)。

    兼容两种 probe 形状：审计 probe（`construction`，`RP.PromptAudit.result()`）与
    prompt_report 落盘形状（`construction_identity`，bench 写出时改名）；两者数值同源。
    """
    construction = probe.get("construction") or probe.get("construction_identity") or {}
    init_forward = probe.get("init_forward") or {}
    records = probe.get("grad_probe") or []
    delay = probe.get("delay") or {}
    alpha_final = float(probe.get("alpha_final") or 0.0)
    gate = val_stats.get("gate") or {}
    streams = val_stats.get("streams") or {}
    dispersion = val_stats.get("dispersion") or {}

    dd1 = bool(construction.get("shared_params_bit_identical")
               and construction.get("global_rng_endpoint_identical")
               and construction.get("extra_keys") == sorted(RP.EXTRA_PARAM_NAMES))
    dd2 = bool(construction.get("alpha_at_construction") == 0.0
               and init_forward.get("bit_identical"))
    frozen_signature = bool(records and records[0].get("alpha") == 0.0
                            and records[0].get("alpha_grad_norm") is None
                            and records[0].get("generator_grad_norm") == 0.0)
    rg_after = delay.get("requires_grad_after_unfreeze") or {}
    dd3 = bool(delay.get("delay_epochs") == DELAY_EPOCHS
               and delay.get("unfreeze_epoch") == UNFREEZE_EPOCH
               and delay.get("alpha_at_construction") == 0.0
               and delay.get("alpha_requires_grad_at_construction") is False
               and delay.get("generator_requires_grad_at_construction") == [False] * 4
               and delay.get("alpha_at_unfreeze") == 0.0
               and delay.get("generator_sha_at_unfreeze") is not None
               and delay.get("generator_sha_at_unfreeze") == delay.get("generator_sha_at_construction")
               and delay.get("unfrozen_once") is True
               and rg_after.get("alpha") is True
               and rg_after.get("generator") == [True] * 4
               and delay.get("optimizer_covers_named_parameters") is True
               and delay.get("unfreeze_rng_endpoint_unchanged") is True
               and frozen_signature)
    if len(records) >= 2:
        unfreeze_probe = records[1]
        last = records[-1]
        dd4 = bool(unfreeze_probe.get("alpha_grad_norm") is not None
                   and float(unfreeze_probe.get("alpha_grad_norm") or 0.0) != 0.0
                   and unfreeze_probe.get("generator_grad_norm") == 0.0
                   and float(last.get("alpha") or 0.0) != 0.0
                   and float(last.get("generator_grad_norm") or 0.0) > 0.0
                   and alpha_final != 0.0)
    else:
        dd4 = False
    ratio_max = [float(r) for r in (streams.get("ratio_max") or [])]
    dd5 = bool(ratio_max) and all(r <= abs(alpha_final) + BOUND_TOL for r in ratio_max)
    ratio_mean = [float(r) for r in (streams.get("ratio_mean") or [])]
    band_low = bool(ratio_mean) and any(r < RATIO_BAND[0] for r in ratio_mean)
    band_high = bool(ratio_mean) and any(r > RATIO_BAND[1] for r in ratio_mean)
    dd6 = bool(ratio_mean) and not band_low and not band_high
    dd7 = bool(float(gate.get("geff_std") or 0.0) > 0.0
               and float(gate.get("geff_max") or 0.0) > 0.0
               and float(gate.get("geff_min") or 0.0) >= 0.0)
    ref_disp = (reference or {}).get("pred_dispersion") or {}
    ref_std = ref_disp.get("pred_std")
    pred_std = float(dispersion.get("pred_std") or 0.0)
    dd8 = bool(pred_std > 0.0 and ref_std is not None and float(ref_std) > 0.0
               and pred_std >= PRED_STD_MIN_RATIO * float(ref_std))
    names = sorted(str(item.get("name")) for item in (params.get("new_param_list") or []))
    dd9 = bool(names == sorted(RP.EXTRA_PARAM_NAMES)
               and int(params.get("new_params_total", -1)) == EXPECTED_NEW_PARAMS_TOTAL
               and int(params.get("head_params", -1)) == EXPECTED_HEAD_PARAMS)
    aux = {"band_low": band_low, "band_high": band_high,
           "frozen_signature": frozen_signature, "ratio_mean": ratio_mean,
           "ratio_max": ratio_max}
    return {"DD1": dd1, "DD2": dd2, "DD3": dd3, "DD4": dd4, "DD5": dd5, "DD6": dd6,
            "DD7": dd7, "DD8": dd8, "DD9": dd9}, aux


def delay_arm_verdict(*, auc_test: float, auc_val: float, probe: dict, val_stats: dict,
                      params: dict, reference: dict | None, protocol_ok: bool = True,
                      baseline_auc_test: float = BASELINE_AUC_TEST,
                      baseline_auc_val: float = BASELINE_AUC_VAL,
                      expected_reference_auc: float = BASELINE_AUC_VAL,
                      expected_reference_pred_std: float = REFERENCE_PRED_STD) -> dict:
    """M0 + DD1–DD9 + U1/U2 判定与三标签结局分类（预注册 §5.2；判据先于结果写定）。"""
    reference = reference or {}
    ref_auc = reference.get("val_auc")
    ref_disp = reference.get("pred_dispersion") or {}
    ref_std = ref_disp.get("pred_std")
    m0_pass = bool(ref_auc is not None and ref_std is not None
                   and abs(float(ref_auc) - expected_reference_auc) <= REFERENCE_IDENTITY_TOL
                   and abs(float(ref_std) - expected_reference_pred_std) <= REFERENCE_IDENTITY_TOL)
    passes, aux = dd_gate_observed(probe=probe, val_stats=val_stats, params=params,
                                   reference=reference)

    delta_test = float(auc_test) - baseline_auc_test
    delta_val = float(auc_val) - baseline_auc_val
    u1_pass = bool(delta_test >= RP.AUC_TEST_DELTA_MIN)
    u2_pass = bool(delta_val > RP.AUC_VAL_DIRECTION_MIN)
    u_pass = bool(u1_pass and u2_pass)

    if not (passes["DD1"] and passes["DD2"] and passes["DD3"] and passes["DD9"]):
        classification, subreason = "MECHANISM_FAIL", "INVALID_IMPLEMENTATION"
    elif not passes["DD4"]:
        classification, subreason = "MECHANISM_FAIL", "MECHANISM_INACTIVE"
    elif aux["band_low"]:
        classification, subreason = "MECHANISM_FAIL", "MECHANISM_SILENT"
    elif aux["band_high"]:
        classification, subreason = "MECHANISM_FAIL", "MECHANISM_OVER_PERTURB"
    elif not passes["DD7"]:
        classification, subreason = "MECHANISM_FAIL", "GATE_DEGENERATE"
    elif not passes["DD8"]:
        classification, subreason = "MECHANISM_FAIL", "PREDICTION_COLLAPSE"
    elif not passes["DD5"]:
        classification, subreason = "MECHANISM_FAIL", "NORM_BOUND_VIOLATION"
    elif not m0_pass:
        classification, subreason = "MECHANISM_FAIL", "REFERENCE_IDENTITY"
    elif not protocol_ok:
        classification, subreason = "MECHANISM_FAIL", "PROTOCOL_INVALID"
    elif u_pass:
        classification, subreason = "VALID_POSITIVE", None
    else:
        classification, subreason = "VALID_NEGATIVE", None

    construction = probe.get("construction") or {}
    init_forward = probe.get("init_forward") or {}
    records = probe.get("grad_probe") or []
    delay = probe.get("delay") or {}
    alpha_final = float(probe.get("alpha_final") or 0.0)
    gate = val_stats.get("gate") or {}
    dispersion = val_stats.get("dispersion") or {}
    pred_std = float(dispersion.get("pred_std") or 0.0)
    return {
        "M0": {"pass": m0_pass,
               "rule": f"|ref_val_auc - {expected_reference_auc}| <= {REFERENCE_IDENTITY_TOL} AND "
                       f"|ref_pred_std - {expected_reference_pred_std}| <= {REFERENCE_IDENTITY_TOL}",
               "observed": {"ref_val_auc": ref_auc, "ref_pred_std": ref_std}},
        "DD1": {"pass": passes["DD1"],
                "rule": "shared bit-identical AND rng endpoint identical AND extra keys exact (5)",
                "observed": {"shared_params_bit_identical": construction.get("shared_params_bit_identical"),
                             "global_rng_endpoint_identical": construction.get("global_rng_endpoint_identical"),
                             "extra_keys": construction.get("extra_keys")}},
        "DD2": {"pass": passes["DD2"],
                "rule": "alpha_at_construction == 0 AND init forward bit-identical",
                "observed": {"alpha_at_construction": construction.get("alpha_at_construction"),
                             "init_bit_identical": init_forward.get("bit_identical")}},
        "DD3": {"pass": passes["DD3"],
                "rule": "delay block exact (epochs/values/sha/rg/optimizer) AND epoch-1 frozen signature "
                        "(alpha==0 AND alpha_grad None AND generator_grad==0)",
                "observed": {"delay": delay, "frozen_signature": aux["frozen_signature"]}},
        "DD4": {"pass": passes["DD4"],
                "rule": "epoch-2 probe alpha_grad != 0 (and not None) AND same probe generator_grad == 0 "
                        "AND last probe alpha != 0 AND last probe generator_grad > 0 AND alpha_final != 0",
                "observed": {"unfreeze_probe": records[1] if len(records) >= 2 else None,
                             "last_probe": records[-1] if records else None,
                             "alpha_final": alpha_final}},
        "DD5": {"pass": passes["DD5"], "rule": f"every stream ratio_max <= |alpha_final| + {BOUND_TOL}",
                "observed": {"ratio_max": aux["ratio_max"], "alpha_abs": abs(alpha_final)}},
        "DD6": {"pass": passes["DD6"], "rule": f"every stream ratio_mean in [{RATIO_BAND[0]}, {RATIO_BAND[1]}]",
                "observed": {"ratio_mean": aux["ratio_mean"], "band_low": aux["band_low"],
                             "band_high": aux["band_high"]}},
        "DD7": {"pass": passes["DD7"], "rule": "geff_std > 0 AND geff_max > 0 AND geff_min >= 0",
                "observed": {"geff_std": gate.get("geff_std"), "geff_min": gate.get("geff_min"),
                             "geff_max": gate.get("geff_max")}},
        "DD8": {"pass": passes["DD8"], "rule": f"pred_std > 0 AND pred_std >= {PRED_STD_MIN_RATIO} * ref_pred_std",
                "observed": {"pred_std": pred_std, "ref_pred_std": ref_std}},
        "DD9": {"pass": passes["DD9"],
                "rule": f"5 exact prompt_ keys AND new_params_total == {EXPECTED_NEW_PARAMS_TOTAL} "
                        f"AND head_params == {EXPECTED_HEAD_PARAMS}",
                "observed": {"names": sorted(str(item.get("name")) for item in (params.get("new_param_list") or [])),
                             "new_params_total": params.get("new_params_total"),
                             "head_params": params.get("head_params")}},
        "U": {"pass": u_pass,
              "U1": {"pass": u1_pass, "rule": f"delta_test >= +{RP.AUC_TEST_DELTA_MIN}",
                     "observed": {"auc_test": float(auc_test), "delta_test": delta_test,
                                  "baseline_auc_test": baseline_auc_test}},
              "U2": {"pass": u2_pass, "rule": "delta_val > 0",
                     "observed": {"auc_val": float(auc_val), "delta_val": delta_val,
                                  "baseline_auc_val": baseline_auc_val}}},
        "protocol_10_1": {"delta_test_ge_0.005": bool(delta_test >= 0.005), "val_positive": u2_pass},
        "classification": classification,
        "subreason": subreason,
        "pass": classification == "VALID_POSITIVE",
    }


# =====================================================================================
# §5.4/§5.5 判定树与二级分类（纯函数；分析器与独立复核共用）
# =====================================================================================

def decide_verdict(*, invalid_subreason, delta_d: float, delta_val_d: float) -> dict:
    """预注册 §5.4 判定树（优先级：INVALID > SUPPORTED > NOT_SUPPORTED）。"""
    if invalid_subreason is not None:
        return {"verdict": VERDICT_INVALID, "subreason": invalid_subreason,
                "rule": "any precondition false -> INVALID (no utility reading)"}
    if delta_d >= MATERIAL_DELTA and delta_val_d > 0.0:
        return {"verdict": VERDICT_SUPPORTED, "subreason": None,
                "rule": f"delta_d >= +{MATERIAL_DELTA} AND delta_val_d > 0"}
    subreason = SUBREASON_BELOW_MATERIALITY if delta_d < MATERIAL_DELTA \
        else SUBREASON_VALIDATION_DISAGREEMENT
    return {"verdict": VERDICT_NOT_SUPPORTED, "subreason": subreason,
            "rule": f"delta_d < +{MATERIAL_DELTA} OR delta_val_d <= 0"}


def classify_secondary(delta_d: float) -> str:
    """用户三标签（§5.5；边界闭端：+0.001 与 −0.02）。"""
    if delta_d >= SECONDARY_POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if delta_d <= SECONDARY_NEGATIVE_MAX:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def half_gap_met(delta_d: float) -> bool:
    """半程回收（§5.3；预声明描述统计；闭端 ≥）。"""
    return bool(delta_d >= HALF_GAP_TARGET)


def headroom_report(*, delta_d: float, delta_val_d: float, probe: dict, delay: dict,
                    best_epoch, epochs_run: int, epochs_recorded: int) -> dict:
    """§5.5 NO_CLEAR 的机械 headroom（因子取该臂记录值）。"""
    alpha_final = float(probe.get("alpha_final") or 0.0)
    streams = (probe.get("val_stats") or (probe.get("streams") or {}))
    ratio_mean = [float(r) for r in (streams.get("ratio_mean") or [])]
    gate = probe.get("gate") or {}
    active = bool(alpha_final != 0.0 and ratio_mean and all(r > 0 for r in ratio_mean)
                  and float(gate.get("geff_max") or 0.0) > 0.0)
    records = probe.get("grad_probe") or []
    engaged = bool(len(records) >= 2 and records[1].get("alpha_grad_norm") is not None
                   and float(records[1].get("alpha_grad_norm") or 0.0) != 0.0)
    censored = bool(best_epoch == epochs_run == epochs_recorded)
    return {
        "mechanism_activity": "UNFROZEN_ACTIVE" if active else "INACTIVE",
        "unfreeze_engagement": "ENGAGED" if engaged else "NOT_ENGAGED",
        "validation_direction": "POSITIVE" if delta_val_d > 0 else "NON_POSITIVE",
        "censoring": "RIGHT_CENSORED_STILL_IMPROVING" if censored else "EARLY_STOPPED",
        "gap_to_positive_threshold": MATERIAL_DELTA - float(delta_d),
        "recovery_ratio": float(delta_d) / DELTA_TEST_LEARNABLE,
        "gap_closure": (float(delta_d) - DELTA_TEST_PINNED) / GAP,
    }


# =====================================================================================
# §5 分析器（只读两 run 记录 + 历史常量 → 全量判定）
# =====================================================================================

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_metrics(runs_root: Path, run_id: str) -> dict:
    return _load_json(runs_root / run_id / "metrics.json")


def _a_class_ok(gate_report: dict) -> bool:
    return all(rec.get("verdict") in ("PASS", "SKIP")
               for key, rec in (gate_report.get("gates") or {}).items() if key.startswith("A"))


def analyze_runs(*, runs_root, baseline_run: str, delay_run: str) -> dict:
    """预注册 §5 机械计算：前置条件（ID/REP_B/FROZEN_ID/A/DD/CC）→ 判定树 → 二级分类 → 描述量。"""
    runs_root = Path(runs_root)
    b = _read_metrics(runs_root, baseline_run)
    d = _read_metrics(runs_root, delay_run)
    d_report = _load_json(runs_root / delay_run / "prompt_report.json")

    # ---- ID ----
    suffixless = not any(b["run_id"].endswith(s) for s in
                         (RP.RUN_ID_SUFFIX, RPP.RUN_ID_SUFFIX_CORRECT, RPP.RUN_ID_SUFFIX_SHUFFLED,
                          RUN_ID_SUFFIX_DELAY))
    d_ref_path = ((d_report.get("reference_dispersion") or {}).get("newtask_checkpoint"))
    ref_head_path = runs_root / BASELINE_RUN_ID / "newtask.pt"
    identity = {
        "stage1_id_match": b.get("stage1_id") == STAGE1_ID and d.get("stage1_id") == STAGE1_ID,
        "model_seed": b.get("model_seed") == MODEL_SEED_PINNED and d.get("model_seed") == MODEL_SEED_PINNED,
        "protocol_point": (b.get("epochs"), b.get("patience"), b.get("tag")) == (SHORT_EPOCHS, SHORT_PATIENCE, "short")
                          and (d.get("epochs"), d.get("patience"), d.get("tag")) == (SHORT_EPOCHS, SHORT_PATIENCE, "short"),
        "clean_tree": b.get("git", {}).get("dirty") is False and d.get("git", {}).get("dirty") is False,
        "baseline_variant": b.get("variant") == RP.BASELINE_VARIANT and suffixless,
        "delay_variant": d.get("variant") == VARIANT_DELAY and d["run_id"].endswith(RUN_ID_SUFFIX_DELAY),
        "reference_path": str(d_ref_path).replace("\\", "/").endswith(BASELINE_RUN_ID + "/newtask.pt") if d_ref_path else False,
        "reference_head_sha": ref_head_path.is_file() and _sha256_file(ref_head_path) == REF_HEAD_SHA,
        "stage1_lineage": all(x.get("backbone_sha256_loaded") == STAGE1_BACKBONE_SHA
                              and x.get("env_ids_sha256") == STAGE1_ENV_IDS_SHA
                              and x.get("fingerprint_sha256") == STAGE1_FINGERPRINT_SHA
                              for x in (b, d)),
    }
    identity_ok = all(identity.values())

    # ---- REP_B ----
    b_vals = [e["val_auc_bsi"] for e in b["per_epoch"]]
    b_losses = [e["train_loss"] for e in b["per_epoch"]]
    reproduction_b = {
        "per_epoch_val": b_vals == BASELINE_PER_EPOCH_VAL,
        "per_epoch_loss": b_losses == BASELINE_PER_EPOCH_LOSS,
        "best_epoch": b.get("best_epoch") == 5,
        "best_val": b.get("best_val_auc_bsi") == BASELINE_AUC_VAL,
        "test": b.get("test_auc_bsi") == BASELINE_AUC_TEST,
        "gate_mean": b.get("gate_mean") == BASELINE_GATE_MEAN,
        "newtask_sha": (runs_root / baseline_run / "newtask.pt").is_file()
                       and _sha256_file(runs_root / baseline_run / "newtask.pt") == REF_HEAD_SHA,
    }
    rep_b_ok = all(reproduction_b.values())

    # ---- FROZEN_ID（三元逐位 + delay 块） ----
    delay_block = d_report.get("delay") or {}
    d_ep1 = d["per_epoch"][0]
    b_ep1 = b["per_epoch"][0]
    rg_after = delay_block.get("requires_grad_after_unfreeze") or {}
    frozen_phase_identity = {
        "ep1_train_loss_triple": d_ep1["train_loss"] == BASELINE_EP1_TRAIN_LOSS
                                 and b_ep1["train_loss"] == BASELINE_EP1_TRAIN_LOSS,
        "ep1_val_triple": d_ep1["val_auc_bsi"] == BASELINE_EP1_VAL
                          and b_ep1["val_auc_bsi"] == BASELINE_EP1_VAL,
        "delay_epochs": delay_block.get("delay_epochs") == DELAY_EPOCHS,
        "unfreeze_epoch": delay_block.get("unfreeze_epoch") == UNFREEZE_EPOCH,
        "alpha_at_construction": delay_block.get("alpha_at_construction") == 0.0,
        "alpha_at_unfreeze": delay_block.get("alpha_at_unfreeze") == 0.0,
        "generator_sha_equal": delay_block.get("generator_sha_at_unfreeze") is not None
                               and delay_block.get("generator_sha_at_unfreeze")
                               == delay_block.get("generator_sha_at_construction"),
        "unfrozen_once_rg": delay_block.get("unfrozen_once") is True
                            and rg_after.get("alpha") is True
                            and rg_after.get("generator") == [True] * 4,
    }
    frozen_ok = all(frozen_phase_identity.values())

    # ---- A 类（两 run） ----
    a_ok = _a_class_ok(_load_json(runs_root / baseline_run / "gate_report.json")) \
        and _a_class_ok(_load_json(runs_root / delay_run / "gate_report.json"))

    # ---- DD（重算 + 记录一致性） ----
    arm = d.get("rp_arm") or {}
    # prompt_report.json 的键名（construction_identity）与审计 probe 形状（construction）归一
    probe_for_gates = {
        "construction": d_report.get("construction_identity") or {},
        "init_forward": d_report.get("init_forward") or {},
        "grad_probe": d_report.get("grad_probe") or [],
        "alpha_final": d_report.get("alpha_final"),
        "delay": d_report.get("delay") or {},
    }
    recomputed, aux = dd_gate_observed(probe=probe_for_gates,
                                       val_stats={"streams": d_report.get("val_stats") or {},
                                                  "gate": d_report.get("gate") or {},
                                                  "dispersion": d_report.get("dispersion") or {}},
                                       params=d_report.get("params") or {},
                                       reference=d_report.get("reference_dispersion") or {})
    recorded_consistent = all(bool((arm.get(g) or {}).get("pass")) == recomputed[g]
                              for g in recomputed)
    mechanism_ok = all(recomputed.values()) and bool((arm.get("M0") or {}).get("pass")) \
        and recorded_consistent

    # ---- CC（对照链：五历史 run 记录 + L 来源文件 + 六轨迹 + 三常量重推） ----
    hist = {tag: _read_metrics(runs_root, run_id) for tag, run_id in
            (("baseline", BASELINE_RUN_ID), ("learnable", LEARNABLE_RUN_ID),
             ("shuf_hist", SHUF_HIST_RUN_ID), ("pinned", PINNED_RUN_ID),
             ("pinned_shuf", PINNED_SHUFFLED_RUN_ID))}
    lt = hist["learnable"]
    pt = hist["pinned"]
    comparator = {
        "hist_baseline": hist["baseline"].get("test_auc_bsi") == BASELINE_AUC_TEST
                         and hist["baseline"].get("best_val_auc_bsi") == BASELINE_AUC_VAL
                         and [e["val_auc_bsi"] for e in hist["baseline"]["per_epoch"]] == BASELINE_PER_EPOCH_VAL,
        "hist_learnable": lt.get("test_auc_bsi") == RPP.CORRECT_RECORD["test_auc_bsi"]
                          and lt.get("best_val_auc_bsi") == RPP.CORRECT_RECORD["best_val_auc_bsi"]
                          and lt.get("variant") == "residual-prompt"
                          and (lt.get("rp_arm") or {}).get("classification") == "VALID_POSITIVE",
        "hist_learnable_metrics_sha": (runs_root / LEARNABLE_RUN_ID / "metrics.json").is_file()
                                      and _sha256_file(runs_root / LEARNABLE_RUN_ID / "metrics.json")
                                      == LEARNABLE_METRICS_SHA,
        "learnable_prompt_report_sha": _sha256_file(runs_root / LEARNABLE_RUN_ID / "prompt_report.json")
                                       == LEARNABLE_PROMPT_REPORT_SHA,
        "learnable_alpha_final": _load_json(runs_root / LEARNABLE_RUN_ID / "prompt_report.json")
                                 .get("alpha_final") == LEARNABLE_ALPHA_FINAL,
        "hist_shuf": hist["shuf_hist"].get("test_auc_bsi") == SHUF_HIST_RECORD["test_auc_bsi"]
                     and (hist["shuf_hist"].get("rp_arm") or {}).get("classification") == "MECHANISM_FAIL"
                     and _sha256_file(runs_root / SHUF_HIST_RUN_ID / "prompt_report.json")
                     == SHUF_HIST_PROMPT_SHA,
        "hist_pinned": pt.get("test_auc_bsi") == PINNED_CORRECT_RECORD["test_auc_bsi"]
                       and pt.get("best_val_auc_bsi") == PINNED_CORRECT_RECORD["best_val_auc_bsi"]
                       and [e["val_auc_bsi"] for e in pt["per_epoch"]] == PINNED_CORRECT_RECORD["per_epoch_val"]
                       and pt.get("variant") == "residual-prompt-pinned"
                       and (pt.get("rp_arm") or {}).get("classification") == "VALID_NEGATIVE",
        "pinned_file_shas": _sha256_file(runs_root / PINNED_RUN_ID / "metrics.json") == PINNED_METRICS_SHA
                            and _sha256_file(runs_root / PINNED_RUN_ID / "newtask.pt") == PINNED_NEWTASK_SHA
                            and _sha256_file(runs_root / PINNED_RUN_ID / "prompt_report.json") == PINNED_PROMPT_REPORT_SHA,
        "hist_pinned_shuf": hist["pinned_shuf"].get("test_auc_bsi") == PINNED_SHUFFLED_RECORD["test_auc_bsi"]
                            and (hist["pinned_shuf"].get("rp_arm") or {}).get("classification") == "VALID_NEGATIVE",
        "delta_l_recompute": (lt.get("test_auc_bsi", 0.0) - hist["baseline"].get("test_auc_bsi", 0.0)) == DELTA_TEST_LEARNABLE,
        "delta_p_recompute": (pt.get("test_auc_bsi", 0.0) - hist["baseline"].get("test_auc_bsi", 0.0)) == DELTA_TEST_PINNED,
        "gap_recompute": (DELTA_TEST_LEARNABLE - DELTA_TEST_PINNED) == GAP,
        "half_gap_target_recompute": (DELTA_TEST_PINNED + HALF_GAP_FRACTION * GAP) == HALF_GAP_TARGET,
        "trajectory_evidence": _trajectory_evidence_ok(runs_root),
    }
    comparator_ok = all(comparator.values())

    # ---- 分量与判定 ----
    b_test, b_val = b["test_auc_bsi"], b["best_val_auc_bsi"]
    d_test, d_val = d["test_auc_bsi"], d["best_val_auc_bsi"]
    delta_d = d_test - b_test
    delta_val_d = d_val - b_val
    first_failure = None
    if not identity_ok:
        first_failure = "IDENTITY_MISMATCH"
    elif not rep_b_ok:
        first_failure = "BASELINE_REPRODUCTION_FAILED"
    elif not frozen_ok:
        first_failure = "FROZEN_PHASE_IDENTITY_FAILED"
    elif not a_ok:
        first_failure = "PROTOCOL_INVALID"
    elif not mechanism_ok:
        first_failure = "MECHANISM_FAIL"
    elif not comparator_ok:
        first_failure = "COMPARATOR_MISMATCH"
    verdict = decide_verdict(invalid_subreason=first_failure, delta_d=delta_d,
                             delta_val_d=delta_val_d)
    secondary = {"class": classify_secondary(delta_d),
                 "boundaries": {"positive_min": SECONDARY_POSITIVE_MIN,
                                "negative_max": SECONDARY_NEGATIVE_MAX}}
    headroom = None
    if secondary["class"] == "NO_CLEAR_IMPROVEMENT":
        headroom = headroom_report(delta_d=delta_d, delta_val_d=delta_val_d, probe=d_report,
                                   delay=delay_block, best_epoch=d.get("best_epoch"),
                                   epochs_run=len(d["per_epoch"]), epochs_recorded=d.get("epochs"))
    return {
        "analyzer": "rp_delay.analyze_runs",
        "inputs": {"baseline_run": baseline_run, "delay_run": delay_run},
        "preconditions": {
            "identity": identity, "identity_ok": identity_ok,
            "reproduction_b": reproduction_b, "rep_b_ok": rep_b_ok,
            "frozen_phase_identity": frozen_phase_identity, "frozen_ok": frozen_ok,
            "a_class_ok": a_ok,
            "mechanism_gates": {**recomputed, "M0": bool((arm.get("M0") or {}).get("pass")),
                                "recorded_consistent": recorded_consistent},
            "mechanism_ok": mechanism_ok,
            "comparator": comparator, "comparator_ok": comparator_ok,
            "all_pass": verdict["verdict"] != VERDICT_INVALID,
        },
        "components": {
            "delta_d": delta_d, "delta_val_d": delta_val_d,
            "delta_l": DELTA_TEST_LEARNABLE, "delta_p": DELTA_TEST_PINNED, "gap": GAP,
            "delta_d_vs_u": delta_d - U_ARM_DELTA,
            "recovery_ratio": delta_d / DELTA_TEST_LEARNABLE,
            "retention_vs_five_seed_mean": delta_d / FIVE_SEED_MEAN_DELTA,
            "gap_closure": (delta_d - DELTA_TEST_PINNED) / GAP,
            "test": {"baseline": b_test, "delay": d_test},
            "val": {"baseline": b_val, "delay": d_val},
            "gate_mean_delay": d.get("gate_mean"),
        },
        "descriptive": {
            "half_gap_met": half_gap_met(delta_d), "half_gap_target": HALF_GAP_TARGET,
            "half_gap_fraction": HALF_GAP_FRACTION,
            "abs_delta_d_ge_historical_0.0055": bool(delta_d >= RP.AUC_TEST_DELTA_MIN),
            "arm_classification": arm.get("classification"), "arm_subreason": arm.get("subreason"),
            "delay_block": delay_block,
            "stopping": {"best_epoch": d.get("best_epoch"), "epochs_run": len(d["per_epoch"]),
                         "epochs_recorded": d.get("epochs"),
                         "right_censored": bool(d.get("best_epoch") == len(d["per_epoch"])
                                                == d.get("epochs"))},
            "context": {"learnable_delta_test": DELTA_TEST_LEARNABLE,
                        "pinned_delta_test": DELTA_TEST_PINNED,
                        "five_seed_mean_delta_test": FIVE_SEED_MEAN_DELTA,
                        "uncond_u_delta_test": U_ARM_DELTA,
                        "budget_chain": BUDGET_CHAIN},
        },
        "verdict": verdict,
        "secondary": secondary,
        "headroom": headroom,
    }


def _trajectory_evidence_ok(runs_root: Path) -> bool:
    """六条学习臂 α 轨迹证据（预注册 §2.2）：probe(ep1) 精确 0/0，probe(ep2) 非零/正。"""
    for tag, spec in LEARNABLE_TRAJECTORIES.items():
        report_path = runs_root / spec["run"] / "prompt_report.json"
        if not report_path.is_file() or _sha256_file(report_path) != spec["prompt_report_sha"]:
            return False
        probe = _load_json(report_path)["grad_probe"]
        if not (probe[0]["alpha"] == 0.0 and probe[0]["generator_grad_norm"] == 0.0):
            return False
        if not (probe[1]["alpha"] != 0.0 and probe[1]["generator_grad_norm"] > 0.0):
            return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="AliCCP 阶段 2 延迟解冻消融分析（预注册 §5；只读两 run 记录 + 历史常量）")
    parser.add_argument("--runs-root", type=Path,
                        default=Path(P.ARTIFACT_ROOT) / "runs")
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--delay-run", type=str, required=True)
    args = parser.parse_args(argv)
    result = analyze_runs(runs_root=args.runs_root, baseline_run=args.baseline_run,
                          delay_run=args.delay_run)
    out_path = Path(args.runs_root) / args.delay_run / "rp_delay_compare.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    v = result["verdict"]
    print(f"verdict={v['verdict']} subreason={v['subreason']} "
          f"delta_d={result['components']['delta_d']!r} "
          f"half_gap_met={result['descriptive']['half_gap_met']} "
          f"secondary={result['secondary']['class']}")
    print(f"report: {out_path}")
    return 0 if v["verdict"] != VERDICT_INVALID else 1


if __name__ == "__main__":
    raise SystemExit(main())
