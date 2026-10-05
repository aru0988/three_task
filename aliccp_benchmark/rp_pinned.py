"""AliCCP 阶段 2 残差 Prompt 的 α 钉死条件对应因果消融（pinned-alpha conditioning；本分支
exp/aliccp-stage2-residual-prompt-alpha-pinned-condition）。

唯一事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-alpha-pinned-condition-design.md

消融语义（预注册 §2）：学习门控 `prompt_gate` 在**两条处理臂**中替换为同一个 fixed signed
`α = PINNED_ALPHA = +0.07101669907569885`（来源 = 已提交 correct seed2 学习臂
`20261004-0325-…-013e105-rpg` 的 `prompt_report.json:alpha_final`，文件 sha 钉死）。α 实现为
**非参数 buffer**（不进入 `named_parameters()` ⇒ 结构性排除于 Adam；训练全程逐位不变；随
checkpoint 落盘）。两臂注入的标量幅度相同，**唯一差异 = 样本↔条件对应**：

  * `F_c`（variant `residual-prompt-pinned`）：条件 = 自身 `dnn_input`（钉死 correct）；
  * `F_s`（variant `residual-prompt-pinned-shuffled`）：条件 = 同 batch 确定性标签无关错排 π 指定的
    `dnn_input[π(i)]`（规格逐字沿用 fbfff09 §2.1：sha256 键推导 + Sattolo；`COND_PERM_SEED=20261005`
    同一张 deck，digest 与历史 shuffled 臂逐位一致 = PG7）。

机制公式与结构性事实（S1 范数受控 / S2 跨流一致 / S3 α=0 恒等）沿用钉死 `residual_prompt.py`（零改动）。

判定（预注册 §5）：`analyze_runs` 只读三 run 记录 → 前置条件（ID/REP/A/PA/PG/CC/DD）→ 因果判定树
（`SAMPLE_CONDITION_SUPPORTED` / `CONDITION_ALIGNMENT_NOT_SUPPORTED` / `INVALID`）+ 两臂二级分类
（`POSITIVE_IMPROVEMENT` / `NO_CLEAR_IMPROVEMENT` / `CLEAR_DEGRADATION`）。学习活性带（原 G5）不适用，
替换为钉死 α 结构门禁 PA1–PA8（§5.3）；历史 +0.0055 判定单列保留。不主张新颖性。
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
from multitaskrec.model import NewTask

# ---- 臂与接线（预注册 §2）----
VARIANT_PINNED = "residual-prompt-pinned"                  # F_c：钉死 α correct-condition
VARIANT_PINNED_SHUFFLED = "residual-prompt-pinned-shuffled"  # F_s：钉死 α shuffled-condition
RUN_ID_SUFFIX_CORRECT = "-rpp"
RUN_ID_SUFFIX_SHUFFLED = "-rpps"
ALL_VARIANTS = (RP.BASELINE_VARIANT, RP.VARIANT, VARIANT_PINNED, VARIANT_PINNED_SHUFFLED)
PINNED_VARIANTS = (VARIANT_PINNED, VARIANT_PINNED_SHUFFLED)


def run_id_suffix_for(variant: str) -> str:
    """臂后缀分派（baseline ⇒ 空；学习 correct ⇒ -rpg；钉死 correct ⇒ -rpp；钉死 shuffled ⇒ -rpps）。"""
    if variant == VARIANT_PINNED:
        return RUN_ID_SUFFIX_CORRECT
    if variant == VARIANT_PINNED_SHUFFLED:
        return RUN_ID_SUFFIX_SHUFFLED
    if variant == RP.VARIANT:
        return RP.RUN_ID_SUFFIX
    return ""


# ---- 冻结判据常量（预注册 §1/§2/§5；只允许在看到结果之前修改）----
PINNED_ALPHA = 0.07101669907569885          # fixed signed α（float32 往返逐位精确；§1.5-pin）
PINNED_ALPHA_SOURCE_RUN = "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"
PINNED_ALPHA_SOURCE_FIELD = "prompt_report.json:alpha_final"
CORRECT_PROMPT_REPORT_SHA = "185df4d0f9910508aeff84ed780e7df4fe8ad4fd6b8451ff429a74927d333c1e"
COND_PERM_SEED = 20261005                  # 错排基数种子（冻结；与 fbfff09 同一张 deck）
MATERIAL_DELTA = 0.001                     # 最小实质差异（= 用户指定；二级分类边界同值）
SECONDARY_POSITIVE_MIN = 0.001             # 二级分类：Δtest ≥ +0.001 ⇒ POSITIVE_IMPROVEMENT（闭）
SECONDARY_NEGATIVE_MAX = -0.02             # Δtest ≤ −0.02 ⇒ CLEAR_DEGRADATION（闭）
PERM_SAMPLE_KEYS = 3
PERM_SAMPLE_LEN = 16

# ---- 钉死 α 结构门禁常量（预注册 §5.3）----
EXPECTED_TRAINABLE_PROMPT_KEYS = ("prompt_generator.0.bias", "prompt_generator.0.weight",
                                  "prompt_generator.2.bias", "prompt_generator.2.weight")
EXPECTED_STATE_EXTRA_KEYS = ("prompt_gate",) + EXPECTED_TRAINABLE_PROMPT_KEYS   # state_dict 新增 5 键
EXPECTED_NEW_PARAMS_TOTAL = 2384           # 2385 − 1（α 不再是参数）
EXPECTED_HEAD_PARAMS = 8129
RATIO_SPREAD_TOL = 1e-6                    # PA6：三流 ratio_mean 两两 spread 上限（S2）
PIN_BOUND_TOL = RP.BOUND_TOL               # PA5：ratio_max ≤ |α| + 1e-6（S1）
PRED_STD_MIN_RATIO = RP.PRED_STD_MIN_RATIO  # PA7：pred_std ≥ 0.5 × ref_pred_std

# ---- 钉死身份（预注册 §1.5；与 verify_pinned_prerun.py 逐位一致）----
STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
STAGE1_BACKBONE_SHA = "e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c"
STAGE1_ENV_IDS_SHA = "5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0"
STAGE1_FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
MODEL_SEED_PINNED = 1688723740
BASELINE_RUN_ID = "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"
CORRECT_RUN_ID = PINNED_ALPHA_SOURCE_RUN    # 学习 correct 臂（只读对照）
SHUF_HIST_RUN_ID = "20261005-0933-p2M-v500k-t1M-m1688723740-short-9d26bc8-rpgs"
REF_HEAD_SHA = "90ee06da387129b50ed7ba0b93742a02db6e87d40d60ae11dbc6b189aeecb94f"
MECHANISM_REL = "aliccp_benchmark/residual_prompt.py"
MECHANISM_PIN_LF_SHA = "b3b93b3aa501277656b57e7e89e5ab38a76343f005d0b312e4dc18c8a98120cc"
BASELINE_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5809347091990792, "test_auc_bsi": 0.5974422649550507,
    "gate_mean": [0.789178, 0.2108221875],
    "per_epoch_val": [0.4645711559431739, 0.48564481224085004, 0.5142344439088465,
                      0.5508809596059387, 0.5809347091990792],
}
CORRECT_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5895572066556973, "test_auc_bsi": 0.6055453782825184,
    "gate_mean": [0.7686165625, 0.23138340625],
    "per_epoch_val": [0.4650886037939688, 0.4904911795680732, 0.5231212372261194,
                      0.5597976191334388, 0.5895572066556973],
    "variant": "residual-prompt", "classification": "VALID_POSITIVE",
    "alpha_final": PINNED_ALPHA,
}
SHUF_HIST_RECORD = {
    "best_epoch": 5, "epochs": 5, "patience": 2, "model_seed": 1688723740,
    "best_val_auc_bsi": 0.5819965357883198, "test_auc_bsi": 0.5989210260551207,
    "gate_mean": [0.7904243125, 0.2095755],
    "per_epoch_val": [0.4646865641709375, 0.48723682075012587, 0.5170954166630655,
                      0.5533860188554449, 0.5819965357883198],
    "variant": "residual-prompt-shuffled", "classification": "MECHANISM_FAIL",
    "alpha_final": 0.010569889098405838,
}
DELTA_TEST_CORRECT = 0.008103113327467715      # 学习 correct 臂 test 增量（钉死，逐位）
DELTA_VAL_CORRECT = 0.008622497456618139
DELTA_TEST_SHUFFLED_HIST = 0.0014787611000699474   # 学习 shuffled 臂 test 增量（钉死，逐位）
HISTORICAL_U1_THRESHOLD = 0.0055               # 历史效用门（§5.7 单列保留，不适用于本实验主判定）

# ---- deck 钉死（预注册 §1.2-A7；F_s 的 PG7 重放比对值）----
DECK_PIN = {
    "base_seed": COND_PERM_SEED,
    "n_perms_total": 1750,
    "per_split": {
        "train": {"n_batches": 1000, "n_rows": 2_000_000,
                  "digest": "50ecd19e5775069bfc1270e26c11f13feff34bfc238d61ae87a5e0f5c9ac4d60"},
        "val": {"n_batches": 250, "n_rows": 500_000,
                "digest": "2d85298c509acc26f2d9ee7654e2fb4accdf8d76d68e280e0946a68171b966c0"},
        "test": {"n_batches": 500, "n_rows": 1_000_000,
                 "digest": "239856ffef561c9a4c5dd19fee383a8c19f218fc92d8ea6aec66cb0acdd622cb"},
    },
    "total_digest": "dcf302c1a80ae7e8023345105e559d7a6daaa1ad492b016c39c22705886599d8",
}
CONTEXT = {                                    # 上下文对照（只读数；§1.2 审计值）
    "learnable_correct": {"run_id": CORRECT_RUN_ID, "delta_test": DELTA_TEST_CORRECT,
                          "delta_val": DELTA_VAL_CORRECT, "alpha_final": PINNED_ALPHA,
                          "classification": "VALID_POSITIVE"},
    "learnable_shuffled": {"run_id": SHUF_HIST_RUN_ID, "delta_test": DELTA_TEST_SHUFFLED_HIST,
                           "delta_val": 0.0010618265892405887, "alpha_final": 0.010569889098405838,
                           "classification": "MECHANISM_FAIL", "shuffled_compare_verdict": "INVALID"},
    "budget_chain": {
        "five_epoch": {"delta_test": DELTA_TEST_CORRECT, "delta_val": DELTA_VAL_CORRECT},
        "ten_epoch": {"delta_test": 0.0074396653390377265,
                      "delta_val": 0.0048363447472773435},
        "twenty_epoch": {"delta_test": 0.004155967377889924,
                         "delta_val": 0.0023134736258212385,
                         "status": "NOT_PERSIST", "claim": True, "ablation_eligible": True},
        "budget_trend_label": "MONOTONE_NARROWING",
    },
}
PRECONDITION_PRIORITY = ("IDENTITY_MISMATCH", "BASELINE_REPRODUCTION_FAILED", "PROTOCOL_INVALID",
                         "PINNED_ALPHA_INVALID", "MECHANISM_FAIL", "COMPARATOR_MISMATCH",
                         "DECK_MISMATCH", "PERMUTATION_INVALID")


# =====================================================================================
# §2.2 错排构造（逐字沿用 fbfff09 §2.1；独立复核按此规格重实现）
# =====================================================================================

def perm_seed_material(split: str, batch_index: int, n: int, base_seed: int = COND_PERM_SEED) -> int:
    """键 → 64-bit 种子（只依赖键；标签/特征/预测一律不参与）。"""
    key = f"{int(base_seed)}|{split}|{int(batch_index)}|{int(n)}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def derange(n: int, split: str, batch_index: int, base_seed: int = COND_PERM_SEED) -> list:
    """Sattolo 洗牌：n≥2 ⇒ 无不动点的均匀 n-轮换（双射）；n≤1 ⇒ 恒等（退化，定义为恒等；batch-size-1 保留）。"""
    n = int(n)
    if n <= 1:
        return list(range(n))
    rng = random.Random(perm_seed_material(split, batch_index, n, base_seed))
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        j = rng.randrange(0, i)
        a[i], a[j] = a[j], a[i]
    return a


class ConditioningShuffler:
    """按 (split, batch_index, n) 键缓存 π；运行期完整性探针 + 逐 split/总 digest（预注册 §5.2；逐字移植）。"""

    def __init__(self, base_seed: int = COND_PERM_SEED):
        self.base_seed = int(base_seed)
        self._perms: dict = {}
        self._tensors: dict = {}
        self._counters = {
            "fixed_points_total": 0,
            "degenerate_identity_batches": 0,
            "bijection_failures": 0,
            "regeneration_mismatches": 0,
            "rng_isolation_violations": 0,
            "key_conflicts": 0,
        }
        self._samples: dict = {}

    def perm(self, split: str, batch_index: int, n: int) -> torch.Tensor:
        split, batch_index, n = str(split), int(batch_index), int(n)
        for (s, b, nn) in self._perms:
            if s == split and b == batch_index and nn != n:
                self._counters["key_conflicts"] += 1
                raise ValueError(f"同键不同 n（键冲突，拒绝出数）: ({split},{batch_index}) n={nn} vs {n}")
        key = (split, batch_index, n)
        if key not in self._perms:
            self._generate(key)
        return self._tensors[key]

    def _generate(self, key) -> None:
        split, batch_index, n = key
        torch_state = torch.get_rng_state()
        py_state = random.getstate()
        values = derange(n, split, batch_index, self.base_seed)
        again = derange(n, split, batch_index, self.base_seed)     # 全新 Random 重生成（PG3 探针）
        if again != values:
            self._counters["regeneration_mismatches"] += 1
        if not torch.equal(torch.get_rng_state(), torch_state) or random.getstate() != py_state:
            self._counters["rng_isolation_violations"] += 1
        if sorted(values) != list(range(n)):
            self._counters["bijection_failures"] += 1
        if n <= 1:
            self._counters["degenerate_identity_batches"] += 1
        else:
            self._counters["fixed_points_total"] += sum(1 for i, v in enumerate(values) if i == v)
        self._perms[key] = values
        self._tensors[key] = torch.tensor(values, dtype=torch.long)
        if len(self._samples.setdefault(split, [])) < PERM_SAMPLE_KEYS:
            head = values[:PERM_SAMPLE_LEN]
            tail_start = max(0, n - 5)
            self._samples[split].append({
                "batch_index": batch_index, "n": n, "perm_head": head,
                "pairs_head": [[i, values[i]] for i in range(min(5, n))],
                "pairs_tail": [[i, values[i]] for i in range(tail_start, n)],
            })

    def report_counters(self) -> dict:
        return dict(self._counters)

    def finalize(self) -> dict:
        """全部用毕后定稿：计数 + 逐 split/总 digest（键排序 ⇒ 与首次生成顺序无关）+ 样例。"""
        per_split = {}
        total = hashlib.sha256()
        for split in sorted({k[0] for k in self._perms}):
            keys = sorted(k for k in self._perms if k[0] == split)
            digest = hashlib.sha256()
            rows = 0
            for (_s, batch_index, n) in keys:
                values = self._perms[(_s, batch_index, n)]
                digest.update(f"{batch_index}|{n}|".encode("utf-8"))
                digest.update(torch.tensor(values, dtype=torch.long).numpy().tobytes())
                rows += n
            per_split[split] = {"n_batches": len(keys), "n_rows": rows, "digest": digest.hexdigest()}
            total.update(f"{split}:".encode("utf-8"))
            total.update(digest.hexdigest().encode("utf-8"))
        report = dict(self._counters)
        report.update({
            "base_seed": self.base_seed,
            "n_perms_total": len(self._perms),
            "per_split": per_split,
            "total_digest": total.hexdigest(),
            "samples": {split: list(items) for split, items in sorted(self._samples.items())},
        })
        return report


def shuffle_coverage_ok(report: dict, budgets: dict, batch_size: int) -> dict:
    """PG2 覆盖核对：逐 split 行数 == 预算、批数 == ceil(预算/批大小)。"""
    coverage = {}
    per_split = report.get("per_split") or {}
    for split in sorted(budgets):
        rec = per_split.get(split)
        if not rec:
            coverage[split] = {"n_batches_ok": False, "n_rows_ok": False}
            continue
        expected_batches = -(-int(budgets[split]) // int(batch_size))
        coverage[split] = {"n_batches_ok": int(rec.get("n_batches", -1)) == expected_batches,
                           "n_rows_ok": int(rec.get("n_rows", -1)) == int(budgets[split])}
    for split in sorted(per_split):
        coverage.setdefault(split, {"n_batches_ok": False, "n_rows_ok": False})
    return coverage


def deck_check(report: dict) -> dict:
    """PG7/DD：F_s deck（逐 split digest ×3 + total + n_perms_total + base_seed）== 历史钉死值。"""
    per_split = report.get("per_split") or {}
    checks = {
        "base_seed": report.get("base_seed") == DECK_PIN["base_seed"],
        "n_perms_total": report.get("n_perms_total") == DECK_PIN["n_perms_total"],
        "total_digest": report.get("total_digest") == DECK_PIN["total_digest"],
    }
    for split, pin in sorted(DECK_PIN["per_split"].items()):
        rec = per_split.get(split) or {}
        checks[f"{split}.digest"] = rec.get("digest") == pin["digest"]
        checks[f"{split}.n_batches_rows"] = (rec.get("n_batches"), rec.get("n_rows")) == \
            (pin["n_batches"], pin["n_rows"])
    return {"checks": checks, "pass": all(checks.values())}


# =====================================================================================
# §2.1/§2.3 钉死 α 头（correct + shuffled；只重写 α 的学习性与条件对应）
# =====================================================================================

_COND_PERM_REQUIRED = object()


class PinnedAlphaResidualPromptNewTask(RP.ResidualPromptNewTask):
    """钉死 α correct-condition 头：`prompt_gate` 由零初始化参数替换为**非参数 buffer**（= PINNED_ALPHA）。

    构造路径逐位同 `ResidualPromptNewTask`（共享参数/RNG 端点/生成器初始化不变）；`prompt_deltas`
    与 forward 逐字继承 ⇒ 公式 `α · (||h||/√d) · m` 不变，唯一差异 = α 不可学习且恒为钉死值。
    """

    def __init__(self, *args, pinned_alpha: float = PINNED_ALPHA, **kwargs):
        super().__init__(*args, **kwargs)
        del self.prompt_gate                                   # 移除零初始化参数（不进 named_parameters）
        self.register_buffer("prompt_gate", torch.tensor(float(pinned_alpha), dtype=torch.float32))


class PinnedAlphaShuffledResidualPromptNewTask(PinnedAlphaResidualPromptNewTask):
    """钉死 α shuffled-condition 头：`prompt_deltas` 一律以 `dnn_input[cond_perm]` 为条件输入；其余逐字继承。

    `cond_perm` 为**必需参数**（哨兵缺省 ⇒ 显式 ValueError，拒绝静默回退为正确条件）；基头
    `super().forward` 仍接收**正确的** `dnn_input`（projection/gate 不受影响）。错排器不在此类上
    （由 bench 持有并经参数传入）⇒ state_dict/参数/构造 RNG 与钉死 correct 头逐位一致。
    """

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs, cond_perm=_COND_PERM_REQUIRED):
        if cond_perm is _COND_PERM_REQUIRED:
            raise ValueError("pinned-shuffled 臂要求显式 cond_perm（拒绝静默回退为正确条件）")
        if not torch.is_tensor(cond_perm) or cond_perm.dtype != torch.long \
                or cond_perm.dim() != 1 or int(cond_perm.shape[0]) != int(dnn_input.shape[0]):
            raise ValueError(f"cond_perm 必须是 shape=[B] 的 long 张量（B={int(dnn_input.shape[0])}）")
        cond = dnn_input[cond_perm]
        deltas, _ = self.prompt_deltas(cond, [gen_rep, *spec_reps])
        gen_rep_p = gen_rep + deltas[0]
        spec_reps_p = [spec + delta for spec, delta in zip(spec_reps, deltas[1:])]
        # 注意：必须显式委托 NewTask.forward——`super()` 会落到 ResidualPromptNewTask.forward（二次注入）。
        # 恒等置换时本方法 ≡ 钉死 correct 头 forward（逐位；等价测试钉死）。
        return NewTask.forward(self, dnn_input, gen_rep_p, spec_reps_p, env_embs)


def build_pinned_newtask(*, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                         hidden: int = RP.PROMPT_HIDDEN, pinned_alpha: float = PINNED_ALPHA):
    return PinnedAlphaResidualPromptNewTask(
        input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=tower_dnn_hidden_units,
        reg_dnn=reg_dnn, device=device, hidden=hidden, pinned_alpha=pinned_alpha)


def build_pinned_shuffled_newtask(*, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                                  hidden: int = RP.PROMPT_HIDDEN, pinned_alpha: float = PINNED_ALPHA):
    return PinnedAlphaShuffledResidualPromptNewTask(
        input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=tower_dnn_hidden_units,
        reg_dnn=reg_dnn, device=device, hidden=hidden, pinned_alpha=pinned_alpha)


def forward_with_conditioning(newtask, dnn_input, gen_rep, spec_reps, env_embs,
                              shuffler, split, batch_index):
    """统一前向分派：`shuffler=None`（baseline/F_c）⇒ 与钉死调用逐字相同。"""
    if shuffler is None:
        return newtask(dnn_input, gen_rep, spec_reps, env_embs)
    if split is None:
        raise ValueError("pinned-shuffled 分派必须给出 split")
    perm = shuffler.perm(split, batch_index, int(dnn_input.shape[0])).to(dnn_input.device)
    return newtask(dnn_input, gen_rep, spec_reps, env_embs, cond_perm=perm)


class PinnedPromptAudit(RP.PromptAudit):
    """PA1–PA4 运行期审计：构造身份（G1 等价，继承）+ α 钉死记录 + 优化器排除 + 零 α 注入纯度探针。

    - 构造记录：α 是 buffer（非参数）、`requires_grad=False`、不在 `named_parameters()`、
      构造值 == PINNED_ALPHA；优化器 param_groups 覆盖全部 named_parameters 且不含 α（对象同一性）。
    - 首 batch 探针（PA4）：(i) α 临时置 0（恢复后逐位校验）⇒ 与同共享权重参照头**逐位相等**；
      (ii) α 钉死值下与参照头 `max_abs_diff > 0`（干预自首步即激活）。F_s 用真实 train/0 π。
    """

    def __init__(self, *args, shuffler=None, optimizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._shuffler = shuffler
        self.shuffle_init_perm_used = None
        gate = self.newtask.prompt_gate
        if optimizer is None:
            optimizer_excludes = optimizer_covers = optimizer_total = None
        else:
            opt_params = [p for group in optimizer.param_groups for p in group["params"]]
            optimizer_excludes = not any(p is gate for p in opt_params)
            optimizer_covers = ({id(p) for p in opt_params}
                                == {id(p) for _, p in self.newtask.named_parameters()})
            optimizer_total = len(opt_params)
        self.pin = {
            "pinned_alpha": PINNED_ALPHA,
            "source_run": PINNED_ALPHA_SOURCE_RUN,
            "source_field": PINNED_ALPHA_SOURCE_FIELD,
            "source_file_sha256": CORRECT_PROMPT_REPORT_SHA,
            "is_parameter": isinstance(gate, torch.nn.Parameter),
            "requires_grad": bool(gate.requires_grad),
            "in_named_parameters": any(name == "prompt_gate"
                                       for name, _ in self.newtask.named_parameters()),
            "alpha_at_construction": float(gate.detach()),
            "optimizer_excludes_prompt_gate": optimizer_excludes,
            "optimizer_covers_named_parameters": optimizer_covers,
            "optimizer_params_total": optimizer_total,
        }

    @torch.no_grad()
    def init_forward_check(self, dnn_input, gen_rep, spec_reps, env_embs) -> None:
        """只在真实第一个训练 batch 上执行一次（幂等）。"""
        if self.init_forward is not None:
            return
        perm = None
        if self._shuffler is not None:
            n = int(dnn_input.shape[0])
            perm = self._shuffler.perm("train", 0, n).to(dnn_input.device)
            self.shuffle_init_perm_used = {"split": "train", "batch_index": 0, "n": n}
        reference = self._reference.to(dnn_input.device).eval()
        was_training = self.newtask.training
        self.newtask.eval()
        gate = self.newtask.prompt_gate
        saved = gate.detach().clone()
        try:
            gate.zero_()                                        # PA4-i：零 α 注入纯度
            out_zero = self.newtask(dnn_input, gen_rep, spec_reps, env_embs) if perm is None \
                else self.newtask(dnn_input, gen_rep, spec_reps, env_embs, cond_perm=perm)
            out_reference = reference(dnn_input, gen_rep, spec_reps, env_embs)
            gate.copy_(saved)                                   # 恢复（逐位校验）
            restored_exact = bool(torch.equal(gate, saved)) and float(gate) == PINNED_ALPHA
            out_pinned = self.newtask(dnn_input, gen_rep, spec_reps, env_embs) if perm is None \
                else self.newtask(dnn_input, gen_rep, spec_reps, env_embs, cond_perm=perm)
        finally:
            gate.copy_(saved)
            self.newtask.train(was_training)
        self.init_forward = {
            "zero_alpha_bit_identical": bool(torch.equal(out_zero, out_reference)),
            "zero_alpha_max_abs_diff": float((out_zero - out_reference).abs().max()),
            "pinned_max_abs_diff": float((out_pinned - out_reference).abs().max()),
            "pinned_differs_from_reference": not bool(torch.equal(out_pinned, out_reference)),
            "alpha_restored_exact": restored_exact,
            "alpha_after_check": float(gate.detach()),
            "n_samples": int(out_pinned.numel()),
            "cond_perm": self.shuffle_init_perm_used,
        }

    def result(self, alpha_final: float) -> dict:
        base = super().result(alpha_final)
        base["pin"] = dict(self.pin)
        base["pin"]["alpha_after_training"] = float(self.newtask.prompt_gate.detach())
        base["pin"]["pin_unchanged_after_training"] = \
            float(self.newtask.prompt_gate.detach()) == PINNED_ALPHA
        return base


@torch.no_grad()
def evaluate_prompt_diagnostics_shuffled(newtask, backbone, loader, device, shuffler,
                                         split: str = "val") -> dict:
    """钉死诊断循环的错排版：deltas/preds 一律以 `dnn_input[perm]` 为条件（逐字移植 fbfff09）。"""
    newtask.eval()
    backbone.eval()
    stats = RP.PromptStats()
    preds = []
    for step, (_, _, _, features) in enumerate(loader):
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, gen_rep, spec_reps, env_embs = backbone.get_infos(features)
        reps = [gen_rep, *spec_reps]
        perm = shuffler.perm(split, step, int(dnn_input.shape[0])).to(dnn_input.device)
        cond = dnn_input[perm]
        deltas, m = newtask.prompt_deltas(cond, reps)
        stats.update(deltas, reps, RP.effective_gate(m, newtask.prompt_gate.detach()))
        preds.append(newtask(dnn_input, gen_rep, spec_reps, env_embs,
                             cond_perm=perm).detach().float().cpu())
    result = stats.result()
    pred = torch.cat(preds).double()
    result["dispersion"] = {
        "pred_mean": float(pred.mean()), "pred_std": float(pred.std(unbiased=False)),
        "pred_min": float(pred.min()), "pred_max": float(pred.max()),
        "pred_q05": float(torch.quantile(pred, 0.05)), "pred_q50": float(torch.quantile(pred, 0.50)),
        "pred_q95": float(torch.quantile(pred, 0.95)), "n_samples": int(pred.numel())}
    result["alpha_final"] = float(newtask.prompt_gate.detach())
    return result


# =====================================================================================
# §5.2/§5.3 置换门禁 PG1–PG7 + 钉死 α 结构机制门禁 PA1–PA8
# =====================================================================================

def pg6_evidence() -> dict:
    """PG6（correct 默认逐位不变）的构建期证据：机制文件 LF sha + 守卫测试名钉死。"""
    path = Path(__file__).resolve().parent / "residual_prompt.py"
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {
        "mechanism_file": MECHANISM_REL,
        "mechanism_file_lf_sha256": digest,
        "mechanism_pin_lf_sha256": MECHANISM_PIN_LF_SHA,
        "mechanism_byte_identical": digest == MECHANISM_PIN_LF_SHA,
        "guards": ["test_residual_prompt_pinned.TestMechanismByteIdentical",
                   "test_residual_prompt_pinned.TestWiringReconstructEquality",
                   "test_residual_prompt_pinned.TestPinnedHeadSemantics."
                   "test_identity_perm_equivalent_to_pinned_correct_head"],
    }


def permutation_gates(report: dict, coverage: dict) -> dict:
    """PG1–PG7（PG1–PG5 由运行期计数机械重算；PG6 由机制文件哈希；PG7 = deck 重放钉死）。"""
    ints = {k: int(report.get(k, -1)) for k in
            ("fixed_points_total", "degenerate_identity_batches", "bijection_failures",
             "regeneration_mismatches", "rng_isolation_violations", "key_conflicts")}
    pg1 = ints["fixed_points_total"] == 0
    pg2 = ints["bijection_failures"] == 0 and bool(coverage) and all(
        v["n_batches_ok"] and v["n_rows_ok"] for v in coverage.values())
    pg3 = ints["regeneration_mismatches"] == 0
    pg4 = ints["rng_isolation_violations"] == 0
    pg5 = ints["key_conflicts"] == 0
    pg6 = pg6_evidence()
    deck = deck_check(report)
    return {
        "PG1": {"pass": pg1, "rule": "运行期每个 π（n≥2）不动点数 == 0",
                "observed": {"fixed_points_total": ints["fixed_points_total"],
                             "degenerate_identity_batches": ints["degenerate_identity_batches"]}},
        "PG2": {"pass": pg2, "rule": "每 π 双射 ∧ 逐 split 条件行覆盖 == 预算、批数 == ceil(预算/批大小)",
                "observed": {"bijection_failures": ints["bijection_failures"], "coverage": coverage}},
        "PG3": {"pass": pg3, "rule": "每个 π 以全新 Random 重生成逐位一致（+独立复核重推 digest）",
                "observed": {"regeneration_mismatches": ints["regeneration_mismatches"]}},
        "PG4": {"pass": pg4, "rule": "π 生成前后 torch 全局 RNG 与 python 全局 RNG 逐位不变",
                "observed": {"rng_isolation_violations": ints["rng_isolation_violations"]}},
        "PG5": {"pass": pg5, "rule": "键无冲突（键唯一性）+ 独立复核以键重推 digest（标签无关）",
                "observed": {"key_conflicts": ints["key_conflicts"]}},
        "PG6": {"pass": bool(pg6["mechanism_byte_identical"]),
                "rule": "机制文件逐字节 == 钉死 blob ∧ shuffler=None 恒等分派 ∧ 等价测试",
                "observed": pg6},
        "PG7": {"pass": bool(deck["pass"]),
                "rule": "F_s deck（逐 split digest ×3 + total + n_perms_total）== 历史钉死值（同一 COND_PERM_SEED）",
                "observed": deck["checks"]},
    }


def pinned_mechanism_gates(*, probe: dict, val_stats: dict, params: dict, reference: dict,
                           expected_reference_auc: float = RP.BASELINE_AUC_VAL,
                           expected_reference_pred_std: float = RP.REFERENCE_PRED_STD) -> dict:
    """M0 + PA1–PA8（预注册 §5.3；判据先于结果写死，不得事后修改）。

    - M0：参照头身份（in-run 复算 val AUC/pred_std == 钉死值）。
    - PA1：α 钉死身份（buffer/非参数/requires_grad False/不在 named_parameters；构造与训练后 == 钉死值）。
    - PA2：优化器排除（不含 α 且覆盖全部参数；逐 epoch 探针无 α 梯度且 α 恒为钉死值）。
    - PA3：头身份（共享参数逐位/RNG 端点/新增 state_dict 键恰 5：4 参数 + α buffer）。
    - PA4：注入纯度（零 α 逐位恒等 + 恢复逐位）+ 激活（钉死 α 下与参照头有差）。
    - PA5：范数界（S1：ratio_max ≤ |α| + 1e-6）。
    - PA6：跨流一致（S2：三流 ratio_mean spread ≤ 1e-6）。
    - PA7：无坍缩（pred_std ≥ 0.5 × ref_pred_std）。
    - PA8：预算/钉死核算（可训练 prompt 键恰 4；new_params_total == 2384；head_params == 8129）。
    """
    construction = probe.get("construction") or {}
    init_forward = probe.get("init_forward") or {}
    records = probe.get("grad_probe") or []
    pin = probe.get("pin") or {}
    alpha_final = float(probe.get("alpha_final") or 0.0)
    gate = val_stats.get("gate") or {}
    dispersion = val_stats.get("dispersion") or {}
    streams = val_stats.get("streams") or {}

    reference = reference or {}
    reference_dispersion = reference.get("pred_dispersion") or {}
    ref_auc = reference.get("val_auc")
    ref_std = reference_dispersion.get("pred_std")
    m0_pass = bool(ref_auc is not None and ref_std is not None
                   and abs(float(ref_auc) - expected_reference_auc) <= RP.REFERENCE_IDENTITY_TOL
                   and abs(float(ref_std) - expected_reference_pred_std) <= RP.REFERENCE_IDENTITY_TOL)

    pa1_pass = bool(pin.get("is_parameter") is False
                    and pin.get("requires_grad") is False
                    and pin.get("in_named_parameters") is False
                    and pin.get("alpha_at_construction") == PINNED_ALPHA
                    and alpha_final == PINNED_ALPHA
                    and pin.get("alpha_after_training") == PINNED_ALPHA
                    and pin.get("pin_unchanged_after_training") is True)
    pa2_pass = bool(pin.get("optimizer_excludes_prompt_gate") is True
                    and pin.get("optimizer_covers_named_parameters") is True
                    and records
                    and all(r.get("alpha_grad_norm") is None for r in records)
                    and all(float(r.get("alpha") or 0.0) == PINNED_ALPHA for r in records))
    pa3_pass = bool(construction.get("shared_params_bit_identical")
                    and construction.get("global_rng_endpoint_identical")
                    and construction.get("extra_keys") == sorted(EXPECTED_STATE_EXTRA_KEYS))
    pa4_pass = bool(init_forward.get("zero_alpha_bit_identical")
                    and init_forward.get("alpha_restored_exact")
                    and float(init_forward.get("zero_alpha_max_abs_diff") or 0.0) == 0.0
                    and init_forward.get("pinned_differs_from_reference"))
    ratio_max = [float(r) for r in (streams.get("ratio_max") or [1e9])]
    pa5_pass = all(r <= abs(PINNED_ALPHA) + PIN_BOUND_TOL for r in ratio_max)
    ratio_mean = [float(r) for r in (streams.get("ratio_mean") or [])]
    spread = (max(ratio_mean) - min(ratio_mean)) if ratio_mean else float("inf")
    pa6_pass = bool(ratio_mean) and spread <= RATIO_SPREAD_TOL
    pred_std = float(dispersion.get("pred_std") or 0.0)
    pa7_pass = bool(pred_std > 0.0 and ref_std is not None and float(ref_std) > 0.0
                    and pred_std >= PRED_STD_MIN_RATIO * float(ref_std))
    names = sorted(str(item["name"]) for item in (params.get("new_param_list") or []))
    pa8_pass = bool(names == sorted(EXPECTED_TRAINABLE_PROMPT_KEYS)
                    and int(params.get("new_params_total", -1)) == EXPECTED_NEW_PARAMS_TOTAL
                    and int(params.get("head_params", -1)) == EXPECTED_HEAD_PARAMS)

    return {
        "M0": {"pass": m0_pass,
               "rule": f"|ref_val_auc - {expected_reference_auc}| <= {RP.REFERENCE_IDENTITY_TOL} AND "
                       f"|ref_pred_std - {expected_reference_pred_std}| <= {RP.REFERENCE_IDENTITY_TOL}",
               "observed": {"ref_val_auc": ref_auc, "ref_pred_std": ref_std}},
        "PA1": {"pass": pa1_pass,
                "rule": "prompt_gate 非参数 buffer ∧ requires_grad False ∧ 不在 named_parameters ∧ "
                        "构造/训练后 == PINNED_ALPHA（精确）",
                "observed": {"is_parameter": pin.get("is_parameter"),
                             "requires_grad": pin.get("requires_grad"),
                             "in_named_parameters": pin.get("in_named_parameters"),
                             "alpha_at_construction": pin.get("alpha_at_construction"),
                             "alpha_after_training": pin.get("alpha_after_training"),
                             "alpha_final": alpha_final,
                             "pinned_alpha": PINNED_ALPHA}},
        "PA2": {"pass": pa2_pass,
                "rule": "optimizer 不含 α ∧ 覆盖全部 named_parameters ∧ 逐 epoch 探针 alpha_grad_norm None ∧ "
                        "α 恒为 PINNED_ALPHA",
                "observed": {"optimizer_excludes_prompt_gate": pin.get("optimizer_excludes_prompt_gate"),
                             "optimizer_covers_named_parameters": pin.get("optimizer_covers_named_parameters"),
                             "optimizer_params_total": pin.get("optimizer_params_total"),
                             "probe_alpha_grad_norms": [r.get("alpha_grad_norm") for r in records]}},
        "PA3": {"pass": pa3_pass,
                "rule": "shared bit-identical AND rng endpoint identical AND extra state keys exact "
                        "(4 generator params + prompt_gate buffer)",
                "observed": {"shared_params_bit_identical": construction.get("shared_params_bit_identical"),
                             "global_rng_endpoint_identical": construction.get("global_rng_endpoint_identical"),
                             "extra_keys": construction.get("extra_keys")}},
        "PA4": {"pass": pa4_pass,
                "rule": "zero-alpha forward bit-identical to reference ∧ alpha restored exact ∧ "
                        "pinned-alpha forward differs from reference (active)",
                "observed": {"zero_alpha_bit_identical": init_forward.get("zero_alpha_bit_identical"),
                             "zero_alpha_max_abs_diff": init_forward.get("zero_alpha_max_abs_diff"),
                             "alpha_restored_exact": init_forward.get("alpha_restored_exact"),
                             "pinned_max_abs_diff": init_forward.get("pinned_max_abs_diff"),
                             "pinned_differs_from_reference": init_forward.get("pinned_differs_from_reference")}},
        "PA5": {"pass": pa5_pass, "rule": f"every stream ratio_max <= |PINNED_ALPHA| + {PIN_BOUND_TOL}",
                "observed": {"ratio_max": ratio_max, "alpha_abs": abs(PINNED_ALPHA)}},
        "PA6": {"pass": pa6_pass, "rule": f"three-stream ratio_mean spread <= {RATIO_SPREAD_TOL}",
                "observed": {"ratio_mean": ratio_mean, "spread": spread}},
        "PA7": {"pass": pa7_pass, "rule": f"pred_std > 0 AND pred_std >= {PRED_STD_MIN_RATIO} * ref_pred_std",
                "observed": {"pred_std": pred_std, "ref_pred_std": ref_std}},
        "PA8": {"pass": pa8_pass,
                "rule": f"4 exact trainable prompt keys AND new_params_total == {EXPECTED_NEW_PARAMS_TOTAL} "
                        f"AND head_params == {EXPECTED_HEAD_PARAMS}",
                "observed": {"names": names, "new_params_total": params.get("new_params_total"),
                             "head_params": params.get("head_params"),
                             "state_extra_keys": sorted(EXPECTED_STATE_EXTRA_KEYS)}},
    }


def pinned_arm_verdict(*, auc_test, auc_val, probe, val_stats, params, reference,
                       protocol_ok=True, variant, shuffle_report=None, budgets=None,
                       batch_size=None) -> dict:
    """PA1–PA8 + U1/U2 判定与结局分类（预注册 §5.3；固定阈值与钉死值口径）。

    F_s（variant=pinned-shuffled）另附 shuffle 段（PG1–PG7）；PG 任一 FAIL ⇒ 分类覆写为
    `MECHANISM_FAIL / PERMUTATION_INVALID`（不覆盖更早的结构失败）。
    """
    gates = pinned_mechanism_gates(probe=probe, val_stats=val_stats, params=params,
                                   reference=reference)
    delta_test = float(auc_test) - RP.BASELINE_AUC_TEST
    delta_val = float(auc_val) - RP.BASELINE_AUC_VAL
    u1_pass = bool(delta_test >= RP.AUC_TEST_DELTA_MIN)
    u2_pass = bool(delta_val > RP.AUC_VAL_DIRECTION_MIN)
    u_pass = bool(u1_pass and u2_pass)

    if not gates["M0"]["pass"]:
        classification, subreason = "MECHANISM_FAIL", "REFERENCE_IDENTITY"
    elif not (gates["PA1"]["pass"] and gates["PA2"]["pass"] and gates["PA8"]["pass"]):
        classification, subreason = "MECHANISM_FAIL", "PINNED_ALPHA_INVALID"
    elif not (gates["PA3"]["pass"] and gates["PA4"]["pass"]):
        classification, subreason = "MECHANISM_FAIL", "INVALID_IMPLEMENTATION"
    elif not gates["PA5"]["pass"]:
        classification, subreason = "MECHANISM_FAIL", "NORM_BOUND_VIOLATION"
    elif not gates["PA6"]["pass"]:
        classification, subreason = "MECHANISM_FAIL", "CROSS_STREAM_INCONSISTENT"
    elif not gates["PA7"]["pass"]:
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
    if variant == VARIANT_PINNED_SHUFFLED and shuffle_report is not None:
        if budgets and batch_size:
            coverage = shuffle_coverage_ok(shuffle_report, budgets=budgets, batch_size=batch_size)
        else:                                     # 无预算上下文（单元测试）⇒ 仅列 per_split 键
            coverage = {split: {"n_batches_ok": True, "n_rows_ok": True}
                        for split in (shuffle_report.get("per_split") or {})}
        pgates = permutation_gates(shuffle_report, coverage)
        shuffle = {"base_seed": shuffle_report.get("base_seed"), "report": shuffle_report,
                   "coverage": coverage, "gates": pgates,
                   "pass": all(g["pass"] for g in pgates.values())}
        if not shuffle["pass"] and arm["classification"] != "MECHANISM_FAIL":
            arm["classification"], arm["subreason"], arm["pass"] = \
                "MECHANISM_FAIL", "PERMUTATION_INVALID", False
        arm["shuffle"] = shuffle
    return arm


# =====================================================================================
# §5.4/§5.5/§5.6 判定（纯函数 + 只读分析器）
# =====================================================================================

def causal_verdict(*, gap: float, delta_val_gap: float, preconditions: dict | None = None) -> dict:
    """预注册 §5.5 因果判定树（先于结果写死）；preconditions 按插入顺序为首个失败项给出 subreason。"""
    gap, delta_val_gap = float(gap), float(delta_val_gap)
    components = {
        "gap": gap, "delta_val_gap": delta_val_gap,
        "gap_material_positive": gap >= MATERIAL_DELTA,
        "validation_agreement": delta_val_gap > 0.0,
        "shuffled_matches_or_exceeds_correct": gap <= 0.0,
    }
    for name, failed in (preconditions or {}).items():
        if failed:
            return {"verdict": "INVALID", "subreason": name, "components": components}
    if gap >= MATERIAL_DELTA and delta_val_gap > 0.0:
        return {"verdict": "SAMPLE_CONDITION_SUPPORTED", "subreason": None, "components": components}
    return {"verdict": "CONDITION_ALIGNMENT_NOT_SUPPORTED",
            "subreason": ("GAP_BELOW_MATERIALITY" if gap < MATERIAL_DELTA
                          else "VALIDATION_DISAGREEMENT"),
            "components": components}


def secondary_classification(delta_test: float) -> str:
    """预注册 §5.6 二级分类（边界先于结果写死；±闭端按预注册）。"""
    delta_test = float(delta_test)
    if delta_test >= SECONDARY_POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if delta_test <= SECONDARY_NEGATIVE_MAX:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def headroom_assessment(*, delta_test: float, delta_val: float, arm_record: dict,
                        prompt_doc: dict) -> dict:
    """预注册 §5.6：NO_CLEAR_IMPROVEMENT 的机械 headroom（仅当落入该区间时适用）。"""
    if secondary_classification(delta_test) != "NO_CLEAR_IMPROVEMENT":
        return {"applicable": False}
    gate = prompt_doc.get("gate") or {}
    streams = prompt_doc.get("val_stats") or {}
    ratio_mean = [float(r) for r in (streams.get("ratio_mean") or [])]
    active = (abs(float(prompt_doc.get("alpha_final") or 0.0)) > 0.0
              and float(gate.get("geff_std") or 0.0) > 0.0
              and bool(ratio_mean) and all(r > 0.0 for r in ratio_mean))
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


def _sign(x: float) -> int:
    return (float(x) > 0.0) - (float(x) < 0.0)


def analyze_runs(*, root, baseline_run, arm_correct_run, arm_shuffled_run) -> dict:
    """只读 B/F_c/F_s + 三个历史对照 run 记录 → 预注册 §5 全量判定（不改任何 run 产物）。"""
    root = Path(root)
    base = _read_json(root / "runs" / baseline_run / "metrics.json")
    base_gate = _read_json(root / "runs" / baseline_run / "gate_report.json")
    fc = _read_json(root / "runs" / arm_correct_run / "metrics.json")
    fc_gate = _read_json(root / "runs" / arm_correct_run / "gate_report.json")
    fc_prompt = _read_json(root / "runs" / arm_correct_run / "prompt_report.json")
    fs = _read_json(root / "runs" / arm_shuffled_run / "metrics.json")
    fs_gate = _read_json(root / "runs" / arm_shuffled_run / "gate_report.json")
    fs_cfg = _read_json(root / "runs" / arm_shuffled_run / "config.json")
    fs_prompt = _read_json(root / "runs" / arm_shuffled_run / "prompt_report.json")
    ctx_base = _read_json(root / "runs" / BASELINE_RUN_ID / "metrics.json")
    ctx_corr = _read_json(root / "runs" / CORRECT_RUN_ID / "metrics.json")
    ctx_corr_prompt = _read_json(root / "runs" / CORRECT_RUN_ID / "prompt_report.json")
    ctx_shuf = _read_json(root / "runs" / SHUF_HIST_RUN_ID / "metrics.json")

    # ---- §5.1-ID：identity ----
    expected_ref_path = root / "runs" / BASELINE_RUN_ID / "newtask.pt"
    fc_ref_recorded = (fc_prompt.get("reference_dispersion") or {}).get("newtask_checkpoint")
    fs_ref_recorded = (fs_prompt.get("reference_dispersion") or {}).get("newtask_checkpoint")
    ref_sha = _sha256_file(expected_ref_path) if expected_ref_path.is_file() else None
    runs = (base, fc, fs)
    identity = {
        "stage1_same": base["stage1_id"] == fc["stage1_id"] == fs["stage1_id"],
        "stage1_pinned": base["stage1_id"] == STAGE1_ID,
        "seed_same_pinned": base["model_seed"] == fc["model_seed"] == fs["model_seed"]
        == MODEL_SEED_PINNED,
        "epochs": base["epochs"] == fc["epochs"] == fs["epochs"] == 5,
        "patience": base["patience"] == fc["patience"] == fs["patience"] == 2,
        "tag": base["tag"] == fc["tag"] == fs["tag"] == "short",
        "dirty_false": all(not r.get("git", {}).get("dirty") for r in runs),
        "baseline_variant": base.get("variant") == RP.BASELINE_VARIANT,
        "baseline_no_suffix": not base["run_id"].endswith(tuple(s for s in
                                                                (RUN_ID_SUFFIX_CORRECT,
                                                                 RUN_ID_SUFFIX_SHUFFLED,
                                                                 RP.RUN_ID_SUFFIX) if s)),
        "arm_correct_variant": fc.get("variant") == VARIANT_PINNED,
        "arm_correct_suffix": fc["run_id"].endswith(RUN_ID_SUFFIX_CORRECT),
        "arm_shuffled_variant": fs.get("variant") == VARIANT_PINNED_SHUFFLED,
        "arm_shuffled_suffix": fs["run_id"].endswith(RUN_ID_SUFFIX_SHUFFLED),
        "arms_ref_path_pinned": (fc_ref_recorded is not None and fs_ref_recorded is not None
                                 and Path(str(fc_ref_recorded)) == expected_ref_path
                                 and Path(str(fs_ref_recorded)) == expected_ref_path),
        "arms_ref_sha": ref_sha == REF_HEAD_SHA,
        "stage1_shas": all(r.get("backbone_sha256_loaded") == STAGE1_BACKBONE_SHA
                           and r.get("env_ids_sha256") == STAGE1_ENV_IDS_SHA
                           and r.get("fingerprint_sha256") == STAGE1_FINGERPRINT_SHA
                           for r in runs),
    }

    # ---- §5.1-REP：B 与历史基线记录逐位一致 ----
    base_pe = [e["val_auc_bsi"] for e in base["per_epoch"]]
    baseline_newtask = root / "runs" / baseline_run / "newtask.pt"
    reproduction = {
        "per_epoch_val": base_pe == BASELINE_RECORD["per_epoch_val"],
        "best_epoch": base["best_epoch"] == BASELINE_RECORD["best_epoch"],
        "best_val": base["best_val_auc_bsi"] == BASELINE_RECORD["best_val_auc_bsi"],
        "test": base["test_auc_bsi"] == BASELINE_RECORD["test_auc_bsi"],
        "gate_mean": base["gate_mean"] == BASELINE_RECORD["gate_mean"],
        "newtask_sha": baseline_newtask.is_file() and _sha256_file(baseline_newtask) == REF_HEAD_SHA,
    }

    # ---- §5.1-A ----
    protocol_a = {"baseline": _a_class_ok(base_gate), "arm_correct": _a_class_ok(fc_gate),
                  "arm_shuffled": _a_class_ok(fs_gate)}

    # ---- §5.1-PA：两处理臂的 PA1–PA8（含 M0；记录值机械重算，不信任记录布尔）----
    def _pa_status(metrics_doc):
        arm = metrics_doc.get("rp_arm") or {}
        return {g: bool((arm.get(g) or {}).get("pass"))
                for g in ("M0", "PA1", "PA2", "PA3", "PA4", "PA5", "PA6", "PA7", "PA8")}

    mechanism = {"arm_correct": _pa_status(fc), "arm_shuffled": _pa_status(fs)}

    # ---- §5.1-PG：由 F_s 记录重算（含覆盖与 deck；不信任记录布尔）----
    shuffle_block = fs_prompt.get("shuffle") or {}
    shuffle_rep = shuffle_block.get("report") or {}
    coverage = shuffle_coverage_ok(shuffle_rep, budgets=fs_cfg["budgets"],
                                   batch_size=fs_cfg["batch_size"])
    perm_gates = permutation_gates(shuffle_rep, coverage)
    recorded_gates = (shuffle_block.get("gates") or {})
    recorded_consistent = all(
        bool((recorded_gates.get(g) or {}).get("pass")) == perm_gates[g]["pass"]
        for g in perm_gates) and bool(shuffle_block.get("pass")) == all(
        g["pass"] for g in perm_gates.values())
    deck = deck_check(shuffle_rep)

    # ---- §5.1-CC：历史对照链 + α 钉死来源 ----
    comparator = {
        "baseline_values": (ctx_base["test_auc_bsi"] == BASELINE_RECORD["test_auc_bsi"]
                            and ctx_base["best_val_auc_bsi"] == BASELINE_RECORD["best_val_auc_bsi"]
                            and [e["val_auc_bsi"] for e in ctx_base["per_epoch"]]
                            == BASELINE_RECORD["per_epoch_val"]
                            and ctx_base["gate_mean"] == BASELINE_RECORD["gate_mean"]),
        "correct_values": (ctx_corr["test_auc_bsi"] == CORRECT_RECORD["test_auc_bsi"]
                           and ctx_corr["best_val_auc_bsi"] == CORRECT_RECORD["best_val_auc_bsi"]
                           and ctx_corr.get("variant") == CORRECT_RECORD["variant"]
                           and (ctx_corr.get("rp_arm") or {}).get("classification")
                           == CORRECT_RECORD["classification"]),
        "shuf_hist_values": (ctx_shuf["test_auc_bsi"] == SHUF_HIST_RECORD["test_auc_bsi"]
                             and ctx_shuf["best_val_auc_bsi"] == SHUF_HIST_RECORD["best_val_auc_bsi"]
                             and ctx_shuf.get("variant") == SHUF_HIST_RECORD["variant"]
                             and (ctx_shuf.get("rp_arm") or {}).get("classification")
                             == SHUF_HIST_RECORD["classification"]),
        "pin_source_alpha": ctx_corr_prompt.get("alpha_final") == PINNED_ALPHA,
        "pinned_arms_alpha_equal_pin": (all(float(r.get("alpha_final") or 0.0) == PINNED_ALPHA
                                            for r in (fc_prompt, fs_prompt))),
        "correct_deltas": (ctx_corr["test_auc_bsi"] - ctx_base["test_auc_bsi"] == DELTA_TEST_CORRECT
                           and ctx_corr["best_val_auc_bsi"] - ctx_base["best_val_auc_bsi"]
                           == DELTA_VAL_CORRECT),
        "shuf_hist_delta": (ctx_shuf["test_auc_bsi"] - ctx_base["test_auc_bsi"]
                            == DELTA_TEST_SHUFFLED_HIST),
    }

    # ---- §5.4 效用分量 ----
    base_test, base_val = base["test_auc_bsi"], base["best_val_auc_bsi"]
    fc_test, fc_val = fc["test_auc_bsi"], fc["best_val_auc_bsi"]
    fs_test, fs_val = fs["test_auc_bsi"], fs["best_val_auc_bsi"]
    gap = fc_test - fs_test
    g_c, g_s = fc_test - base_test, fs_test - base_test
    delta_val_gap = fc_val - fs_val
    delta_val_c, delta_val_s = fc_val - base_val, fs_val - base_val
    pin_ratio_c = (fc_prompt.get("val_stats") or {}).get("ratio_mean") or []
    pin_ratio_s = (fs_prompt.get("val_stats") or {}).get("ratio_mean") or []
    ratio_mean_fc = float(sum(pin_ratio_c) / len(pin_ratio_c)) if pin_ratio_c else None
    ratio_mean_fs = float(sum(pin_ratio_s) / len(pin_ratio_s)) if pin_ratio_s else None
    ratio_rel_diff = (abs(ratio_mean_fc - ratio_mean_fs) / ratio_mean_fc
                      if ratio_mean_fc and ratio_mean_fs is not None and ratio_mean_fc > 0 else None)

    preconditions = {
        "IDENTITY_MISMATCH": not all(identity.values()),
        "BASELINE_REPRODUCTION_FAILED": not all(reproduction.values()),
        "PROTOCOL_INVALID": not all(protocol_a.values()),
        "PINNED_ALPHA_INVALID": not (mechanism["arm_correct"]["PA1"] and mechanism["arm_correct"]["PA2"]
                                     and mechanism["arm_correct"]["PA8"]
                                     and mechanism["arm_shuffled"]["PA1"]
                                     and mechanism["arm_shuffled"]["PA2"]
                                     and mechanism["arm_shuffled"]["PA8"]),
        "MECHANISM_FAIL": not all(v for status in mechanism.values() for v in status.values()),
        "COMPARATOR_MISMATCH": not all(comparator.values()),
        "DECK_MISMATCH": not bool(deck["pass"]),
        "PERMUTATION_INVALID": not all(g["pass"] for g in perm_gates.values()),
    }
    causal = causal_verdict(gap=gap, delta_val_gap=delta_val_gap, preconditions=preconditions)

    components = causal["components"]
    components.update({
        "baseline_test": base_test, "baseline_val": base_val,
        "arm_correct_test": fc_test, "arm_correct_val": fc_val,
        "arm_shuffled_test": fs_test, "arm_shuffled_val": fs_val,
        "g_c": g_c, "g_s": g_s,
        "retained_ratio_vs_baseline": (g_s / g_c) if g_c > 0 else None,
        "delta_val_c": delta_val_c, "delta_val_s": delta_val_s,
        "retention_correct_vs_hist": g_c / DELTA_TEST_CORRECT if DELTA_TEST_CORRECT else None,
        "retention_shuffled_vs_hist_correct": g_s / DELTA_TEST_CORRECT if DELTA_TEST_CORRECT else None,
        "shuffled_pinned_vs_learnable": (g_s / DELTA_TEST_SHUFFLED_HIST
                                         if DELTA_TEST_SHUFFLED_HIST else None),
        "G_c_ge_historical_0.0055": g_c >= HISTORICAL_U1_THRESHOLD,
        "G_s_ge_historical_0.0055": g_s >= HISTORICAL_U1_THRESHOLD,
        "pin_ratio_mean_correct": ratio_mean_fc, "pin_ratio_mean_shuffled": ratio_mean_fs,
        "pin_ratio_relative_diff": ratio_rel_diff,
    })

    secondary = {"arm_correct": secondary_classification(g_c),
                 "arm_shuffled": secondary_classification(g_s)}
    headroom = {"arm_correct": headroom_assessment(delta_test=g_c, delta_val=delta_val_c,
                                                   arm_record=fc, prompt_doc=fc_prompt),
                "arm_shuffled": headroom_assessment(delta_test=g_s, delta_val=delta_val_s,
                                                    arm_record=fs, prompt_doc=fs_prompt)}
    result = {
        "analysis": "rp_pinned_compare",
        "root": str(root), "baseline_run": baseline_run,
        "arm_correct_run": arm_correct_run, "arm_shuffled_run": arm_shuffled_run,
        "context_baseline_run": BASELINE_RUN_ID, "context_correct_run": CORRECT_RUN_ID,
        "context_shuffled_run": SHUF_HIST_RUN_ID,
        "pinned_alpha": PINNED_ALPHA,
        "identity": {"checks": identity, "pass": all(identity.values())},
        "reproduction": {"checks": reproduction, "pass": all(reproduction.values())},
        "protocol_a": {**protocol_a, "pass": all(protocol_a.values())},
        "mechanism": {"checks": mechanism, "pass": all(v for s in mechanism.values() for v in s.values())},
        "comparator": {"checks": comparator, "pass": all(comparator.values())},
        "permutation_gates": {**perm_gates, "pass": all(g["pass"] for g in perm_gates.values()),
                              "coverage": coverage, "recorded_gates_consistent": recorded_consistent},
        "deck": deck,
        "preconditions": preconditions,
        "components": components,
        "verdict": causal["verdict"], "subreason": causal["subreason"],
        "secondary": secondary, "headroom": headroom,
        "context": {"learnable_arms": CONTEXT["learnable_correct"], "learnable_shuffled": CONTEXT["learnable_shuffled"],
                    "budget_chain": CONTEXT["budget_chain"],
                    "historical_u1_threshold": HISTORICAL_U1_THRESHOLD},
        "per_epoch": {"baseline": [e["val_auc_bsi"] for e in base["per_epoch"]],
                      "arm_correct": [e["val_auc_bsi"] for e in fc["per_epoch"]],
                      "arm_shuffled": [e["val_auc_bsi"] for e in fs["per_epoch"]]},
        "records": {name: {"run_id": doc["run_id"], "commit": doc.get("commit"), "git": doc.get("git"),
                           "best_epoch": doc.get("best_epoch"),
                           "epochs_run": len(doc.get("per_epoch") or []),
                           "stopped_early": len(doc.get("per_epoch") or []) < doc.get("epochs", 0),
                           "wall_seconds": doc.get("wall_seconds"),
                           "peak_vram_mb": doc.get("peak_vram_mb"),
                           "hard_pass": doc.get("hard_pass")}
                   for name, doc in (("baseline", base), ("arm_correct", fc), ("arm_shuffled", fs))},
        "inherited_b_gates": {name: {g: (doc.get("gates", {}).get(g) or {}).get("verdict")
                                     for g in ("B1", "B2", "B3", "B4")}
                              for name, doc in (("baseline", base_gate), ("arm_correct", fc_gate),
                                                ("arm_shuffled", fs_gate))},
        "a_gates": {name: {g: (doc.get("gates", {}).get(g) or {}).get("verdict")
                           for g in ("A1", "A2", "A3", "A4", "A5", "A6")}
                    for name, doc in (("baseline", base_gate), ("arm_correct", fc_gate),
                                      ("arm_shuffled", fs_gate))},
    }
    out_path = root / "runs" / arm_shuffled_run / "rp_pinned_compare.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# =====================================================================================
# CLI（纯分析；恰一次）
# =====================================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="AliCCP pinned-alpha 条件对应因果消融分析（预注册 §5；只读三 run 记录）")
    parser.add_argument("--root", type=str, default="artifacts/aliccp_bench")
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--arm-correct-run", type=str, required=True)
    parser.add_argument("--arm-shuffled-run", type=str, required=True)
    args = parser.parse_args(argv)
    result = analyze_runs(root=Path(args.root), baseline_run=args.baseline_run,
                          arm_correct_run=args.arm_correct_run,
                          arm_shuffled_run=args.arm_shuffled_run)
    comp = result["components"]
    print(f"verdict={result['verdict']} subreason={result['subreason']} "
          f"secondary={result['secondary']}")
    print(f"GAP={comp['gap']!r} G_c={comp['g_c']!r} G_s={comp['g_s']!r} "
          f"R={comp['retained_ratio_vs_baseline']!r} dval_gap={comp['delta_val_gap']!r}")
    print(f"validation_agreement={comp['validation_agreement']} "
          f"shuffled_matches_or_exceeds_correct={comp['shuffled_matches_or_exceeds_correct']}")
    print(f"report -> {Path(args.root) / 'runs' / args.arm_shuffled_run / 'rp_pinned_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
