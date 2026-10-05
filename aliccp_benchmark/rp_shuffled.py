"""AliCCP 阶段 2 残差 Prompt 的条件对应因果消融（shuffled conditioning；本分支 exp/aliccp-stage2-residual-prompt-shuffled-condition）。

唯一事实来源：docs/superpowers/specs/2026-10-05-aliccp-stage2-residual-prompt-shuffled-condition-design.md

消融语义（预注册 §2）：残差 prompt 的其余一切逐字节不变，**唯一变化** = 目标样本 i 的 prompt
条件特征由自身 `dnn_input[i]` 改为同 batch 内一个**确定性标签无关错排** π 指定的
`dnn_input[π(i)]`（`P(x_{π(i)})` 生成作用在样本 i 上的方向）。基头（projection/gate）仍接收
正确的 `dnn_input`；标签、目标、基表征、优化器更新、数据顺序、全局 RNG 一律不动。

错排构造（冻结，§2.1；独立复核脚本按此规格重实现）：
  * 键 = (split, batch_index, n)；`seed_material = sha256(f"{COND_PERM_SEED}|{split}|{batch_index}|{n}")[:16]`；
  * `random.Random(int(seed_material, 16))`（局部实例，不触碰任何全局 RNG）上的 Sattolo 洗牌
    （`for i in range(n-1, 0, -1): j = rng.randrange(0, i); swap(a[i], a[j])`）⇒ n≥2 时无不动点的
    均匀 n-轮换；n≤1 ⇒ 恒等（退化，如实记录）；同键（含跨 epoch/跨评估）永远同一 π。

结构性事实（沿用钉死机制）：α=0 ⇒ 输出与基线逐位一致（G2）；范数界 ratio ≤ |α| 与跨流一致性
S2 在置换下保持（与条件来源无关）；correct-condition 默认路径逐字节不变（`residual_prompt.py`
零适配 + `shuffler=None` 恒等分派，PG6）。

判定（预注册 §5）：analyze_runs 只读 P0/配对/对照 run 记录 → 前置条件（identity/复现/A/机制/
对照/置换）→ 因果判定树（SAMPLE_CONDITION_SUPPORTED / CONDITION_ALIGNMENT_NOT_SUPPORTED /
INVALID）+ 二级分类（POSITIVE_IMPROVEMENT / NO_CLEAR_IMPROVEMENT / CLEAR_DEGRADATION）。
历史 +0.0055 判定单列保留（§5.7）。不主张新颖性。
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

# ---- 臂与接线（预注册 §2/§3）----
VARIANT_SHUFFLED = "residual-prompt-shuffled"
RUN_ID_SUFFIX = "-rpgs"

# ---- 冻结判据常量（预注册 §5；只允许在看到结果之前修改）----
COND_PERM_SEED = 20261005                 # 错排基数种子（冻结；§2.1）
MATERIAL_DELTA = 0.001                    # 最小实质差异（= 用户指定；二分类边界同值）
SECONDARY_POSITIVE_MIN = 0.001            # 二级分类：Δtest ≥ +0.001 ⇒ POSITIVE_IMPROVEMENT（闭）
SECONDARY_NEGATIVE_MAX = -0.02            # Δtest ≤ −0.02 ⇒ CLEAR_DEGRADATION（闭）
PERM_SAMPLE_KEYS = 3                      # 逐 split 样例保留的键数（人审用）
PERM_SAMPLE_LEN = 16                      # 样例 π 前若干元素

# ---- 钉死身份（预注册 §1.5；与 verify_shuffled_prerun.py 逐位一致）----
STAGE1_ID = "s1-5c060b9c-m1688723740-e3-4e1b5c6f"
STAGE1_BACKBONE_SHA = "e5e7e610f9dcbe9b748820079187dfbd70510dfe41fa46dd85a6baeba97d990c"
STAGE1_ENV_IDS_SHA = "5cd198f1a22e829de1ac5cabfdf03fb80d95caec6775534bf00918b89acf2ac0"
STAGE1_FINGERPRINT_SHA = "5c060b9c5c9d0e235ec815e1b488b9dec222fc37887ad2eedd2afadc82bdd0d8"
MODEL_SEED_PINNED = 1688723740
BASELINE_RUN_ID = "20261003-0624-p2M-v500k-t1M-m1688723740-short-79b5e07"
CORRECT_RUN_ID = "20261004-0325-p2M-v500k-t1M-m1688723740-short-013e105-rpg"
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
    "alpha_final": 0.07101669907569885,
}
DELTA_TEST_CORRECT = 0.008103113327467715      # G_c 钉死值（预注册 §1.2-A1 重推逐位）
DELTA_VAL_CORRECT = 0.008622497456618139
HISTORICAL_U1_THRESHOLD = 0.0055               # 历史效用门（§5.7 单列保留，不适用于 P1 主判定）
CONTEXT = {                                    # 预算链 context（只读数；§1.2-A2 审计值）
    "five_epoch": {"delta_test": 0.008103113327467715, "delta_val": 0.008622497456618139,
                   "arm_run": CORRECT_RUN_ID,
                   "baseline_run": BASELINE_RUN_ID},
    "ten_epoch": {"delta_test": 0.0074396653390377265, "delta_val": 0.0048363447472773435,
                  "arm_run": "20261004-0431-p2M-v500k-t1M-m1688723740-long-f2ccec2-rpg",
                  "baseline_run": "20261004-0427-p2M-v500k-t1M-m1688723740-long-2b1d585"},
    "twenty_epoch": {"delta_test": 0.004155967377889924, "delta_val": 0.0023134736258212385,
                     "status": "NOT_PERSIST", "claim": True, "ablation_eligible": True,
                     "arm_run": "20261005-0829-p2M-v500k-t1M-m1688723740-xlong-eafc336-rpg",
                     "baseline_run": "20261005-0820-p2M-v500k-t1M-m1688723740-xlong-45ab7e7"},
    "budget_trend_label": "MONOTONE_NARROWING",
}
PRECONDITION_PRIORITY = ("IDENTITY_MISMATCH", "BASELINE_REPRODUCTION_FAILED", "PROTOCOL_INVALID",
                         "MECHANISM_FAIL", "COMPARATOR_MISMATCH", "PERMUTATION_INVALID")


# =====================================================================================
# §2.1 错排构造（冻结；独立复核按此规格重实现）
# =====================================================================================

def perm_seed_material(split: str, batch_index: int, n: int, base_seed: int = COND_PERM_SEED) -> int:
    """键 → 64-bit 种子（只依赖键；标签/特征/预测一律不参与）。"""
    key = f"{int(base_seed)}|{split}|{int(batch_index)}|{int(n)}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest()[:16], 16)


def derange(n: int, split: str, batch_index: int, base_seed: int = COND_PERM_SEED) -> list:
    """Sattolo 洗牌：n≥2 ⇒ 无不动点的均匀 n-轮换（双射）；n≤1 ⇒ 恒等（退化，定义为恒等）。"""
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
    """按 (split, batch_index, n) 键缓存 π；运行期完整性探针 + 逐 split/总 digest（预注册 §5.2）。"""

    def __init__(self, base_seed: int = COND_PERM_SEED):
        self.base_seed = int(base_seed)
        self._perms: dict = {}                # key -> list[int]
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

    # ---- 取 π（唯一入口；首次生成时做全部运行期探针）----
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

    # ---- 报告 ----
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


# =====================================================================================
# §2.2 shuffled 头（只重写 forward 的"条件对应"一行）
# =====================================================================================

_COND_PERM_REQUIRED = object()


class ShuffledConditionResidualPromptNewTask(RP.ResidualPromptNewTask):
    """条件对应消融头：`prompt_deltas` 一律以 `dnn_input[cond_perm]` 为条件输入；其余逐字继承。

    `cond_perm` 为**必需参数**（哨兵缺省 ⇒ 显式 ValueError，拒绝静默回退为正确条件）；
    基头 `super().forward` 仍接收**正确的** `dnn_input`（projection/gate 不受影响）。
    错排器不在此类上（由 bench 持有并经参数传入）⇒ state_dict/参数/构造 RNG 与 correct 头逐位一致。
    """

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs, cond_perm=_COND_PERM_REQUIRED):
        if cond_perm is _COND_PERM_REQUIRED:
            raise ValueError("shuffled-condition 臂要求显式 cond_perm（拒绝静默回退为正确条件）")
        if not torch.is_tensor(cond_perm) or cond_perm.dtype != torch.long \
                or cond_perm.dim() != 1 or int(cond_perm.shape[0]) != int(dnn_input.shape[0]):
            raise ValueError(f"cond_perm 必须是 shape=[B] 的 long 张量（B={int(dnn_input.shape[0])}）")
        cond = dnn_input[cond_perm]
        deltas, _ = self.prompt_deltas(cond, [gen_rep, *spec_reps])
        gen_rep_p = gen_rep + deltas[0]
        spec_reps_p = [spec + delta for spec, delta in zip(spec_reps, deltas[1:])]
        # 注意：必须显式委托 NewTask.forward——`super()` 会落到 ResidualPromptNewTask.forward（二次注入）。
        # 恒等置换时本方法 ≡ ResidualPromptNewTask.forward（逐位；PG6-c 等价测试钉死）。
        return NewTask.forward(self, dnn_input, gen_rep_p, spec_reps_p, env_embs)


def build_shuffled_newtask(*, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                           hidden: int = RP.PROMPT_HIDDEN):
    return ShuffledConditionResidualPromptNewTask(
        input_size=input_size, rep_dim=rep_dim, tower_dnn_hidden_units=tower_dnn_hidden_units,
        reg_dnn=reg_dnn, device=device, hidden=hidden)


def forward_with_conditioning(newtask, dnn_input, gen_rep, spec_reps, env_embs,
                              shuffler, split, batch_index):
    """统一前向分派：`shuffler=None`（baseline/correct 臂）⇒ 与钉死调用逐字相同。"""
    if shuffler is None:
        return newtask(dnn_input, gen_rep, spec_reps, env_embs)
    if split is None:
        raise ValueError("shuffled 分派必须给出 split")
    perm = shuffler.perm(split, batch_index, int(dnn_input.shape[0])).to(dnn_input.device)
    return newtask(dnn_input, gen_rep, spec_reps, env_embs, cond_perm=perm)


class ShuffledPromptAudit(RP.PromptAudit):
    """G1 构造审计逐字继承；G2 首 batch 前向检查以**真实训练 π**（train/0）调用错排头。"""

    def __init__(self, *args, shuffler, **kwargs):
        super().__init__(*args, **kwargs)
        self._shuffler = shuffler
        self.shuffle_init_perm_used = None

    @torch.no_grad()
    def init_forward_check(self, dnn_input, gen_rep, spec_reps, env_embs) -> None:
        if self.init_forward is not None:
            return
        n = int(dnn_input.shape[0])
        perm = self._shuffler.perm("train", 0, n).to(dnn_input.device)
        self.shuffle_init_perm_used = {"split": "train", "batch_index": 0, "n": n}
        reference = self._reference.to(dnn_input.device).eval()
        was_training = self.newtask.training
        self.newtask.eval()
        try:
            out_variant = self.newtask(dnn_input, gen_rep, spec_reps, env_embs, cond_perm=perm)
            out_reference = reference(dnn_input, gen_rep, spec_reps, env_embs)
        finally:
            self.newtask.train(was_training)
        self.init_forward = {
            "bit_identical": bool(torch.equal(out_variant, out_reference)),
            "max_abs_diff": float((out_variant - out_reference).abs().max()),
            "n_samples": int(out_variant.numel()),
            "cond_perm": self.shuffle_init_perm_used,
        }


@torch.no_grad()
def evaluate_prompt_diagnostics_shuffled(newtask, backbone, loader, device, shuffler,
                                         split: str = "val") -> dict:
    """钉死诊断循环的错排版：deltas/preds 一律以 `dnn_input[perm]` 为条件（同一原则）。"""
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
# §5.2 置换门禁 + §5.3 机制门禁（pinned arm_verdict 包装）
# =====================================================================================

def pg6_evidence() -> dict:
    """PG6（correct-condition 默认逐位不变）的构建期证据：机制文件 LF sha + 守卫测试名钉死。"""
    path = Path(__file__).resolve().parent / "residual_prompt.py"
    digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {
        "mechanism_file": MECHANISM_REL,
        "mechanism_file_lf_sha256": digest,
        "mechanism_pin_lf_sha256": MECHANISM_PIN_LF_SHA,
        "mechanism_byte_identical": digest == MECHANISM_PIN_LF_SHA,
        "guards": ["test_residual_prompt_shuffled.TestMechanismByteIdentical",
                   "test_residual_prompt_shuffled.TestWiringReconstructEquality",
                   "test_residual_prompt_shuffled.TestShuffledHeadSemantics."
                   "test_identity_perm_equivalent_to_correct_head_at_nonzero_alpha"],
    }


def permutation_gates(report: dict, coverage: dict) -> dict:
    """PG1–PG6（PG1–PG5 由运行期计数机械重算；PG6 由机制文件哈希）。"""
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
                "rule": "机制文件逐字节 == 钉死 blob ∧ shuffler=None 恒等分派 ∧ 恒等置换等价测试",
                "observed": pg6},
    }


def shuffled_arm_verdict(*, auc_test, auc_val, probe, val_stats, params, reference,
                         protocol_ok=True, shuffle_report, budgets=None, batch_size=None) -> dict:
    """钉死 `RP.arm_verdict`（M0+G1–G8+U）+ shuffle 段（PG1–PG6）；PG 任一 FAIL ⇒ 分类覆写。"""
    arm = RP.arm_verdict(auc_test=auc_test, auc_val=auc_val, probe=probe, val_stats=val_stats,
                         params=params, reference=reference, protocol_ok=protocol_ok)
    if budgets and batch_size:
        coverage = shuffle_coverage_ok(shuffle_report, budgets=budgets, batch_size=batch_size)
    else:                                     # 无预算上下文（单元测试）⇒ 仅列 per_split 键
        coverage = {split: {"n_batches_ok": True, "n_rows_ok": True}
                    for split in (shuffle_report.get("per_split") or {})}
    gates = permutation_gates(shuffle_report, coverage)
    shuffle = {"base_seed": shuffle_report.get("base_seed"), "report": shuffle_report,
               "coverage": coverage, "gates": gates,
               "pass": all(g["pass"] for g in gates.values())}
    if not shuffle["pass"] and arm["classification"] != "MECHANISM_FAIL":
        arm["classification"], arm["subreason"], arm["pass"] = "MECHANISM_FAIL", "PERMUTATION_INVALID", False
    arm["shuffle"] = shuffle
    return arm


# =====================================================================================
# §5.4/§5.5/§5.6 判定（纯函数 + 只读分析器）
# =====================================================================================

def causal_verdict(*, g_c: float, g_s: float, preconditions: dict | None = None) -> dict:
    """预注册 §5.5 因果判定树（先于结果写死）；preconditions 按插入顺序为首个失败项给出 subreason。"""
    g_c, g_s = float(g_c), float(g_s)
    components = {
        "g_c": g_c, "g_s": g_s, "L": g_c - g_s,
        "retained_ratio": (g_s / g_c) if g_c > 0 else None,
        "shuffled_above_baseline": g_s > 0.0,
        "shuffled_material_positive": g_s >= MATERIAL_DELTA,
        "shuffled_exceeds_correct": g_s > g_c,
    }
    for name, failed in (preconditions or {}).items():
        if failed:
            return {"verdict": "INVALID", "subreason": name, "components": components}
    if g_c < MATERIAL_DELTA:
        return {"verdict": "INVALID", "subreason": "COMPARATOR_GAIN_NOT_MATERIAL",
                "components": components}
    if g_c - g_s >= MATERIAL_DELTA:
        sub = "FULL_GAIN_REQUIRES_ALIGNMENT" if g_s < MATERIAL_DELTA else "PARTIAL_ALIGNMENT_CONTRIBUTION"
        return {"verdict": "SAMPLE_CONDITION_SUPPORTED", "subreason": sub, "components": components}
    return {"verdict": "CONDITION_ALIGNMENT_NOT_SUPPORTED", "subreason": "SHUFFLED_RETAINS_GAIN",
            "components": components}


def secondary_classification(g_s: float) -> str:
    """预注册 §5.6 二级分类（边界先于结果写死；±闭端按预注册）。"""
    g_s = float(g_s)
    if g_s >= SECONDARY_POSITIVE_MIN:
        return "POSITIVE_IMPROVEMENT"
    if g_s <= SECONDARY_NEGATIVE_MAX:
        return "CLEAR_DEGRADATION"
    return "NO_CLEAR_IMPROVEMENT"


def headroom_assessment(*, g_s: float, delta_val_s: float, arm_record: dict, prompt_doc: dict) -> dict:
    """预注册 §5.6：NO_CLEAR_IMPROVEMENT 的机械 headroom（仅当落入该区间时适用）。"""
    if secondary_classification(g_s) != "NO_CLEAR_IMPROVEMENT":
        return {"applicable": False}
    grad_probe = prompt_doc.get("grad_probe") or [{}]
    active = (abs(float(prompt_doc.get("alpha_final") or 0.0)) > 0.0
              and float(grad_probe[-1].get("generator_grad_norm") or 0.0) > 0.0
              and float((prompt_doc.get("gate") or {}).get("geff_std") or 0.0) > 0.0)
    epochs_run = len(arm_record.get("per_epoch") or [])
    right_censored = (arm_record.get("best_epoch") == arm_record.get("epochs") == epochs_run)
    return {
        "applicable": True,
        "mechanism_activity": "ACTIVE" if active else "INACTIVE",
        "validation_direction": "POSITIVE" if float(delta_val_s) > 0.0 else "NON_POSITIVE",
        "epoch_trajectory": "RIGHT_CENSORED_STILL_IMPROVING" if right_censored else "EARLY_STOPPED",
        "gap_to_positive_threshold": SECONDARY_POSITIVE_MIN - float(g_s),
        "retained_ratio": (float(g_s) / DELTA_TEST_CORRECT) if DELTA_TEST_CORRECT > 0 else None,
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


def analyze_runs(*, root, baseline_run, arm_run) -> dict:
    """只读 P0/P1/对照 run 记录 → 预注册 §5 全量判定（不改任何 run 产物；报告落 P1 run 目录）。"""
    root = Path(root)
    base = _read_json(root / "runs" / baseline_run / "metrics.json")
    base_gate = _read_json(root / "runs" / baseline_run / "gate_report.json")
    base_cfg = _read_json(root / "runs" / baseline_run / "config.json")
    arm = _read_json(root / "runs" / arm_run / "metrics.json")
    arm_gate = _read_json(root / "runs" / arm_run / "gate_report.json")
    arm_cfg = _read_json(root / "runs" / arm_run / "config.json")
    arm_prompt = _read_json(root / "runs" / arm_run / "prompt_report.json")
    ctx_base = _read_json(root / "runs" / BASELINE_RUN_ID / "metrics.json")
    ctx_corr = _read_json(root / "runs" / CORRECT_RUN_ID / "metrics.json")
    ctx_corr_prompt = _read_json(root / "runs" / CORRECT_RUN_ID / "prompt_report.json")

    # ---- §5.1-ID：identity（14 项）----
    expected_ref_path = root / "runs" / BASELINE_RUN_ID / "newtask.pt"
    ref_recorded = (arm_prompt.get("reference_dispersion") or {}).get("newtask_checkpoint")
    ref_path_ok = ref_recorded is not None and Path(str(ref_recorded)) == expected_ref_path
    ref_sha = _sha256_file(expected_ref_path) if expected_ref_path.is_file() else None
    identity = {
        "stage1_same": base["stage1_id"] == arm["stage1_id"],
        "stage1_pinned": base["stage1_id"] == arm["stage1_id"] == STAGE1_ID,
        "seed_same_pinned": base["model_seed"] == arm["model_seed"] == MODEL_SEED_PINNED,
        "epochs": base["epochs"] == arm["epochs"] == 5,
        "patience": base["patience"] == arm["patience"] == 2,
        "tag": base["tag"] == arm["tag"] == "short",
        "dirty_false": (not base.get("git", {}).get("dirty")) and (not arm.get("git", {}).get("dirty")),
        "baseline_variant": base.get("variant") == RP.BASELINE_VARIANT,
        "baseline_no_arm_suffix": not base["run_id"].endswith(RUN_ID_SUFFIX),
        "arm_variant": arm.get("variant") == VARIANT_SHUFFLED,
        "arm_suffix": arm["run_id"].endswith(RUN_ID_SUFFIX),
        "arm_ref_path_pinned": ref_path_ok,
        "arm_ref_sha": ref_sha == REF_HEAD_SHA,
        "stage1_shas": (base.get("backbone_sha256_loaded") == STAGE1_BACKBONE_SHA
                        and arm.get("backbone_sha256_loaded") == STAGE1_BACKBONE_SHA
                        and base.get("env_ids_sha256") == arm.get("env_ids_sha256") == STAGE1_ENV_IDS_SHA
                        and base.get("fingerprint_sha256") == arm.get("fingerprint_sha256")
                        == STAGE1_FINGERPRINT_SHA),
    }

    # ---- §5.1-REP：P0 与历史基线记录逐位一致 ----
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

    # ---- §5.1-A / §5.1-M / §5.1-CC ----
    protocol_a = {"baseline": _a_class_ok(base_gate), "arm": _a_class_ok(arm_gate)}
    arm_arm = arm.get("rp_arm") or {}
    mechanism = {g: bool((arm_arm.get(g) or {}).get("pass"))
                 for g in ("M0", "G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")}
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
                           == CORRECT_RECORD["classification"]
                           and ctx_corr_prompt.get("alpha_final") == CORRECT_RECORD["alpha_final"]),
        "correct_deltas": (ctx_corr["test_auc_bsi"] - ctx_base["test_auc_bsi"] == DELTA_TEST_CORRECT
                           and ctx_corr["best_val_auc_bsi"] - ctx_base["best_val_auc_bsi"]
                           == DELTA_VAL_CORRECT),
    }

    # ---- §5.2-PG：由 P1 记录重算（含覆盖；不信任记录布尔）----
    shuffle_block = arm_prompt.get("shuffle") or {}
    shuffle_rep = shuffle_block.get("report") or {}
    coverage = shuffle_coverage_ok(shuffle_rep, budgets=arm_cfg["budgets"],
                                   batch_size=arm_cfg["batch_size"])
    perm_gates = permutation_gates(shuffle_rep, coverage)
    recorded_gates = (shuffle_block.get("gates") or {})
    recorded_consistent = all(
        bool((recorded_gates.get(g) or {}).get("pass")) == perm_gates[g]["pass"]
        for g in perm_gates) and bool(shuffle_block.get("pass")) == all(
        g["pass"] for g in perm_gates.values())

    # ---- §5.4 效用分量 ----
    base_test, base_val = base["test_auc_bsi"], base["best_val_auc_bsi"]
    arm_test, arm_val = arm["test_auc_bsi"], arm["best_val_auc_bsi"]
    corr_test, corr_val = ctx_corr["test_auc_bsi"], ctx_corr["best_val_auc_bsi"]
    g_c, g_s = corr_test - base_test, arm_test - base_test
    delta_val_s, delta_val_c = arm_val - base_val, corr_val - base_val

    preconditions = {
        "IDENTITY_MISMATCH": not all(identity.values()),
        "BASELINE_REPRODUCTION_FAILED": not all(reproduction.values()),
        "PROTOCOL_INVALID": not all(protocol_a.values()),
        "MECHANISM_FAIL": not all(mechanism.values()),
        "COMPARATOR_MISMATCH": not all(comparator.values()),
        "PERMUTATION_INVALID": not all(g["pass"] for g in perm_gates.values()),
    }
    causal = causal_verdict(g_c=g_c, g_s=g_s, preconditions=preconditions)
    components = causal["components"]
    components.update({
        "baseline_test": base_test, "baseline_val": base_val,
        "arm_test": arm_test, "arm_val": arm_val,
        "correct_test": corr_test, "correct_val": corr_val,
        "delta_val_s": delta_val_s, "delta_val_c": delta_val_c,
        "validation_agreement": delta_val_s > 0.0,
        "val_sign_matches_correct": _sign(delta_val_s) == _sign(delta_val_c),
        "G_s_ge_historical_0.0055": g_s >= HISTORICAL_U1_THRESHOLD,   # §5.7 报告项（不判定）
    })

    secondary = secondary_classification(g_s)
    headroom = headroom_assessment(g_s=g_s, delta_val_s=delta_val_s, arm_record=arm,
                                   prompt_doc=arm_prompt)
    result = {
        "analysis": "rp_shuffled_compare",
        "root": str(root), "baseline_run": baseline_run, "arm_run": arm_run,
        "context_baseline_run": BASELINE_RUN_ID, "context_correct_run": CORRECT_RUN_ID,
        "identity": {"checks": identity, "pass": all(identity.values())},
        "reproduction": {"checks": reproduction, "pass": all(reproduction.values())},
        "protocol_a": {**protocol_a, "pass": all(protocol_a.values())},
        "mechanism": {"checks": mechanism, "pass": all(mechanism.values())},
        "comparator": {"checks": comparator, "pass": all(comparator.values())},
        "permutation_gates": {**perm_gates, "pass": all(g["pass"] for g in perm_gates.values()),
                              "coverage": coverage, "recorded_gates_consistent": recorded_consistent},
        "preconditions": preconditions,
        "components": components,
        "verdict": causal["verdict"], "subreason": causal["subreason"],
        "secondary": secondary, "headroom": headroom,
        "context": {"budget_chain": CONTEXT,
                    "historical_u1_threshold": HISTORICAL_U1_THRESHOLD,
                    "G_s_ge_historical_0.0055": g_s >= HISTORICAL_U1_THRESHOLD},
        "per_epoch": {"baseline": [e["val_auc_bsi"] for e in base["per_epoch"]],
                      "arm": [e["val_auc_bsi"] for e in arm["per_epoch"]]},
        "records": {"baseline": {"run_id": base["run_id"], "commit": base.get("commit"),
                                 "git": base.get("git"), "best_epoch": base.get("best_epoch"),
                                 "epochs_run": len(base.get("per_epoch") or []),
                                 "stopped_early": len(base.get("per_epoch") or []) < base.get("epochs", 0),
                                 "wall_seconds": base.get("wall_seconds"),
                                 "peak_vram_mb": base.get("peak_vram_mb"),
                                 "hard_pass": base.get("hard_pass")},
                    "arm": {"run_id": arm["run_id"], "commit": arm.get("commit"),
                            "git": arm.get("git"), "best_epoch": arm.get("best_epoch"),
                            "epochs_run": len(arm.get("per_epoch") or []),
                            "stopped_early": len(arm.get("per_epoch") or []) < arm.get("epochs", 0),
                            "wall_seconds": arm.get("wall_seconds"),
                            "peak_vram_mb": arm.get("peak_vram_mb"),
                            "hard_pass": arm.get("hard_pass")}},
        "inherited_b_gates": {
            "baseline": {g: (base_gate.get("gates", {}).get(g) or {}).get("verdict")
                         for g in ("B1", "B2", "B3", "B4")},
            "arm": {g: (arm_gate.get("gates", {}).get(g) or {}).get("verdict")
                    for g in ("B1", "B2", "B3", "B4")}},
        "a_gates": {
            "baseline": {g: (base_gate.get("gates", {}).get(g) or {}).get("verdict")
                         for g in ("A1", "A2", "A3", "A4", "A5", "A6")},
            "arm": {g: (arm_gate.get("gates", {}).get(g) or {}).get("verdict")
                    for g in ("A1", "A2", "A3", "A4", "A5", "A6")}},
    }
    out_path = root / "runs" / arm_run / "rp_shuffled_compare.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# =====================================================================================
# CLI（纯分析；恰一次）
# =====================================================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="AliCCP shuffled-condition 因果消融分析（预注册 §5；只读 run 记录）")
    parser.add_argument("--root", type=str, default="artifacts/aliccp_bench")
    parser.add_argument("--baseline-run", type=str, required=True)
    parser.add_argument("--arm-run", type=str, required=True)
    args = parser.parse_args(argv)
    result = analyze_runs(root=Path(args.root), baseline_run=args.baseline_run, arm_run=args.arm_run)
    comp = result["components"]
    print(f"verdict={result['verdict']} subreason={result['subreason']} "
          f"secondary={result['secondary']}")
    print(f"G_c={comp['g_c']!r} G_s={comp['g_s']!r} L={comp['L']!r} R={comp['retained_ratio']!r}")
    print(f"shuffled_above_baseline={comp['shuffled_above_baseline']} "
          f"validation_agreement={comp['validation_agreement']}")
    print(f"report -> {Path(args.root) / 'runs' / args.arm_run / 'rp_shuffled_compare.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
