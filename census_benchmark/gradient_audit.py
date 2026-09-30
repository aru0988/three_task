"""阶段 1 梯度冲突审计（**诊断专用**：不改变任何训练更新，不主张任何性能提升）。

背景与定位
----------
梯度手术 / 梯度冲突消解（PCGrad、GradNorm、CAGrad、Gradient Vaccine 及其变体）是**既有文献**，
不是本项目的贡献。本模块只回答一个问题：MPT-Rec 阶段 1（CensusIncome）的 Income / Marital /
环境目标三个损失分量，在**真正共享**的参数上是否已经存在可测的梯度冲突，从而值得**另开分支**
去试梯度手术。阈值在 `docs/superpowers/specs/2026-09-30-census-stage1-gradient-conflict-audit-prereg.md`
中预注册，先于任何结果写定。

三个不可协商的约束
------------------
1. **只读**：梯度一律用 `torch.autograd.grad(..., retain_graph=True, allow_unused=True)` 取，
   不写 `.grad`、不动优化器状态、不消耗 RNG、不改参数。
2. **逐位保真**：`GradientAuditTrainManager` 在 `audit=None` 时与父类 `MPTRecTrainManager.train_two_task`
   产生逐位相同（`torch.equal`）的参数；开启采集时同样逐位相同（由单元测试守护）。
3. **只在真正共享的参数上谈冲突**：`shared_trunk` = embedding + general DNN，
   任务独有塔 / gate / env_embedding 行 / env 分类器一律不算共享。

损失分量（与 `train_two_task` 逐字对应）
--------------------------------------
`total = fused_0 + fused_1 + uni_coe*(uni_0 + uni_1) + env_coe*env + l2_reg`，按"对 total 的实际贡献"切：

- `income`  = `fused_loss_0 + uni_coe * uni_loss_0`
- `marital` = `fused_loss_1 + uni_coe * uni_loss_1`
- `env`     = `env_coe * env_loss`
- `reg`     = `model.get_l2_reg()`

`env` 的梯度**包含** `ReverseLayerF` 的反转与 `alpha` 缩放，即优化器实际看到的那一支；
这是**构造性对抗**，故 task-vs-env 的负余弦有设计上的先验，只作诊断、单独不构成行动依据。

未定义量：`fused_rep_i` 的 gate 权重依赖样本，损失空间不存在 general/specific 的可加分解，
故按"only if mathematically defined"原则**不产出**该复合量（见 `UNDEFINED_COMPOSITES`）。
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from multitaskrec.train import MPTRecTrainManager

# ============================ 常量（与预注册一致，测试校验） ============================
SCHEMA = "census-stage1-gradient-audit/1"
ARTIFACT_ROOT = Path("artifacts/census_stage1_grad")
PREREG_SPEC = "docs/superpowers/specs/2026-09-30-census-stage1-gradient-conflict-audit-prereg.md"
SUMMARY_COLUMNS = ["run_id", "commit", "smoke", "n_steps", "n_epochs", "neg_cos_rate",
                   "mean_cos", "nonzero_rate", "status"]

# ---- 损失分量 ----
COMPONENTS = ("income", "marital", "env", "reg")
CONFLICT_PAIRS = (("income", "marital"), ("income", "env"), ("marital", "env"))
UNDEFINED_COMPOSITES = ("general_vs_specific",)

# ---- 参数分组 ----
GROUP_EMBEDDINGS = "shared_embeddings"
GROUP_SHARED_EXPERT = "shared_expert"
GROUP_SHARED_TRUNK = "shared_trunk"            # 合成组 = 上两组之并
GROUP_ENV_CLASSIFIER = "env_classifier"
GROUP_TASK_EXCLUSIVE = "task_exclusive"
GROUP_UNCLASSIFIED = "unclassified"

#: 互斥划分：每个参数恰好属于其中一组
PARTITION_GROUPS = (GROUP_EMBEDDINGS, GROUP_SHARED_EXPERT, GROUP_ENV_CLASSIFIER,
                    GROUP_TASK_EXCLUSIVE, GROUP_UNCLASSIFIED)
#: 由划分导出的合成组（不参与"划分"校验，但报告里要用）
COMPOSITE_GROUPS = (GROUP_SHARED_TRUNK,)
#: step_record / aggregate 报告的组（unclassified 单独在 meta 里报，避免与空组混淆）
REPORTED_GROUPS = (GROUP_EMBEDDINGS, GROUP_SHARED_EXPERT, GROUP_SHARED_TRUNK,
                   GROUP_ENV_CLASSIFIER, GROUP_TASK_EXCLUSIVE)

#: 参数名前缀 → 组（前缀互不重叠，顺序不影响结果）
PREFIX_GROUPS = (
    ("embedding_network.", GROUP_EMBEDDINGS),
    ("shared_expert_network.", GROUP_SHARED_EXPERT),
    ("env_classifier.", GROUP_ENV_CLASSIFIER),
    ("specific_expert_networks.", GROUP_TASK_EXCLUSIVE),
    ("gate_networks.", GROUP_TASK_EXCLUSIVE),
    ("tower_networks.", GROUP_TASK_EXCLUSIVE),
    ("env_embedding_network.", GROUP_TASK_EXCLUSIVE),
)

# ---- 采样节奏（预注册 §3：确定性、非随机）----
WARMUP_STEPS = 10        # 每个 epoch 采前 N 个 batch（捕捉初始冲突区）
SPACED_STEPS = 25        # 每个 epoch 再均匀间隔采 N 个 batch
EPOCHS_FOR_SCHEDULE = 2  # 与实际阶段 1 的 STAGE1_EPOCHS 一致，用于"步数够不够"的自检

# ---- 门禁（预注册 §2；只允许在运行前修改）----
MIN_SAMPLED_STEPS = 50
GATE_NEG_COS_RATE = 0.25
GATE_MEAN_COS = -0.05
GATE_NONZERO_STEP_RATE = 0.95
GATE_GROUP = GROUP_SHARED_TRUNK
GATE_PAIR = ("income", "marital")

SMOKE_STATUS = "NOT_EVALUATED_SMOKE"

_BCE = nn.BCELoss()
_NLL = nn.NLLLoss()


def prereg_block() -> dict:
    """预注册阈值的结构化自述块（写进产物，防止事后改阈值）。"""
    return {
        "spec": PREREG_SPEC,
        "gate_group": GATE_GROUP,
        "gate_pair": list(GATE_PAIR),
        "min_sampled_steps": MIN_SAMPLED_STEPS,
        "neg_cos_rate_min": GATE_NEG_COS_RATE,
        "mean_cos_max": GATE_MEAN_COS,
        "nonzero_step_rate_min": GATE_NONZERO_STEP_RATE,
        "warmup_steps": WARMUP_STEPS,
        "spaced_steps": SPACED_STEPS,
        "epochs_for_schedule": EPOCHS_FOR_SCHEDULE,
        "env_conflicts_are_diagnostic_only": True,
        "undefined_composites": list(UNDEFINED_COMPOSITES),
        "component_definitions": {
            "income": "fused_loss_0 + uni_coe * uni_loss_0",
            "marital": "fused_loss_1 + uni_coe * uni_loss_1",
            "env": "env_coe * env_loss (gradient includes ReverseLayerF negation and alpha scaling)",
            "reg": "model.get_l2_reg()",
        },
        "decision_rule": "G0 and G1 and G2 and G3 -> ACTIONABLE else NO_ACTION",
    }


# ============================ 参数分组 ============================
def parameter_groups(model) -> dict:
    """返回参数名分组：5 个互斥组（含 `unclassified`）+ 合成组 `shared_trunk`。

    `shared_trunk` 是 `shared_embeddings ∪ shared_expert` 的**并集视图**，与划分组有重叠，
    因此"每个参数恰好属于一组"的划分校验只对 `PARTITION_GROUPS` 成立（见测试）。

    为什么不是 shared：`tower_networks[i]` 同时服务 `gen_preds[i]` 与 `fused_preds[i]`，
    但只服务任务 i；`env_embedding_network.weight` 第 i 行只被任务 i 使用；
    `env_classifier` 只被 env 分量可达 —— 三者都不是跨分量的共享主干。
    """
    groups = {name: [] for name in PARTITION_GROUPS}
    for name, _ in model.named_parameters():
        for prefix, group in PREFIX_GROUPS:
            if name.startswith(prefix):
                groups[group].append(name)
                break
        else:
            groups[GROUP_UNCLASSIFIED].append(name)
    groups[GROUP_SHARED_TRUNK] = sorted(groups[GROUP_EMBEDDINGS] + groups[GROUP_SHARED_EXPERT])
    return groups


def group_sizes(groups: dict) -> dict:
    return {group: len(members) for group, members in groups.items()}


# ============================ 损失分解 ============================
def decompose_losses(model, output, *, y_income, y_marital, env_ids, uni_coe, env_coe,
                     loss_func=_BCE, env_loss_func=_NLL) -> dict:
    """把 `train_two_task` 的单 batch 目标拆成 4 个可加分量（浮点再结合，非逐位相等）。

    恒等式 `sum(分量) ≈ total` 由单元测试守护；误差只来自加法结合顺序，无任何语义改动。
    """
    device = output["gen_preds"][0].device
    y_income = y_income.float().to(device)
    y_marital = y_marital.float().to(device)
    env_ids = env_ids.to(device)

    uni_income = loss_func(output["gen_preds"][0], y_income)
    uni_marital = loss_func(output["gen_preds"][1], y_marital)
    fused_income = loss_func(output["fused_preds"][0], y_income)
    fused_marital = loss_func(output["fused_preds"][1], y_marital)
    env_loss = env_loss_func(output["env_pred"], env_ids)

    return {
        "income": fused_income + uni_coe * uni_income,
        "marital": fused_marital + uni_coe * uni_marital,
        "env": env_coe * env_loss,
        "reg": model.get_l2_reg(),
    }


def total_loss(components: dict) -> torch.Tensor:
    """分量之和；应 ≈ `train_two_task` 里那个 `loss`（同一分量的不同加法结合顺序）。"""
    return components["income"] + components["marital"] + components["env"] + components["reg"]


def component_grads(components: dict, params, *, retain_graph: bool = True) -> dict:
    """逐分量取梯度：**不写 .grad、不动优化器状态、不消耗 RNG、不改参数**。

    `allow_unused=True` 使"该分量不可达此参数"显式返回 `None`（与精确零梯度区分开，
    但仍按零计入统计）。`retain_graph=True` 保证调用方之后仍可对同一图做真正的 `backward()`。
    """
    result = {}
    for name in COMPONENTS:
        grads = torch.autograd.grad(components[name], params, retain_graph=retain_graph,
                                    allow_unused=True)
        result[name] = list(grads)
    return result


# ============================ 向量化与统计 ============================
def _flatten(tensors) -> torch.Tensor:
    """把一组梯度张量拼成一个 float64 一维向量（组内余弦按拼接整体定义）。"""
    if not tensors:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat([t.detach().reshape(-1).to(device="cpu", dtype=torch.float64) for t in tensors])


def _cosine(a: torch.Tensor, b: torch.Tensor):
    """余弦；任一侧范数为 0 时返回 None（未定义，而非 NaN）。"""
    norm_a, norm_b = float(a.norm()), float(b.norm())
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return float(torch.dot(a, b)) / (norm_a * norm_b)


def _is_exactly_zero(grad) -> bool:
    """`None`（不可达）或精确全零张量都算零梯度。"""
    return grad is None or not bool(grad.any())


def pair_key(a: str, b: str) -> str:
    return f"{a}_vs_{b}"


def cos_key(a: str, b: str, group: str) -> str:
    return f"cos_{a}_{b}__{group}"


def norm_key(component: str, group: str) -> str:
    return f"n_{component}__{group}"


def zero_key(component: str, group: str) -> str:
    return f"z_{component}__{group}"


def _zero_slots(grads, params=None) -> dict:
    """每个参数下标 → 一个"该参数的精确零张量"模板，用于不可达（`None`）的槽位。

    形状/设备/dtype 优先取真实参数 `params[i]`（调用方给了就是逐位同形）；否则取该下标上
    任意一个非 `None` 梯度（autograd 保证同一参数的梯度与参数同形）。模板只依赖下标、与分量
    无关，因此同一组的各分量向量必然逐位对齐、长度一致。
    """
    slots = {}
    if params is not None:
        for i, param in enumerate(params):
            slots[i] = torch.zeros_like(param)
    for component in COMPONENTS:
        for i, grad in enumerate(grads[component]):
            if grad is not None and i not in slots:
                slots[i] = torch.zeros_like(grad)
    return slots


def step_record(*, epoch, step, alpha, grads, param_names, groups, params=None,
                reported_groups=REPORTED_GROUPS) -> dict:
    """单个采样步的**扁平标量**记录（约 58 个键；不做逐参数 dump）。

    `grads` 为 `{component: [Tensor|None, ...]}`，与 `param_names`（及可选的 `params`）按
    下标对齐。组向量只取 `param_names ∩ groups[group]`，因此"未纳入本次采集的参数"不会被计入。

    **参数对齐**：不可达（`None`）的槽位以该参数的精确零张量占位后再拼接，故同一组的各分量
    向量长度一致、逐位落在同一参数空间上。若把 `None` 直接丢掉，两个分量在同一组上会拼出
    长度不同（长度恰好相同时则**逐位错位**）的向量，余弦要么抛错、要么静默地比较了不相干的
    参数。`params` 给出时占位张量与真实参数同 numel/device/dtype；未给出时按下标上任一非
    `None` 梯度推断；全部分量都不可达且未给 `params` 时该槽位对所有分量一致地跳过（长度仍
    然一致，只是不占位）。
    """
    index = {name: i for i, name in enumerate(param_names)}
    members = {group: [index[n] for n in groups.get(group, []) if n in index]
               for group in reported_groups}
    slots = _zero_slots(grads, params)

    record = {"epoch": int(epoch), "step": int(step), "alpha": float(alpha)}
    vectors = {}
    for component in COMPONENTS:
        grads_c = grads[component]
        for group in reported_groups:
            kept, zero_count = [], 0
            for i in members[group]:
                grad = grads_c[i]
                if _is_exactly_zero(grad):
                    zero_count += 1
                slot = grad if grad is not None else slots.get(i)
                if slot is not None:
                    kept.append(slot)
            vector = _flatten(kept)
            vectors[(component, group)] = vector
            record[norm_key(component, group)] = float(vector.norm())
            record[zero_key(component, group)] = zero_count

    for a, b in CONFLICT_PAIRS:
        for group in reported_groups:
            record[cos_key(a, b, group)] = _cosine(vectors[(a, group)], vectors[(b, group)])
    return record


def _mean(values):
    return (sum(values) / len(values)) if values else None


def _pair_stats(records, a: str, b: str, group: str) -> dict:
    """一对分量在某参数组上的冲突统计（预注册 §4 的定义）。"""
    cosines, ratios, dominances, norm_a, norm_b = [], [], [], [], []
    nonzero_a = nonzero_b = 0
    for record in records:
        na, nb = record[norm_key(a, group)], record[norm_key(b, group)]
        norm_a.append(na)
        norm_b.append(nb)
        nonzero_a += int(na > 0.0)
        nonzero_b += int(nb > 0.0)
        cosine = record[cos_key(a, b, group)]
        if cosine is not None:
            cosines.append(cosine)
        if na > 0.0 and nb > 0.0:
            ratios.append(na / nb)
            dominances.append(na / (na + nb))

    n_steps, defined = len(records), len(cosines)
    negative = sum(1 for c in cosines if c < 0)
    mean_a, mean_b = _mean(norm_a), _mean(norm_b)
    return {
        "n_steps": n_steps,
        "n_defined": defined,
        "n_undefined": n_steps - defined,
        "n_negative": negative,
        "n_positive": sum(1 for c in cosines if c > 0),
        "n_zero_cos": sum(1 for c in cosines if c == 0),
        "neg_cos_rate": (negative / defined) if defined else None,
        "mean_cos": _mean(cosines),
        "min_cos": min(cosines) if cosines else None,
        "max_cos": max(cosines) if cosines else None,
        "nonzero_rate_a": (nonzero_a / n_steps) if n_steps else None,
        "nonzero_rate_b": (nonzero_b / n_steps) if n_steps else None,
        "mean_norm_a": mean_a,
        "mean_norm_b": mean_b,
        # 逐步比值再平均（对 ‖g_b‖=0 的步跳过）vs 先平均再比值——两者一起看才不会被单步爆炸带偏
        "mean_of_ratio": _mean(ratios),
        "ratio_of_means": (mean_a / mean_b) if (mean_a is not None and mean_b) else None,
        # ∈[0,1]，0.5 为均衡；按 (a, b) 顺序非对称
        "mean_dominance_ratio": _mean(dominances),
    }


def _zero_stats(records, component: str, group: str):
    epochs = [str(int(record["epoch"])) for record in records]
    norm_by_epoch = dict.fromkeys(epochs, 0)
    tensor_by_epoch = dict.fromkeys(epochs, 0)
    total_norm = total_tensor = 0
    for record in records:
        key = str(int(record["epoch"]))
        if record[norm_key(component, group)] == 0.0:
            norm_by_epoch[key] += 1
            total_norm += 1
        zero_tensors = record[zero_key(component, group)]
        total_tensor += zero_tensors
        if zero_tensors > 0:
            tensor_by_epoch[key] += 1
    n_steps = len(records)
    zero_norm = {"by_epoch": norm_by_epoch, "total": total_norm,
                 "rate": (total_norm / n_steps) if n_steps else None}
    steps_with_zero = sum(tensor_by_epoch.values())
    zero_tensor = {"by_epoch": tensor_by_epoch, "total": total_tensor,
                   "steps_with_zero": steps_with_zero,
                   # "frac" = 出现零梯度张量的**步**占比（组大小不随记录传递，故不按张量占比）
                   "frac": (steps_with_zero / n_steps) if n_steps else None}
    return zero_norm, zero_tensor


def aggregate(records) -> dict:
    """跨采样步汇总。等价输入必须逐字节可复现（无时间戳、无路径、无随机）。"""
    summary = {"n_steps": len(records), "n_steps_by_epoch": {},
               "pairs": {}, "zero_norm_steps": {}, "zero_tensor": {}}
    for record in records:
        key = str(int(record["epoch"]))
        summary["n_steps_by_epoch"][key] = summary["n_steps_by_epoch"].get(key, 0) + 1

    for a, b in CONFLICT_PAIRS:
        summary["pairs"][pair_key(a, b)] = {group: _pair_stats(records, a, b, group)
                                            for group in REPORTED_GROUPS}
    for component in COMPONENTS:
        summary["zero_norm_steps"][component] = {}
        summary["zero_tensor"][component] = {}
        for group in REPORTED_GROUPS:
            zero_norm, zero_tensor = _zero_stats(records, component, group)
            summary["zero_norm_steps"][component][group] = zero_norm
            summary["zero_tensor"][component][group] = zero_tensor
    return summary


# ============================ 门禁判定 ============================
def judge_gate(summary: dict) -> dict:
    """预注册 §2 的门禁。**只**用 shared_trunk 上的 income-vs-marital。"""
    stats = summary["pairs"][pair_key(*GATE_PAIR)][GATE_GROUP]
    rate = stats["neg_cos_rate"]
    mean_cos = stats["mean_cos"]
    nonzero = min(stats["nonzero_rate_a"] or 0.0, stats["nonzero_rate_b"] or 0.0)

    checks = {
        "G0": (summary["n_steps"] >= MIN_SAMPLED_STEPS, summary["n_steps"],
               f">= {MIN_SAMPLED_STEPS}", "采样步数不足（证据不够，不得据此行动）"),
        "G1": (rate is not None and rate >= GATE_NEG_COS_RATE, rate,
               f">= {GATE_NEG_COS_RATE}", "income-vs-marital 负余弦占比（shared_trunk）"),
        "G2": (mean_cos is not None and mean_cos <= GATE_MEAN_COS, mean_cos,
               f"<= {GATE_MEAN_COS}", "income-vs-marital 平均余弦（shared_trunk）"),
        "G3": (nonzero >= GATE_NONZERO_STEP_RATE, nonzero,
               f">= {GATE_NONZERO_STEP_RATE}", "两侧梯度范数同时非零的步占比（shared_trunk）"),
    }
    criteria = {code: {"code": code, "pass": bool(passed), "observed": observed,
                       "requirement": requirement, "note": note}
                for code, (passed, observed, requirement, note) in checks.items()}
    reasons = [f"{item['code']} 未过：{item['note']}；observed={item['observed']}，"
               f"requirement {item['requirement']}"
               for item in criteria.values() if not item["pass"]]

    actionable = all(item["pass"] for item in criteria.values())
    env_diagnostic = {pair_key(a, b): summary["pairs"][pair_key(a, b)]
                      for a, b in CONFLICT_PAIRS if "env" in (a, b)}
    return {
        "status": "ACTIONABLE" if actionable else "NO_ACTION",
        "actionable": actionable,
        "criteria": criteria,
        "reasons": reasons,
        "gate_group": GATE_GROUP,
        "gate_pair": list(GATE_PAIR),
        # 环境目标经梯度反转被**构造性**设成与任务相反，负余弦是机制而非新发现
        "env_diagnostic_only": True,
        "env_diagnostic": env_diagnostic,
        "decision_rule": "G0 且 G1 且 G2 且 G3 → ACTIONABLE（另开分支试 PCGrad）；否则 NO_ACTION（主线继续）",
    }


def smoke_gate() -> dict:
    """smoke 模式的门禁占位：**永远不产生判定**，防止误引用。"""
    return {
        "status": SMOKE_STATUS,
        "actionable": False,
        "criteria": None,
        "reasons": ["smoke 模式（--max-batches）只验证链路，不产生门禁判定，不得当作审计结果"],
        "gate_group": GATE_GROUP,
        "gate_pair": list(GATE_PAIR),
        "env_diagnostic_only": True,
        "env_diagnostic": None,
        "decision_rule": "smoke 模式不判定",
    }


# ============================ 采样节奏 ============================
def sample_schedule(n_batches: int, warmup: int = WARMUP_STEPS, spaced: int = SPACED_STEPS):
    """前 `warmup` 个 batch + 均匀间隔 `spaced` 个 batch（确定性整数公式，去重升序）。

    间隔位置为 `(k * n_batches) // spaced`，k = 0..spaced-1；与 warmup 重叠处自动去重，
    故实际步数为 `len(返回值)`，调用方需按 `EPOCHS_FOR_SCHEDULE` 自行核对是否够 50 步。
    """
    if n_batches <= 0:
        return []
    picked = set(range(min(warmup, n_batches)))
    if spaced > 0:
        picked.update((k * n_batches) // spaced for k in range(spaced))
    return sorted(i for i in picked if 0 <= i < n_batches)


# ============================ 采集器 ============================
class GradientAudit:
    """只读采集器：持有模型的参数/分组视图，按采样点记录标量。

    `model=None` 时先构造、后 `attach(model)`；未绑定就 `record` 会显式抛错，
    绝不允许"静默跳过采集"导致得出空结论。
    """

    def __init__(self, model=None, *, uni_coe, env_coe, verify_identity=True):
        self.uni_coe = float(uni_coe)
        self.env_coe = float(env_coe)
        self.verify_identity = bool(verify_identity)
        self.records: list[dict] = []
        self.model = None
        self.param_names: list[str] = []
        self.params: list[torch.Tensor] = []
        self.groups: dict = {}
        #: 第一个采样步上 `sum(分量)` 与"照抄 train_two_task 写法重算的 total"的相对偏差。
        #: 这不是装饰：它证明审计用的分解**就是**训练目标本身（否则整份审计无效）。
        self.identity_gap = None
        if model is not None:
            self.attach(model)

    @property
    def ready(self) -> bool:
        return self.model is not None and bool(self.param_names)

    def attach(self, model) -> "GradientAudit":
        self.model = model
        self.param_names = [name for name, _ in model.named_parameters()]
        self.params = [param for _, param in model.named_parameters()]
        self.groups = parameter_groups(model)
        return self

    def identity_gap_on(self, output, y_income, y_marital, env_ids, components) -> float:
        """照抄 `train_two_task` 的写法重算 total，与 `sum(分量)` 比相对偏差。

        只用**同一次 forward** 的 `output` 重算，不额外 forward、不消耗 RNG、不改图。
        """
        device = output["gen_preds"][0].device
        y_income = y_income.float().to(device)
        y_marital = y_marital.float().to(device)
        env_ids = env_ids.to(device)
        reference = (_BCE(output["fused_preds"][0], y_income)
                     + _BCE(output["fused_preds"][1], y_marital)
                     + self.uni_coe * (_BCE(output["gen_preds"][0], y_income)
                                       + _BCE(output["gen_preds"][1], y_marital))
                     + self.env_coe * _NLL(output["env_pred"], env_ids)
                     + self.model.get_l2_reg())
        gap = (total_loss(components) - reference).abs()
        return float(gap) / (float(reference.abs()) + 1e-12)

    def record(self, *, epoch, step, alpha, output, y_income, y_marital, env_ids) -> dict:
        """记录一个采样步。只读：不写 .grad、不动优化器、不消耗 RNG。"""
        if not self.ready:
            raise RuntimeError("GradientAudit 尚未绑定模型：请先 attach(model) 或传入 model=...")
        components = decompose_losses(self.model, output, y_income=y_income,
                                      y_marital=y_marital, env_ids=env_ids,
                                      uni_coe=self.uni_coe, env_coe=self.env_coe)
        if self.verify_identity and self.identity_gap is None:
            self.identity_gap = self.identity_gap_on(output, y_income, y_marital, env_ids,
                                                     components)
        grads = component_grads(components, self.params)
        record = step_record(epoch=epoch, step=step, alpha=alpha, grads=grads,
                             param_names=self.param_names, groups=self.groups,
                             params=self.params)
        self.records.append(record)
        return record


# ============================ 训练循环插桩 ============================
class GradientAuditTrainManager(MPTRecTrainManager):
    """`train_two_task` 的**逐字复刻 + 一个只读采样点**。

    为什么是复刻而不是包装：采样需要 `step` / `epoch` / `alpha` / 同一次 forward 的 `output`
    与当批标签，父类循环没有可插入的钩子点。复刻的风险（父类改动后静默漂移）由
    `test_gradient_audit.TestInstrumentationFidelity` 的逐位一致测试兜住：
    `audit=None` 与开启采集两种情形都必须与父类产生 `torch.equal` 的参数。

    逐位保真的关键：插桩复用**同一次 forward** 的 `output`（不额外 forward → 不消耗 RNG），
    且 `component_grads` 用 `retain_graph=True`（不破坏随后 `loss.backward()` 用的图）。
    """

    def __init__(self, *args, audit=None, audit_schedule=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit = audit
        self.audit_schedule = (list(audit_schedule) if audit_schedule is not None
                               else sample_schedule(len(self.train_loader)))
        self._audit_steps = set(self.audit_schedule)
        if audit is not None and not audit.ready:
            audit.attach(self.model)

    def train_two_task(self):
        earlystop_count = 0
        best_auc_score = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            uni_loss_0_sum = 0
            uni_loss_1_sum = 0
            fused_loss_0_sum = 0
            fused_loss_1_sum = 0
            env_loss_sum = 0

            tepoch = tqdm(self.train_loader, unit="batch")
            for step, (y_0, y_1, _, features) in enumerate(tepoch):
                for key in features.keys():
                    features[key] = features[key].to(self.device)
                p = float(step + (epoch - 1) * self.batch_size) / float(
                    self.epochs * self.batch_size
                )
                alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0

                output = self.model(features, alpha)

                batch_env_ids = self.env_ids[
                    self.batch_size * step : self.batch_size * (step + 1)
                ]
                device = output["gen_preds"][0].device
                uni_loss_0 = self.loss_func(output["gen_preds"][0], y_0.float().to(device))
                uni_loss_1 = self.loss_func(output["gen_preds"][1], y_1.float().to(device))
                fused_loss_0 = self.loss_func(
                    output["fused_preds"][0], y_0.float().to(device)
                )
                fused_loss_1 = self.loss_func(
                    output["fused_preds"][1], y_1.float().to(device)
                )
                env_loss = self.env_loss_func(output["env_pred"], batch_env_ids.to(device))
                loss = fused_loss_0 + fused_loss_1 + self.uni_coe * (uni_loss_0 + uni_loss_1) + \
                       self.env_coe * env_loss + self.model.get_l2_reg()

                uni_loss_0_sum += uni_loss_0
                uni_loss_1_sum += uni_loss_1
                fused_loss_0_sum += fused_loss_0
                fused_loss_1_sum += fused_loss_1
                env_loss_sum += env_loss

                # ---- 唯一的插桩点：只读采样，复用同一次 forward 的 output ----
                if self.audit is not None and step in self._audit_steps:
                    self.audit.record(epoch=epoch, step=step, alpha=alpha, output=output,
                                      y_income=y_0, y_marital=y_1, env_ids=batch_env_ids)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            uni_loss_0_sum /= len(self.train_loader)
            uni_loss_1_sum /= len(self.train_loader)
            fused_loss_0_sum /= len(self.train_loader)
            fused_loss_1_sum /= len(self.train_loader)
            env_loss_sum /= len(self.train_loader)

            print(
                "uni_loss_0:{:.4f}, uni_loss_1:{:.4f}, fuse_loss_0:{:.4f}, fuse_loss_1:{:.4f}, env_loss:{:.4f}".format(
                    uni_loss_0_sum,
                    uni_loss_1_sum,
                    fused_loss_0_sum,
                    fused_loss_1_sum,
                    env_loss_sum,
                )
            )

            self.uni_loss_0_list.append(uni_loss_0_sum.item())
            self.uni_loss_1_list.append(uni_loss_1_sum.item())
            self.fused_loss_0_list.append(fused_loss_0_sum.item())
            self.fused_loss_1_list.append(fused_loss_1_sum.item())
            self.env_loss_list.append(env_loss_sum.item())

            if epoch % 2 == 0:
                self.env_ids = self.cluster_2()

            # auc_train = self.evaluation_two_task(self.train_loader)
            auc_val = self.evaluation_two_task(self.val_loader)
            print(
                "Epoch:{}, AUC-Val-{}:{:.4f}, AUC-Val-{}:{:.4f}, ".format(
                    epoch, self.task_name[0], auc_val[0], self.task_name[1], auc_val[1]
                )
            )

            if sum(auc_val) > best_auc_score:
                earlystop_count = 0
                best_auc_score = sum(auc_val)
                self.best_weight = copy.deepcopy(self.model.state_dict())
            else:
                earlystop_count += 1
                print("EarlyStopping count {}".format(earlystop_count))
                if earlystop_count == self.patience:
                    print("EarlyStopping at epoch {}".format(epoch))
                    break


# ============================ 产物 ============================
def build_payload(*, step_records, meta: dict, smoke: bool) -> dict:
    """审计产物：**确定性**（无时间戳/路径/墙钟），同输入逐字节可复现。"""
    summary = aggregate(step_records)
    return {
        "schema": SCHEMA,
        "smoke": bool(smoke),
        "meta": dict(meta),
        "prereg": prereg_block(),
        "steps": list(step_records),
        "aggregate": summary,
        "gate": smoke_gate() if smoke else judge_gate(summary),
        "notes": [
            "梯度手术（PCGrad/GradNorm/CAGrad 等）是既有文献，本审计不主张任何新颖性。",
            "本审计只读梯度，不改变训练更新，不主张任何性能提升。",
            "env 分量含 ReverseLayerF 反转：task-vs-env 的负余弦有设计上的先验，只作诊断。",
            f"未定义的复合量（损失空间不存在可加分解）：{', '.join(UNDEFINED_COMPOSITES)}。",
            "组内余弦按该组所有参数梯度拼接后的整体向量定义，不是逐参数余弦再平均。",
        ],
    }


def interpretation_lines(payload: dict) -> list[str]:
    """结论解读：只用 payload 里已有的数字（预注册 §4：不引入额外数字）。

    为什么把解读写进产物而不是留给读者：门禁未过有两种**截然不同**的成因——
    ①逐步冲突根本没出现（G1 不过）；②冲突出现、但在均值上相互抵消（G1 过、G2 不过）。
    只有后者需要点明"半数步冲突 ≠ 持续的方向性对立"，否则容易被拿去主张梯度手术的必要性。
    """
    if payload["smoke"]:
        return ["smoke 模式：只验证链路，**不产生门禁判定**，不得作为审计结果引用。"]
    gate, summary = payload["gate"], payload["aggregate"]
    stats = summary["pairs"][pair_key(*GATE_PAIR)][GATE_GROUP]
    criteria = gate["criteria"] or {}
    failed = [code for code in ("G0", "G1", "G2", "G3")
              if criteria.get(code) and not criteria[code]["pass"]]
    lines = []
    if gate["actionable"]:
        lines.append("门禁 G0–G3 全过 → `ACTIONABLE`：按预注册允许**另开分支**试 PCGrad；"
                     "本分支不实现、不试，也不主张任何性能提升。")
    else:
        lines.append(f"门禁未过（{', '.join(failed) or 'n/a'}）→ `NO_ACTION`：按预注册"
                     "**不做梯度手术**，主线继续；阈值事后不回改（要改只能新开预注册、从零重跑）。")
        rate, mean_cos = stats["neg_cos_rate"], stats["mean_cos"]
        if (rate is not None and rate > 0.0 and mean_cos is not None
                and criteria.get("G1", {}).get("pass") and not criteria.get("G2", {}).get("pass")
                and stats["min_cos"] is not None and stats["max_cos"] is not None):
            lines.append(
                f"关键区分：负余弦出现在 {rate:.1%} 的采样步（{stats['n_negative']}/{stats['n_defined']}），"
                f"但平均余弦 = {mean_cos:+.6f}（逐步幅度 min {stats['min_cos']:+.3f} / "
                f"max {stats['max_cos']:+.3f}）——冲突是**双向抖动、在均值上相互抵消**，"
                "而非持续的方向性对立；「半数步冲突」本身不构成 PCGrad 的依据。")
        if not criteria.get("G0", {}).get("pass", True):
            lines.append("采样步数不足：证据量不够，本 run 的统计量只作参考，不得据此行动。")
    lines.append("env 相关对（`income_vs_env` / `marital_vs_env`）的负余弦是 `ReverseLayerF` 的"
                 "**机制本身**：只作诊断，不参与门禁，也不构成行动依据。")
    return lines


def _fmt(value, digits=4):
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(payload: dict) -> str:
    """紧凑 markdown 报告（与 JSON 同源，不引入额外数字）。"""
    gate, summary, meta = payload["gate"], payload["aggregate"], payload["meta"]
    stats = summary["pairs"][pair_key(*GATE_PAIR)][GATE_GROUP]
    lines = [
        "# 阶段 1 梯度冲突审计报告",
        "",
        "> 梯度手术（PCGrad / GradNorm / CAGrad / Gradient Vaccine 等）是**既有文献**，"
        "本审计不主张新颖性；审计本身**不改变训练更新**，也不主张任何性能提升。",
        "",
        f"- 状态：**{gate['status']}**（actionable={gate['actionable']}，smoke={payload['smoke']}）",
        f"- 采样步数：{summary['n_steps']}，按 epoch：{summary['n_steps_by_epoch']}",
        f"- 判定组：`{GATE_GROUP}`，判定对：`{pair_key(*GATE_PAIR)}`",
        f"- meta：{json.dumps(meta, ensure_ascii=False, sort_keys=True)}",
        "",
        "## 门禁判据（预注册阈值）",
        "",
        "| 编号 | 观测值 | 要求 | 通过 |",
        "|---|---|---|---|",
    ]
    criteria = gate["criteria"] or {}
    for code in ("G0", "G1", "G2", "G3"):
        item = criteria.get(code)
        if item is None:
            lines.append(f"| {code} | - | - | 未评估（smoke） |")
        else:
            lines.append(f"| {code} | {item['observed']} | {item['requirement']} | "
                         f"{'是' if item['pass'] else '否'} |")
    lines += [
        "",
        f"- 未过原因：{'; '.join(gate['reasons']) if gate['reasons'] else '无'}",
        f"- 决策规则：{gate['decision_rule']}",
        "  未过门禁即 `NO_ACTION`：不做梯度手术，主线继续。",
        "",
        "## 结论解读",
        "",
    ]
    lines += [f"- {line}" for line in interpretation_lines(payload)]
    lines += [
        "",
        "## income-vs-marital（shared_trunk，门禁所用）",
        "",
        "| 统计量 | 值 |",
        "|---|---|",
        f"| mean_cos | {_fmt(stats['mean_cos'], 6)} |",
        f"| neg_cos_rate | {_fmt(stats['neg_cos_rate'], 6)} |",
        f"| n_negative / n_defined | {stats['n_negative']} / {stats['n_defined']} |",
        f"| nonzero_rate_a (income) | {_fmt(stats['nonzero_rate_a'], 6)} |",
        f"| nonzero_rate_b (marital) | {_fmt(stats['nonzero_rate_b'], 6)} |",
        f"| mean_of_ratio (‖g_a‖/‖g_b‖) | {_fmt(stats['mean_of_ratio'], 6)} |",
        f"| ratio_of_means | {_fmt(stats['ratio_of_means'], 6)} |",
        f"| mean_dominance_ratio | {_fmt(stats['mean_dominance_ratio'], 6)} |",
        f"| mean_norm_a / mean_norm_b | {_fmt(stats['mean_norm_a'], 6)} / {_fmt(stats['mean_norm_b'], 6)} |",
        "",
        "## 各参数组（income-vs-marital）",
        "",
        "| 组 | mean_cos | neg_cos_rate | nonzero_a | nonzero_b | dominance |",
        "|---|---|---|---|---|---|",
    ]
    for group in REPORTED_GROUPS:
        item = summary["pairs"][pair_key(*GATE_PAIR)][group]
        lines.append(f"| `{group}` | {_fmt(item['mean_cos'], 6)} | {_fmt(item['neg_cos_rate'], 6)} | "
                     f"{_fmt(item['nonzero_rate_a'], 6)} | {_fmt(item['nonzero_rate_b'], 6)} | "
                     f"{_fmt(item['mean_dominance_ratio'], 6)} |")
    lines += [
        "",
        "## 环境分量（**仅诊断**，不参与门禁）",
        "",
        "| 对 | 组 | mean_cos | neg_cos_rate |",
        "|---|---|---|---|",
    ]
    for key in (pair_key("income", "env"), pair_key("marital", "env")):
        for group in REPORTED_GROUPS:
            item = summary["pairs"][key][group]
            lines.append(f"| `{key}` | `{group}` | {_fmt(item['mean_cos'], 6)} | "
                         f"{_fmt(item['neg_cos_rate'], 6)} |")
    lines += [
        "",
        "env 支路经 `ReverseLayerF` 反转，与任务的负余弦是**机制本身**而非新发现；"
        "因此环境冲突单独不构成任务间 PCGrad 的依据。",
        "",
        "## 零梯度计数（精确零：embedding 行未被 batch 命中 / 不可达参数 / alpha=0）",
        "",
        "| 分量 | 组 | 范数为零的步（by epoch） | 占比 | 零梯度张量对数 | 出现的步占比 |",
        "|---|---|---|---|---|---|",
    ]
    for component in COMPONENTS:
        for group in REPORTED_GROUPS:
            z_norm = summary["zero_norm_steps"][component][group]
            z_tensor = summary["zero_tensor"][component][group]
            lines.append(f"| `{component}` | `{group}` | {z_norm['by_epoch']} | "
                         f"{_fmt(z_norm['rate'], 6)} | {z_tensor['total']} | {_fmt(z_tensor['frac'], 6)} |")
    lines += ["", "## 说明", ""]
    lines += [f"- {note}" for note in payload["notes"]]
    lines += ["", f"- 预注册：`{payload['prereg']['spec']}`"]
    return "\n".join(lines) + "\n"


def write_audit(run_dir: Path, payload: dict, prereg_text: str = "", stdout_text: str = "") -> Path:
    """落盘 audit.json（确定性）+ report.md + prereg.md 快照 + stdout.log（非确定性内容只进 log）。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (run_dir / "report.md").write_text(render_markdown(payload), encoding="utf-8")
    if prereg_text:
        (run_dir / "prereg.md").write_text(prereg_text, encoding="utf-8")
    if stdout_text:
        (run_dir / "stdout.log").write_text(stdout_text, encoding="utf-8")
    return run_dir


def append_summary_row(path: Path, row: dict) -> None:
    """只追加，永不重写已有行（含 NO_ACTION 的 run，不得只报告成功的 run）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("| " + " | ".join(SUMMARY_COLUMNS) + " |\n"
                        + "|" + "---|" * len(SUMMARY_COLUMNS) + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(str(row[column]) for column in SUMMARY_COLUMNS) + " |\n")


def summary_row(*, run_id: str, commit: str, payload: dict) -> dict:
    gate, summary = payload["gate"], payload["aggregate"]
    stats = summary["pairs"][pair_key(*GATE_PAIR)][GATE_GROUP]
    nonzero = min(stats["nonzero_rate_a"] or 0.0, stats["nonzero_rate_b"] or 0.0)
    return {
        "run_id": run_id, "commit": commit, "smoke": payload["smoke"],
        "n_steps": summary["n_steps"], "n_epochs": len(summary["n_steps_by_epoch"]),
        "neg_cos_rate": _fmt(stats["neg_cos_rate"]), "mean_cos": _fmt(stats["mean_cos"]),
        "nonzero_rate": _fmt(nonzero), "status": gate["status"],
    }
