"""CensusIncome 阶段 2 低秩残差适配（LoRA-style residual adapter）——独立实验模块，零协议改动。

预注册文档：docs/superpowers/specs/2026-09-30-stage2-lora-adapter-design.md
父协议：census_benchmark/protocol.py + census_benchmark/metrics.py（只读引用，绝不修改）

机制（预注册，只此一种）：

    h' = h + scale · up(down(h)),   scale = alpha / rank = 1.0

- 只作用于**冻结的 Stage-1 表征**（`gen_rep` 与每个 `spec_rep`），在既有 NewTask 融合之前；
- gen / 各 source 三路**共享同一份**适配器权重（参数效率 + 单一机制可解释性，见文档 §2.3）；
- `up` 零初始化 → 初始 delta ≡ 0 → 初始预测与基线逐元素相等（torch.equal）；
- 不给 backbone 增参、不原地改写 backbone 的 Linear 权重 → backbone_sha256 不变、backbone 梯度恒为 None；
- 适配器初始化包在 `isolated_cpu_rng()` 内 → 构造结束后全局 RNG 端点与基线一致
  （与 `protocol.make_env_ids` 用独立 Generator 是同一思路，spec 5.3.2）。

**本模块不主张方法新颖性**：低秩 / 残差适配由已发表工作提出（Rebuffi et al. 2017；Houlsby et al. 2019；
Hu et al. 2022），这里只做协议内的工程消融。
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from multitaskrec.model import NewTask

# ---- 预注册常量（看到结果之前不得修改；改动必须同步文档并单独提交）----
RANK = 4
ALPHA = 4.0
SCALE = ALPHA / RANK                      # 1.0：LoRA 惯例 alpha/r
RATIO_MIN, RATIO_MAX = 0.005, 0.5         # R1：适配器输出占比的合格带
GRAD_EPS = 0.0                            # 梯度“非零”判据：范数 > GRAD_EPS
AUC_TEST_MIN = 0.8521                     # S1：主指标接受阈值
BASELINE_AUC = 0.8500685307175756         # 基线 AUC-Test-Education（见 BASELINE_RUN_ID）
BASELINE_RUN_ID = "20260929-1735-s20260929-m1685480945-short-904f8d0"
BASELINE_STAGE1_ID = "s1-096f8f16-m1685480945-e2-cb2094b3"
MECHANISM = "lora-r4-residual-adapter"


def verdict(passed: bool, **detail) -> dict:
    """与 metrics.judge 内部同形的判定结构（本模块自带，避免改动 metrics.py）。"""
    return {"pass": bool(passed), "detail": detail}


@contextlib.contextmanager
def isolated_cpu_rng():
    """块内消耗的全局 CPU 随机性不影响块外：进入时保存 RNG 现场，退出时原样恢复。

    与 `protocol.make_env_ids` 用独立 Generator 是同一目的（spec 5.3.2：不干扰模型初始化的随机性）。
    此处需要复用全局 RNG 的默认分布（`nn.Linear` 的标准初始化），故用“保存—恢复”。
    LoRANewTask/LowRankResidualAdapter 的构造都在 CPU 上创建参数，所以只需 CPU RNG；
    若将来改成在 CUDA 上构造模块，此上下文不足以隔离 CUDA RNG。
    """
    state = torch.get_rng_state()
    try:
        yield
    finally:
        torch.set_rng_state(state)


class LowRankResidualAdapter(nn.Module):
    """h' = h + scale · up(down(h))；`up` 零初始化 → 初始为恒等映射。"""

    def __init__(self, dim: int, rank: int = RANK, alpha: float = ALPHA):
        super().__init__()
        if int(rank) < 1:
            raise ValueError(f"rank 必须 ≥ 1，收到 {rank}")
        if int(dim) < 1:
            raise ValueError(f"dim 必须 ≥ 1，收到 {dim}")
        self.dim, self.rank, self.alpha = int(dim), int(rank), float(alpha)
        self.scale = self.alpha / self.rank
        self.down = nn.Linear(self.dim, self.rank, bias=False)   # A：(r, d)，默认 Linear 初始化
        self.up = nn.Linear(self.rank, self.dim, bias=False)     # B：(d, r)
        nn.init.zeros_(self.up.weight)                           # B=0 → delta ≡ 0（初始逐元素等同基线）

    def delta(self, h: torch.Tensor) -> torch.Tensor:
        """残差项 scale · B(A(h))，形状与 h 相同。"""
        return self.scale * self.up(self.down(h))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.delta(h)

    @torch.no_grad()
    def effective_fro(self) -> float:
        """‖scale · B A‖_F：等效线性映射（未加恒等项）的 Frobenius 范数。"""
        return float(self.scale * (self.up.weight @ self.down.weight).norm())

    def extra_repr(self) -> str:
        return f"dim={self.dim}, rank={self.rank}, alpha={self.alpha}, scale={self.scale}"


class LoRANewTask(NewTask):
    """共享一份低秩残差适配器的 NewTask；除适配器外结构与语义与 NewTask 完全一致。"""

    def __init__(self, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                 rank: int = RANK, alpha: float = ALPHA):
        super().__init__(input_size=input_size, rep_dim=rep_dim,
                         tower_dnn_hidden_units=tower_dnn_hidden_units,
                         reg_dnn=reg_dnn, device=device)
        # 顺序关键：先 super().__init__()（与基线 NewTask 消费同一段 RNG 前缀 → 共享参数逐元素相等），
        # 再构造适配器；isolated_cpu_rng 隔离其初始化消耗并在退出时恢复现场
        # → 真实构造结束后全局 RNG 端点仍与基线一致（见模块 docstring）。
        with isolated_cpu_rng():
            self.rep_adapter = LowRankResidualAdapter(rep_dim, rank=rank, alpha=alpha)

    def _adapt(self, gen_rep, spec_reps):
        return self.rep_adapter(gen_rep), [self.rep_adapter(spec) for spec in spec_reps]

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs):
        """签名与 NewTask.forward 完全一致 → metrics.evaluate_newtask 无需改动即可复用。"""
        adapted_gen, adapted_specs = self._adapt(gen_rep, spec_reps)
        return super().forward(dnn_input, adapted_gen, adapted_specs, env_embs)

    def adapt_with_delta(self, gen_rep, spec_reps):
        """诊断用：返回 (适配后 gen, 适配后 specs, gen 的 delta, 各 spec 的 delta)。"""
        gen_delta = self.rep_adapter.delta(gen_rep)
        spec_deltas = [self.rep_adapter.delta(spec) for spec in spec_reps]
        return (gen_rep + gen_delta, [spec + delta for spec, delta in zip(spec_reps, spec_deltas)],
                gen_delta, spec_deltas)


def param_report(newtask: LoRANewTask) -> dict:
    """可训练参数量与占比：适配器 / 新任务头（不含适配器）/ 合计。"""
    adapter_params = sum(param.numel() for param in newtask.rep_adapter.parameters())
    total = sum(param.numel() for param in newtask.parameters())
    head = total - adapter_params
    return {"mechanism": MECHANISM,
            "dim": newtask.rep_adapter.dim, "rank": newtask.rep_adapter.rank,
            "alpha": newtask.rep_adapter.alpha, "scale": newtask.rep_adapter.scale,
            "adapter_params": int(adapter_params), "newtask_head_params": int(head),
            "newtask_total_params": int(total),
            "adapter_ratio_of_newtask": adapter_params / max(total, 1)}


def construction_identity_report(*, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn,
                                 rank: int = RANK, alpha: float = ALPHA) -> dict:
    """从同一 RNG 起点分别构造基线 NewTask 与 LoRANewTask，验证构造期的两条恒等性（A6）。

    两个构造都在 isolated_cpu_rng 内完成 → 本函数不消耗全局 RNG，也不影响真实构造的后续随机性。
    NewTask/LoRANewTask 的模块参数一律先在 CPU 上创建（再 `.to(device)`），故此处固定用 CPU
    构造，结论与真实构造（任意 device）一致，且不会在 CUDA 上留下临时显存。
    """
    cpu = torch.device("cpu")
    with isolated_cpu_rng():
        baseline = NewTask(input_size=input_size, rep_dim=rep_dim,
                           tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn, device=cpu)
        baseline_endpoint = torch.get_rng_state().clone()
    with isolated_cpu_rng():
        adapted = LoRANewTask(input_size=input_size, rep_dim=rep_dim,
                              tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn,
                              device=cpu, rank=rank, alpha=alpha)
        adapted_endpoint = torch.get_rng_state().clone()

    baseline_state, adapted_state = baseline.state_dict(), adapted.state_dict()
    extra_keys = sorted(key for key in adapted_state if key not in baseline_state)
    shared_identical = all(torch.equal(adapted_state[key], value) for key, value in baseline_state.items())
    return {"shared_params_bit_identical": bool(shared_identical),
            "extra_keys": extra_keys,
            "global_rng_endpoint_identical": bool(torch.equal(baseline_endpoint, adapted_endpoint)),
            "baseline_state_numel": int(sum(v.numel() for v in baseline_state.values())),
            "adapted_state_numel": int(sum(v.numel() for v in adapted_state.values()))}


def actual_instance_identity_report(newtask, *, rng_state_before, rng_state_after, input_size, rep_dim,
                                    tower_dnn_hidden_units, reg_dnn, device=None) -> dict:
    """针对**真实构造出来的** LoRANewTask 实例做恒等审计（A6 的强证据）。

    调用约定：`state = torch.get_rng_state()` → 构造 `LoRANewTask` → `after = torch.get_rng_state()`
    → 把两者传进来。本函数在 isolated_cpu_rng 内把全局 RNG 拨回 `state`，用同一顺序构造基线
    NewTask 并逐键比较；退出时恢复现场，故**审计本身不改变全局 RNG**，也不影响其后的训练。
    """
    with isolated_cpu_rng():
        torch.set_rng_state(rng_state_before)
        baseline = NewTask(input_size=input_size, rep_dim=rep_dim,
                           tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn, device=device)
        baseline_endpoint = torch.get_rng_state().clone()
    baseline_state, actual_state = baseline.state_dict(), newtask.state_dict()
    extra_keys = sorted(key for key in actual_state if key not in baseline_state)
    shared_identical = all(torch.equal(actual_state[key].detach().cpu(), value)
                           for key, value in baseline_state.items())
    return {"shared_params_bit_identical": bool(shared_identical),
            "extra_keys": extra_keys,
            "global_rng_endpoint_identical": bool(torch.equal(baseline_endpoint, rng_state_after)),
            "up_fro_at_construction": float(newtask.rep_adapter.up.weight.detach().norm()),
            "baseline_state_numel": int(sum(v.numel() for v in baseline_state.values())),
            "adapted_state_numel": int(sum(v.numel() for v in actual_state.values()))}


class AdapterStats:
    """逐流统计适配器输出占比与余弦：fp64、CPU 累加、显存 O(1)（与 metrics.GateStats 同口径）。

    流顺序 = ("gen", "spec_0", ..., "spec_{num_tasks-1}")，统计点在**注入点**（融合之前）。
    占比定义：ratio_i = ‖delta_i‖₂ / max(‖h_i‖₂, denom_floor)，逐样本算完再取均值。
    """

    def __init__(self, num_tasks: int, denom_floor: float = 1e-12):
        self.names = ("gen",) + tuple(f"spec_{i}" for i in range(num_tasks))
        self.denom_floor = float(denom_floor)
        n = len(self.names)
        self.ratio_sum = torch.zeros(n, dtype=torch.float64)
        self.ratio_sq_sum = torch.zeros(n, dtype=torch.float64)
        self.ratio_max = torch.zeros(n, dtype=torch.float64)
        self.cos_sum = torch.zeros(n, dtype=torch.float64)
        self.delta_norm_sum = torch.zeros(n, dtype=torch.float64)
        self.rep_norm_sum = torch.zeros(n, dtype=torch.float64)
        self.count = 0
        self.n_zero_rep = 0

    @torch.no_grad()
    def update(self, gen_rep, gen_delta, spec_reps, spec_deltas) -> None:
        reps = (gen_rep,) + tuple(spec_reps)
        deltas = (gen_delta,) + tuple(spec_deltas)
        for i, (rep, delta) in enumerate(zip(reps, deltas)):
            rep64, delta64 = rep.detach().double(), delta.detach().double()
            rep_norm, delta_norm = rep64.norm(dim=1), delta64.norm(dim=1)
            self.n_zero_rep += int((rep_norm <= self.denom_floor).sum())
            ratio = delta_norm / rep_norm.clamp_min(self.denom_floor)
            # 累加器恒在 CPU：先在原设备归约，再显式 .cpu()（同 metrics.GateStats 的写法）
            self.ratio_sum[i] += ratio.sum().cpu()
            self.ratio_sq_sum[i] += (ratio * ratio).sum().cpu()
            self.ratio_max[i] = torch.maximum(self.ratio_max[i], ratio.max().cpu())
            self.cos_sum[i] += F.cosine_similarity(delta64, rep64, dim=1, eps=self.denom_floor).sum().cpu()
            self.delta_norm_sum[i] += delta_norm.sum().cpu()
            self.rep_norm_sum[i] += rep_norm.sum().cpu()
        self.count += int(gen_rep.shape[0])

    def result(self) -> dict:
        n = max(self.count, 1)
        mean = self.ratio_sum / n
        std = (self.ratio_sq_sum / n - mean * mean).clamp_min(0).sqrt()
        return {"streams": list(self.names),
                "ratio_mean": mean.tolist(), "ratio_std": std.tolist(),
                "ratio_max": self.ratio_max.tolist(),
                "cos_mean": (self.cos_sum / n).tolist(),
                "delta_norm_mean": (self.delta_norm_sum / n).tolist(),
                "rep_norm_mean": (self.rep_norm_sum / n).tolist(),
                "count": int(self.count), "n_zero_rep": int(self.n_zero_rep)}


@torch.no_grad()
def evaluate_adapter_stats(newtask: LoRANewTask, backbone, loader, device) -> dict:
    """val 集上逐流统计适配器输出（独立一遍前向；不改动 metrics.evaluate_newtask）。"""
    newtask.eval()
    backbone.eval()
    stats = AdapterStats(num_tasks=len(backbone.specific_expert_networks))
    for _, _, _, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        _, gen_rep, spec_reps, _ = backbone.get_infos(features)          # 三件套之 3：no_grad 抽取
        _, _, gen_delta, spec_deltas = newtask.adapt_with_delta(gen_rep, spec_reps)
        stats.update(gen_rep, gen_delta, spec_reps, spec_deltas)
    return stats.result()


def _grad_norm(param: torch.nn.Parameter):
    return None if param.grad is None else float(param.grad.detach().norm())


class AdapterGradProbe:
    """记录适配器梯度范数：全局第 1 次 backward（更新前）与每个 epoch 的末次 backward。"""

    def __init__(self, adapter_module: LowRankResidualAdapter):
        self.adapter = adapter_module
        self.first: dict | None = None
        self.last: dict | None = None
        self.per_epoch: list[dict] = []

    def observe(self, *, epoch: int, step: int, is_first: bool = False) -> dict:
        row = {"epoch": int(epoch), "step": int(step),
               "down_grad_norm": _grad_norm(self.adapter.down.weight),
               "up_grad_norm": _grad_norm(self.adapter.up.weight)}
        self.last = row
        if is_first:
            self.first = row
        return row

    def end_epoch(self) -> None:
        if self.last is not None:
            self.per_epoch.append(self.last)


@torch.no_grad()
def adapter_norms(adapter_module: LowRankResidualAdapter) -> dict:
    return {"down_fro": float(adapter_module.down.weight.norm()),
            "up_fro": float(adapter_module.up.weight.norm()),
            "effective_fro": adapter_module.effective_fro(),
            "scale": adapter_module.scale, "rank": adapter_module.rank, "dim": adapter_module.dim}


# ---- 实验门禁（写在 gate_report.json 的 "adapter" 子对象里，不并入协议 A/B 的 overall_pass）----
def identity_gate(report: dict) -> dict:
    """A6：构造期恒等——共享参数逐元素相等 + 全局 RNG 端点一致 + 新增键仅为适配器。"""
    only_adapter = report["extra_keys"] == ["rep_adapter.down.weight", "rep_adapter.up.weight"]
    return verdict(report["shared_params_bit_identical"] and report["global_rng_endpoint_identical"]
                   and only_adapter,
                   shared_params_bit_identical=report["shared_params_bit_identical"],
                   global_rng_endpoint_identical=report["global_rng_endpoint_identical"],
                   extra_keys=report["extra_keys"])


def ratio_gate(stats: dict) -> dict:
    """R1：每一路的平均输出占比都 ∈ [RATIO_MIN, RATIO_MAX]（未静默，也未反客为主）。"""
    ratios = list(stats["ratio_mean"])
    ok = bool(ratios) and all(RATIO_MIN <= ratio <= RATIO_MAX for ratio in ratios)
    return verdict(ok, streams=list(stats["streams"]), ratio_mean=ratios,
                   ratio_min=RATIO_MIN, ratio_max=RATIO_MAX,
                   ratio_max_per_stream=list(stats["ratio_max"]),
                   n_zero_rep=int(stats["n_zero_rep"]))


def grad_gate(first: dict | None, per_epoch: list[dict]) -> dict:
    """G：适配器梯度活性。

    预期（B=0 的固有性质）：第 1 次 backward 时 up（零初始化）梯度非零、down 梯度**恒为 0**
    （dL/dA ∝ B = 0）；一旦 up 被更新（≥1 步），down 梯度即非零。二者与“backbone 梯度全为 None”
    共同构成本次适配确实在工作的证据；若第 1 步 down 梯度非零，说明 B 不是零初始化（恒等性已破）。
    """
    last = per_epoch[-1] if per_epoch else None
    up_first = None if first is None else first["up_grad_norm"]
    down_first = None if first is None else first["down_grad_norm"]
    down_last = None if last is None else last["down_grad_norm"]
    up_first_ok = up_first is not None and up_first > GRAD_EPS
    down_first_zero = down_first == 0.0
    down_late_ok = down_last is not None and down_last > GRAD_EPS
    return verdict(up_first_ok and down_first_zero and down_late_ok,
                   first_backward=first, last_epoch_backward=last,
                   up_grad_nonzero_at_first_step=bool(up_first_ok),
                   down_grad_zero_at_first_step_expected=bool(down_first_zero),
                   down_grad_nonzero_after_updates=bool(down_late_ok))
