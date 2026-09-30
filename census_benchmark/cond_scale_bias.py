"""阶段 1：Router 条件化 Scale+Bias（**消融 / 工程诊断**，见 spec 第 5 节）。

唯一事实来源：docs/superpowers/specs/2026-09-30-stage2-cond-scale-bias-design.md

公式（逐样本；$x$ = `dnn_input`，$H(x)$ = `projection_network(x)`，$E$ = stack(env_embs, 1)）：

    W(x)     = softmax(H(x) E / T)                      # 既有源 router，原样保留
    e_new(x) = Σ_k W_k(x) spec_rep_k                    # 既有 router 融合（基线 new_spec_rep），原样保留
    γ(x)     = Σ_k W_k(x) γ_k,  β(x) = Σ_k W_k(x) β_k   # 本消融的唯一新增量（γ, β ∈ R^{K×r}）
    env_aware = s ⊙ (e_new(x) + γ(x)) + β(x)            # s = 既有 new_env_emb（既有 scale，原样保留）

γ=β=0（零初始化）⇒ env_aware = s ⊙ e_new，与基线**逐位一致**（测试锁定：共享参数 / 前向 / 首个训练步梯度）。

硬约束：
  * **不改 `multitaskrec/model.py`**（spec P2）：本模块以**子类**扩展 NewTask，只重写 forward。
  * **不影响其它分支 / 基线臂**：默认路径仍是基线 NewTask；基线臂的 config / metrics 键集不变。
  * **不主张新颖性**：条件化调制（FiLM / 条件化归一化 / prompt 调制）是已有大类，本方向只回答
    "在有样本级差异的 router 之上加一路条件化 scale+bias 是否带来可复现收益"。

用法（接线见 run_census_benchmark.run_stage2）：

    python run_census_benchmark.py stage2 --stage1-dir <stage1_dir> --variant cond-scale-bias --tag short
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from census_benchmark import protocol as P
from multitaskrec.model import NewTask

# ---- 臂与接线（spec 5.5）----
BASELINE_VARIANT = "baseline"
VARIANT = "cond-scale-bias"
VARIANTS = (BASELINE_VARIANT, VARIANT)
RUN_ID_SUFFIX = "-csb"                     # 沿用 null-expert / attn 分支的"协议 run_id + 臂后缀"约定

# ---- 预注册判据（spec 5.3）：只允许在看到结果之前修改 ----
AUC_TEST_MIN = 0.8521                      # C1：AUC-Test-Education 下限（基线实测 0.850069）
MODULATION_RATIO_BOUNDS = (0.01, 1.0)      # C2：有效调制比 sanity 带（含端点）


class CondScaleBiasNewTask(NewTask):
    """`NewTask` 的子类：在既有 router 融合表征上加 router 条件化的 scale+bias。

    `num_envs` 必须等于冻结 backbone 的 env 数（CensusIncome 为 2）：γ/β 的行数即 env 数，
    前向遇到不一致的 env 数直接 `ValueError` 拒绝出数（同 `router_probe.probe_loader` 的口径）。
    """

    def __init__(self, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn, device=None,
                 num_envs: int = P.NUM_ENVS):
        super().__init__(input_size=input_size, rep_dim=rep_dim,
                         tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn, device=device)
        self.num_envs = int(num_envs)
        # 零初始化且不消耗全局 RNG（zeros / Parameter 都不抽样）→ 构造顺序与基线逐位同序
        self.gamma = nn.Parameter(torch.zeros(self.num_envs, rep_dim))
        self.beta = nn.Parameter(torch.zeros(self.num_envs, rep_dim))

    def router_weights(self, dnn_input: torch.Tensor, env_embs: list[torch.Tensor]) -> torch.Tensor:
        """既有源 router：与基线 `NewTask.forward`、`router_probe.router_weights` 同一表达式。"""
        exist_env_embs = torch.stack(env_embs, dim=1)
        if exist_env_embs.shape[-1] != self.num_envs:
            raise ValueError(f"env 数 {exist_env_embs.shape[-1]} 与本模块 γ/β 的 {self.num_envs} 不一致，拒绝出数")
        h_out = self.projection_network(dnn_input)
        return F.softmax(torch.mm(h_out, exist_env_embs) / self.temperature, dim=-1)

    def modulation_terms(self, dnn_input: torch.Tensor, spec_reps: list[torch.Tensor],
                         env_embs: list[torch.Tensor]) -> dict:
        """γ(x)/β(x) 与它们所调制的 e_new(x)（纯函数：无副作用、不改参数；诊断与 forward 共用）。"""
        w = self.router_weights(dnn_input, env_embs)                                   # [B, K]
        e_new = torch.matmul(torch.stack(spec_reps, dim=2), w.unsqueeze(2)).squeeze()  # [B, r]
        return {"w": w, "e_new": e_new,
                "gamma_x": torch.mm(w, self.gamma), "beta_x": torch.mm(w, self.beta)}

    def forward(self, dnn_input, gen_rep, spec_reps, env_embs):
        new_env_emb = self.env_embedding_network(self.new_env_idx).squeeze(0)          # 既有 scale s，原样保留
        terms = self.modulation_terms(dnn_input, spec_reps, env_embs)
        env_aware_rep = new_env_emb * (terms["e_new"] + terms["gamma_x"]) + terms["beta_x"]
        gate_out = self.gate_network(dnn_input).unsqueeze(dim=2)                       # 既有 gate，原样保留
        all_reps = torch.stack([env_aware_rep, gen_rep], dim=2)
        fused_rep = torch.matmul(all_reps, gate_out).squeeze()
        return self.tower_network(fused_rep).squeeze()                                 # 既有 tower，原样保留


def _row_norms(tensor: torch.Tensor) -> torch.Tensor:
    """逐样本 L2 范数（CPU float64；1-D 输入视作单样本）——累加器的设备/精度教训同 GateStats。"""
    return torch.atleast_2d(tensor.detach().double()).norm(dim=1).cpu()


class ModulationStats:
    """调制项的样本级统计（spec 5.2；流式累加，显存 O(1)）。

    口径：
      * `gamma_x_norm_mean` / `beta_x_norm_mean`：mean_b ‖γ(x)_b‖₂ / ‖β(x)_b‖₂
      * `e_new_norm_mean`：mean_b ‖e_new(x)_b‖₂（调制的参照尺度）
      * `modulation_ratio = gamma_x_norm_mean / e_new_norm_mean`：**有效调制比**（C2 判据）
      * `bias_ratio = beta_x_norm_mean / e_new_norm_mean`
    退化输入 `e_new_norm_mean = 0`：γ 也为 0 记 0.0，否则记 `inf`（判失败，见 spec 5.2）。
    """

    def __init__(self) -> None:
        self.gamma_sum = self.beta_sum = self.e_new_sum = 0.0
        self.count = 0

    def update(self, terms: dict) -> None:
        self.gamma_sum += float(_row_norms(terms["gamma_x"]).sum())
        self.beta_sum += float(_row_norms(terms["beta_x"]).sum())
        self.e_new_sum += float(_row_norms(terms["e_new"]).sum())
        self.count += int(torch.atleast_2d(terms["gamma_x"]).shape[0])

    def result(self) -> dict:
        n = max(self.count, 1)
        gamma_norm, beta_norm, e_new_norm = self.gamma_sum / n, self.beta_sum / n, self.e_new_sum / n

        def ratio(numerator: float) -> float:
            if e_new_norm > 0:
                return numerator / e_new_norm
            return 0.0 if numerator == 0 else float("inf")          # 有调制却无表征 → 判失败

        return {"gamma_x_norm_mean": gamma_norm, "beta_x_norm_mean": beta_norm,
                "e_new_norm_mean": e_new_norm, "modulation_ratio": ratio(gamma_norm),
                "bias_ratio": ratio(beta_norm), "n_samples": self.count}


@torch.no_grad()
def param_norms(newtask: CondScaleBiasNewTask) -> dict:
    """逐 env 的 γ/β 参数范数（不随样本变化）：分辨"参数是否真的离开零点"与"是否只有单侧在动"。"""
    return {"gamma_param_norm_per_env": newtask.gamma.detach().double().norm(dim=1).cpu().tolist(),
            "beta_param_norm_per_env": newtask.beta.detach().double().norm(dim=1).cpu().tolist()}


@torch.no_grad()
def evaluate_modulation(newtask, backbone, loader, device) -> dict:
    """在给定划分上复算调制项统计（spec 5.5 公平性）。

    调用点固定在**选点完成、best_state 已载入之后**，只用 val、只前向：不训练、不反传、
    不参与 early stop / 选点 / test 评测，因此不可能改变 A/B 门禁或 C1 的结果。
    """
    if not isinstance(newtask, CondScaleBiasNewTask):
        raise ValueError(f"调制诊断只适用于 {CondScaleBiasNewTask.__name__}，收到 {type(newtask).__name__}")
    newtask.eval()
    backbone.eval()                                                       # 三件套之 2：backbone 恒为 eval
    stats = ModulationStats()
    for _, _, _, features in loader:
        features = {key: value.to(device) for key, value in features.items()}
        dnn_input, _, spec_reps, env_embs = backbone.get_infos(features)  # 三件套之 3：no_grad 抽表征
        stats.update(newtask.modulation_terms(dnn_input, spec_reps, env_embs))
    return {**stats.result(), **param_norms(newtask)}


def modulation_ratio_verdict(ratio: float, bounds=MODULATION_RATIO_BOUNDS) -> dict:
    """C2（spec 5.3）：有效调制比必须落在带内（含端点）。带外即判失败并记录 observed。"""
    low, high = bounds
    return {"rule": f"{low} <= modulation_ratio <= {high}", "bounds": [float(low), float(high)],
            "observed": float(ratio), "pass": bool(low <= ratio <= high)}


def arm_verdict(test_auc: float, modulation_ratio: float, *, auc_min: float = AUC_TEST_MIN,
                bounds=MODULATION_RATIO_BOUNDS) -> dict:
    """本消融的臂级判定（spec 5.3）：C1 与 C2 **同时**满足才通过；不并入 judge() 的 overall_pass。"""
    c1 = {"rule": f"auc_test_education >= {auc_min}", "threshold": float(auc_min),
          "observed": float(test_auc), "pass": bool(test_auc >= auc_min)}
    c2 = modulation_ratio_verdict(modulation_ratio, bounds)
    return {"C1": c1, "C2": c2, "pass": bool(c1["pass"] and c2["pass"])}


def build_newtask(variant: str, *, input_size, rep_dim, tower_dnn_hidden_units, reg_dnn,
                  device=None, num_envs: int = P.NUM_ENVS):
    """变体工厂：`baseline` → 基线 `NewTask`（逐位同 master）；`cond-scale-bias` → 本消融的处理臂。"""
    if variant == BASELINE_VARIANT:
        return NewTask(input_size=input_size, rep_dim=rep_dim,
                       tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn, device=device)
    if variant == VARIANT:
        return CondScaleBiasNewTask(input_size=input_size, rep_dim=rep_dim,
                                    tower_dnn_hidden_units=tower_dnn_hidden_units, reg_dnn=reg_dnn,
                                    device=device, num_envs=num_envs)
    raise ValueError(f"未知 variant: {variant!r}（可选 {VARIANTS}）")
